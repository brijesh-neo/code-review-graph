"""``check-rules`` flags forbidden dependencies and community import rings."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path
from code_review_graph.rules import RulesError, _glob_regex, check_rules

FILES = {
    "src/ui/view.py": "from src.db.store import save\n\n\ndef show():\n    return save(1)\n",
    "src/db/store.py": "def save(x):\n    return x\n",
    "src/core/logic.py": "from src.db.store import save\n\n\ndef run():\n    return save(2)\n",
}

RULES = '''
[[forbid]]
name = "ui does not touch db"
from = "src/ui/**"
to = "src/db/**"
'''


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for rel, text in FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    for pkg in ("src", "src/ui", "src/db", "src/core"):
        (repo / pkg / "__init__.py").write_text("", encoding="utf-8")
    store = GraphStore(get_db_path(repo))
    try:
        full_build(repo, store)
    finally:
        store.close()
    (repo / ".code-review-graph" / "rules.toml").write_text(RULES, encoding="utf-8")
    return repo.resolve()


def test_glob_semantics() -> None:
    assert _glob_regex("src/ui/**").match("src/ui/a/b.py")
    assert not _glob_regex("src/ui/*").match("src/ui/a/b.py")
    assert _glob_regex("**/test_*.py").match("test_x.py")
    assert not _glob_regex("src/ui/**").match("src/uix/a.py")


def test_forbidden_import_and_call_are_reported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    report = check_rules(repo)
    hits = {(v["kind"], v["file"]) for v in report["violations"]}
    assert ("IMPORTS_FROM", "src/ui/view.py") in hits
    assert ("CALLS", "src/ui/view.py") in hits
    assert all(v["file"] != "src/core/logic.py" for v in report["violations"])

    assert check_rules(repo, changed_files=["src/core/logic.py"])["violations"] == []


def test_community_ring_is_reported(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    rules = repo / ".code-review-graph" / "rules.toml"
    rules.write_text('[no_cycles]\nbetween = "communities"\n', encoding="utf-8")
    db = get_db_path(repo)
    conn = sqlite3.connect(str(db))
    with conn:
        conn.execute("DELETE FROM communities")
        conn.executemany("INSERT INTO communities (id, name) VALUES (?, ?)",
                         [(1, "ui"), (2, "db")])
        conn.execute("UPDATE nodes SET community_id = 1 WHERE file_path = './src/ui/view.py'")
        conn.execute("UPDATE nodes SET community_id = 2 WHERE file_path = './src/db/store.py'")
        # A back-edge db -> ui closes the ring.
        conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line, "
            "updated_at) VALUES ('IMPORTS_FROM', './src/db/store.py', './src/ui/view.py', "
            "'./src/db/store.py', 1, 0)"
        )
    conn.close()
    assert check_rules(repo)["community_cycles"] == [["ui", "db"]]


def test_bad_rules_file_is_an_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / ".code-review-graph" / "rules.toml").write_text(
        "[[forbid]]\nfrom = 'a'\n", encoding="utf-8")
    with pytest.raises(RulesError, match="needs both"):
        check_rules(repo)


def test_cli_exit_codes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    cmd = [sys.executable, "-m", "code_review_graph", "check-rules", "--repo", str(repo)]
    broken = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo))
    assert broken.returncode == 1, broken.stderr
    assert "ui does not touch db: src/ui/view.py" in broken.stdout
    (repo / ".code-review-graph" / "rules.toml").write_text(
        RULES.replace("src/db/**", "src/nowhere/**"), encoding="utf-8")
    clean = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo))
    assert clean.returncode == 0 and "No rule is broken" in clean.stdout
