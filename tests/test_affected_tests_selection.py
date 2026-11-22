"""``affected-tests`` selects the test files a change reaches, and falls
back to every test whenever the graph cannot be trusted for the change.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from code_review_graph.affected_tests import affected_tests
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path

FILES = {
    "lib.py": "def helper(x):\n    return x + 1\n",
    "other.py": "def lone():\n    return 0\n",
    "tests/test_lib.py": "from lib import helper\n\n\ndef test_helper():\n"
                         "    assert helper(1) == 2\n",
    "tests/test_other.py": "from other import lone\n\n\ndef test_lone():\n"
                           "    assert lone() == 0\n",
    "README.md": "# demo\n",
}


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=CRG test",
         "-c", "user.email=crg-test@example.invalid", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".code-review-graph/\n", encoding="utf-8")
    for rel, text in FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo.resolve()


def _build(repo: Path) -> None:
    store = GraphStore(get_db_path(repo))
    try:
        full_build(repo, store)
    finally:
        store.close()


def test_only_the_test_of_the_changed_function_is_selected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "lib.py").write_text("def helper(x):\n    return x + 2\n", encoding="utf-8")
    _build(repo)
    result = affected_tests(repo)
    assert result["mode"] == "selected"
    assert result["tests"] == ["tests/test_lib.py"]


def test_stale_graph_runs_everything(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _build(repo)
    (repo / "lib.py").write_text("def helper(x):\n    return x + 2\n", encoding="utf-8")
    result = affected_tests(repo)
    assert result["mode"] == "all" and "stale" in result["reason"]


def test_unknown_deleted_and_config_files_run_everything(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _build(repo)
    for changed, word in ((["new.py"], "not in the graph"),
                          (["gone.py"], "deleted"),
                          (["tests/conftest.py"], "configuration")):
        if changed == ["new.py"]:
            (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
        result = affected_tests(repo, changed=changed)
        assert result["mode"] == "all" and word in result["reason"], result


def test_docs_only_change_selects_nothing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _build(repo)
    result = affected_tests(repo, changed=["README.md"])
    assert result == {"mode": "selected", "reason": "only inert files changed",
                      "changed": ["README.md"], "tests": []}


def test_pytest_plugin_deselects_unaffected_tests(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "lib.py").write_text("def helper(x):\n    return x + 0 + 1\n", encoding="utf-8")
    _build(repo)
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "code_review_graph.pytest_plugin",
         "--crg-affected", "-q", "-p", "no:cacheprovider", "--rootdir", str(repo), str(repo)],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed, 1 deselected" in run.stdout

    plain = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "code_review_graph.pytest_plugin",
         "-q", "-p", "no:cacheprovider", "--rootdir", str(repo), str(repo)],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert "2 passed" in plain.stdout


def test_cli_lists_selected_files(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "other.py").write_text("def lone():\n    return 1\n", encoding="utf-8")
    _build(repo)
    run = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "affected-tests", "--repo", str(repo)],
        capture_output=True, text=True, cwd=str(repo),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "tests/test_other.py"
