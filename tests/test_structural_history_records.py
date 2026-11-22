"""``history`` records when edges appear and go, and when a symbol loses
its last test, one record per run, tagged with the commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.history import HistoryError, record, show
from code_review_graph.incremental import full_build, get_db_path

LIB = "def helper(x):\n    return x + 1\n"
APP = "def main():\n    return 1\n"
TEST = "from lib import helper\n\n\ndef test_helper():\n    assert helper(1) == 2\n"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=CRG test",
         "-c", "user.email=crg-test@example.invalid", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    )


def _commit(repo: Path, files: dict[str, str | None], msg: str) -> None:
    for rel, text in files.items():
        if text is None:
            (repo / rel).unlink()
        else:
            (repo / rel).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    store = GraphStore(get_db_path(repo))
    try:
        full_build(repo, store)
    finally:
        store.close()


def test_dependency_appearing_and_last_test_lost_are_recorded(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, {".gitignore": ".code-review-graph/\n", "lib.py": LIB, "app.py": APP,
                   "test_lib.py": TEST}, "initial")
    with pytest.raises(HistoryError, match="no history"):
        show(repo)
    assert record(repo)["baseline"] is True

    _commit(repo, {"app.py": "from lib import helper\n\n\ndef main():\n"
                             "    return helper(1)\n"}, "app uses lib")
    first = record(repo)
    assert first["baseline"] is False and first["added"] >= 2

    _commit(repo, {"test_lib.py": None}, "drop the test")
    second = record(repo)
    assert second["lost_last_test"] == ["lib.py::helper"]

    deps = show(repo, source="app.py", target="lib.py", kind="IMPORTS_FROM")["events"]
    assert [(e["change"], e["subject"]) for e in deps] == [("+", "app uses lib")]
    lost = show(repo, lost_tests=True)["events"]
    assert [(e["source"], e["subject"]) for e in lost] == [("lib.py::helper", "drop the test")]

    cli = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "history", "show", "--lost-tests",
         "--repo", str(repo)], capture_output=True, text=True, cwd=str(repo),
    )
    assert cli.returncode == 0, cli.stderr
    assert "lib.py::helper lost its last test  (drop the test)" in cli.stdout
