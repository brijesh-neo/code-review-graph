"""The conflict radar finds changes in two worktrees that touch each other.

Git merges these cleanly: one branch edits ``helper`` in ``lib.py`` while
another adds a caller of it in ``app.py``. The radar must report them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from code_review_graph.conflicts import find_conflicts
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path

LIB = '''def helper(x):
    return x + 1


def untouched():
    return 0
'''

APP = '''from lib import helper


def main():
    return 1
'''


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=CRG test",
            "-c", "user.email=crg-test@example.invalid",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _build(root: Path) -> None:
    store = GraphStore(get_db_path(root))
    try:
        full_build(root, store)
    finally:
        store.close()


def _two_worktrees(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".code-review-graph/\n", encoding="utf-8")
    (repo / "lib.py").write_text(LIB, encoding="utf-8")
    (repo / "app.py").write_text(APP, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    other = tmp_path / "other"
    _git(repo, "worktree", "add", "-q", "-b", "feature", str(other))

    # Here: change helper's body (uncommitted edits count too).
    (repo / "lib.py").write_text(LIB.replace("x + 1", "x + 2"), encoding="utf-8")
    # There: a new caller of helper, committed on its branch.
    (other / "app.py").write_text(
        APP + "\n\ndef new_caller():\n    return helper(3)\n", encoding="utf-8",
    )
    _git(other, "commit", "-q", "-am", "call helper")
    _build(repo)
    _build(other)
    return repo.resolve(), other.resolve()


def test_new_caller_of_a_changed_function_is_found_both_ways(tmp_path: Path) -> None:
    repo, other = _two_worktrees(tmp_path)

    report = find_conflicts(repo)
    (entry,) = report["others"]
    assert entry["worktree"] == str(other)
    assert entry["branch"] == "feature"
    assert entry["both"] == []
    assert entry["uses"] == []
    assert entry["used_by"] == [
        {"symbol": "app.py::new_caller", "edge": "CALLS", "changed_there": "lib.py::helper"},
    ]

    mirrored = find_conflicts(other)
    (back,) = mirrored["others"]
    assert back["uses"] == [
        {"symbol": "app.py::new_caller", "edge": "CALLS", "changed_there": "lib.py::helper"},
    ]
    assert back["used_by"] == []


def test_same_function_changed_in_both_is_found(tmp_path: Path) -> None:
    repo, other = _two_worktrees(tmp_path)
    (other / "lib.py").write_text(LIB.replace("x + 1", "x * 2"), encoding="utf-8")
    _git(other, "commit", "-q", "-am", "also change helper")
    _build(other)

    (entry,) = find_conflicts(repo)["others"]
    assert entry["both"] == ["lib.py::helper"]
    assert "lib.py::untouched" not in entry["both"]


def test_worktree_without_a_graph_is_skipped_not_fatal(tmp_path: Path) -> None:
    repo, other = _two_worktrees(tmp_path)
    get_db_path(other).unlink()

    (entry,) = find_conflicts(repo)["others"]
    assert "no graph" in entry["skipped"]


def test_cli_prints_json_and_exit_code(tmp_path: Path) -> None:
    repo, _ = _two_worktrees(tmp_path)
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "conflicts", "--repo", str(repo),
         "--json", "--exit-code"],
        capture_output=True, text=True, env=env, cwd=str(repo),
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["others"][0]["used_by"][0]["symbol"] == "app.py::new_caller"

    text = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "conflicts", "--repo", str(repo)],
        capture_output=True, text=True, env=env, cwd=str(repo),
    )
    assert text.returncode == 0, text.stderr
    assert "app.py::new_caller calls lib.py::helper, which you change" in text.stdout
