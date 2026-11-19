"""``structural-diff`` reports what a branch did to the graph's shape.

The base worktree has ``helper`` tested and ``app.py`` importing only
``lib``. The branch drops the test, adds ``util.py`` and an import of it,
and a new call. Each change must show in its section.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path
from code_review_graph.postprocessing import run_post_processing
from code_review_graph.structural_diff import StructuralDiffError, structural_diff

LIB = "def helper(x):\n    return x + 1\n"
APP = "from lib import helper\n\n\ndef main():\n    return helper(1)\n"
TEST = "from lib import helper\n\n\ndef test_helper():\n    assert helper(1) == 2\n"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=CRG test",
         "-c", "user.email=crg-test@example.invalid", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    )


def _build(root: Path) -> None:
    store = GraphStore(get_db_path(root))
    try:
        full_build(root, store)
        run_post_processing(store)
    finally:
        store.close()


def _pair(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    base.mkdir()
    _git(base, "init", "-q", "-b", "main")
    (base / ".gitignore").write_text(".code-review-graph/\n", encoding="utf-8")
    (base / "lib.py").write_text(LIB, encoding="utf-8")
    (base / "app.py").write_text(APP, encoding="utf-8")
    (base / "test_lib.py").write_text(TEST, encoding="utf-8")
    _git(base, "add", ".")
    _git(base, "commit", "-q", "-m", "initial")
    branch = tmp_path / "branch"
    _git(base, "worktree", "add", "-q", "-b", "feature", str(branch))
    (branch / "test_lib.py").unlink()
    (branch / "util.py").write_text("def tidy(x):\n    return x\n", encoding="utf-8")
    (branch / "app.py").write_text(
        "from lib import helper\nfrom util import tidy\n\n\n"
        "def main():\n    return tidy(helper(1))\n",
        encoding="utf-8",
    )
    _build(base)
    _build(branch)
    return base.resolve(), branch.resolve()


def test_sections_report_each_structural_change(tmp_path: Path) -> None:
    base, branch = _pair(tmp_path)
    report = structural_diff(branch, base)

    assert report["lost_all_tests"] == ["lib.py::helper"]
    assert {"file": "app.py", "imports": "util.py"} in report["new_file_dependencies"]
    assert {"source": "app.py::main", "target": "util.py::tidy", "kind": "CALLS"} in (
        report["edges_added"]
    )
    assert report["edges_removed_by_kind"].get("TESTED_BY", 0) >= 1

    # The mirror image: from the branch's side nothing lost its tests.
    assert structural_diff(base, branch)["lost_all_tests"] == []


def test_identical_graphs_have_no_difference(tmp_path: Path) -> None:
    base, _ = _pair(tmp_path)
    report = structural_diff(base, get_db_path(base))
    assert report["edges_added_by_kind"] == {} and report["edges_removed_by_kind"] == {}
    assert report["symbols_added"] == report["symbols_removed"] == 0


def test_missing_base_graph_is_an_error(tmp_path: Path) -> None:
    base, branch = _pair(tmp_path)
    get_db_path(base).unlink()
    with pytest.raises(StructuralDiffError, match="no graph"):
        structural_diff(branch, base)


def test_cli_text_and_json(tmp_path: Path) -> None:
    base, branch = _pair(tmp_path)
    env = {**os.environ, "PYTHONUTF8": "1"}
    cmd = [sys.executable, "-m", "code_review_graph", "structural-diff", str(base),
           "--repo", str(branch)]
    text = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(branch))
    assert text.returncode == 0, text.stderr
    assert "Lost every test (1):\n  lib.py::helper" in text.stdout
    as_json = subprocess.run(cmd + ["--json"], capture_output=True, text=True, env=env,
                             cwd=str(branch))
    assert json.loads(as_json.stdout)["lost_all_tests_count"] == 1
