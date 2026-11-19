"""``seed`` starts a graph from another checkout's, and ``export-absolute``
writes a copy releases before schema v16 open.

A seeded graph must equal a full build of the target checkout: same nodes,
same edges, and no hint left pointing at the source checkout.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path
from code_review_graph.portable import PortableGraphError, export_absolute, seed_graph
from code_review_graph.postprocessing import run_post_processing

LIB = '''class Store:
    def save(self, x):
        return x


def helper(x):
    return Store().save(x + 1)
'''

APP = '''from lib import helper, Store


def main():
    s = Store()
    s.save(1)
    return helper(2)
'''

TEST = '''from lib import helper


def test_helper():
    assert helper(1) == 2
'''


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


def _snapshot(root: Path) -> tuple[set, set]:
    store = GraphStore(get_db_path(root, read_only=True), read_only=True)
    try:
        nodes = {(n.qualified_name, n.kind, n.line_start, n.line_end)
                 for n in store.get_all_nodes(exclude_files=False)}
        edges = {(e.source_qualified, e.target_qualified, e.kind)
                 for e in store.get_all_edges()}
    finally:
        store.close()
    return nodes, edges


def _two_worktrees(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".code-review-graph/\n", encoding="utf-8")
    (repo / "lib.py").write_text(LIB, encoding="utf-8")
    (repo / "app.py").write_text(APP, encoding="utf-8")
    (repo / "test_lib.py").write_text(TEST, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    other = tmp_path / "other"
    _git(repo, "worktree", "add", "-q", "-b", "feature", str(other))
    (other / "app.py").write_text(APP + "\n\ndef extra():\n    return helper(5)\n",
                                  encoding="utf-8")
    (other / "new.py").write_text("def fresh():\n    return 1\n", encoding="utf-8")
    _build(repo)
    return repo.resolve(), other.resolve()


def test_seeded_graph_equals_a_full_build(tmp_path: Path) -> None:
    repo, other = _two_worktrees(tmp_path)

    result = seed_graph(other, repo)
    assert result["files_reparsed"] == 2  # app.py differs, new.py is new
    seeded = _snapshot(other)

    get_db_path(other).unlink()
    _build(other)
    assert seeded == _snapshot(other)

    conn = sqlite3.connect(str(get_db_path(repo)))
    conn.close()
    seeded_conn = sqlite3.connect(f"file:{get_db_path(other)}?mode=ro", uri=True)
    try:
        leaked = seeded_conn.execute(
            "SELECT count(*) FROM edges WHERE instr(extra, ?) > 0", (str(repo) + "/",),
        ).fetchone()[0]
    finally:
        seeded_conn.close()
    assert leaked == 0


def test_seed_refuses_to_replace_a_graph_without_force(tmp_path: Path) -> None:
    repo, other = _two_worktrees(tmp_path)
    _build(other)
    with pytest.raises(PortableGraphError, match="--force"):
        seed_graph(other, repo)
    seed_graph(other, repo, force=True)


def test_seed_from_a_checkout_without_a_graph_fails_cleanly(tmp_path: Path) -> None:
    repo, other = _two_worktrees(tmp_path)
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "seed", str(other), "--repo", str(repo),
         "--force"],
        capture_output=True, text=True, env=env, cwd=str(repo),
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr and "no graph" in result.stderr


def test_absolute_export_is_schema_15_with_absolute_paths(tmp_path: Path) -> None:
    repo, _ = _two_worktrees(tmp_path)
    before = _snapshot(repo)
    out = tmp_path / "old.db"
    export_absolute(repo, out)

    conn = sqlite3.connect(str(out))
    try:
        meta = dict(conn.execute("SELECT key, value FROM metadata"))
        files = {r[0] for r in conn.execute("SELECT DISTINCT file_path FROM nodes")}
        fts = conn.execute(
            "SELECT n.qualified_name FROM nodes_fts f JOIN nodes n ON n.id = f.rowid "
            "WHERE nodes_fts MATCH 'helper'"
        ).fetchall()
    finally:
        conn.close()
    assert meta["schema_version"] == "15"
    assert "path_mode" not in meta
    assert files and all(Path(f).is_absolute() for f in files)
    assert any(q.startswith(str(repo)) for (q,) in fts)

    # The checkout's own graph is untouched.
    assert _snapshot(repo) == before
    with pytest.raises(PortableGraphError, match="exists"):
        export_absolute(repo, out)


def test_current_release_reads_an_absolute_export_back(tmp_path: Path) -> None:
    repo, _ = _two_worktrees(tmp_path)
    before = _snapshot(repo)
    out = tmp_path / "old.db"
    export_absolute(repo, out)
    os.replace(out, get_db_path(repo))
    for suffix in ("-wal", "-shm"):
        side = get_db_path(repo).with_name("graph.db" + suffix)
        if side.exists():
            side.unlink()
    GraphStore(get_db_path(repo)).close()  # a writable open migrates v15 -> v16
    assert _snapshot(repo) == before
