"""A full build and a schema upgrade leave query planner statistics behind.

Without ``sqlite_stat1`` SQLite guesses table sizes, and on a large graph it
picks the low-selectivity ``kind`` indexes and drives joins from ``edges``.

From PR #5.
"""

import logging
import sqlite3
import subprocess

from code_review_graph import migrations
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build
from code_review_graph.migrations import LATEST_VERSION
from code_review_graph.parser import EdgeInfo, NodeInfo


def _stat_tables(conn: sqlite3.Connection) -> set[str]:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'sqlite_stat1'"
    ).fetchone()
    if not exists:
        return set()
    return {row[0] for row in conn.execute("SELECT DISTINCT tbl FROM sqlite_stat1")}


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_full_build_writes_stats_for_nodes_and_edges(tmp_path):
    (tmp_path / "a.py").write_text("def helper():\n    return 1\n")
    (tmp_path / "b.py").write_text("from a import helper\n\ndef run():\n    return helper()\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@example.com", "-c", "user.name=T", "add", "-A")
    _git(tmp_path, "-c", "user.email=t@example.com", "-c", "user.name=T",
         "commit", "-qm", "initial")
    store = GraphStore(tmp_path / ".code-review-graph" / "graph.db")
    try:
        assert "edges" not in _stat_tables(store._conn)
        full_build(tmp_path, store)
        assert {"nodes", "edges"} <= _stat_tables(store._conn)
    finally:
        store.close()


def test_schema_upgrade_writes_stats(tmp_path):
    path = tmp_path / "graph.db"
    store = GraphStore(path)
    file_path = (tmp_path / "app.py").as_posix()
    store.upsert_node(NodeInfo("File", file_path, file_path, 1, 2, "python"))
    store.upsert_node(NodeInfo("Function", "run", file_path, 1, 2, "python"))
    store.upsert_edge(EdgeInfo("CONTAINS", file_path, f"{file_path}::run", file_path))
    store._conn.execute("DROP TABLE IF EXISTS sqlite_stat1")
    store._conn.execute(
        "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
        (str(LATEST_VERSION - 1),),
    )
    store.commit()
    store.close()

    store = GraphStore(path)  # reopening runs the pending migration
    try:
        assert {"nodes", "edges"} <= _stat_tables(store._conn)
    finally:
        store.close()


def test_a_held_write_lock_skips_the_stats_without_failing(tmp_path, caplog):
    path = tmp_path / "graph.db"
    GraphStore(path).close()
    holder = sqlite3.connect(path)
    holder.execute("BEGIN IMMEDIATE")
    conn = sqlite3.connect(path, timeout=0)
    try:
        with caplog.at_level(logging.WARNING, logger="code_review_graph.migrations"):
            migrations.refresh_planner_stats(conn)
        assert "planner statistics" in caplog.text
    finally:
        conn.close()
        holder.rollback()
        holder.close()
