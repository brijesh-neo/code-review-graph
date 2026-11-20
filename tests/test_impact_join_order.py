"""One impact hop is driven by the frontier, not by a scan of every edge.

With a plain ``JOIN`` SQLite made ``edges`` the outer loop of the two edge
branches of ``_impact_candidate_sql`` and probed the frontier for each row, so
every hop read the whole table: 99 seconds against 0.15 on a graph of 10
million edges. ``CROSS JOIN`` pins the frontier as the outer loop.

From PR #5.
"""

from code_review_graph import graph as graph_module
from code_review_graph.graph import GraphStore, _impact_candidate_sql
from code_review_graph.parser import EdgeInfo, NodeInfo

FILES = 60
FUNCS = 10


def _store(tmp_path) -> GraphStore:
    """A graph of a few thousand edges with no ``sqlite_stat1``."""
    store = GraphStore(tmp_path / "graph.db")
    for f in range(FILES):
        path = (tmp_path / f"mod{f}.py").as_posix()
        store.upsert_node(NodeInfo("File", path, path, 1, 100, "python"))
        for k in range(FUNCS):
            store.upsert_node(NodeInfo("Function", f"fn{k}", path, k + 1, k + 1, "python"))
    for f in range(FILES):
        path = (tmp_path / f"mod{f}.py").as_posix()
        for k in range(FUNCS):
            for step in (1, 7, 13, 29):
                callee = (tmp_path / f"mod{(f + step) % FILES}.py").as_posix()
                store.upsert_edge(EdgeInfo(
                    "CALLS", f"{path}::fn{k}", f"{callee}::fn{(k + step) % FUNCS}", path, k,
                ))
        test_path = (tmp_path / f"test_mod{f}.py").as_posix()
        store.upsert_node(NodeInfo("Test", "test_it", test_path, 1, 2, "python", is_test=True))
        store.upsert_edge(EdgeInfo("TESTED_BY", f"{path}::fn0", f"{test_path}::test_it", path))
    store.commit()
    store._conn.execute("DROP TABLE IF EXISTS sqlite_stat1")
    return store


def _plan(store: GraphStore, sql: str) -> list[str]:
    for table, key in (
        ("_impact_frontier", "node_qn"),
        ("_impact_next", "node_qn"),
        ("_impact_frontier_dirs", "dir"),
    ):
        store._conn.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {table} "
            f"({key} TEXT PRIMARY KEY, score REAL NOT NULL)"
        )
    store._conn.execute(
        "CREATE TEMP TABLE IF NOT EXISTS _impact_policies "
        "(kind TEXT PRIMARY KEY, weight REAL NOT NULL, direction TEXT NOT NULL)"
    )
    return [
        row["detail"]
        for row in store._conn.execute("EXPLAIN QUERY PLAN " + sql, (None,) * sql.count("?"))
    ]


def _summary(result: dict) -> tuple:
    return (
        sorted(n.qualified_name for n in result["impacted_nodes"]),
        sorted(result["impacted_files"]),
        sorted((qn, round(score, 9)) for qn, score in result["impact_scores"].items()),
    )


def test_edge_branches_never_scan_the_edges_table(tmp_path):
    store = _store(tmp_path)
    try:
        plan = _plan(store, _impact_candidate_sql(""))
    finally:
        store.close()
    assert not any(line.startswith("SCAN e") for line in plan), plan
    assert any("idx_edges_source_kind" in line for line in plan), plan
    assert any("idx_edges_target_kind" in line for line in plan), plan


def test_pinned_join_order_returns_the_same_impact(tmp_path, monkeypatch):
    store = _store(tmp_path)
    changed = [(tmp_path / "mod0.py").as_posix(), (tmp_path / "mod31.py").as_posix()]
    try:
        pinned = store.get_impact_radius_sql(changed, max_depth=3, max_nodes=10_000)
        monkeypatch.setattr(
            graph_module,
            "_impact_candidate_sql",
            lambda guard: _impact_candidate_sql(guard).replace(
                "CROSS JOIN edges e ON e.", "JOIN edges e ON e.",
            ),
        )
        unpinned = store.get_impact_radius_sql(changed, max_depth=3, max_nodes=10_000)
    finally:
        store.close()
    assert pinned["impacted_nodes"], "fixture reaches nothing; the comparison proves nothing"
    assert _summary(pinned) == _summary(unpinned)
