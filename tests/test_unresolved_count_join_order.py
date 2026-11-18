"""The unresolved call-site count is driven by the seed names.

``count_unresolved_call_sites`` joined ``edges`` to the seed-name table with a
plain ``JOIN``, so SQLite walked every CALLS and REFERENCES edge through
``idx_edges_kind`` and probed the names: 13.9 seconds on a graph of 10
million edges. ``CROSS JOIN`` makes the names the outer loop, one seek each.
"""

from code_review_graph.graph import GraphStore
from code_review_graph.migrations import TARGET_RESOLUTION_EXPR
from code_review_graph.parser import EdgeInfo, NodeInfo

#: The query as it stood before the join order was pinned: the oracle.
_UNPINNED_SQL = (
    "SELECT COUNT(*) FROM edges "
    "JOIN _impact_seed_names s ON s.name = edges.target_qualified "
    "WHERE edges.kind IN ('CALLS', 'REFERENCES') "
    f"AND COALESCE(edges.target_resolution, {TARGET_RESOLUTION_EXPR}) = 'unresolved'"
)

NAMES = 600  # more than one 450-name insert batch


def _store(tmp_path) -> GraphStore:
    """Resolved, unresolved, wrong-kind and unrelated edges, some unclassified."""
    store = GraphStore(tmp_path / "graph.db")
    path = (tmp_path / "app.py").as_posix()
    store.upsert_node(NodeInfo("File", path, path, 1, 1000, "python"))
    for i in range(NAMES):
        store.upsert_node(NodeInfo("Function", f"name{i}", path, i + 1, i + 1, "python"))
    caller = f"{path}::name0"
    for i in range(NAMES):
        # resolved: the target is an indexed node
        store.upsert_edge(EdgeInfo("CALLS", caller, f"{path}::name{i}", path, i))
        # unresolved bare names, as CALLS and as REFERENCES, twice for even i
        store.upsert_edge(EdgeInfo("CALLS", caller, f"name{i}", path, i))
        if i % 2 == 0:
            store.upsert_edge(EdgeInfo("REFERENCES", caller, f"name{i}", path, i + 1))
        # a bare target on a kind the count ignores
        store.upsert_edge(EdgeInfo("IMPORTS_FROM", path, f"name{i}", path, i))
        # unresolved, but not one of the names asked about
        store.upsert_edge(EdgeInfo("CALLS", caller, f"other{i}", path, i))
    store.commit()
    store.refresh_target_resolution()
    # Rows written after the last refresh keep a NULL target_resolution and are
    # classified by the fallback expression; leave some that way.
    store._conn.execute(
        "UPDATE edges SET target_resolution = NULL WHERE id % 3 = 0"
    )
    store._conn.execute("DROP TABLE IF EXISTS sqlite_stat1")
    store.commit()
    return store


def _unpinned_count(store: GraphStore, names: set[str]) -> int:
    conn = store._conn
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _impact_seed_names (name TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _impact_seed_names")
    conn.executemany(
        "INSERT OR IGNORE INTO _impact_seed_names (name) VALUES (?)", [(n,) for n in names],
    )
    try:
        return int(conn.execute(_UNPINNED_SQL).fetchone()[0])
    finally:
        conn.execute("DELETE FROM _impact_seed_names")


def _plan_of_next_count(store: GraphStore) -> list[str]:
    """Capture the plan of the COUNT query the method actually runs."""
    plans: list[str] = []
    conn = store._conn

    def trace(sql: str) -> None:
        if sql.startswith("SELECT COUNT(*)") and "_impact_seed_names" in sql:
            plans.extend(
                row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)
            )

    conn.set_trace_callback(trace)
    try:
        store.count_unresolved_call_sites({"name1"})
    finally:
        conn.set_trace_callback(None)
    return plans


def test_count_matches_the_unpinned_query(tmp_path):
    store = _store(tmp_path)
    try:
        for names in (
            {"name1", "name2"},
            {f"name{i}" for i in range(NAMES)},  # 600 names: two insert batches
            {"name3", "other3", "missing"},
        ):
            expected = _unpinned_count(store, names)
            assert expected > 0
            assert store.count_unresolved_call_sites(names) == expected
    finally:
        store.close()


def test_every_name_over_several_batches_is_counted(tmp_path):
    store = _store(tmp_path)
    try:
        # One CALLS per name plus one REFERENCES per even name.
        count = store.count_unresolved_call_sites({f"name{i}" for i in range(NAMES)})
    finally:
        store.close()
    assert count == NAMES + NAMES // 2


def test_empty_names_count_zero(tmp_path):
    store = _store(tmp_path)
    try:
        assert store.count_unresolved_call_sites(set()) == 0
    finally:
        store.close()


def test_count_is_driven_by_the_seed_names(tmp_path):
    store = _store(tmp_path)
    try:
        plan = _plan_of_next_count(store)
    finally:
        store.close()
    tables = [line for line in plan if line.startswith(("SCAN", "SEARCH"))]
    assert tables, plan
    assert tables[0].startswith("SCAN s"), plan
    assert any("edges USING INDEX idx_edges_target_kind" in line for line in plan), plan
