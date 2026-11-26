"""An update's grouped writes leave the rows one commit per file leaves.

Under CRG_FAST_ENGINE an update commits its stored files a group at a time
and, when they replace most of the edge rows, writes them with the edge
indexes dropped. Both must leave the same rows, ids, indexes and planner
statistics as one transaction per file, and a failure must leave the graph
as it was.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo, NodeInfo

FILES = [f"/repo/m{i}.py" for i in range(6)]


def _file(path: str, version: int) -> tuple[list[NodeInfo], list[EdgeInfo]]:
    nodes = [
        NodeInfo(kind="File", name=path, file_path=path, line_start=1, line_end=40,
                 language="python"),
        NodeInfo(kind="Function", name="f", file_path=path, line_start=2, line_end=9,
                 language="python"),
        NodeInfo(kind="Function", name="g", file_path=path, line_start=10, line_end=20,
                 language="python"),
    ]
    edges = [
        EdgeInfo(kind="CONTAINS", source=path, target=f"{path}::f", file_path=path, line=2),
        EdgeInfo(kind="CONTAINS", source=path, target=f"{path}::g", file_path=path, line=10),
        EdgeInfo(kind="CALLS", source=f"{path}::f", target=f"{path}::g", file_path=path,
                 line=3 + version),
        EdgeInfo(kind="IMPORTS_FROM", source=path, target=FILES[(version + 1) % len(FILES)],
                 file_path=path, line=1),
    ]
    return nodes, edges


def _seed(db: Path) -> GraphStore:
    store = GraphStore(str(db))
    for path in FILES:
        nodes, edges = _file(path, 0)
        store.store_file_nodes_edges(path, nodes, edges, "h0")
    store._conn.execute("ANALYZE")
    store._conn.commit()
    return store


def _rewrite(store: GraphStore) -> None:
    """Re-save half the files' edges and replace the other half whole."""
    for i, path in enumerate(FILES):
        nodes, edges = _file(path, 1)
        if i % 2:
            store.store_files_edges([(path, edges)])
        else:
            store.store_file_nodes_edges(path, nodes, edges, "h1")


def _state(store: GraphStore) -> tuple:
    conn = store._conn
    return (
        conn.execute(
            "SELECT id, kind, source_qualified, target_qualified, file_path, line, extra "
            "FROM edges ORDER BY id"
        ).fetchall(),
        conn.execute(
            "SELECT id, qualified_name, file_path, file_hash FROM nodes ORDER BY id"
        ).fetchall(),
        conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE tbl_name = 'edges' ORDER BY name"
        ).fetchall(),
        conn.execute("SELECT rowid, tbl, idx, stat FROM sqlite_stat1 ORDER BY rowid").fetchall(),
        conn.execute("SELECT name, seq FROM sqlite_sequence ORDER BY name").fetchall(),
    )


def _rows(state: tuple) -> tuple:
    return tuple([tuple(row) for row in part] for part in state)


@pytest.fixture
def expected(tmp_path: Path) -> tuple:
    store = _seed(tmp_path / "per_file.db")
    _rewrite(store)
    state = _rows(_state(store))
    store.close()
    return state


@pytest.mark.parametrize("bulk_edges", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 256])
def test_grouped_writes_leave_the_rows_of_one_commit_per_file(
    tmp_path: Path, expected: tuple, limit: int, bulk_edges: bool,
) -> None:
    store = _seed(tmp_path / "grouped.db")
    with store.grouped_writes(limit, bulk_edges=bulk_edges):
        _rewrite(store)
    assert _rows(_state(store)) == expected
    store.close()


def test_bulk_edge_writes_drop_the_edge_indexes_while_they_run(tmp_path: Path) -> None:
    store = _seed(tmp_path / "grouped.db")
    indexes = "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND tbl_name = 'edges'"
    before = store._conn.execute(indexes).fetchone()[0]
    with store.grouped_writes(256, bulk_edges=True):
        assert store._conn.execute(indexes).fetchone()[0] == 0
        _rewrite(store)
    assert store._conn.execute(indexes).fetchone()[0] == before > 0
    store.close()


@pytest.mark.parametrize("bulk_edges", [False, True])
def test_failed_grouped_write_leaves_the_graph_as_it_was(
    tmp_path: Path, bulk_edges: bool,
) -> None:
    store = _seed(tmp_path / "grouped.db")
    before = _rows(_state(store))
    with pytest.raises(RuntimeError):
        with store.grouped_writes(256, bulk_edges=bulk_edges):
            _rewrite(store)
            raise RuntimeError("parse loop failed")
    assert _rows(_state(store)) == before
    store.close()


def test_edge_row_share_counts_the_rows_of_the_given_files(tmp_path: Path) -> None:
    store = _seed(tmp_path / "grouped.db")
    assert store.edge_row_share(FILES[:3]) == pytest.approx(0.5)
    assert store.edge_row_share([]) == 0.0
    store.close()
