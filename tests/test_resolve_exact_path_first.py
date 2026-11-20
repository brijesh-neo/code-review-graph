"""Changed-file resolution tries the exact stored path before a suffix scan.

``_resolve_graph_file_paths_with_unmatched`` ran a ``LIKE '%suffix'`` query
over every node row for each input, even when the absolute path was stored
exactly: 3.4 seconds per call on a large graph. The exact path is now one
index seek, and the suffix scan only runs for an input it misses.

From PR #5.
"""

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.parser import NodeInfo
from code_review_graph.tools._common import (
    _resolve_graph_file_paths,
    _resolve_graph_file_paths_with_unmatched,
)


@pytest.fixture
def store(tmp_path):
    store = GraphStore(tmp_path / "graph.db")
    for rel in ("src/app.py", "src/b.py"):
        path = (tmp_path / rel).as_posix()
        store.upsert_node(NodeInfo("File", path, path, 1, 2, "python"))
        store.upsert_node(NodeInfo("Function", "fn", path, 1, 2, "python"))
    # Rows keyed by another checkout of the same repository, as a graph copied
    # from the main worktree into a linked one would be.
    other = "/elsewhere/checkout/src/lib.py"
    store.upsert_node(NodeInfo("File", other, other, 1, 2, "python"))
    store.commit()
    yield store
    store.close()


def _like_queries(store: GraphStore, call):
    seen: list[str] = []
    store._conn.set_trace_callback(seen.append)
    try:
        result = call()
    finally:
        store._conn.set_trace_callback(None)
    return result, [sql for sql in seen if " LIKE " in sql]


@pytest.mark.parametrize("spelling", ["relative", "absolute"])
def test_exact_hit_skips_the_suffix_scan(store, tmp_path, spelling):
    absolute = (tmp_path / "src" / "app.py").as_posix()
    given = "src/app.py" if spelling == "relative" else absolute
    result, likes = _like_queries(
        store, lambda: _resolve_graph_file_paths(store, tmp_path, [given]),
    )
    assert result == [absolute]
    assert likes == []


def test_suffix_fallback_still_resolves_a_path_from_another_root(store, tmp_path):
    (resolved, unmatched), likes = _like_queries(
        store,
        lambda: _resolve_graph_file_paths_with_unmatched(store, tmp_path, ["src/lib.py"]),
    )
    assert resolved == ["/elsewhere/checkout/src/lib.py"]
    assert unmatched == []
    assert likes, "a miss on the exact path must fall back to the suffix scan"


def test_duplicates_are_removed_and_order_kept(store, tmp_path):
    app = (tmp_path / "src" / "app.py").as_posix()
    b = (tmp_path / "src" / "b.py").as_posix()
    result = _resolve_graph_file_paths(
        store, tmp_path, ["src/b.py", "src/app.py", app, "src/b.py", "src/lib.py", b],
    )
    assert result == [b, app, "/elsewhere/checkout/src/lib.py"]


def test_unknown_path_is_still_reported(store, tmp_path):
    resolved, unmatched = _resolve_graph_file_paths_with_unmatched(
        store, tmp_path, ["src/app.py", "src/ghost.py"],
    )
    assert resolved == [(tmp_path / "src" / "app.py").as_posix()]
    assert unmatched == ["src/ghost.py"]
