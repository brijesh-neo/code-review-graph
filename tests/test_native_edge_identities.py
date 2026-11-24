"""The native identity pass settles edges as ``_resolve_edge_identities`` does (CRG_FAST_ENGINE).

``GraphStore._aligned_edges`` checks every endpoint of a file's edges against
the collision identities already stored. Under the fast engine
``crg_fast.settle_edges`` returns the edges whose endpoints all stay as they
are, and hands every other edge to the Python call with the same cache. The
two must give equal edges in the same order: local endpoints, names with no
``::``, one stored identity that is the name itself, one suffixed survivor,
several collisions on the called end and on the other end, endpoints that
are not plain ``str``, and a file's own collisions aligned first.
"""

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo

crg_fast = pytest.importorskip("crg_fast")

_S1 = ":S0123456789ab"
_S2 = ":Sba9876543210"
_STORED = (
    "b.py::solo",
    "b.py::dup", f"b.py::dup{_S1}", f"b.py::dup{_S2}",
    f"b.py::left{_S1}",
    f"b.py::many{_S1}", f"b.py::many{_S2}",
    "b.py::sel:with:",  # a colon-bearing name, not a collision identity
)


class _Text(str):
    """A ``str`` subclass, which the native pass leaves to Python."""


def _edges():
    targets = (
        "a.py::own", "plain", "b.py::solo", "b.py::dup", "b.py::left", "b.py::many",
        "b.py::missing", "b.py::sel:with:", "ünï.py::x", _Text("b.py::dup"),
    )
    edges = []
    for kind in ("CALLS", "REFERENCES", "TESTED_BY", "IMPORTS_FROM", "CONTAINS"):
        for line, target in enumerate(targets):
            extra = {"confidence": 0.5} if line % 3 == 0 else {}
            edges.append(EdgeInfo(kind, "a.py::own", target, "a.py", line, dict(extra)))
            edges.append(EdgeInfo(kind, target, "a.py::caller", "a.py", line, dict(extra)))
    return edges


def _store(tmp_path, monkeypatch, fast):
    monkeypatch.setenv("CRG_FAST_ENGINE", "1" if fast else "0")
    path = tmp_path / ("fast" if fast else "python")
    path.mkdir()
    store = GraphStore(path / "graph.db")
    for qualified in _STORED:
        store._conn.execute(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, updated_at) "
            "VALUES ('Function', ?, ?, 'b.py', 0)",
            (qualified.split("::")[1], qualified),
        )
    store._conn.commit()
    return store


def _shape(edges):
    return [
        (type(edge.source), edge.kind, edge.source, edge.target, edge.file_path,
         edge.line, edge.extra)
        for edge in edges
    ]


@pytest.mark.parametrize("collisions", [
    {},
    {"a.py::own": [(0, 4, "a.py::own"), (5, 9, f"a.py::own{_S1}")]},
])
def test_native_settle_matches_python(tmp_path, monkeypatch, collisions):
    python = _store(tmp_path, monkeypatch, False)
    expected = python._aligned_edges("a.py", _edges(), collisions)
    fast = _store(tmp_path, monkeypatch, True)
    edges = _edges()
    got = fast._aligned_edges("a.py", edges, collisions)
    assert _shape(got) == _shape(expected)
    # Some edges change, so the comparison covers the hand-over too.
    assert _shape(got) != _shape(edges)
    if not collisions:
        # An edge that stays as it is comes back as the same object.
        assert any(new is old for new, old in zip(got, edges))


def test_a_prefix_that_is_not_utf8_goes_to_python(tmp_path, monkeypatch):
    fast = _store(tmp_path, monkeypatch, True)
    calls = []

    def resolve(edge, prefix, cache):
        calls.append(prefix)
        return edge

    edges = _edges()[:3]
    got = crg_fast.settle_edges(
        edges, "\udcff.py::", {}, fast._stored_identity_candidates, resolve,
    )
    assert got == edges
    assert calls == ["\udcff.py::"] * 3
