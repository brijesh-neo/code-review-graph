"""The native flow trace gives the flows the Python trace gives (CRG_FAST_ENGINE).

``incremental_trace_flows`` re-traces each flow that reaches a changed file.
Under the fast engine one native pass on the same connection walks the calls
and counts what criticality needs; Python still scores it. Every flow must
match ``_trace_single_flow`` over the lazy adjacency, field for field,
including cycles, calls out of the graph, repeated calls, the depth cap and
names with security keywords in odd case.
"""

import dataclasses

import pytest

from code_review_graph import flows
from code_review_graph.graph import GraphStore

crg_fast = pytest.importorskip("crg_fast")


def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    store = GraphStore(tmp_path / "graph.db")
    assert store._native_sql
    return store


def _fill(store):
    conn = store._conn
    nodes = [
        ("a.py", "a.py::main"),
        ("a.py", "a.py::LoginHandler"),
        ("b.py", "b.py::helper"),
        ("b.py", "b.py::loop_one"),
        ("c.py", "c.py::loop_two"),
        ("c.py", "c.py::dup"),
        ("d.py", "d.py::check_Token"),
        ("e.py", "e.py::tést"),
        ("e.py", "e.py::lonely"),
    ]
    for path, qualified in nodes:
        conn.execute(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, updated_at) "
            "VALUES ('Function', ?, ?, ?, 0)",
            (qualified.split("::")[1], qualified, path),
        )
    chain = [f"f{i}.py::step{i}" for i in range(20)]
    for i, qualified in enumerate(chain):
        conn.execute(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, updated_at) "
            "VALUES ('Function', ?, ?, ?, 0)",
            (qualified.split("::")[1], qualified, f"f{i}.py"),
        )
    calls = [
        ("a.py::main", "b.py::helper"),
        ("a.py::main", "a.py::LoginHandler"),
        ("a.py::main", "os.path.join"),  # out of the graph
        ("a.py::main", "b.py::helper"),  # twice
        ("b.py::helper", "b.py::loop_one"),
        ("b.py::loop_one", "c.py::loop_two"),
        ("c.py::loop_two", "b.py::loop_one"),  # a cycle
        ("c.py::loop_two", "c.py::dup"),
        ("c.py::dup", "d.py::check_Token"),
        ("c.py::dup", "requests.get"),
        ("a.py::LoginHandler", "e.py::tést"),
        ("e.py::tést", "a.py::main"),  # back to the entry point
    ]
    calls += list(zip(chain, chain[1:]))
    for line, (source, target) in enumerate(calls):
        conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line, "
            "updated_at) VALUES ('CALLS', ?, ?, ?, ?, 0)",
            (source, target, source.split("::")[0], line),
        )
    for target in ("b.py::helper", "c.py::dup", "f3.py::step3"):
        conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line, "
            "updated_at) VALUES ('TESTED_BY', ?, 'tests/t.py::test_it', 't.py', 1, 0)",
            (target,),
        )
    conn.commit()


def _entry_points(store):
    rows = store._conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()
    return [store._row_to_node(row) for row in rows]


def _python_flows(store, entry_points, max_depth):
    adj = flows._lazy_flow_adjacency(store)
    assert adj is not None
    traced = []
    for ep in entry_points:
        flow = flows._trace_single_flow(adj, ep, max_depth)
        if flow is not None:
            flow.pop("files")
            traced.append(flow)
    return traced


@pytest.mark.parametrize("max_depth", [0, 1, 3, 15])
def test_native_trace_matches_the_python_trace(tmp_path, monkeypatch, max_depth):
    store = _store(tmp_path, monkeypatch)
    _fill(store)
    entry_points = _entry_points(store)
    native = flows._native_traced_flows(store, entry_points, max_depth)
    assert native is not None
    python = _python_flows(store, entry_points, max_depth)
    assert native == python
    if max_depth >= 3:
        assert any(flow["criticality"] > 0 for flow in native)


def test_an_entry_point_with_no_node_row_still_traces_the_same(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    _fill(store)
    ghost_node = dataclasses.replace(
        _entry_points(store)[0], id=10_000, name="ghost", file_path="z.py",
    )
    native = flows._native_traced_flows(store, [ghost_node], 15)
    assert native == _python_flows(store, [ghost_node], 15)


def test_a_crg_fast_without_the_pass_hands_over_to_python(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    _fill(store)
    monkeypatch.delattr(crg_fast, "stage_flow_trace")
    assert flows._native_traced_flows(store, _entry_points(store), 15) is None
