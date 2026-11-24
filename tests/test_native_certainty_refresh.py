"""The native certainty refresh writes what the SQL writes (CRG_FAST_ENGINE).

``GraphStore.refresh_target_resolution`` rewrites ``target_resolution``,
``confidence_tier`` and ``confidence`` with four UPDATEs. Under the fast
engine one native pass on the same connection does it instead. Both must
leave every row identical and return the same count, including where SQL is
easy to get wrong: ``LIKE`` folds ASCII case only and reads ``_`` as any one
character, a NULL ``extra`` never matches, and ``$.confidence`` is whatever
SQLite's own JSON functions say. A row the native pass cannot read exactly
makes it write nothing and hand over to the SQL.
"""

import itertools

import pytest

from code_review_graph.graph import GraphStore

crg_fast = pytest.importorskip("crg_fast")

_NODES = ("m.py::known", "m.py::Known", "m.py::tést")

_EXTRAS = (
    None,
    "{}",
    '{"ambiguous_targets": ["a", "b"]}',
    '{"AMBIGUOUS_TARGETS": 1}',
    '{"bare_call_target": "x"}',
    '{"BARE_Call_Target": "x"}',
    '{"bareXcall_target": "x"}',  # "_" matches any one character
    '{"bareécall_target": "x"}',  # ... including one that is not ASCII
    '{"barécall_target": "x"}',  # one character short
    '{"bare_cÁll_target": "x"}',  # non-ASCII letters do not fold
    '{"scoped_resolved": true, "confidence": 0.65}',
    '{"confidence": 0.3}',
    '{"confidence": 1}',
    '{"confidence": null}',
    '{"confidence": 0.3',  # mentions the key, but invalid JSON
    '{"CONFIDENCE": 0.3}',
    '{"spring_resolved": 1, "ambiguous_targets": []}',
)
_TIERS = (None, "EXTRACTED", "INFERRED", "AMBIGUOUS", "UNRESOLVED", "custom")
_RESOLUTIONS = (None, "direct", "unresolved", "other")
_CONFIDENCES = (None, 1.0, 0.2, 0.3)


def _store(tmp_path, monkeypatch, name):
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    (tmp_path / name).mkdir()
    store = GraphStore(tmp_path / name / "graph.db")
    assert store._native_sql
    conn = store._conn
    for qualified in _NODES:
        conn.execute(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, updated_at) "
            "VALUES ('Function', ?, ?, 'm.py', 0)",
            (qualified.split("::")[1], qualified),
        )
    return store


def _fill(store, extras):
    rows = []
    endpoints = ("m.py::known", "m.py::KNOWN", "m.py::missing", "m.py::tést")
    combos = itertools.product(
        ("CALLS", "REFERENCES", "TESTED_BY", "IMPORTS_FROM"), endpoints, extras,
        _TIERS, _RESOLUTIONS, _CONFIDENCES,
    )
    for index, (kind, endpoint, extra, tier, resolution, confidence) in enumerate(combos):
        if index % 7:  # a spread of the product keeps the test quick
            continue
        rows.append((kind, endpoint, endpoint, "m.py", index, extra, confidence, tier,
                     resolution))
    store._conn.executemany(
        "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line, "
        "extra, confidence, confidence_tier, target_resolution, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        rows,
    )
    store._conn.commit()


def _edges(store):
    return [
        tuple(row) for row in store._conn.execute(
            "SELECT id, kind, source_qualified, target_qualified, extra, confidence, "
            "typeof(confidence), confidence_tier, target_resolution FROM edges ORDER BY id"
        )
    ]


def _refresh(store, native):
    store._begin_immediate()
    count = store._native_certainty_refresh() if native else store._certainty_refresh_sql()
    store._conn.commit()
    return count


def test_native_refresh_matches_the_sql(tmp_path, monkeypatch):
    native = _store(tmp_path, monkeypatch, "native")
    sql = _store(tmp_path, monkeypatch, "sql")
    for store in (native, sql):
        _fill(store, _EXTRAS)
    assert _edges(native) == _edges(sql)

    assert _refresh(native, True) == _refresh(sql, False) > 0
    assert _edges(native) == _edges(sql)
    # A second pass finds nothing left to change.
    assert _refresh(native, True) == _refresh(sql, False) == 0
    assert _edges(native) == _edges(sql)


@pytest.mark.parametrize("odd", ['{"confidence": "high"}', '{"confidence": {"nested": 1}}'])
def test_a_value_it_cannot_read_exactly_hands_over_to_the_sql(tmp_path, monkeypatch, odd):
    native = _store(tmp_path, monkeypatch, "native")
    sql = _store(tmp_path, monkeypatch, "sql")
    for store in (native, sql):
        _fill(store, (odd, *_EXTRAS))
    before = _edges(native)

    assert _refresh(native, True) == -1
    assert _edges(native) == before

    assert native.refresh_target_resolution() == sql.refresh_target_resolution()
    assert _edges(native) == _edges(sql)
