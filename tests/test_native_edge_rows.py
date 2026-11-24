"""Native edge rows match the Python ones exactly (CRG_FAST_ENGINE).

``crg_fast.edge_rows`` builds the ``edges`` rows of ``GraphStore._insert_edges``
and ``crg_fast.PathStripper`` gives each row the stored form
``PathCodec.strip_params`` would. Both must agree with the Python code value
for value and type, including the call-site collapse (the last edge at a site
wins, first-seen order is kept) and strings the codec must escape.
"""

import json

import pytest

from code_review_graph.parser import EdgeInfo
from code_review_graph.relpaths import PathCodec

crg_fast = pytest.importorskip("crg_fast")


def _python_rows(edges, now):
    by_site = {}
    for edge in edges:
        extra_dict = edge.extra if edge.extra else {}
        site = (edge.kind, edge.source, edge.target, edge.file_path, edge.line,
                extra_dict.get("source_offset"))
        by_site[site] = (
            edge.kind, edge.source, edge.target, edge.file_path, edge.line,
            json.dumps(extra_dict) if extra_dict else "{}",
            float(extra_dict.get("confidence", 1.0)),
            str(extra_dict.get("confidence_tier", "EXTRACTED")), now,
        )
    return list(by_site.values())


def _edges(root):
    return [
        EdgeInfo("CALLS", f"{root}/a.py::f", f"{root}/b.py::g", f"{root}/a.py", 3),
        EdgeInfo("CALLS", f"{root}/a.py::f", f"{root}/b.py::g", f"{root}/a.py", 3,
                 {"confidence": 0.5, "confidence_tier": "INFERRED"}),
        EdgeInfo("CALLS", f"{root}/a.py::f", "h", f"{root}/a.py", 4,
                 {"source_offset": 7}),
        EdgeInfo("CALLS", f"{root}/a.py::f", "h", f"{root}/a.py", 4,
                 {"source_offset": 9, "confidence": 1}),
        EdgeInfo("IMPORTS_FROM", f"{root}/a.py", "./looks/relative.py", f"{root}/a.py", 1,
                 {"resolved": f"{root}/c.py", "note": "x" * 400}),
        EdgeInfo("CONTAINS", root, f"{root}/a.py", f"{root}/a.py", 0),
        EdgeInfo("REFERENCES", f"{root}/a.py::f", "!escaped", f"{root}/a.py", 5),
    ]


def test_rows_match_python(tmp_path):
    root = str(tmp_path)
    edges = _edges(root)
    assert crg_fast.edge_rows(edges, 12.5, json.dumps) == _python_rows(edges, 12.5)


def test_stored_rows_match_strip_params(tmp_path):
    root = str(tmp_path)
    edges = _edges(root)
    codec = PathCodec(root)
    expected = [codec.strip_params(row) for row in _python_rows(edges, 12.5)]
    stripper = codec.native_stripper()
    for _ in range(2):  # the second pass reads the cache
        got = crg_fast.edge_rows(edges, 12.5, json.dumps, stripper)
        assert got == expected
        assert [[type(v) for v in row] for row in got] == \
            [[type(v) for v in row] for row in expected]
