"""TOON output leaves out only the fields the rest give back exactly.

``drop_repeated_fields`` runs before ``encode`` in the CLI and the MCP server;
``restore_repeated_fields`` after ``decode`` rebuilds every field ``--json``
shows, so the shorter output loses nothing.

- An edge that repeats its result row is folded into the row as its id, plus
  its tier and confidence when it is not a plain extracted one. The kind,
  which end the row is and the other end are written once. An edge that does
  not repeat its row stays as it is.
- A node's ``qualified_name`` is left out when it is ``<file>::<name>`` or
  ``<file>::<parent>.<name>``; one that says more stays.
- A node's ``is_test`` is written into ``kind`` (``Function (test)``).
- ``language`` is left out of a list only when every node's file extension
  maps to it.
- Real answers from every query pattern and from other tools come back
  field for field.

From PR #3.
"""

from __future__ import annotations

import pytest

from code_review_graph.toon import decode, drop_repeated_fields, encode, restore_repeated_fields


def _node(name, path="app/pay.py", **extra):
    node = {
        "id": 7, "kind": "Function", "name": name, "qualified_name": f"{path}::{name}",
        "file_path": path, "line_start": 3, "line_end": 9, "language": "python",
        "parent_name": None, "is_test": False,
    }
    node.update(extra)
    return node


def _without_nulls(value):
    # TOON writes a field a table row lacks as null, so "absent" and "null"
    # read back the same; compare without either.
    if isinstance(value, dict):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_nulls(v) for v in value]
    return value


def _round_trip(value):
    return restore_repeated_fields(decode(encode(drop_repeated_fields(value))))


def _assert_lossless(value):
    assert _without_nulls(_round_trip(value)) == _without_nulls(decode(encode(value)))


def _edge(source, target, line, **extra):
    edge = {
        "id": line * 10, "kind": "CALLS", "source": source, "target": target,
        "file_path": "app/pay.py", "line": line, "confidence": 1.0,
        "confidence_tier": "EXTRACTED",
    }
    edge.update(extra)
    return edge


def test_an_edge_that_repeats_its_row_is_folded_into_it():
    response = {
        "results": [
            {**_node("charge"), "call_site": {"line": 5}},
            {**_node("refund", id=8), "call_site": {"line": 12}},
        ],
        "edges": [
            _edge("app/pay.py::charge", "app/db.py::save", 5),
            _edge("app/pay.py::refund", "app/db.py::save", 12,
                  confidence=0.8, confidence_tier="INFERRED"),
            _edge("app/cron.py::nightly", "app/db.py::save", 2),
        ],
    }

    out = drop_repeated_fields(response)

    charge, refund = out["results"]
    assert charge["edge"] == {"id": 50}
    assert refund["edge"] == {"id": 120, "tier": "INFERRED", "confidence": 0.8}
    assert out["edges"] == [response["edges"][2]]
    assert out["edges_in_rows"] == {
        "kind": "CALLS", "row_end": "source", "other": "app/db.py::save",
    }
    assert "id" in response["results"][0] and "edge" not in response["results"][0]
    _assert_lossless(response)


def test_an_edge_that_says_more_than_its_row_stays():
    response = {
        "results": [{**_node("charge"), "call_site": {"line": 5}}],
        "edges": [_edge("app/pay.py::charge", "app/db.py::save", 6)],
    }

    out = drop_repeated_fields(response)

    assert out["edges"] == response["edges"]
    assert "edge" not in out["results"][0]
    _assert_lossless(response)


def test_node_fields_the_rest_give_back_are_left_out():
    rows = [
        _node("charge"),
        _node("check_charge", is_test=True),
        _node("test_charge", kind="Test", is_test=True),
        _node("save", parent_name="Ledger", qualified_name="app/pay.py::Ledger.save"),
    ]

    out = drop_repeated_fields({"results": rows})["results"]

    assert out[0] == {
        "id": 7, "kind": "Function", "name": "charge", "file_path": "app/pay.py",
        "line_start": 3, "line_end": 9, "parent_name": None,
    }
    assert [row["kind"] for row in out] == ["Function", "Function (test)", "Test", "Function"]
    assert all("qualified_name" not in row for row in out)
    _assert_lossless({"results": rows})


@pytest.mark.parametrize(
    ("odd", "kept"),
    [
        ({"qualified_name": "app/pay.py::save#2"}, ["qualified_name"]),
        ({"kind": "Test", "is_test": False}, ["kind", "is_test"]),
        ({"language": None}, ["language"]),
        ({"file_path": "include/pay.h", "qualified_name": "include/pay.h::charge",
          "language": "cpp"}, ["language"]),
    ],
    ids=["overload-suffix", "test-kind-not-a-test", "no-language", "header-not-c"],
)
def test_a_field_the_rest_cannot_give_back_stays(odd, kept):
    rows = [_node("charge", **odd), _node("refund")]

    out = drop_repeated_fields({"results": rows})["results"]

    for key in kept:
        assert out[0][key] == odd[key]
    _assert_lossless({"results": rows})


def test_other_records_keep_every_field():
    rows = [_node("charge"), {"name": "bool", "qualified_name": "bool", "kind": "Function"}]
    other = {"id": 1, "name": "x", "language": "go"}

    out = drop_repeated_fields({"results": rows, "other": other})

    assert out["results"][1] == rows[1]
    assert out["other"] == other
    _assert_lossless({"results": rows, "other": other})


_REPO = {
    "app/__init__.py": "",
    "app/db.py": (
        "class Ledger:\n"
        "    def save(self, row):\n"
        "        return row\n\n"
        "def connect():\n"
        "    return Ledger()\n"
    ),
    "app/pay.py": (
        "from app.db import connect\n\n"
        "def charge(amount):\n"
        "    ledger = connect()\n"
        "    ledger.save(amount)\n"
        "    print(amount)\n"
        "    return refund(amount)\n\n"
        "def refund(amount):\n"
        "    return -amount\n"
    ),
    "tests/test_pay.py": (
        "from app.pay import charge\n\n"
        "def test_charge():\n"
        "    assert charge(2) == -2\n"
    ),
}


@pytest.fixture(scope="module")
def built_repo(tmp_path_factory):
    from code_review_graph.graph import GraphStore
    from code_review_graph.incremental import full_build

    root = tmp_path_factory.mktemp("lossless")
    (root / ".git").mkdir()
    for rel, text in _REPO.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    (root / ".code-review-graph").mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CRG_SERIAL_PARSE", "1")
        with GraphStore(root / ".code-review-graph" / "graph.db") as store:
            full_build(root, store)
    return root


def _query_answers(root):
    from code_review_graph.tools import query_graph
    from code_review_graph.tools.query import _QUERY_PATTERNS

    for pattern in sorted(_QUERY_PATTERNS):
        for target in ("charge", "app/pay.py", "Ledger", "connect"):
            extra = {"to": "save"} if pattern == "path_to" else {}
            yield f"{pattern}:{target}", query_graph(pattern, target, repo_root=str(root), **extra)


def test_every_query_answer_comes_back_field_for_field(built_repo):
    folded = 0
    for label, answer in _query_answers(built_repo):
        assert _without_nulls(_round_trip(answer)) == _without_nulls(
            decode(encode(answer))
        ), label
        folded += "edges_in_rows" in drop_repeated_fields(answer)
    assert folded, "no answer folded an edge, so the check proved nothing"


def test_other_tool_answers_come_back_field_for_field(built_repo):
    from code_review_graph import tools

    root = str(built_repo)
    answers = {
        "impact": tools.get_impact_radius(changed_files=["app/pay.py"], repo_root=root),
        "search": tools.semantic_search_nodes(query="charge", repo_root=root),
        "large": tools.find_large_functions(min_lines=1, repo_root=root),
        "hubs": tools.get_hub_nodes_func(repo_root=root),
        "stats": tools.list_graph_stats(repo_root=root),
    }
    for label, answer in answers.items():
        assert answer.get("status") == "ok", (label, answer)
        assert _without_nulls(_round_trip(answer)) == _without_nulls(
            decode(encode(answer))
        ), label
