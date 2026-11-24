"""The native row factory returns what ``PathCodec.row_factory`` does (CRG_FAST_ENGINE).

``crg_fast.RowRestorer`` gives every row of a relative graph its root back.
Each value must come out equal to the Python factory's, with the same type,
for the common ``./x`` form and every rarer one it hands to ``restore``:
the root token, escapes, JSON text, metadata columns left alone, and values
that are not text.
"""

import sqlite3

import pytest

from code_review_graph.relpaths import PathCodec, TranslatingConnection

crg_fast = pytest.importorskip("crg_fast")

_VALUES = [
    "./a.py", "./pkg/b.py::Helper.ping", ".", "./", ".hidden", "..", "../up",
    "!./missing", "!!./missing", "!.", "!plain", "!", "plain", "", "ünï/./x",
    '{"a": "./b.py"}', '{"a": "!./x", "b": "."}', '{"a": 1}', '["./x", "y"]',
    '"./x"', "[1, 2]", "{bad json \".", None, 7, 2.5, b"./bytes",
]


def _rows(factory, sql, params=()):
    conn = sqlite3.connect(":memory:", factory=TranslatingConnection)
    conn.row_factory = factory
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [[(type(value), value) for value in row] for row in rows], [
        tuple(row.keys()) for row in rows
    ]


@pytest.mark.parametrize("root", ["/repo", "/r/ü ber"])
def test_rows_match_the_python_factory(root):
    codec = PathCodec(root)
    restorer = crg_fast.RowRestorer(codec.root, codec.restore, codec._spec, sqlite3.Row)
    placeholders = ", ".join(f"? AS c{index}" for index in range(len(_VALUES)))
    queries = [
        (f"SELECT {placeholders}", _VALUES),
        ("SELECT ? AS key, ? AS value, ? AS path", ("./k", "./v", "./p")),
        ("SELECT ? AS x UNION ALL SELECT ? UNION ALL SELECT ?", (".", "./a", "!./b")),
    ]
    for sql, params in queries:
        assert _rows(restorer, sql, params) == _rows(codec.row_factory, sql, params)


def test_a_new_statement_gets_its_own_columns():
    codec = PathCodec("/repo")
    restorer = crg_fast.RowRestorer(codec.root, codec.restore, codec._spec, sqlite3.Row)
    conn = sqlite3.connect(":memory:", factory=TranslatingConnection)
    conn.row_factory = restorer
    assert tuple(conn.execute("SELECT './a' AS value, './b' AS p").fetchone()) == (
        "./a", "/repo/b",
    )
    assert tuple(conn.execute("SELECT './a' AS p, './b' AS key").fetchone()) == (
        "/repo/a", "./b",
    )
