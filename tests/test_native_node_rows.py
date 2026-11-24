"""The native node rows are the rows the Python store writes (CRG_FAST_ENGINE).

``crg_fast.node_rows`` builds a file's ``nodes`` rows: JSON ``extra``, the
``symbol``, the indexed docstring summary and the ``name_tokens`` camelCase
splits. Every stored column must come out as the Python store writes it,
with the same type, in an absolute and in a relative graph: acronyms, digits
and non-ASCII names, ``File`` nodes, empty parents and identity names, ``::``
in a path, docstrings with every ASCII separator Python splits at, long
ones, non-ASCII ones and ones that are not text, and nodes the native pass
hands back to Python (a ``str`` subclass or ``dict`` subclass, a stem that
is not UTF-8).
"""

import random

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.parser import NodeInfo

crg_fast = pytest.importorskip("crg_fast")

_NAMES = (
    "hashPassword", "HTTPServer", "getHTTPResponseCode", "parse2Json", "x86Asm",
    "ABC", "a", "", "_private_name", "ünïcodeName", "camelCase_withSnake",
    "Shapes.area", "ALLCaps2Go", "aB", "AbC", "iOSApp", "mixedHTTPAndXML",
)
_DOCSTRINGS = (
    None, "", "  Summary  line\twith\ttabs\n", "a\x0bb\x0cc\x1cd\x1de\x1ff\x1eg",
    "x" * 450, "word " * 120, "ünï  côdé nbsp sep", 7, ["list"], "   ",
)


class _Text(str):
    """A ``str`` subclass, which the native pass leaves to Python."""


def _nodes(path):
    rng = random.Random(4)
    nodes = [NodeInfo("File", path, path, 1, 90, "python")]
    for index in range(160):
        extra = {}
        docstring = _DOCSTRINGS[index % len(_DOCSTRINGS)]
        if index % 4:
            extra["docstring"] = docstring
        if index % 5 == 0:
            extra["decorators"] = ["route"]
        name = rng.choice(_NAMES) + (str(index) if index % 3 else "")
        nodes.append(NodeInfo(
            "Test" if index % 7 == 0 else "Function" if index % 2 else "Class",
            _Text(name) if index == 11 else name,
            path, index + 2, index + 3, "python",
            parent_name=(None, "", "OuterHTTPThing", "pkg.mod")[index % 4],
            params="(self)" if index % 2 else None,
            return_type="int" if index % 3 == 0 else None,
            is_test=index % 7 == 0,
            extra=extra,
            identity_name=(None, "", f"renamed{index}")[index % 3],
        ))
    return nodes


def _columns(store):
    return [
        tuple((type(value), value) for value in row)
        for row in store._conn.execute(
            "SELECT kind, name, qualified_name, file_path, line_start, line_end, language, "
            "parent_name, params, return_type, modifiers, is_test, file_hash, extra, symbol, "
            "docstring, name_tokens FROM nodes ORDER BY qualified_name"
        )
    ]


def _stored(tmp_path, monkeypatch, fast, root, path):
    monkeypatch.setenv("CRG_FAST_ENGINE", "1" if fast else "0")
    folder = tmp_path / ("fast" if fast else "python")
    folder.mkdir()
    store = GraphStore(folder / "graph.db")
    if root:
        store.bind_repo_root(root)
    store.store_file_nodes_edges(path, _nodes(path), [], "hash1")
    return _columns(store)


@pytest.mark.parametrize("root, path", [
    (None, "/repo/pkg/mod_HTTPThing.py"),
    ("/repo", "/repo/pkg/mod_HTTPThing.py"),
    ("/repo", "/repo/a::b/serverName.py"),
])
def test_native_rows_match_the_python_store(tmp_path, monkeypatch, root, path):
    python = _stored(tmp_path, monkeypatch, False, root, path)
    fast = _stored(tmp_path, monkeypatch, True, root, path)
    assert len(fast) == 161
    assert fast == python


def test_a_node_it_cannot_read_is_built_by_python():
    calls = []

    def node_row(node, qualified):
        calls.append(qualified)
        return ("python",)

    nodes = [
        NodeInfo("Function", _Text("f"), "a.py", 1, 2),
        NodeInfo("Function", "g", "a.py", 1, 2, extra=_Dict(a=1)),
        NodeInfo("Function", "h", "a.py", 1, 2, parent_name=3),
        NodeInfo("Function", "k", "a.py", 1, 2),
    ]
    rows = crg_fast.node_rows(
        nodes, ["a.py::f", "a.py::g", "a.py::h", "a.py::k"], "", 0.0, "a",
        lambda value: "{}", lambda extra: None, node_row,
    )
    assert calls == ["a.py::f", "a.py::g", "a.py::h"]
    assert rows[:3] == [("python",)] * 3
    assert rows[3][:3] == ("Function", "k", "a.py::k")

    calls.clear()
    rows = crg_fast.node_rows(
        nodes[3:], ["a.py::k"], "", 0.0, "\udcff", lambda value: "{}",
        lambda extra: None, node_row,
    )
    assert calls == ["a.py::k"]
    assert rows == [("python",)]


class _Dict(dict):
    """A ``dict`` subclass, which the native pass leaves to Python."""
