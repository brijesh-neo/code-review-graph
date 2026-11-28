"""The ``TYPE_CHECKING`` shadowing scan skips scopes that cannot shadow.

Finding calls under ``if TYPE_CHECKING:`` asks each function and class whether
it rebinds an alias of ``TYPE_CHECKING``. A scope can only bind a name its own
text spells, so the check skips the binding scan where the bytes hold none of
the aliases, unless the file could spell one another way (an NFKC lookalike, a
byte order mark, a lone carriage return). These cases pin that the answers
equal the full scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_graph import parser as parser_module

CASES = {
    "shadowed in a function": (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    a.b()\n"
        "def f():\n    TYPE_CHECKING = 1\n    if TYPE_CHECKING:\n        c.d()\n"
        "def g():\n    if TYPE_CHECKING:\n        e.f()\n"
    ),
    "module alias shadowed in a class": (
        "import typing\n"
        "class K:\n    typing = 3\n    if typing.TYPE_CHECKING:\n        x.y()\n"
        "if typing.TYPE_CHECKING:\n    z.w()\n"
    ),
    "shadowed at module level": (
        "from typing import TYPE_CHECKING as TC\nTC = True\nif TC:\n    q.r()\n"
    ),
    "parameter": (
        "from typing import TYPE_CHECKING\n"
        "def g(TYPE_CHECKING):\n    if TYPE_CHECKING:\n        m.n()\n"
        "def h():\n    if not TYPE_CHECKING:\n        o.p()\n"
    ),
    "other non-ASCII text": (
        "from typing import TYPE_CHECKING\ns = 'é→'\n"
        "def f():\n    if TYPE_CHECKING:\n        c.d()\n"
        "class K:\n    TYPE_CHECKING = 0\n    if TYPE_CHECKING:\n        e.f()\n"
    ),
    "fullwidth lookalike": (
        "from typing import TYPE_CHECKING\n"
        "def f():\n    ＴＹＰＥ_CHECKING = 1\n    if TYPE_CHECKING:\n        c.d()\n"
    ),
    "byte order mark": (
        "\ufefffrom typing import TYPE_CHECKING\n"
        "def f():\n    TYPE_CHECKING = 1\n    if TYPE_CHECKING:\n        c.d()\n"
    ),
    "carriage returns": (
        "from typing import TYPE_CHECKING\r\n"
        "def f():\r\n    TYPE_CHECKING = 1\r\n    if TYPE_CHECKING:\r\n        c.d()\r\n"
    ),
}


def _full_scan(source: bytes, visitor_cls, extra: frozenset[str]) -> frozenset:
    tree = parser_module._python_module_ast(source)
    if tree is None:
        return frozenset()
    names, modules = parser_module._python_type_checking_aliases(tree)
    visitor = visitor_cls(names | extra, modules)
    visitor.visit(tree)
    return frozenset(visitor.positions)


@pytest.mark.parametrize("name", sorted(CASES))
def test_prefiltered_scan_equals_full_scan(name: str) -> None:
    source = CASES[name].encode("utf-8")
    assert parser_module._python_unreachable_call_positions.__wrapped__(source) == (
        _full_scan(source, parser_module._PythonUnreachableCallVisitor, frozenset())
    )
    assert parser_module._python_unreachable_import_positions.__wrapped__(source) == (
        _full_scan(
            source, parser_module._PythonUnreachableImportVisitor,
            frozenset({"TYPE_CHECKING"}),
        )
    )


def test_shadowing_is_still_seen() -> None:
    source = CASES["shadowed in a function"].encode()
    dead = parser_module._python_unreachable_call_positions.__wrapped__(source)
    assert (3, 4) in dead  # a.b() under the module's TYPE_CHECKING
    assert (10, 8) in dead  # e.f() in g, which does not shadow it
    assert (7, 8) not in dead  # c.d() in f, where TYPE_CHECKING = 1
    lookalike = CASES["fullwidth lookalike"].encode()
    assert parser_module._python_unreachable_call_positions.__wrapped__(lookalike) == frozenset()


def test_package_sources_read_the_same() -> None:
    package = Path(parser_module.__file__).parent
    for path in sorted(package.glob("*.py")):
        source = path.read_bytes()
        if parser_module._python_module_ast(source) is None:
            continue
        assert parser_module._python_unreachable_call_positions.__wrapped__(source) == (
            _full_scan(source, parser_module._PythonUnreachableCallVisitor, frozenset())
        ), path.name
