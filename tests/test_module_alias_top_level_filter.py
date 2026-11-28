"""Under CRG_FAST_ENGINE a module swap is looked for only where Python runs one.

``_read_python_module_alias`` follows ``sys.modules[__name__] = ...`` only as a
module-level statement. With the fast engine a file whose ``sys.modules``
never starts a module-level statement (an indented docstring example, a
nested block) is answered without ``ast.parse``. Every spelling Python
accepts at module level still reaches ``ast`` and is followed.
"""

import itertools
from pathlib import Path

import pytest

from code_review_graph import parser as parser_mod

read_alias = parser_mod.CodeParser._read_python_module_alias

HEADS = [
    "", "﻿", "x = 1; ", "x = 1;", "(", "(\n  ", "( # note\n ", "((",
    " \\\n", "\f", "x = 1; \\\n ",
]
EOLS = ["\n", "\r\n", "\r"]


def _swap(head: str, eol: str) -> bytes:
    body = f"import importlib{eol}import sys{eol}{head}sys.modules[__name__]"
    body += ")" * head.count("(") + " = importlib.import_module('app.real')" + eol
    return body.encode("utf-8")


@pytest.mark.parametrize("fast", ["0", "1"])
def test_every_module_level_spelling_is_followed(monkeypatch, fast):
    monkeypatch.setenv("CRG_FAST_ENGINE", fast)
    checked = 0
    for head, eol in itertools.product(HEADS, EOLS):
        source = _swap(head, eol)
        try:
            compile(source, "shim.py", "exec")
        except SyntaxError:
            continue
        checked += 1
        assert read_alias(Path("shim.py"), source) == "app.real", (head, eol)
    assert checked > 20


def test_an_indented_example_is_answered_without_ast(monkeypatch):
    source = (
        b'"""Shim docs.\n\n'
        b"    import importlib, sys\n"
        b"    sys.modules[__name__] = importlib.import_module('app.real')\n"
        b'"""\n\n\ndef f():\n    return 1\n'
    )
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")

    def no_parse(*_args, **_kwargs):
        raise AssertionError("ast.parse was called")

    monkeypatch.setattr(parser_mod.ast, "parse", no_parse)
    assert read_alias(Path("shim.py"), source) is None


def test_a_nested_swap_is_declined_either_way(monkeypatch):
    source = (
        b"import importlib, sys\n\nif True:\n"
        b"    sys.modules[__name__] = importlib.import_module('app.real')\n"
    )
    for fast in ("0", "1"):
        monkeypatch.setenv("CRG_FAST_ENGINE", fast)
        assert read_alias(Path("shim.py"), source) is None
