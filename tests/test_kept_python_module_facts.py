"""Facts kept across processes: Python module skeletons and annotations, and
the record of grammar probes that passed.

An update parses the modules a changed Python file imports to read their star
exports and annotations. Those answers depend on the module's bytes alone, so
they are kept beside the source copies under the module's sha256 and read back
by the next update. The grammar crash probe is likewise recorded per
interpreter and grammar library, so each update process does not start it
again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from code_review_graph import parser as parser_module
from code_review_graph.parser import CodeParser

MODULE = b'''\
"""Doc."""
import os
from .impl import charge as pay
from typing import Optional

__all__ = ["Shape", "area", "pay", "helper"]

LIMIT: int = 3


@decorator(1)
def area(x: int) -> "Shape":
    return x * LIMIT


class Shape:
    @property
    def size(self) -> int:
        return 1

    def grow(self) -> "Shape":
        return self


async def helper() -> Optional[int]:
    if os.environ:
        return 1
    return None


def _private():
    pass
'''


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    parser_module._clear_parser_probe_cache()
    yield
    parser_module._clear_parser_probe_cache()


def _module(tmp_path: Path, data: bytes = MODULE) -> Path:
    pkg = tmp_path / "repo" / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_bytes(b"")
    (pkg / "impl.py").write_bytes(b"def charge():\n    pass\n")
    path = pkg / "mod.py"
    path.write_bytes(data)
    return path


def _facts(parser: CodeParser, path: Path) -> tuple:
    return (
        parser._read_python_star_exports(path, frozenset()),
        parser._read_python_property_return_annotations(path),
        parser._read_python_return_annotations(path),
    )


def test_kept_facts_read_the_same_as_parsing(tmp_path: Path) -> None:
    path = _module(tmp_path)
    plain = _facts(CodeParser(tmp_path / "repo"), path)
    facts = tmp_path / "facts"

    first = CodeParser(tmp_path / "repo")
    first.use_python_skeletons(facts, "print-a")
    assert _facts(first, path) == plain
    digest = hashlib.sha256(MODULE).hexdigest()
    assert sorted(p.name for p in facts.iterdir()) == [
        f"{digest}.props", f"{digest}.rets", f"{digest}.skel",
    ]
    skeleton = (facts / f"{digest}.skel").read_bytes()
    assert b"return x * LIMIT" not in skeleton

    second = CodeParser(tmp_path / "repo")
    second.use_python_skeletons(facts, "print-a")
    assert _facts(second, path) == plain


def test_facts_kept_by_other_code_or_for_other_bytes_are_not_read(tmp_path: Path) -> None:
    path = _module(tmp_path)
    facts = tmp_path / "facts"
    first = CodeParser(tmp_path / "repo")
    first.use_python_skeletons(facts, "print-a")
    _facts(first, path)
    digest = hashlib.sha256(MODULE).hexdigest()
    # A wrong answer under the old fingerprint: other code must not read it.
    (facts / f"{digest}.rets").write_bytes(b'crg-skel 1 print-a\n[["area", "Wrong"]]')

    other = CodeParser(tmp_path / "repo")
    other.use_python_skeletons(facts, "print-b")
    assert other._read_python_return_annotations(path) == (
        CodeParser(tmp_path / "repo")._read_python_return_annotations(path)
    )

    changed = MODULE.replace(b"-> int:", b"-> str:")
    _module(tmp_path, changed)
    fresh = CodeParser(tmp_path / "repo")
    fresh.use_python_skeletons(facts, "print-a")
    assert _facts(fresh, path) == _facts(CodeParser(tmp_path / "repo"), path)


def test_a_module_that_does_not_parse_cleanly_keeps_no_skeleton(tmp_path: Path) -> None:
    broken = MODULE + b"\ndef broken(:\n"
    path = _module(tmp_path, broken)
    facts = tmp_path / "facts"
    parser = CodeParser(tmp_path / "repo")
    parser.use_python_skeletons(facts, "print-a")
    assert parser._read_python_star_exports(path, frozenset()) == (
        CodeParser(tmp_path / "repo")._read_python_star_exports(path, frozenset())
    )
    digest = hashlib.sha256(broken).hexdigest()
    assert not (facts / f"{digest}.skel").exists()


def test_a_recorded_probe_pass_spares_the_next_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[str] = []
    identity = ["lib-v1"]

    def fake_probe(grammar: str, _timeout: float) -> bool:
        probes.append(grammar)
        return grammar != "broken"

    monkeypatch.setattr(parser_module, "_run_parser_load_probe", fake_probe)
    monkeypatch.setattr(
        parser_module, "_parser_probe_identity", lambda grammar: f"{grammar}:{identity[0]}",
    )
    record = tmp_path / "parser-probes.json"

    def new_process() -> None:
        parser_module._clear_parser_probe_cache()
        parser_module.use_parser_probe_record(record)

    new_process()
    assert parser_module._parser_load_probe_succeeds("python")
    assert not parser_module._parser_load_probe_succeeds("broken")
    new_process()
    assert parser_module._parser_load_probe_succeeds("python")
    assert not parser_module._parser_load_probe_succeeds("broken")
    # A failure is never recorded, so it is probed again.
    assert probes == ["python", "broken", "broken"]

    identity[0] = "lib-v2"  # The grammar library changed.
    new_process()
    assert parser_module._parser_load_probe_succeeds("python")
    assert probes[-1] == "python"

    parser_module._clear_parser_probe_cache()  # No record in use: probe again.
    assert parser_module._parser_load_probe_succeeds("python")
    assert probes.count("python") == 3
