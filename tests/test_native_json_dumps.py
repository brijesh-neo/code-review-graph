"""The native JSON text is ``json.dumps``'s own (CRG_FAST_ENGINE).

``crg_fast.edge_rows`` and ``crg_fast.node_rows`` write each ``extra`` dict as
JSON text without calling ``json.dumps`` when the value is plain. The text
must be exactly what ``json.dumps`` writes with its defaults: every escape
``ensure_ascii`` makes (quotes, backslashes, control characters, DEL,
non-ASCII and astral characters), integers beyond 64 bits, float repr and
``NaN``/``Infinity``, tuples as arrays, insertion order and nesting. Values it
does not handle (subclasses, other key types, lone surrogates, cycles) must
reach the Python ``dumps``, so their text or error is Python's.
"""

import json
import math
import random

import pytest

crg_fast = pytest.importorskip("crg_fast")

_TEXT = "aZ09 \"\\/\b\f\n\r\t\x00\x1f\x7f\x80éü€￿\U0001f600\U0010ffff~"


def _value(rng, depth=0):
    choice = rng.randrange(12 if depth < 4 else 7)
    if choice == 0:
        return None
    if choice == 1:
        return rng.choice((True, False))
    if choice == 2:
        return rng.choice((0, -1, 7, 2**63 - 1, -(2**63), 2**64, -(10**40)))
    if choice == 3:
        return rng.choice((0.0, -0.0, 0.1, 1e16, 1e-7, 1.5e300, -2.5, math.inf,
                           -math.inf, math.nan, 123456789.123))
    if choice in (4, 5, 6):
        return "".join(rng.choice(_TEXT) for _ in range(rng.randrange(8)))
    if choice in (7, 8):
        return [_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    if choice == 9:
        return tuple(_value(rng, depth + 1) for _ in range(rng.randrange(3)))
    return {
        "".join(rng.choice(_TEXT) for _ in range(rng.randrange(5))): _value(rng, depth + 1)
        for _ in range(rng.randrange(5))
    }


def _native(value):
    def fallback(_value):
        raise AssertionError("handed to Python")

    return crg_fast.json_dumps(value, fallback)


def test_plain_values_match_json_dumps():
    rng = random.Random(7)
    for _ in range(3000):
        value = {"k": _value(rng), "confidence": 0.5, "tier": "INFERRED"}
        assert _native(value) == json.dumps(value)
    assert _native({}) == "{}"
    assert _native([]) == "[]"


class _Text(str):
    pass


class _Dict(dict):
    pass


class _Int(int):
    pass


@pytest.mark.parametrize("value", [
    {"a": _Text("x")},
    {"a": _Dict()},
    {"a": _Int(3)},
    {1: "int key"},
    {None: "none key"},
    {"a": "\udcff"},
    {"a": {1, 2}},
    {"a": b"bytes"},
])
def test_other_values_go_to_python(value):
    calls = []

    def fallback(given):
        calls.append(given)
        return "python"

    assert crg_fast.json_dumps(value, fallback) == "python"
    assert calls == [value]


def test_a_cycle_raises_as_python_does():
    value: dict = {}
    value["self"] = value
    with pytest.raises(ValueError, match="Circular reference"):
        crg_fast.json_dumps(value, json.dumps)
