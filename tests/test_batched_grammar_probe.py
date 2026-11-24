"""Grammar probes share one child process under CRG_FAST_ENGINE.

Before the parse pool starts, ``_prewarm_parser_probes`` checks every grammar
the build needs in a disposable interpreter. With the fast engine one child
loads them all in turn. Only a grammar that child reports as loaded is
recorded. A grammar that fails, crashes the child or is not reached before
the timeout is left for its own per-grammar probe, so its failure detail and
warning are the same as before.
"""

import threading
from types import SimpleNamespace

import pytest

from code_review_graph import incremental
from code_review_graph import parser as parser_mod

pytest.importorskip("tree_sitter_language_pack")


def _run_parser_load_probe(grammar, timeout):  # same qualname as the real probe
    raise AssertionError("not called by the batch")


def _parser_mod(timeout=30.0):
    return SimpleNamespace(
        _run_parser_load_probe=_run_parser_load_probe,
        _parser_load_timeout_seconds=lambda: timeout,
        _parser_probe_env=parser_mod._parser_probe_env,
        _PARSER_PROBE_LOCK=threading.Lock(),
        _PARSER_PROBE_RESULTS={},
        _PARSER_PROBE_FAILURE_DETAILS={"go": "stale detail"},
    )


def test_loaded_grammars_are_recorded_and_the_rest_left_to_their_own_probe():
    mod = _parser_mod()
    left = incremental._batch_probe_grammars(mod, ["python", "no_such_grammar_x", "go"])
    assert left == ["no_such_grammar_x"]
    assert mod._PARSER_PROBE_RESULTS == {"python": True, "go": True}
    assert mod._PARSER_PROBE_FAILURE_DETAILS == {}


def test_a_crash_leaves_every_grammar_after_it_unprobed(monkeypatch):
    monkeypatch.setattr(incremental, "_BATCH_PROBE_CODE", (
        "import os, sys\n"
        "print(sys.argv[1], flush=True)\n"
        "os.abort()\n"
    ))
    mod = _parser_mod()
    assert incremental._batch_probe_grammars(mod, ["python", "go", "c"]) == ["go", "c"]
    assert mod._PARSER_PROBE_RESULTS == {"python": True}


def test_a_hang_keeps_what_loaded_before_the_timeout(monkeypatch):
    monkeypatch.setattr(incremental, "_BATCH_PROBE_CODE", (
        "import sys, time\n"
        "print(sys.argv[1], flush=True)\n"
        "sys.stdout.write('go')\n"  # an unfinished line is not a report
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    ))
    mod = _parser_mod(timeout=1.0)
    assert incremental._batch_probe_grammars(mod, ["python", "go"]) == ["go"]
    assert mod._PARSER_PROBE_RESULTS == {"python": True}


def test_a_replaced_probe_decides_every_grammar_itself():
    mod = _parser_mod()
    mod._run_parser_load_probe = lambda grammar, timeout: True
    assert incremental._batch_probe_grammars(mod, ["python", "go"]) == ["python", "go"]
    assert mod._PARSER_PROBE_RESULTS == {}


def test_prewarm_gives_the_per_grammar_results(monkeypatch):
    files = ["a.py", "b.go", "c.rs", "d.unknownext"]
    results = {}
    for fast in ("0", "1"):
        monkeypatch.setenv("CRG_FAST_ENGINE", fast)
        parser_mod._clear_parser_probe_cache()
        incremental._prewarm_parser_probes(parser_mod.CodeParser(), files)
        results[fast] = dict(parser_mod._PARSER_PROBE_RESULTS)
    parser_mod._clear_parser_probe_cache()
    assert results["1"] == results["0"]
    assert {"python", "go", "rust"} <= set(results["1"])
