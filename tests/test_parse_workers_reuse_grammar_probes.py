"""Parse workers start with the grammar probe results the parent already has.

Each grammar is probed once in a throwaway interpreter before it is loaded. A
spawned worker starts with an empty cache, so every worker probed every
grammar again: one extra interpreter per grammar per worker on each build and
update. The pool now hands the parent's results to each worker.
"""

from code_review_graph import incremental, parser


def test_workers_start_with_the_parents_probe_results(monkeypatch):
    monkeypatch.setattr(incremental, "_select_executor_kind", lambda: "process")
    parser._clear_parser_probe_cache()
    try:
        parser.seed_parser_probe_results({"python": True, "not_a_grammar": False})
        with incremental._make_executor(1) as executor:
            in_worker = executor.submit(parser.parser_probe_results).result(timeout=60)
    finally:
        parser._clear_parser_probe_cache()
    assert in_worker == {"python": True, "not_a_grammar": False}


def test_seeding_keeps_results_this_process_already_has():
    parser._clear_parser_probe_cache()
    try:
        parser.seed_parser_probe_results({"python": False})
        parser.seed_parser_probe_results({"python": True, "go": True})
        assert parser.parser_probe_results() == {"python": False, "go": True}
    finally:
        parser._clear_parser_probe_cache()
