"""Run only the tests the graph says a change can reach.

Load it explicitly; it is not registered as an entry point, so installing
code-review-graph changes no test run::

    pytest -p code_review_graph.pytest_plugin --crg-affected [--crg-base REF]

Whenever selection is unsure (see :mod:`code_review_graph.affected_tests`)
every test runs, and the reason is printed in the header.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("code-review-graph")
    group.addoption("--crg-affected", action="store_true", default=False,
                    help="Run only test files the code-review-graph graph links to the change")
    group.addoption("--crg-base", default="HEAD",
                    help="Git ref the change is measured from (default HEAD)")


def _repo_root(config: pytest.Config) -> Path:
    from .incremental import find_project_root

    return find_project_root(Path(str(config.rootpath)))


def pytest_configure(config: pytest.Config) -> None:
    if not config.getoption("crg_affected"):
        return
    from .affected_tests import affected_tests

    root = _repo_root(config)
    result = affected_tests(root, config.getoption("crg_base"))
    config.stash[_RESULT] = (root, result)


_RESULT = pytest.StashKey[tuple]()


def pytest_report_header(config: pytest.Config) -> list[str]:
    if _RESULT not in config.stash:
        return []
    _root, result = config.stash[_RESULT]
    if result["mode"] == "all":
        return [f"code-review-graph: running every test ({result['reason']})"]
    return [f"code-review-graph: {len(result['tests'])} affected test file(s) "
            f"({result['reason']})"]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    if _RESULT not in config.stash:
        return
    root, result = config.stash[_RESULT]
    if result["mode"] == "all":
        return
    wanted = {(root / t).resolve() for t in result["tests"]}
    keep: list[pytest.Item] = []
    drop: list[pytest.Item] = []
    for item in items:
        (keep if Path(str(item.path)).resolve() in wanted else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
