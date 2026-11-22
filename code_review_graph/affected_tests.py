"""The test files a change can reach, or "run everything" when unsure.

Selection walks the impact radius of the changed files (callers, importers,
TESTED_BY) and keeps the test files it reaches. A wrong "skip" is worse than
a slow run, so any doubt selects every test:

* a changed file the graph does not know, or one deleted since the build;
* the graph is stale: a changed file's bytes differ from what it recorded;
* test configuration changed (``conftest.py``, ``pyproject.toml`` ...);
* the impact walk stopped at its node cap;
* the changed files could not be listed at all.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

#: Files whose change can alter any test's outcome.
_CONFIG_NAMES = frozenset({
    "conftest.py", "pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini",
    "setup.py", "uv.lock", "poetry.lock", "Pipfile.lock", "noxfile.py",
    "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
    "jest.config.js", "jest.config.ts", "vitest.config.ts", "vitest.config.js",
})
_CONFIG_PREFIXES = ("requirements",)
#: Changes to these never change a test's outcome on their own.
_INERT_SUFFIXES = frozenset({".md", ".rst", ".txt", ".png", ".jpg", ".svg", ".gif"})


def _all(reason: str, changed: list[str]) -> dict:
    return {"mode": "all", "reason": reason, "changed": changed, "tests": []}


def _is_config(rel: str) -> bool:
    name = PurePosixPath(rel).name
    return name in _CONFIG_NAMES or name.startswith(_CONFIG_PREFIXES)


def affected_tests(
    repo_root: Path, base: str = "HEAD", changed: Optional[list[str]] = None,
) -> dict:
    """``{"mode": "selected"|"all", "reason", "changed", "tests"}``; ``tests``
    holds repository-relative test file paths when ``mode`` is ``selected``."""
    from .errors import CodeReviewGraphError
    from .graph import GraphStore
    from .incremental import get_changed_files, get_db_path
    from .parser import is_test_file

    repo_root = repo_root.resolve()
    if changed is None:
        try:
            changed = get_changed_files(repo_root, base, strict=True)
        except (CodeReviewGraphError, RuntimeError, OSError) as exc:
            return _all(f"changed files could not be listed: {exc}", [])
    changed = sorted({Path(c).as_posix() for c in changed})
    if not changed:
        return {"mode": "selected", "reason": "nothing changed", "changed": [], "tests": []}
    config = [c for c in changed if _is_config(c)]
    if config:
        return _all(f"test configuration changed: {config[0]}", changed)

    db = get_db_path(repo_root, read_only=True)
    if not db.is_file():
        return _all("no graph; run `code-review-graph build`", changed)
    try:
        store = GraphStore(db, read_only=True)
    except (CodeReviewGraphError, OSError) as exc:
        return _all(f"graph could not be opened: {exc}", changed)
    try:
        recorded = store.get_file_hashes()
        seeds: list[str] = []
        tests: set[str] = set()
        for rel in changed:
            full = repo_root / rel
            if PurePosixPath(rel).suffix.lower() in _INERT_SUFFIXES:
                continue
            if not full.is_file():
                return _all(f"{rel} was deleted; its importers are no longer in the graph",
                            changed)
            stored = recorded.get(str(full))
            if stored is None:
                return _all(f"{rel} is not in the graph", changed)
            if hashlib.sha256(full.read_bytes()).hexdigest() != stored:
                return _all(
                    f"the graph is stale for {rel}; run `code-review-graph update`", changed,
                )
            if is_test_file(rel, repo_root):
                tests.add(rel)
            seeds.append(str(full))
        if not seeds:
            return {"mode": "selected", "reason": "only inert files changed",
                    "changed": changed, "tests": sorted(tests)}
        impact = store.get_impact_radius(seeds)
    finally:
        store.close()
    if impact.get("truncated"):
        return _all("the impact walk reached its node cap", changed)
    prefix = str(repo_root) + "/"
    for node in impact.get("impacted_nodes", []) + impact.get("changed_nodes", []):
        path = getattr(node, "file_path", "") or ""
        if path.startswith(prefix):
            rel = path[len(prefix):]
            if is_test_file(rel, repo_root):
                tests.add(rel)
    return {"mode": "selected", "reason": f"{len(changed)} changed file(s)",
            "changed": changed, "tests": sorted(tests)}


def format_affected(result: dict) -> str:
    if result["mode"] == "all":
        return f"Run every test: {result['reason']}."
    if not result["tests"]:
        return f"No test reaches the change ({result['reason']})."
    return "\n".join(result["tests"])
