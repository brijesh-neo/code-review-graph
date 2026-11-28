"""Ignore matching and repository-relative paths answer as they did.

``_should_ignore`` now sorts each pattern list by kind once and matches the
globs with one regular expression on POSIX, and the hot relative-path
spellings skip pathlib when a plain prefix answers. Both are checked against
the code they replaced on awkward spellings. Pinned for PR #13 on the fork.
"""

from __future__ import annotations

import fnmatch
import itertools
import os
from pathlib import Path, PurePosixPath

import pytest

from code_review_graph.incremental import (
    DEFAULT_IGNORE_PATTERNS,
    _relative_posix,
    _should_ignore,
)


def _reference_should_ignore(path: str, patterns: list[str]) -> bool:
    """``_should_ignore`` as it was before the fast path."""
    normalized = path.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts
    for pattern in patterns:
        anchored = pattern.startswith("/")
        candidate = pattern[1:] if anchored else pattern
        if candidate.startswith("**/") and candidate.endswith("/**"):
            segment = candidate[3:-3]
            if segment and segment in parts:
                return True
            continue
        if candidate.endswith("/**"):
            prefix = tuple(part for part in candidate[:-3].split("/") if part)
            if not prefix:
                continue
            if anchored or len(prefix) > 1:
                if parts[: len(prefix)] == prefix:
                    return True
            elif prefix[0] in parts:
                return True
            continue
        if fnmatch.fnmatch(normalized, candidate):
            return True
    return False


_PATTERNS = [
    *DEFAULT_IGNORE_PATTERNS,
    "**/**", "**//**", "/**", "**/a/b/**", "/build/**", "build/**", "src/gen/**",
    "*.min.js", "/top.py", "docs/*.md", "[ab]*.py", "?x.py", "**/*.lock", "a/**/b.py",
    "*", "", "/", "weird(.py", "x|y.py", "[!c]*.txt",
]
_PATHS = [
    "", ".", "/", "a.py", "/a.py", "//a.py", "a//b.py", "./a.py", "a/./b.py", "a/",
    "build/x.py", "src/build/x.py", "/build/x.py", "src/gen/x.py", "gen/src/x.py",
    "node_modules/p/i.js", "x/node_modules/i.js", "app.min.js", "lib/app.min.js",
    "top.py", "sub/top.py", "docs/a.md", "docs/sub/a.md", "a.py", "bx.py", "c.txt",
    "d.txt", "a/b/c.py", "a/x/b.py", "a\\b\\c.py", "x|y.py", "weird(.py", "..",
    "../a.py", "a/../b.py", "Cargo.lock", "deep/Cargo.lock", "a\nb.py", ".git/x",
]


def test_ignore_matching_agrees_with_the_previous_code() -> None:
    pattern_sets = [list(DEFAULT_IGNORE_PATTERNS), []]
    pattern_sets += [[pattern] for pattern in _PATTERNS]
    pattern_sets += [list(pair) for pair in itertools.combinations(_PATTERNS[-20:], 2)]
    pattern_sets.append(_PATTERNS)
    for patterns in pattern_sets:
        for path in _PATHS:
            expected = _reference_should_ignore(path, patterns)
            assert _should_ignore(path, patterns) is expected, (path, patterns)


@pytest.mark.parametrize("root", ["/repo", "/repo/sub", "/", "//repo", "/re"])
def test_relative_paths_agree_with_pathlib(root: str) -> None:
    def outcome(function, path: Path) -> object:
        try:
            return function(path)
        except ValueError:
            return ValueError

    base = Path(root)
    spellings = [
        "/repo", "/repo/a.py", "/repo/sub/b.py", "/repo//a.py", "/repo/./a.py",
        "/repo/a/", "/repository/a.py", "/rep", "/repo/../x.py", "//repo/a.py",
        "/a.py", "relative/a.py", "/REPO/a.py", "/repo/sub", "/re/a",
    ]
    for spelling in spellings:
        path = Path(spelling)
        assert outcome(lambda p: _relative_posix(p, base), path) == outcome(
            lambda p: p.relative_to(base).as_posix(), path,
        ), (root, spelling)


@pytest.mark.skipif(os.name != "posix", reason="the fast path is POSIX only")
def test_the_fast_path_is_taken_for_a_plain_spelling(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pathlib was asked")

    monkeypatch.setattr(Path, "relative_to", refuse)
    assert _relative_posix(Path("/repo/a/b.py"), Path("/repo")) == "a/b.py"
