"""Resolving changed paths a directory at a time gives ``Path.resolve``'s answers.

``expand_changed_file_paths`` resolves every changed file; doing it with one
``realpath`` per directory instead of one per file must not change a result,
whatever links, missing entries or ``..`` steps the paths hold. Pinned for
PR #13 on the fork.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from code_review_graph.flows import _resolver


@pytest.mark.skipif(os.name != "posix", reason="the shortcut is POSIX only")
def test_each_path_resolves_as_path_resolve_does(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("")
    (root / "pkg" / "sub" / "b.py").write_text("")
    (root / "linked").symlink_to(root / "pkg")
    (root / "pkg" / "alias.py").symlink_to(root / "pkg" / "sub" / "b.py")
    (root / "pkg" / "dangling.py").symlink_to(root / "nowhere.py")
    (root / "loop_a").symlink_to(root / "loop_b")
    (root / "loop_b").symlink_to(root / "loop_a")
    names = [
        "pkg/a.py", "pkg/sub/b.py", "linked/a.py", "linked/sub/b.py",
        "pkg/alias.py", "linked/alias.py", "pkg/dangling.py", "pkg/missing.py",
        "missing/dir/x.py", "pkg/../pkg/a.py", "pkg/sub/..", "pkg/.", "linked",
        "loop_a/x.py", "loop_a",
    ]
    def outcome(resolve, path: Path) -> object:
        try:
            return resolve(path)
        except (OSError, RuntimeError) as exc:  # a loop, as this Python reports it
            return type(exc)

    resolve = _resolver()
    for name in names * 2:  # the second pass reads the remembered directories
        path = root / name
        assert outcome(resolve, path) == outcome(Path.resolve, path), name
