"""An update settles unbound endpoints as a fresh build does.

With the native engine the update's unbound scans read only the rows the
certainty capture holds: rows written or rewritten since it began, and rows
naming a symbol a node gained or lost. A call whose target vanishes or
appears in another file is in that set through the name, so an update ends
with the edges a build of the same tree writes. Pinned for PR #13 on the fork.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from code_review_graph.tools.build import build_or_update_graph

_HELPER = (
    "class Helper:\n"
    "    def run(self):\n        return 1\n\n"
    "    def keep(self):\n        return 2\n"
)
_CALLER = (
    "from lib.helper import Helper\n\n\n"
    "class Local:\n"
    "    def go(self):\n        return self.run()\n\n"
    "    def run(self):\n        return Helper().run()\n\n\n"
    "def use():\n    h = Helper()\n    return h.run(), h.keep()\n"
)
_BASE = "class Base:\n    def shared(self):\n        return 0\n"
_KID = (
    "from lib.base import Base\n\n\n"
    "class Kid(Base):\n    def go(self):\n        return self.shared()\n"
)
_TEST = (
    "from app.caller import use\n\n\n"
    "def test_use():\n    assert use()\n"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=T", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _edges(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return sorted(conn.execute(
            "SELECT kind, source_qualified, target_qualified, extra, confidence_tier, "
            "target_resolution FROM edges WHERE kind IN ('CALLS', 'REFERENCES', 'TESTED_BY')"
        ))
    finally:
        conn.close()


def _write(repo: Path, files: dict[str, str]) -> None:
    for rel_path, text in files.items():
        target = repo / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


@pytest.mark.parametrize("edit", [
    # ``run`` goes, so ``Helper.run`` names no node any more.
    {"lib/helper.py": _HELPER.replace("    def run(self):\n        return 1\n\n", "")},
    # ``Local.run`` goes, so ``self.run()`` stops naming a node in its own file.
    {"app/caller.py": _CALLER.replace("    def run(self):\n        return Helper().run()\n\n", "")},
    # A method appears that a call already named.
    {"lib/helper.py": _HELPER + "\n    def extra(self):\n        return 3\n",
     "app/other.py": "from lib.helper import Helper\n\n\ndef f():\n    return Helper().extra()\n"},
    # ``keep`` goes, so the bare pass's ``Helper.keep`` names no node.
    {"lib/helper.py": _HELPER.replace("\n    def keep(self):\n        return 2\n", "")},
    # The inherited ``shared`` goes, so the call moved onto ``Base`` comes back.
    {"lib/base.py": "class Base:\n    def other(self):\n        return 0\n"},
    # ``Kid`` declares ``shared`` itself, so the call binds in its own file.
    {"app/kid.py": _KID + "\n    def shared(self):\n        return 1\n"},
], ids=[
    "target-removed", "own-method-removed", "target-added", "bare-target-removed",
    "inherited-removed", "inherited-shadowed",
])
def test_an_update_settles_endpoints_as_a_build_does(
    edit: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    repo = tmp_path / "repo"
    _write(repo, {
        "lib/__init__.py": "", "lib/helper.py": _HELPER,
        "app/__init__.py": "", "app/caller.py": _CALLER,
        "app/other.py": "def f():\n    return 0\n",
        "tests/test_caller.py": _TEST,
        "lib/base.py": _BASE, "app/kid.py": _KID,
    })
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    updated_data = tmp_path / "updated"
    monkeypatch.setenv("CRG_DATA_DIR", str(updated_data))
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"

    _write(repo, edit)
    assert build_or_update_graph(repo_root=str(repo))["status"] == "ok"

    built_data = tmp_path / "built"
    monkeypatch.setenv("CRG_DATA_DIR", str(built_data))
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    assert _edges(updated_data / "graph.db") == _edges(built_data / "graph.db")
