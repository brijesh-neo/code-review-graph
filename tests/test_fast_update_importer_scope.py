"""Which importers a CRG_FAST_ENGINE update re-parses after commits.

These tests pin the cases where the update can still tell what changed:

- The graph was built a few commits back. The old bytes are read at the
  stored commit, so the update does not have to re-parse every importer.
- A commit also edits a file no parser indexes, such as a changelog.
- A return or property annotation changes. Only importers whose source spells
  the annotated name are re-parsed.
- The module imports one more name from a module it already imports. Only importers that spell the imported name are
  re-parsed, and every importer is when one of those is imported in turn.
- The module gains a class with a base, a public function, an import inside
  a function or an import from outside the repository. Importers that do not
  spell the new name keep their rows, also when one spells a re-exported name
  only as an attribute or mentions a wildcard import in a comment.
- A commit also adds, edits or deletes a file nothing depends on. Every
  importer is re-parsed only when a Python file spells the module or folder
  that came or went.

In each case the graph must equal the one a legacy update stores.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

import pytest

from code_review_graph import incremental
from code_review_graph.tools.build import build_or_update_graph
from tests.test_fast_update_matches_full_reparse import _dump, _git

LIB = """\
class Box:
    @property
    def inner(self) -> Box:
        return self


class Crate:
    pass


def make() -> Box:
    return Box()


def other():
    return 2


from helpers import kept
"""

FILES = {
    "lib.py": LIB,
    "calls_make.py": (
        "from lib import make\n\n\n"
        "def use():\n"
        "    return make().inner\n"
    ),
    "uses_inner.py": (
        "from lib import Box\n\n\n"
        "def use(box: Box):\n"
        "    return box.inner\n"
    ),
    "calls_other.py": (
        "from lib import other\n\n"
        "try:\n"
        "    import optional_mod\n"
        "except ImportError:\n"
        "    optional_mod = None\n\n\n"
        "def use():\n"
        "    return other()\n"
    ),
    "helpers.py": (
        "def helper():\n    return 3\n\n\n"
        "def extra():\n    return 4\n\n\n"
        "def kept():\n    return 7\n"
    ),
    "forwards.py": (
        "import lib\n"
        "from lib import make\n"
        "from lib import extra\n\n"
        "# A comment, not a wildcard import, from lib import *\n\n\n"
        "def use():\n"
        "    return make()\n\n\n"
        "def also():\n"
        "    return lib.helper()\n"
    ),
    "uses_forwards.py": "from forwards import extra\n\n\ndef use():\n    return extra()\n",
    "standalone.py": "def alone():\n    return 5\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for name, text in FILES.items():
        (repo / name).write_text(text)
    (repo / "CHANGELOG.txt").write_text("start\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _commit(repo: Path, files: dict[str, Optional[str]]) -> None:
    for name, text in files.items():
        if text is None:
            (repo / name).unlink()
            continue
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "edit")


def _updated(
    base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, commits: list[dict[str, Optional[str]]], flag: bool,
) -> Path:
    repo = tmp_path / label
    shutil.copytree(base, repo)
    data = tmp_path / f"{label}-data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    for files in commits:
        _commit(repo, files)
    # Only the commit the graph records can still give the old bytes.
    shutil.rmtree(data / "source-copies", ignore_errors=True)
    if flag:
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    result = build_or_update_graph(repo_root=str(repo))
    assert result["status"] == "ok", result
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    return data / "graph.db"


def _selections(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    commits: list[dict[str, Optional[str]]],
) -> list[object]:
    selections: list[object] = []
    original = incremental._importers_to_reparse

    def spy(*args, **kwargs):
        selected = original(*args, **kwargs)
        selections.append(selected)
        return selected

    monkeypatch.setattr(incremental, "_importers_to_reparse", spy)
    legacy = _updated(repo, tmp_path, monkeypatch, "off", commits, False)
    assert selections == []
    fast = _updated(repo, tmp_path, monkeypatch, "on", commits, True)
    assert _dump(fast) == _dump(legacy)
    return selections


IMPORTERS = {"calls_make.py", "uses_inner.py", "calls_other.py"}
BODY: dict[str, Optional[str]] = {"lib.py": LIB.replace("return 2", "return 22")}


def _none(selected: Optional[set[str]]) -> bool:
    return selected is None


CASES: dict[
    str, tuple[list[dict[str, Optional[str]]], Callable[[Optional[set[str]]], bool]]
] = {
    # Two commits since the build: neither HEAD nor the index holds the
    # bytes the graph stored, but the commit it records does.
    "built_two_commits_back": (
        [BODY, {"lib.py": LIB.replace("return 2", "return 222")}],
        lambda selected: not selected & IMPORTERS,
    ),
    "unindexed_file_changed_too": (
        [{**BODY, "CHANGELOG.txt": "start\nmore\n"}],
        lambda selected: not selected & IMPORTERS,
    ),
    "return_annotation_changed": (
        [{"lib.py": LIB.replace("def make() -> Box", "def make() -> Crate")}],
        lambda selected: "calls_make.py" in selected and "calls_other.py" not in selected,
    ),
    "property_annotation_changed": (
        [{"lib.py": LIB.replace("def inner(self) -> Box", "def inner(self) -> Crate")}],
        lambda selected: {"calls_make.py", "uses_inner.py"} <= selected
        and "calls_other.py" not in selected,
    ),
    "import_added": (
        [{"lib.py": LIB.replace("import kept", "import kept, helper")}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    # forwards.py spells the new name, and uses_forwards.py reads it through
    # forwards.py's exports, which the update does not compare.
    "import_added_and_passed_on": (
        [{"lib.py": LIB.replace("import kept", "import kept, extra")}],
        _none,
    ),
    "class_with_base_added": (
        [{"lib.py": LIB + "\n\nclass Fresh(Crate):\n    pass\n"}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    "public_function_added": (
        [{"lib.py": LIB + "\n\ndef fresh():\n    return 9\n"}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    "deferred_import_added": (
        [{"lib.py": LIB.replace("    return 2", "    import standalone\n    return 2")}],
        lambda selected: selected is not None and "calls_make.py" not in selected,
    ),
    "outside_import_added": (
        [{"lib.py": LIB + "\nimport json\n"}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    "dependent_free_file_edited": (
        [{**BODY, "standalone.py": "def alone():\n    return 55\n"}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    "unspelled_module_added": (
        [{**BODY, "brand_new.py": "def fresh():\n    return 6\n"}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    "unspelled_module_deleted": (
        [{**BODY, "standalone.py": None}],
        lambda selected: selected is not None and not selected & IMPORTERS,
    ),
    # calls_other.py's import of optional_mod resolves once it exists.
    "spelled_module_added": ([{**BODY, "optional_mod.py": "VALUE = 1\n"}], _none),
    # A folder of that name makes it a namespace package.
    "spelled_folder_added": ([{**BODY, "optional_mod/notes.txt": "x\n"}], _none),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_commit_update_reparses_only_affected_importers(
    case: str, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits, expected = CASES[case]
    selections = _selections(repo, tmp_path, monkeypatch, commits)
    assert len(selections) == 1
    assert expected(selections[0]), selections[0]
