"""Which importers a CRG_FAST_ENGINE update re-parses after commits.

These tests pin three cases where the update can still tell what changed:

- The graph was built a few commits back. The old bytes are read at the
  stored commit, so the update does not have to re-parse every importer.
- A commit also edits a file no parser indexes, such as a changelog.
- A return or property annotation changes. Only importers whose source spells
  the annotated name are re-parsed.

In each case the graph must equal the one a legacy update stores.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

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
        "from lib import other\n\n\n"
        "def use():\n"
        "    return other()\n"
    ),
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


def _commit(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "edit")


def _updated(
    base: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, commits: list[dict[str, str]], flag: bool,
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
    commits: list[dict[str, str]],
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
BODY = {"lib.py": LIB.replace("return 2", "return 22")}

CASES: dict[str, tuple[list[dict[str, str]], Callable[[set[str]], bool]]] = {
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
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_commit_update_reparses_only_affected_importers(
    case: str, repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commits, expected = CASES[case]
    selections = _selections(repo, tmp_path, monkeypatch, commits)
    assert len(selections) == 1
    assert isinstance(selections[0], set), selections
    assert expected(selections[0]), selections[0]
