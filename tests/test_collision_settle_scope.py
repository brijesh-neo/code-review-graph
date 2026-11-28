"""A fast update settles duplicated-name edges from the rows that can hold one.

With ``CRG_FAST_ENGINE`` and the certainty capture on, the collision pass that
ends ``refresh_target_resolution`` reads only the edges that matched
``endpoint"`` when the endpoint rows were kept, the edges updated since the
capture began, and the edges inserted during the update, instead of scanning
every edge's ``extra``. These cases add, remove and keep duplicates that a
file the update does not parse again calls, and require the graph a legacy
update and a full rebuild store.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Callable

import pytest

from code_review_graph.graph import GraphStore

from .test_fast_update_matches_full_reparse import _dump, _git, _updated_graph

LIB = """\
def dup(x):
    return x


def dup(x, y):
    return x + y


def plain():
    return 0
"""

EDITS: dict[str, Callable[[str], str]] = {
    "duplicate_removed": lambda text: text.replace(
        "def dup(x, y):\n    return x + y\n", "def gone(x, y):\n    return x + y\n"
    ),
    "duplicate_added": lambda text: text + "\n\ndef dup(a, b, c):\n    return a\n",
    "unrelated_body": lambda text: text.replace("return 0", "return 10"),
    "second_name_duplicated": lambda text: text + "\n\ndef plain(y):\n    return y\n",
}


@pytest.fixture
def collision_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "collision-base"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "lib.py").write_text(LIB)
    (repo / "a.py").write_text("from lib import dup\n\n\ndef use():\n    return dup(1, 2)\n")
    (repo / "b.py").write_text("from lib import plain\n\n\ndef use_b():\n    return plain()\n")
    (repo / "test_b.py").write_text(
        "from b import use_b\n\n\ndef test_use_b():\n    assert use_b() == 0\n"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _rebuilt(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str) -> Path:
    from code_review_graph.tools.build import build_or_update_graph

    data = tmp_path / f"{label}-data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    return data / "graph.db"


def _edges(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return sorted(
            (kind, source, target, path, line, json.dumps(json.loads(extra or "{}"),
                                                          sort_keys=True), rest)
            for kind, source, target, path, line, extra, *rest in (
                (*row[:6], tuple(row[6:])) for row in conn.execute(
                    "SELECT kind, source_qualified, target_qualified, file_path, line, "
                    "extra, confidence, confidence_tier, target_resolution FROM edges"
                )
            )
        )
    finally:
        conn.close()


@pytest.mark.parametrize("case", sorted(EDITS))
def test_scoped_collision_settle_stores_the_legacy_and_rebuilt_graph(
    case: str, collision_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    floors: list[object] = []
    original = GraphStore._settle_collision_endpoints

    def spy(self, scope_floor=None):
        floors.append(scope_floor)
        return original(self, scope_floor)

    monkeypatch.setattr(GraphStore, "_settle_collision_endpoints", spy)

    def apply(repo: Path) -> None:
        (repo / "lib.py").write_text(EDITS[case](LIB))

    legacy = _updated_graph(collision_repo, tmp_path, monkeypatch, "off", apply, False)
    floors.clear()
    fast = _updated_graph(collision_repo, tmp_path, monkeypatch, "on", apply, True)
    fast_floors = list(floors)
    assert _dump(fast) == _dump(legacy)
    rebuilt = _rebuilt(tmp_path / "on", tmp_path, monkeypatch, "re")
    # Updates and a rebuild may write the keys of ``extra`` in another order.
    assert _edges(fast) == _edges(rebuilt)
    try:
        import crg_fast  # noqa: F401
    except ImportError:
        return
    # With the native module the update's own pass ran scoped.
    assert any(floor is not None for floor in fast_floors)


def test_without_a_floor_every_row_is_scanned(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.db")
    try:
        assert store._settle_collision_endpoints() == 0
        assert store._settle_collision_endpoints(None) == 0
    finally:
        store.close()
