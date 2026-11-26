"""The CRG_FAST_ENGINE update path stores the graph a legacy update stores.

These tests pin the CRG_FAST_ENGINE update path. With the flag on, an update
re-parses an edited Python module's importers only where their stored edges
name a definition that changed, and rewrites only the search-index rows whose
node changed. Each case runs the same edit with the flag off and on and
requires identical nodes and edges, an index that matches a full rebuild, and
a connection that holds no transaction afterwards, so watch mode lets the
write-ahead log checkpoint.

The flag only selects which work the update skips, so the tests pass with or
without the native ``crg_fast`` module installed.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from code_review_graph import incremental
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, incremental_update
from code_review_graph.tools.build import build_or_update_graph

LIB = """\
def helper():
    return 1


def other():
    return 2


class Box:
    def size(self):
        return helper()
"""

FILES = {
    "lib.py": LIB,
    "a.py": (
        "from lib import Box, helper, other\n"
        "import lib\n\n\n"
        "def use_a():\n"
        "    helper()\n"
        "    other()\n"
        "    return Box().size()\n"
    ),
    "b.py": (
        "from lib import other\n\n\n"
        "def use_b():\n"
        "    return other()\n"
    ),
    "c.py": (
        "import lib\n\n\n"
        "def use_c():\n"
        "    return lib.later()\n"
    ),
    "d.py": (
        "from lib import Box\n\n\n"
        "def use_d():\n"
        "    return Box()\n"
    ),
    "test_a.py": (
        "from a import use_a\n\n\n"
        "def test_use_a():\n"
        "    assert use_a() == 1\n"
    ),
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=T", *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def base_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for name, text in FILES.items():
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _remove_symbol(repo: Path) -> None:
    (repo / "lib.py").write_text(LIB.replace("def other():\n    return 2\n\n\n", ""))


def _rename_symbol(repo: Path) -> None:
    (repo / "lib.py").write_text(LIB.replace("helper", "helper_two"))


def _add_symbol(repo: Path) -> None:
    (repo / "lib.py").write_text(LIB + "\n\ndef later():\n    return 3\n")


def _change_body(repo: Path) -> None:
    (repo / "lib.py").write_text(LIB.replace("return 2", "return 22"))


def _delete_file(repo: Path) -> None:
    (repo / "lib.py").unlink()


def _rename_file(repo: Path) -> None:
    _git(repo, "mv", "lib.py", "lib2.py")


EDITS: dict[str, Callable[[Path], None]] = {
    "symbol_removed": _remove_symbol,
    "symbol_renamed": _rename_symbol,
    "symbol_added_resolves_call": _add_symbol,
    "body_changed": _change_body,
    "file_deleted": _delete_file,
    "file_renamed": _rename_file,
}


def _dump(db: Path) -> dict[str, list[str]]:
    conn = sqlite3.connect(db)
    try:
        out: dict[str, list[str]] = {}
        for table in ("nodes", "edges"):
            cols = [
                row[1] for row in conn.execute(f"PRAGMA table_info({table})")
                if row[1] not in ("id", "updated_at")
            ]
            rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()  # nosec B608
            out[table] = sorted("\t".join(repr(v) for v in row) for row in rows)
        return out
    finally:
        conn.close()


def _fts_entries(db: Path) -> list[tuple]:
    """Every term occurrence the search index holds, by node."""
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, nodes_fts, instance)"
        )
        return sorted(conn.execute("SELECT term, doc, col, offset FROM temp.vocab"))
    finally:
        conn.close()


def _updated_graph(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, edit: Callable[[Path], None], flag: bool,
) -> Path:
    repo = tmp_path / label
    shutil.copytree(base_repo, repo)
    data = tmp_path / f"{label}-data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    edit(repo)
    if flag:
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    result = build_or_update_graph(repo_root=str(repo))
    assert result["status"] == "ok", result
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    return data / "graph.db"


@pytest.mark.parametrize("case", sorted(EDITS))
def test_flag_on_update_stores_the_legacy_graph(
    case: str, base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections: list[object] = []
    original = incremental._importers_to_reparse

    def spy(*args, **kwargs):
        selected = original(*args, **kwargs)
        selections.append(selected)
        return selected

    monkeypatch.setattr(incremental, "_importers_to_reparse", spy)
    legacy = _updated_graph(base_repo, tmp_path, monkeypatch, "off", EDITS[case], False)
    assert selections == []
    fast = _updated_graph(base_repo, tmp_path, monkeypatch, "on", EDITS[case], True)

    assert _dump(fast) == _dump(legacy)
    fast_index = _fts_entries(fast)
    rebuilt = tmp_path / "rebuilt.db"
    shutil.copyfile(fast, rebuilt)
    from code_review_graph.search import rebuild_fts_index

    store = GraphStore(rebuilt)
    try:
        rebuild_fts_index(store)
    finally:
        store.close()
    assert fast_index == _fts_entries(rebuilt)

    if case in {"symbol_removed", "symbol_renamed", "symbol_added_resolves_call"}:
        # The edit changes names only some importers use, so the fast path
        # ran and left at least one importer's rows alone.
        assert selections and isinstance(selections[0], set)
        assert {"a.py", "b.py", "c.py", "d.py"} - selections[0]


@pytest.mark.parametrize("case", ["symbol_added_resolves_call", "symbol_renamed"])
def test_wildcard_and_reexport_importers_match_legacy(
    case: str, base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A wildcard importer reads the module's public names, and a package that
    # re-exports a name hands it on to a second hop of importers.
    (base_repo / "star.py").write_text(
        "from lib import *\n\n\ndef use_star():\n    return later() + helper()\n"
    )
    (base_repo / "pkg").mkdir()
    (base_repo / "pkg" / "__init__.py").write_text("from lib import helper\n")
    (base_repo / "via_pkg.py").write_text(
        "from pkg import helper\n\n\ndef use_pkg():\n    return helper()\n"
    )
    _git(base_repo, "add", "-A")
    _git(base_repo, "commit", "-qm", "importers")

    legacy = _updated_graph(base_repo, tmp_path, monkeypatch, "off", EDITS[case], False)
    fast = _updated_graph(base_repo, tmp_path, monkeypatch, "on", EDITS[case], True)
    assert _dump(fast) == _dump(legacy)


def test_flag_on_update_leaves_no_transaction_open(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from code_review_graph.postprocessing import run_post_processing

    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    db = tmp_path / "graph.db"
    store = GraphStore(db)
    try:
        full_build(base_repo, store)
        run_post_processing(store)
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
        for edit in (_add_symbol, _remove_symbol, _change_body):
            edit(base_repo)
            # The way watch mode drives it: one open store, update, then
            # the post-processing callback.
            incremental_update(
                base_repo, store, changed_files=["lib.py"], reconcile_stale=False,
            )
            run_post_processing(store)
            assert store._conn.in_transaction is False

            other = sqlite3.connect(db)
            try:
                busy, log_frames, checkpointed = other.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
            finally:
                other.close()
            assert busy == 0
            assert checkpointed == log_frames
    finally:
        store.close()


def _twice_updated_graph(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, flag: bool,
) -> Path:
    """Add a symbol, update, then rename one before committing, and update."""
    repo = tmp_path / label
    shutil.copytree(base_repo, repo)
    data = tmp_path / f"{label}-data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    if flag:
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    _add_symbol(repo)
    assert build_or_update_graph(repo_root=str(repo))["status"] == "ok"
    (repo / "lib.py").write_text(
        (LIB + "\n\ndef later():\n    return 3\n").replace("helper", "helper_two")
    )
    assert build_or_update_graph(repo_root=str(repo))["status"] == "ok"
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    return data / "graph.db"


def test_second_save_before_a_commit_skips_importers_and_matches_legacy(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The graph holds bytes Git never saw after the first update, so the
    # second one reads them from the copy the first update kept.
    selections: list[object] = []
    original = incremental._importers_to_reparse

    def spy(*args, **kwargs):
        selected = original(*args, **kwargs)
        selections.append(selected)
        return selected

    monkeypatch.setattr(incremental, "_importers_to_reparse", spy)
    legacy = _twice_updated_graph(base_repo, tmp_path, monkeypatch, "off", False)
    assert selections == []
    fast = _twice_updated_graph(base_repo, tmp_path, monkeypatch, "on", True)

    assert _dump(fast) == _dump(legacy)
    assert len(selections) == 2
    assert isinstance(selections[1], set)
    assert {"a.py", "b.py", "c.py", "d.py"} - selections[1]
    # Only the copy of the bytes the graph now holds is left.
    stored = (tmp_path / "on" / "lib.py").read_bytes()
    import hashlib

    copies = sorted(p.name for p in (fast.parent / "source-copies").iterdir())
    assert copies == [hashlib.sha256(stored).hexdigest()]


def test_a_copy_that_does_not_match_its_name_is_not_read(tmp_path: Path) -> None:
    import hashlib

    data = b"def f():\n    return 1\n"
    digest = hashlib.sha256(data).hexdigest()
    (tmp_path / digest).write_bytes(data)
    assert incremental._read_source_copy(tmp_path, digest) == data
    (tmp_path / digest).write_bytes(data + b"# changed\n")
    assert incremental._read_source_copy(tmp_path, digest) is None
    assert incremental._read_source_copy(tmp_path, "../" + digest) is None
    assert incremental._read_source_copy(None, digest) is None


def test_a_full_build_drops_copies_no_file_holds(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(base_repo))["status"] == "ok"
    copies = data / "source-copies"
    copies.mkdir()
    import hashlib

    held = hashlib.sha256((base_repo / "lib.py").read_bytes()).hexdigest()
    (copies / held).write_bytes((base_repo / "lib.py").read_bytes())
    (copies / ("0" * 64)).write_bytes(b"gone\n")
    assert build_or_update_graph(full_rebuild=True, repo_root=str(base_repo))["status"] == "ok"
    assert sorted(p.name for p in copies.iterdir()) == [held]


# Two definitions of one name: the second one in the file takes a collision
# identity, and which one an edge names follows their lines.
DUP_LIB = """\
def dup(x):
    return x


def dup(x, y):
    return x + y


def plain():
    return 0
"""


@pytest.fixture
def dup_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "dup-base"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "lib.py").write_text(DUP_LIB)
    (repo / "a.py").write_text("from lib import dup\n\n\ndef use():\n    return dup(1, 2)\n")
    (repo / "b.py").write_text("from lib import plain\n\n\ndef use_b():\n    return plain()\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


@pytest.mark.parametrize(
    ("edit", "reparsed"),
    [
        # Nothing before the two definitions moves: a.py keeps its rows.
        (lambda text: text + "\n\ndef later():\n    return 3\n", set()),
        (lambda text: text.replace("return 0", "return 10"), set()),
        # A line above them shifts both, so a.py's call is parsed again.
        (lambda text: "def first():\n    return 9\n\n\n" + text, {"a.py"}),
        (lambda text: text.replace("def dup(x, y):", "def dup(x, z):"), {"a.py"}),
    ],
    ids=["appended", "unrelated_body", "shifted", "one_changed"],
)
def test_same_named_definitions_count_as_changed_only_when_they_move(
    edit: Callable[[str], str], reparsed: set[str],
    dup_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selections: list[object] = []
    original = incremental._importers_to_reparse

    def spy(*args, **kwargs):
        selected = original(*args, **kwargs)
        selections.append(selected)
        return selected

    monkeypatch.setattr(incremental, "_importers_to_reparse", spy)

    def apply(repo: Path) -> None:
        (repo / "lib.py").write_text(edit(DUP_LIB))

    legacy = _updated_graph(dup_repo, tmp_path, monkeypatch, "off", apply, False)
    fast = _updated_graph(dup_repo, tmp_path, monkeypatch, "on", apply, True)
    assert _dump(fast) == _dump(legacy)
    assert len(selections) == 1 and isinstance(selections[0], set)
    assert selections[0] & {"a.py"} == reparsed

