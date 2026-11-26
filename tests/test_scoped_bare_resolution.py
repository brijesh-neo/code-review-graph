"""An update under CRG_FAST_ENGINE revisits only the bare edges it can move.

A complete graph is a fixed point of the bare-endpoint pass, so after an
update only edges that touch a change can get a different answer: edges the
update wrote, edges whose bare name belongs to a Function or Test node that
changed, and edges whose caller changed. ``begin_change_capture`` records the
changes in TEMP tables (nothing is stored in the graph). A changed import
reaches the edges written in the importing file, because import evidence is
read per file. A change to Class or File nodes, to inheritance edges, to a
scoped import, or to the imports of a file whose classes have parents, sends
the pass back to a full run. Each case runs the same edit with the scoped pass
and with the full pass and requires identical nodes and edges.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.tools.build import build_or_update_graph

LIB = """\
def helper():
    return 1


def other():
    return 2
"""

FILES = {
    "lib.py": LIB,
    "lib3.py": "def third():\n    return 3\n",
    "star.py": "from lib import *\n\n\ndef use_star():\n    return later()\n",
    "both.py": (
        "import lib\nimport lib3\n\n\n"
        "def use_both(x):\n"
        "    return x.third()\n"
    ),
    "a.py": (
        "from lib import helper, other\n\n\n"
        "def use_a():\n"
        "    helper()\n"
        "    return other()\n"
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


def _append(name: str, text: str) -> Callable[[Path], None]:
    def edit(repo: Path) -> None:
        path = repo / name
        path.write_text(path.read_text() + text)
    return edit


def _body_only(repo: Path) -> None:
    (repo / "lib.py").write_text(LIB.replace("return 2", "return 22"))


def _remove_helper(repo: Path) -> None:
    (repo / "lib.py").write_text(LIB.replace("def helper():\n    return 1\n\n\n", ""))


# name -> (edit, whether the scoped pass applies)
EDITS: dict[str, tuple[Callable[[Path], None], bool]] = {
    "body_changed": (_body_only, True),
    "function_added_resolves_star_call": (
        _append("lib.py", "\n\ndef later():\n    return 3\n"), True,
    ),
    "same_name_elsewhere_makes_call_ambiguous": (
        _append("lib3.py", "\n\ndef helper():\n    return 4\n"), True,
    ),
    "method_name_added": (
        _append("lib.py", "\n\ndef third():\n    return 5\n"), True,
    ),
    "function_removed": (_remove_helper, True),
    "class_added": (
        _append("lib.py", "\n\nclass Box:\n    def third(self):\n        return 6\n"), False,
    ),
    "import_added": (_append("a.py", "\nimport lib3\n"), True),
    "import_removed": (
        lambda repo: (repo / "both.py").write_text(FILES["both.py"].replace("import lib3\n", "")),
        True,
    ),
    "import_added_beside_a_parented_class": (
        _append("a.py", "\nimport lib3\n\n\nclass Kid(lib3.Base):\n    pass\n"), False,
    ),
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


def _updated(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, edit: Callable[[Path], None], allow_scope: bool,
) -> tuple[Path, list[bool]]:
    repo = tmp_path / label
    shutil.copytree(base_repo, repo)
    data = tmp_path / f"{label}-data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    edit(repo)
    scopes: list[bool] = []
    original_end = GraphStore.end_change_capture

    def end(self: GraphStore) -> bool:
        scoped = original_end(self)
        scopes.append(scoped)
        return scoped

    with monkeypatch.context() as patch:
        patch.setattr(GraphStore, "end_change_capture", end)
        if not allow_scope:
            patch.setattr(GraphStore, "begin_change_capture", lambda self: False)
        result = build_or_update_graph(repo_root=str(repo))
    assert result["status"] == "ok", result
    return data / "graph.db", scopes


@pytest.mark.parametrize("case", sorted(EDITS))
def test_scoped_pass_stores_what_the_full_pass_stores(
    case: str, base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    edit, applies = EDITS[case]
    full, full_scopes = _updated(base_repo, tmp_path, monkeypatch, "full", edit, False)
    scoped, scopes = _updated(base_repo, tmp_path, monkeypatch, "scoped", edit, True)
    assert full_scopes == [False]
    assert scopes == [applies]
    assert _dump(scoped) == _dump(full)

    # And the result is the pass's fixed point: a full run changes nothing.
    store = GraphStore(scoped)
    try:
        assert store.resolve_bare_call_targets() == 0
        before = _dump(scoped)
        store.resolve_bare_tested_by_sources()
    finally:
        store.close()
    assert _dump(scoped) == before


def test_capture_is_not_started_for_an_incomplete_graph(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    assert build_or_update_graph(
        full_rebuild=True, repo_root=str(base_repo), postprocess="none",
    )["status"] == "ok"
    _body_only(base_repo)
    started: list[bool] = []
    original = GraphStore.begin_change_capture

    def begin(self: GraphStore) -> bool:
        started.append(True)
        return original(self)

    monkeypatch.setattr(GraphStore, "begin_change_capture", begin)
    assert build_or_update_graph(repo_root=str(base_repo))["status"] == "ok"
    assert started == []


def test_capture_is_not_started_without_the_flag(
    base_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(base_repo))["status"] == "ok"
    _body_only(base_repo)
    started: list[bool] = []
    monkeypatch.setattr(
        GraphStore, "begin_change_capture", lambda self: started.append(True) or False,
    )
    assert build_or_update_graph(repo_root=str(base_repo))["status"] == "ok"
    assert started == []


def _store(tmp_path: Path) -> GraphStore:
    store = GraphStore(tmp_path / "g.db")
    store._conn.execute(
        "INSERT INTO nodes (kind, name, qualified_name, file_path, language, "
        "parent_name, extra, updated_at) VALUES "
        "('Class', 'Box', 'm.py::Box', 'm.py', 'python', NULL, '{}', 0), "
        "('Function', 'f', 'm.py::f', 'm.py', 'python', NULL, '{}', 0)"
    )
    return store


def test_a_row_that_leaves_and_returns_unchanged_cancels_out(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        conn = store._conn
        assert store.begin_change_capture()
        rows = conn.execute(
            "SELECT kind, name, qualified_name, file_path, language, parent_name, extra "
            "FROM nodes"
        ).fetchall()
        conn.execute("DELETE FROM nodes")
        conn.executemany(
            "INSERT INTO nodes (kind, name, qualified_name, file_path, language, "
            "parent_name, extra, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            [tuple(row) for row in rows],
        )
        # A column the pass does not read is not a change.
        conn.execute("UPDATE nodes SET line_start = 9, community_id = 3")
        assert store.end_change_capture()
        assert conn.execute("SELECT COUNT(*) FROM temp.crg_dirty_name").fetchone()[0] == 0
        store.drop_change_scope()
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_temp_master WHERE name LIKE 'crg_%'"
        ).fetchone()[0] == 0
    finally:
        store.close()


def test_a_changed_class_sends_the_pass_back_to_full(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        assert store.begin_change_capture()
        store._conn.execute("UPDATE nodes SET extra = '{\"x\": 1}' WHERE kind = 'Class'")
        assert not store.end_change_capture()
    finally:
        store.close()


def test_a_changed_function_is_dirty_by_name_and_by_qualified_name(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        conn = store._conn
        assert store.begin_change_capture()
        conn.execute("UPDATE nodes SET name = 'initialize' WHERE kind = 'Function'")
        assert store.end_change_capture()
        names = {row[0] for row in conn.execute("SELECT name FROM temp.crg_dirty_name")}
        # ``X.new`` is looked up as ``initialize``, so it is dirty too.
        assert names == {"f", "initialize", "new"}
        assert [row[0] for row in conn.execute("SELECT qn FROM temp.crg_dirty_qn")] == [
            "m.py::f",
        ]
    finally:
        store.drop_change_scope()
        store.close()


# The cases below build rows by hand: a parsed update reparses most callers,
# so their edges are dirty anyway. Here the caller's file never changes, and
# only the name, the stored bare name or the caller can bring the edge back.

def _graph(tmp_path: Path, nodes: list[tuple], edges: list[tuple]) -> GraphStore:
    store = GraphStore(tmp_path / "hand.db")
    conn = store._conn
    conn.executemany(
        "INSERT INTO nodes (kind, name, qualified_name, file_path, language, "
        "parent_name, extra, updated_at) VALUES (?, ?, ?, ?, ?, ?, '{}', 0)",
        nodes,
    )
    conn.executemany(
        "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, "
        "line, extra, updated_at) VALUES (?, ?, ?, ?, 1, ?, 0)",
        edges,
    )
    # Start from the pass's fixed point, as an update from a complete graph does.
    store.resolve_bare_call_targets()
    store.resolve_bare_tested_by_sources()
    return store


_FILES = [
    ("File", f, f, f, "python", None) for f in ("x.py", "y.py", "z.py")
]
_IMPORTS = [
    ("IMPORTS_FROM", "x.py", target, "x.py", "{}") for target in ("y.py", "z.py")
]


def _call(store: GraphStore) -> tuple[str, dict]:
    import json

    row = store._conn.execute(
        "SELECT target_qualified, extra FROM edges WHERE kind = 'CALLS'"
    ).fetchone()
    return row[0], json.loads(row[1])


def _scoped_then_full(store: GraphStore, change: str) -> tuple[str, dict]:
    assert store.begin_change_capture()
    store._conn.execute(change)
    assert store.end_change_capture()
    try:
        store.resolve_bare_call_targets(scoped=True)
        store.resolve_bare_tested_by_sources(scoped=True)
    finally:
        store.drop_change_scope()
    after = _call(store)
    assert store.resolve_bare_call_targets() == 0
    assert _call(store) == after
    return after


def test_a_new_candidate_reaches_a_bare_edge_by_its_name(tmp_path: Path) -> None:
    store = _graph(
        tmp_path,
        _FILES + [("Function", "caller", "x.py::caller", "x.py", "python", None)],
        _IMPORTS + [("CALLS", "x.py::caller", "later", "x.py", "{}")],
    )
    try:
        assert _call(store)[0] == "later"
        target, _ = _scoped_then_full(
            store,
            "INSERT INTO nodes (kind, name, qualified_name, file_path, language, "
            "updated_at) VALUES ('Function', 'later', 'y.py::later', 'y.py', 'python', 0)",
        )
        assert target == "y.py::later"
    finally:
        store.close()


def test_a_second_candidate_reaches_a_resolved_edge_by_its_stored_bare_name(
    tmp_path: Path,
) -> None:
    store = _graph(
        tmp_path,
        _FILES + [
            ("Function", "caller", "x.py::caller", "x.py", "python", None),
            ("Function", "later", "y.py::later", "y.py", "python", None),
        ],
        _IMPORTS + [("CALLS", "x.py::caller", "later", "x.py", "{}")],
    )
    try:
        target, extra = _call(store)
        assert target == "y.py::later" and extra["bare_call_target"] == "later"
        target, extra = _scoped_then_full(
            store,
            "INSERT INTO nodes (kind, name, qualified_name, file_path, language, "
            "updated_at) VALUES ('Function', 'later', 'z.py::later', 'z.py', 'python', 0)",
        )
        assert target == "later"
        assert extra["ambiguous_targets"] == ["y.py::later", "z.py::later"]
    finally:
        store.close()


def test_a_changed_caller_reaches_its_edges_by_qualified_name(tmp_path: Path) -> None:
    store = _graph(
        tmp_path,
        _FILES + [
            ("Function", "caller", "x.py::caller", "x.py", "python", None),
            ("Function", "map", "y.py::Match.map", "y.py", "python", "Match"),
        ],
        _IMPORTS + [("CALLS", "x.py::caller", "map", "x.py", '{"receiver": "v"}')],
    )
    try:
        assert _call(store)[0] == "y.py::Match.map"
        # Rust records a receiver's name, never its type: ``v.map()`` says
        # nothing about ``Match``.
        target, extra = _scoped_then_full(
            store, "UPDATE nodes SET language = 'rust' WHERE name = 'caller'",
        )
        assert target == "map"
        assert "unresolved_targets" in extra
    finally:
        store.close()


def test_an_import_change_reaches_the_bare_edges_of_its_file(tmp_path: Path) -> None:
    store = _graph(
        tmp_path,
        _FILES + [
            ("Function", "caller", "x.py::caller", "x.py", "python", None),
            ("Function", "later", "y.py::later", "y.py", "python", None),
            ("Function", "later", "z.py::later", "z.py", "python", None),
        ],
        _IMPORTS + [("CALLS", "x.py::caller", "later", "x.py", "{}")],
    )
    try:
        before, _ = _call(store)
        # Neither ``later`` nor the caller changes: only the import does.
        target, _ = _scoped_then_full(
            store,
            "DELETE FROM edges WHERE kind = 'IMPORTS_FROM' AND target_qualified = 'z.py'",
        )
        assert target != before
        assert target == "y.py::later"
    finally:
        store.close()


def test_a_scoped_import_sends_the_pass_back_to_full(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        assert store.begin_change_capture()
        store._conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, "
            "line, extra, updated_at) VALUES ('IMPORTS_FROM', 'm.py', 'pkg', 'm.py', 1, "
            "'{\"import_scope\": \"pkg\"}', 0)"
        )
        assert not store.end_change_capture()
    finally:
        store.close()


def test_an_import_change_beside_a_parented_class_sends_the_pass_back_to_full(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    try:
        conn = store._conn
        conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, "
            "line, extra, updated_at) VALUES ('INHERITS', 'm.py::Box', 'Base', 'm.py', 1, "
            "'{}', 0)"
        )
        assert store.begin_change_capture()
        conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, "
            "line, extra, updated_at) VALUES ('IMPORTS_FROM', 'm.py', 'n.py', 'm.py', 1, "
            "'{}', 0)"
        )
        assert not store.end_change_capture()
    finally:
        store.close()
