"""The update's TEMP-table community capture remaps what the dict capture does.

An update captures each re-stored file's ``qualified_name -> community_id``
before replacing its nodes, then puts the IDs back on the new rows. The
update keeps the capture in a TEMP table and remaps it in one statement, so
thousands of names skip the connection's path translation. These cases pin
that the IDs and the count of nodes updated equal the dict capture's, with
and without relative path storage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_graph.communities import (
    capture_community_assignments,
    capture_community_assignments_in_sql,
    remap_captured_community_assignments,
    remap_community_assignments,
)
from code_review_graph.graph import GraphStore
from code_review_graph.parser import NodeInfo


def _node(name: str, path: str, line: int) -> NodeInfo:
    return NodeInfo(
        kind="Function", name=name, file_path=path,
        line_start=line, line_end=line + 1, language="python",
    )


def _run(tmp_path: Path, label: str, sql: bool, relative: bool) -> tuple[int, int, list]:
    root = tmp_path / label
    root.mkdir()
    store = GraphStore(root / "graph.db")
    try:
        if relative:
            store._conn.bind_root(str(root.resolve()))
        auth = (root / "auth.py").resolve().as_posix()
        other = (root / "other.py").resolve().as_posix()
        for index, name in enumerate(["login", "logout", "shared"]):
            store.upsert_node(_node(name, auth, 1 + 3 * index), file_hash="old")
        store.upsert_node(_node("keep", other, 1), file_hash="old")
        store.commit()
        conn = store._conn
        for name in ("a", "b"):
            conn.execute(
                "INSERT INTO communities (name, level, cohesion, size, "
                "dominant_language, description) VALUES (?, 0, 1.0, 1, 'python', '')",
                (name,),
            )
        first, second = [row[0] for row in conn.execute("SELECT id FROM communities ORDER BY id")]
        conn.execute("UPDATE nodes SET community_id = ? WHERE file_path = ?", (first, auth))
        conn.execute(
            "UPDATE nodes SET community_id = ? WHERE qualified_name = ?",
            (second, f"{auth}::logout"),
        )
        conn.execute("UPDATE nodes SET community_id = ? WHERE file_path = ?", (second, other))
        store.commit()

        if sql:
            captured = capture_community_assignments_in_sql(store, [auth, other])
        else:
            captured = len(capture_community_assignments(store, [auth, other]))
            mapping = capture_community_assignments(store, [auth, other])

        # Re-store auth.py with one node gone and one new; other.py stays.
        store.remove_file_data(auth)
        store.commit()
        for index, name in enumerate(["login", "shared", "added"]):
            store.upsert_node(_node(name, auth, 1 + 3 * index), file_hash="new")
        store.commit()

        if sql:
            updated = remap_captured_community_assignments(store)
        else:
            updated = remap_community_assignments(store, mapping)
        rows = [
            (row[0].replace(str(root.resolve()), "ROOT"), row[1])
            for row in conn.execute(
                "SELECT qualified_name, community_id FROM nodes ORDER BY qualified_name"
            )
        ]
        return captured, updated, rows
    finally:
        store.close()


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_sql_capture_remaps_like_the_dict_capture(tmp_path: Path, relative: bool) -> None:
    expected = _run(tmp_path, "dict", sql=False, relative=relative)
    assert _run(tmp_path, "sql", sql=True, relative=relative) == expected
    captured, updated, rows = expected
    assert captured == 4 and updated == 2
    assert ("ROOT/auth.py::added", None) in rows


def test_an_empty_capture_updates_nothing(tmp_path: Path) -> None:
    store = GraphStore(tmp_path / "graph.db")
    try:
        assert capture_community_assignments_in_sql(store, []) == 0
        assert remap_captured_community_assignments(store) == 0
        # The capture is dropped after a remap; a second one finds nothing.
        assert remap_captured_community_assignments(store) == 0
    finally:
        store.close()
