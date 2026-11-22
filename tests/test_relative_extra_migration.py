"""A v16 graph's absolute ``extra`` JSON is stored relative on its first v17 open.

The conversion runs in batches that commit with their cursor, so a killed run
resumes without converting a row twice, and answers that read ``extra``
(``ambiguous_targets`` here) are the same before and after.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from code_review_graph import relpath_migration
from code_review_graph.graph import GraphStore
from code_review_graph.migrations import LATEST_VERSION
from code_review_graph.relpaths import EXTRA_MODE_KEY, PathCodec
from tests.test_php_guessed_test_links import _laravel, _qn


def _as_v16(db: Path, root: Path) -> None:
    """Rewrite *db* the way a v16 release stored it: ``extra`` absolute, no mark."""
    codec = PathCodec(str(root))
    conn = sqlite3.connect(db)
    for table in ("nodes", "edges"):
        rows = conn.execute(f"SELECT id, extra FROM {table} WHERE extra IS NOT NULL").fetchall()
        conn.executemany(
            f"UPDATE {table} SET extra = ? WHERE id = ?",
            [(codec.restore_json(extra), row_id) for row_id, extra in rows],
        )
    conn.execute("DELETE FROM metadata WHERE key = ?", (EXTRA_MODE_KEY,))
    conn.execute("UPDATE metadata SET value = '16' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()


def _raw(db: Path, sql: str) -> list:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _target(tmp_path: Path) -> str:
    return _qn(
        tmp_path, "app/Actions/CompleteSupplierCount.php", "CompleteSupplierCount.execute",
    )


def test_a_v16_graph_migrates_and_still_resolves_ambiguous_targets(tmp_path):
    store = _laravel(tmp_path)
    expected = sorted(store.get_ambiguous_call_tests(_target(tmp_path)))
    store.close()
    assert expected
    db = tmp_path / ".code-review-graph" / "graph.db"
    _as_v16(db, tmp_path)
    assert any(str(tmp_path) in r[0] for r in _raw(
        db, "SELECT extra FROM edges WHERE extra LIKE '%ambiguous_targets%'",
    ))

    store = GraphStore(db)
    try:
        assert store.get_metadata("schema_version") == str(LATEST_VERSION)
        assert store.get_metadata(EXTRA_MODE_KEY) == "relative"
        assert sorted(store.get_ambiguous_call_tests(_target(tmp_path))) == expected
        # Read through the store, extra names this checkout again.
        targets = [
            t for r in store._conn.execute(
                "SELECT extra FROM edges WHERE extra LIKE '%ambiguous_targets%'",
            ) for t in json.loads(r[0])["ambiguous_targets"]
        ]
        assert targets and all(t.startswith(f"{tmp_path}/") for t in targets)
    finally:
        store.close()
    root_spellings = {str(tmp_path), str(tmp_path.resolve())}
    assert not any(
        spelling in r[0]
        for r in _raw(db, "SELECT extra FROM edges WHERE extra IS NOT NULL")
        for spelling in root_spellings
    )


def test_a_killed_extra_conversion_resumes_without_converting_twice(tmp_path, monkeypatch):
    store = _laravel(tmp_path)
    store.close()
    db = tmp_path / ".code-review-graph" / "graph.db"
    _as_v16(db, tmp_path)
    # A raw value that reads as relative: converted twice it would be escaped twice.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, line, "
        "extra, updated_at) VALUES ('IMPORTS_FROM', 'a', 'b', 'c', 1, ?, 0)",
        (json.dumps({"specifier": "./missing", "receiver_module": f"{tmp_path}/m.php"}),),
    )
    conn.commit()
    conn.close()
    once = tmp_path / "once.db"
    once.write_bytes(db.read_bytes())

    calls = {"n": 0}
    real = PathCodec.strip_json

    def dying(self, value):
        calls["n"] += 1
        if calls["n"] > 5:
            raise KeyboardInterrupt
        return real(self, value)

    conn = sqlite3.connect(db, isolation_level=None)
    monkeypatch.setattr(PathCodec, "strip_json", dying)
    with pytest.raises(KeyboardInterrupt):
        relpath_migration.convert_extra_to_relative(conn, str(tmp_path), batch=2)
    monkeypatch.setattr(PathCodec, "strip_json", real)
    assert _raw(db, f"SELECT value FROM metadata WHERE key = '{EXTRA_MODE_KEY}'") == []
    relpath_migration.convert_extra_to_relative(conn, str(tmp_path), batch=2)
    conn.close()

    conn = sqlite3.connect(once, isolation_level=None)
    relpath_migration.convert_extra_to_relative(conn, str(tmp_path))
    conn.close()
    sql = "SELECT id, extra FROM edges ORDER BY id"
    assert _raw(db, sql) == _raw(once, sql)
    assert ('{"specifier": "!./missing", "receiver_module": "./m.php"}',) in [
        (r[1],) for r in _raw(db, sql)
    ]
    assert _raw(db, "SELECT key FROM metadata WHERE key LIKE 'relpath_extra_cursor%' "
                    "OR key LIKE 'relpath_migration%'") == []
