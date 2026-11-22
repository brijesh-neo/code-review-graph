"""Pin the CRG_FAST_ENGINE full-build path: it must write the same graph.

With ``CRG_FAST_ENGINE`` on, a full build into an empty graph writes its edges
with the edge indexes dropped and rebuilds them once, and the postprocess
loads the flow adjacency once for flows and communities. These tests build
one small repository with the flag off and on and require every table, the
schema text and each table's index list to match. They hold whether or not
the native ``crg_fast`` module is installed: without it the parser falls back
to the default engine and only the store and postprocess paths differ.

They also pin the safety net: a build stopped while the indexes are dropped
gets them back on the next open, a failing load puts them back at once, and a
finished build leaves no transaction open and no large WAL behind.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build
from code_review_graph.tools.build import build_or_update_graph

# Columns that hold wall-clock times, and metadata keys that do.
_TIME_COLUMNS = {"updated_at", "created_at", "built_at", "timestamp"}
_TIME_METADATA = {
    "last_updated", "last_postprocessed_at", "last_build_duration", "postprocess_duration",
}

_FILES = {
    "pkg/__init__.py": "from .core import Engine\n",
    "pkg/core.py": (
        "from .util import helper, shout\n\n\n"
        "class Base:\n"
        "    def run(self):\n"
        "        return helper(1)\n\n\n"
        "class Engine(Base):\n"
        "    def start(self):\n"
        "        self.run()\n"
        "        return shout('go')\n\n"
        "    def stop(self):\n"
        "        return self.start()\n"
    ),
    "pkg/util.py": (
        "def helper(x):\n    return twice(x)\n\n\n"
        "def twice(x):\n    return x * 2\n\n\n"
        "def shout(word):\n    return word.upper()\n\n\n"
        "def helper(x):\n    return x\n"
    ),
    "pkg/cli.py": (
        "from .core import Engine\n\n\n"
        "def main():\n    Engine().start()\n    unknown_call()\n\n\n"
        "if __name__ == '__main__':\n    main()\n"
    ),
    "web/app.js": (
        "import { render } from './view.js';\n"
        "export function boot() { return render(1); }\n"
        "function idle() { return boot(); }\n"
    ),
    "web/view.js": (
        "export function render(n) { return paint(n); }\n"
        "function paint(n) { return n; }\n"
    ),
    "tests/test_core.py": (
        "from pkg.core import Engine\nfrom pkg.util import helper\n\n\n"
        "def test_start():\n    assert Engine().start() == 'GO'\n\n\n"
        "def test_helper():\n    assert helper(2) == 4\n"
    ),
}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    root = tmp_path / "repo"
    for rel, text in _FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    return root


def _dump(db: Path) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        lines = [
            "SCHEMA " + sql for (sql,) in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
            )
        ]
        tables = [
            name for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        for table in tables:
            cols = [
                row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")
                if row[1] not in _TIME_COLUMNS
            ]
            try:
                rows = conn.execute(
                    f"SELECT {', '.join(cols)} FROM '{table}'"  # noqa: S608
                ).fetchall()
            except sqlite3.OperationalError:
                continue  # FTS shadow tables that cannot be read directly
            body = sorted(
                "\t".join(repr(v) for v in row) for row in rows
                if not (table == "metadata" and row[0] in _TIME_METADATA)
            )
            lines.append(f"TABLE {table} {len(body)}")
            lines.extend(body)
            indexes = conn.execute(f"PRAGMA index_list({table!r})").fetchall()
            lines.append(f"INDEXES {table} {indexes}")
        return lines
    finally:
        conn.close()


def _build(repo: Path, monkeypatch: pytest.MonkeyPatch, fast: bool) -> list[str]:
    shutil.rmtree(repo / ".code-review-graph", ignore_errors=True)
    if fast:
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    else:
        monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    result = build_or_update_graph(full_rebuild=True, repo_root=str(repo))
    assert result["status"] == "ok", result
    return _dump(repo / ".code-review-graph" / "graph.db")


def _edge_indexes(db: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' "
            "AND tbl_name = 'edges' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    finally:
        conn.close()


def test_fast_build_writes_the_same_graph_as_the_default_build(repo, monkeypatch):
    legacy = _build(repo, monkeypatch, fast=False)
    fast = _build(repo, monkeypatch, fast=True)
    assert any(line.startswith("TABLE edges ") and line != "TABLE edges 0" for line in legacy)
    assert any(line.startswith("TABLE flows ") and line != "TABLE flows 0" for line in legacy)
    assert fast == legacy


def test_fast_build_keeps_every_edge_index(repo, monkeypatch):
    _build(repo, monkeypatch, fast=False)
    legacy = _edge_indexes(repo / ".code-review-graph" / "graph.db")
    applied: list[bool] = []
    original = GraphStore.bulk_edge_load

    @contextmanager
    def spy(self):
        with original(self) as active:
            applied.append(active)
            yield active

    monkeypatch.setattr(GraphStore, "bulk_edge_load", spy)
    _build(repo, monkeypatch, fast=True)
    assert applied == [True]
    assert _edge_indexes(repo / ".code-review-graph" / "graph.db") == legacy
    assert {name for name, _sql in legacy} >= {
        "idx_edges_kind", "idx_edges_target_kind", "idx_edges_source_kind",
        "idx_edges_file", "idx_edges_composite", "idx_edges_kind_target_resolution",
    }


def test_build_stopped_during_the_bulk_load_gets_its_indexes_back(repo, monkeypatch):
    _build(repo, monkeypatch, fast=False)
    db = repo / ".code-review-graph" / "graph.db"
    expected = _edge_indexes(db)

    # A process that dies with the indexes dropped: no finally block runs.
    script = (
        "import os, sys\n"
        "from code_review_graph.graph import GraphStore\n"
        "store = GraphStore(sys.argv[1])\n"
        "store._conn.execute('DELETE FROM edges')\n"
        "store._conn.commit()\n"
        "load = store.bulk_edge_load()\n"
        "assert load.__enter__() is True\n"
        "os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(db)], check=True)
    assert _edge_indexes(db) == []

    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    GraphStore(db).close()
    assert _edge_indexes(db) == expected


def test_failing_bulk_load_puts_the_indexes_back(repo, monkeypatch):
    _build(repo, monkeypatch, fast=False)
    db = repo / ".code-review-graph" / "graph.db"
    expected = _edge_indexes(db)
    shutil.rmtree(repo / ".code-review-graph")
    db.parent.mkdir()
    monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    monkeypatch.setattr(
        GraphStore, "store_file_batch",
        lambda self, batch: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O error")),
    )
    store = GraphStore(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            full_build(repo, store)
        assert not store._conn.in_transaction
    finally:
        store.close()
    assert _edge_indexes(db) == expected


def test_bulk_load_does_not_apply_to_a_graph_that_has_edges(repo, monkeypatch):
    _build(repo, monkeypatch, fast=False)
    db = repo / ".code-review-graph" / "graph.db"
    expected = _edge_indexes(db)
    store = GraphStore(db)
    try:
        with store.bulk_edge_load() as active:
            assert active is False
            assert _edge_indexes(db) == expected
    finally:
        store.close()


def test_fast_build_leaves_no_transaction_and_no_large_wal(repo, monkeypatch):
    seen: list[tuple[bool, int]] = []
    original_close = GraphStore.close

    def close(self):
        wal = Path(str(self.db_path) + "-wal")
        seen.append((self._conn.in_transaction, wal.stat().st_size if wal.exists() else 0))
        original_close(self)

    monkeypatch.setattr(GraphStore, "close", close)
    _build(repo, monkeypatch, fast=True)
    assert seen
    in_transaction, wal_size = seen[-1]
    assert in_transaction is False
    assert wal_size == 0


def test_wal_truncate_does_not_wait_for_a_reader(repo, monkeypatch):
    _build(repo, monkeypatch, fast=False)
    db = repo / ".code-review-graph" / "graph.db"
    store = GraphStore(db)
    reader = sqlite3.connect(db)
    try:
        store.set_metadata("probe", "1")
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM nodes").fetchone()
        store.set_metadata("probe", "2")
        assert store.truncate_wal() is False
        assert int(store._conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 5000
        reader.rollback()
        assert store.truncate_wal() is True
    finally:
        reader.close()
        store.close()
