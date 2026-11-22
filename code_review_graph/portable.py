"""Move a graph between checkouts, and back to releases before schema v16.

``seed``: start this checkout's graph from another one's instead of a full
build. The source may be a linked worktree, a fork or clone at a nearby
commit, or a ``graph.db`` file (one a CI run built, say). Nodes and edges
store repository-relative paths since schema v16, so a copy already answers
with this checkout's paths. Two things still name the source: edge ``extra``
JSON (resolver hints such as ``receiver_module``) and the root metadata.
Both are rewritten, then only the files whose bytes differ from what the
graph recorded are parsed again.

``export_absolute``: a copy of the graph in the absolute-path layout, marked
schema v15, which releases before v16 open. v16 and v17 added no tables or
columns, so the paths (and, since v17, the paths inside ``extra``) are the
only difference.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from .relpaths import (
    EXTRA_MODE_KEY,
    MOVED_FROM_KEY,
    PATH_MODE_KEY,
    PATH_MODE_RELATIVE,
    ROOT_SPELLING_KEY,
    PathCodec,
    absolute_root,
    root_variants,
)

logger = logging.getLogger(__name__)

#: The schema an absolute-path export is marked with: the last one before
#: repository-relative paths.
ABSOLUTE_SCHEMA_VERSION = 15

#: Columns that hold paths or qualified names, per table.
_PATH_COLUMNS: dict[str, tuple[str, ...]] = {
    "nodes": ("qualified_name", "file_path"),
    "edges": ("source_qualified", "target_qualified", "file_path"),
}


class PortableGraphError(Exception):
    """A seed or export that cannot go ahead; the message says why."""


def _source_db(source: Path) -> Path:
    """The graph file *source* names: the file itself, or a checkout's graph."""
    from .incremental import get_db_path

    if source.is_file():
        return source
    if source.is_dir():
        db = get_db_path(source, read_only=True)
        if db.is_file():
            return db
        raise PortableGraphError(f"{source} has no graph; build one there first")
    raise PortableGraphError(f"{source} does not exist")


def _meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _copy(src: Path, dst: Path) -> None:
    """A consistent copy, WAL included, through SQLite's backup API."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".seeding")
    if tmp.exists():
        tmp.unlink()
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        target = sqlite3.connect(str(tmp))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()
    os.replace(tmp, dst)


def _rewrite_extra_roots(conn: sqlite3.Connection, old_root: str, new_root: str) -> int:
    """Point ``extra`` JSON strings that name *old_root* at *new_root*."""
    changed = 0
    for table in ("nodes", "edges"):
        for variant in root_variants(old_root):
            cursor = conn.execute(
                f"UPDATE {table} SET extra = replace(extra, ?, ?) "  # nosec B608
                "WHERE instr(extra, ?) > 0",
                ('"' + variant + "/", '"' + new_root + "/", '"' + variant + "/"),
            )
            changed += cursor.rowcount
    return changed


def _files_to_reparse(repo_root: Path, stored: dict[str, str]) -> list[str]:
    """Repository-relative paths whose bytes differ from the recorded hash,
    plus files the graph lacks and recorded files that are gone."""
    from .incremental import collect_all_files

    on_disk = set(collect_all_files(repo_root))
    recorded: dict[str, str] = {}
    for path, digest in stored.items():
        try:
            recorded[Path(path).relative_to(repo_root).as_posix()] = digest
        except ValueError:
            continue
    out = set(on_disk - set(recorded))
    for rel, digest in recorded.items():
        full = repo_root / rel
        try:
            if hashlib.sha256(full.read_bytes()).hexdigest() != digest:
                out.add(rel)
        except OSError:
            out.add(rel)  # gone, or unreadable: the update drops it
    return sorted(out)


def seed_graph(repo_root: Path, source: Path, *, force: bool = False) -> dict:
    """Start *repo_root*'s graph from *source*'s, then update what differs."""
    from .graph import GraphStore
    from .incremental import get_db_path, incremental_update
    from .postprocessing import run_post_processing

    repo_root = repo_root.resolve()
    src_db = _source_db(source.expanduser().resolve())
    dst_db = get_db_path(repo_root)
    if src_db.resolve() == dst_db.resolve():
        raise PortableGraphError("the source is this checkout's own graph")
    if dst_db.exists() and not force:
        raise PortableGraphError(
            f"{repo_root} already has a graph; pass --force to replace it"
        )

    probe = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    try:
        mode = _meta(probe, PATH_MODE_KEY)
        old_root = _meta(probe, ROOT_SPELLING_KEY) or _meta(probe, "repo_root")
    except sqlite3.Error as exc:
        raise PortableGraphError(f"{src_db} is not a graph: {exc}") from exc
    finally:
        probe.close()
    if mode != PATH_MODE_RELATIVE or not old_root:
        raise PortableGraphError(
            f"{src_db} stores absolute paths; open it once with this release "
            "(any command in its checkout) to convert it, then seed from it"
        )

    for suffix in ("-wal", "-shm"):
        stale = dst_db.with_name(dst_db.name + suffix)
        if stale.exists():
            stale.unlink()
    _copy(src_db, dst_db)

    new_root = absolute_root(str(repo_root))
    conn = sqlite3.connect(str(dst_db))
    try:
        with conn:
            rewritten = _rewrite_extra_roots(conn, old_root, new_root)
            for key, value in (
                (ROOT_SPELLING_KEY, new_root),
                ("repo_root", new_root),
                (MOVED_FROM_KEY, ""),
            ):
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value),
                )
    finally:
        conn.close()

    store = GraphStore(dst_db)
    try:
        store.bind_repo_root(repo_root)
        changed = _files_to_reparse(repo_root, store.get_file_hashes())
        result = incremental_update(repo_root, store, changed_files=changed)
        post = run_post_processing(store, changed_files=changed)
    finally:
        store.close()
    return {
        "source": str(src_db),
        "graph": str(dst_db),
        "extra_rows_rewritten": rewritten,
        "files_reparsed": len(changed),
        "files_updated": result.get("files_updated", 0),
        "total_nodes": result.get("total_nodes"),
        "total_edges": result.get("total_edges"),
        "errors": result.get("errors", []),
        "warnings": post.get("warnings", []),
    }


def export_absolute(repo_root: Path, out: Path) -> dict:
    """Write *repo_root*'s graph to *out* with absolute paths, as schema v15."""
    from .incremental import get_db_path

    repo_root = repo_root.resolve()
    src_db = get_db_path(repo_root, read_only=True)
    if not src_db.is_file():
        raise PortableGraphError(f"{repo_root} has no graph")
    out = out.expanduser().resolve()
    if out == src_db.resolve():
        raise PortableGraphError("the export must not overwrite the graph itself")
    if out.exists():
        raise PortableGraphError(f"{out} exists; pick another path")
    _copy(src_db, out)
    conn = sqlite3.connect(str(out))
    try:
        mode = _meta(conn, PATH_MODE_KEY)
        root = _meta(conn, ROOT_SPELLING_KEY) or absolute_root(str(repo_root))
        # Since v17 the paths inside ``extra`` JSON are relative too.
        extra = ("extra",) if _meta(conn, EXTRA_MODE_KEY) == PATH_MODE_RELATIVE else ()
        rows = 0
        with conn:
            if mode == PATH_MODE_RELATIVE:
                codec = PathCodec(root)
                for table, path_columns in _PATH_COLUMNS.items():
                    columns = path_columns + extra
                    select = ", ".join(columns)
                    data = conn.execute(
                        f"SELECT id, {select} FROM {table}"  # nosec B608
                    ).fetchall()
                    assign = ", ".join(f"{c} = ?" for c in columns)
                    conn.executemany(
                        f"UPDATE {table} SET {assign} WHERE id = ?",  # nosec B608
                        [
                            tuple(codec.restore(v) for v in row[1:]) + (row[0],)
                            for row in data
                        ],
                    )
                    rows += len(data)
                tables = {
                    r[0] for r in conn.execute("SELECT name FROM sqlite_master")
                }
                if "embeddings" in tables:
                    conn.execute(
                        "UPDATE OR IGNORE embeddings SET qualified_name = ? || "
                        "substr(qualified_name, 2) WHERE qualified_name LIKE './%'",
                        (codec.root,),
                    )
                if "nodes_fts" in tables:
                    # External-content FTS indexed the relative spellings.
                    conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
                (str(ABSOLUTE_SCHEMA_VERSION),),
            )
            conn.execute(
                "DELETE FROM metadata WHERE key IN (?, ?, ?, ?)",
                (PATH_MODE_KEY, ROOT_SPELLING_KEY, MOVED_FROM_KEY, EXTRA_MODE_KEY),
            )
        conn.execute("VACUUM")
    finally:
        conn.close()
    return {"graph": str(src_db), "out": str(out), "rows_rewritten": rows,
            "schema_version": ABSOLUTE_SCHEMA_VERSION}
