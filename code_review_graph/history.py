"""Structural history: when an edge appeared or went, commit by commit.

``history record`` compares the graph with the edges it saw last time and
writes what was added and removed, tagged with ``HEAD``'s commit, to
``.code-review-graph/history.db``. Run it after each commit (a ``post-commit``
hook line ``code-review-graph history record`` is enough; ``install`` does
not add one). The first run only stores a baseline.

``history show`` answers questions a text log cannot: when did ``a.py``
start depending on ``b.py``, when did a function lose its last test.
"""
from __future__ import annotations

import logging
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HISTORY_FILE = "history.db"
#: Events per ``show`` answer unless asked for more.
DEFAULT_LIMIT = 100
LOST_LAST_TEST = "LOST_LAST_TEST"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_sha TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    dirty INTEGER NOT NULL DEFAULT 0,
    recorded_at TEXT NOT NULL,
    added INTEGER NOT NULL DEFAULT 0,
    removed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    record_id INTEGER NOT NULL REFERENCES records(id),
    change TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_target ON events(target);
CREATE TABLE IF NOT EXISTS current_edges (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    kind TEXT NOT NULL,
    PRIMARY KEY (source, target, kind)
) WITHOUT ROWID;
"""


class HistoryError(Exception):
    """History that cannot be recorded or read; the message says why."""


def _strip(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True,
            timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git %s failed: %s", " ".join(args), exc)
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _history_db(repo_root: Path) -> Path:
    from .incremental import get_db_path

    return get_db_path(repo_root, read_only=True).parent / HISTORY_FILE


def _graph_edges(repo_root: Path) -> set[tuple[str, str, str]]:
    from .incremental import get_db_path

    db = get_db_path(repo_root, read_only=True)
    if not db.is_file():
        raise HistoryError(f"{repo_root} has no graph; run `code-review-graph build`")
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {
            (_strip(s), _strip(t), k) for s, t, k in conn.execute(
                "SELECT source_qualified, target_qualified, kind FROM edges "
                "WHERE kind != 'CONTAINS'"
            )
        }
    except sqlite3.Error as exc:
        raise HistoryError(f"the graph could not be read: {exc}") from exc
    finally:
        conn.close()


def record(repo_root: Path) -> dict:
    """Store what changed in the graph since the last record."""
    repo_root = repo_root.resolve()
    now = _graph_edges(repo_root)
    sha = _git(repo_root, "rev-parse", "HEAD") or "unknown"
    subject = _git(repo_root, "log", "-1", "--format=%s") or ""
    dirty = bool(_git(repo_root, "status", "--porcelain", "--untracked-files=no"))
    path = _history_db(repo_root)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        with conn:
            baseline = conn.execute("SELECT count(*) FROM records").fetchone()[0] == 0
            before = set(conn.execute("SELECT source, target, kind FROM current_edges"))
            added = now - before
            removed = before - now
            if baseline:
                added, removed = set(), set()
            cursor = conn.execute(
                "INSERT INTO records (commit_sha, subject, dirty, recorded_at, added, removed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sha, subject, int(dirty), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 len(added), len(removed)),
            )
            rid = cursor.lastrowid
            rows = [(rid, "+", k, s, t) for s, t, k in sorted(added)]
            rows += [(rid, "-", k, s, t) for s, t, k in sorted(removed)]
            still_tested = {s for s, _t, k in now if k == "TESTED_BY"}
            lost = sorted({s for s, _t, k in removed if k == "TESTED_BY"} - still_tested)
            rows += [(rid, "-", LOST_LAST_TEST, s, "") for s in lost]
            conn.executemany(
                "INSERT INTO events (record_id, change, kind, source, target) "
                "VALUES (?, ?, ?, ?, ?)", rows,
            )
            conn.execute("DELETE FROM current_edges")
            conn.executemany(
                "INSERT INTO current_edges (source, target, kind) VALUES (?, ?, ?)",
                sorted(now),
            )
    except sqlite3.Error as exc:
        raise HistoryError(f"{path} could not be written: {exc}") from exc
    finally:
        conn.close()
    return {
        "history": str(path), "commit": sha, "dirty": dirty, "baseline": baseline,
        "added": len(added), "removed": len(removed), "lost_last_test": lost,
        "added_by_kind": dict(sorted(Counter(k for *_x, k in added).items())),
        "removed_by_kind": dict(sorted(Counter(k for *_x, k in removed).items())),
    }


def show(
    repo_root: Path, *, source: Optional[str] = None, target: Optional[str] = None,
    kind: Optional[str] = None, lost_tests: bool = False, limit: int = DEFAULT_LIMIT,
) -> dict:
    """Events whose source starts with *source* and target with *target*."""
    path = _history_db(repo_root.resolve())
    if not path.is_file():
        raise HistoryError("no history yet; run `code-review-graph history record`")
    where, params = [], []
    if source:
        where.append("substr(e.source, 1, ?) = ?")
        params += [len(source), source]
    if target:
        where.append("substr(e.target, 1, ?) = ?")
        params += [len(target), target]
    if lost_tests:
        where.append("e.kind = ?")
        params.append(LOST_LAST_TEST)
    elif kind:
        where.append("e.kind = ?")
        params.append(kind.upper())
    sql = (
        "SELECT r.id, r.commit_sha, r.subject, r.dirty, r.recorded_at, e.change, e.kind, "
        "e.source, e.target FROM events e JOIN records r ON r.id = e.record_id"
        + (" WHERE " + " AND ".join(where) if where else "")
        + " ORDER BY r.id, e.change DESC, e.kind, e.source, e.target LIMIT ?"
    )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql, (*params, limit + 1)).fetchall()  # nosec B608
        records = conn.execute("SELECT count(*) FROM records").fetchone()[0]
    except sqlite3.Error as exc:
        raise HistoryError(f"{path} could not be read: {exc}") from exc
    finally:
        conn.close()
    events = [
        {"record": r[0], "commit": r[1], "subject": r[2], "dirty": bool(r[3]),
         "recorded_at": r[4], "change": r[5], "kind": r[6], "source": r[7], "target": r[8]}
        for r in rows[:limit]
    ]
    return {"records": records, "events": events, "truncated": len(rows) > limit}


def format_events(result: dict) -> str:
    if not result["events"]:
        return f"No matching change in {result['records']} record(s)."
    lines = []
    for e in result["events"]:
        when = f"{e['commit'][:12]}{'+dirty' if e['dirty'] else ''} {e['recorded_at']}"
        if e["kind"] == LOST_LAST_TEST:
            lines.append(f"{when}  {e['source']} lost its last test  ({e['subject']})")
        else:
            verb = "added" if e["change"] == "+" else "removed"
            lines.append(f"{when}  {verb} {e['source']} {e['kind']} {e['target']}"
                         f"  ({e['subject']})")
    if result["truncated"]:
        lines.append("... more; raise --limit")
    return "\n".join(lines)
