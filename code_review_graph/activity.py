"""What an agent read, edited and ran, and the impact it has not checked.

The opt-in ``enrich`` hook (PreToolUse on Read and Bash) appends the files a
session reads and the test commands it runs to
``.code-review-graph/activity.jsonl``; once that file exists, the edit hook
appends the files it edits. Nothing is logged in a repository where the log
was never started (``enrich`` starts it; so does ``unchecked --start``).

``unchecked`` joins the log with the graph: for every symbol changed since
``HEAD``, its callers in files the session never read and its tests the
session never ran.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ACTIVITY_FILE = "activity.jsonl"
#: The log is cut back to its newer half past this size.
MAX_BYTES = 1_000_000
#: Rows listed per symbol.
LIST_LIMIT = 10

_TEST_RUNNERS = ("pytest", "py.test", "jest", "vitest", "mocha", "rspec", "phpunit")
_TEST_COMMANDS = (("go", "test"), ("cargo", "test"), ("npm", "test"), ("yarn", "test"),
                  ("pnpm", "test"), ("python", "-m", "pytest"), ("python3", "-m", "pytest"),
                  ("uv", "run", "pytest"), ("mvn", "test"), ("gradle", "test"))


def activity_path(repo_root: Path) -> Path:
    from .incremental import get_db_path

    return get_db_path(repo_root, read_only=True).parent / ACTIVITY_FILE


def _rel(repo_root: Path, path: str) -> Optional[str]:
    try:
        full = Path(path)
        if not full.is_absolute():
            full = repo_root / full
        return full.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def is_test_command(command: str) -> bool:
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    names = [os.path.basename(w) for w in words]
    for i in range(len(names)):
        if names[i] in _TEST_RUNNERS:
            return True
        for pattern in _TEST_COMMANDS:
            if tuple(names[i:i + len(pattern)]) == pattern:
                return True
    return False


def log_event(
    repo_root: Path, kind: str, *, session: str = "", path: str = "",
    command: str = "", create: bool = False,
) -> bool:
    """Append one event; ``False`` when the log is not started (and not *create*)."""
    log = activity_path(repo_root)
    if not log.parent.is_dir() or (not create and not log.is_file()):
        return False
    row: dict[str, Any] = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session, "kind": kind,
    }
    if path:
        rel = _rel(repo_root, path)
        if rel is None:
            return False
        row["path"] = rel
    if command:
        row["command"] = command[:2000]
    try:
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if log.stat().st_size > MAX_BYTES:
            lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
            log.write_text("".join(lines[len(lines) // 2:]), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write the activity log %s: %s", log, exc)
        return False
    return True


def record_tool_use(repo_root: Path, payload: dict) -> None:
    """Log a Read or a test-running Bash call from a PreToolUse payload."""
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    session = str(payload.get("session_id", ""))
    if not isinstance(tool_input, dict):
        return
    if tool == "Read" and tool_input.get("file_path"):
        log_event(repo_root, "read", session=session, path=str(tool_input["file_path"]),
                  create=True)
    elif tool == "Bash":
        command = str(tool_input.get("command", ""))
        if is_test_command(command):
            log_event(repo_root, "test", session=session, command=command, create=True)


def read_activity(repo_root: Path, session: Optional[str] = None) -> dict:
    """Files read and edited and test commands run, for *session* (default:
    the session of the newest event)."""
    log = activity_path(repo_root)
    events = []
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                events.append(row)
    if session is None and events:
        session = str(events[-1].get("session", ""))
    mine = [e for e in events if session is None or str(e.get("session", "")) == session]
    return {
        "session": session,
        "read": {e["path"] for e in mine if e.get("kind") == "read" and e.get("path")},
        "edited": {e["path"] for e in mine if e.get("kind") == "edit" and e.get("path")},
        "tests": [e["command"] for e in mine if e.get("kind") == "test" and e.get("command")],
        "logged": log.is_file(),
    }


def _test_was_run(test_file: str, commands: list[str]) -> bool:
    for command in commands:
        try:
            words = shlex.split(command)
        except ValueError:
            words = command.split()
        targets = [w for w in words if not w.startswith("-") and (
            "/" in w or w.endswith((".py", ".ts", ".js")) or "::" in w)]
        if not targets:
            return True  # a bare `pytest` runs every test
        for t in targets:
            spec = t.split("::", 1)[0].rstrip("/")
            if test_file == spec or test_file.startswith(spec + "/") or \
                    re.search(r"(^|/)" + re.escape(spec) + r"$", test_file):
                return True
    return False


def unchecked(repo_root: Path, *, base: str = "HEAD", session: Optional[str] = None) -> dict:
    """Changed symbols with their unread callers and unrun tests."""
    from .changes import map_changes_to_nodes, parse_git_diff_ranges
    from .conflicts import _innermost_changed
    from .graph import GraphStore
    from .incremental import get_db_path

    repo_root = repo_root.resolve()
    activity = read_activity(repo_root, session)
    db = get_db_path(repo_root, read_only=True)
    if not db.is_file():
        return {"error": "no graph; run `code-review-graph build`", "symbols": []}
    ranges = parse_git_diff_ranges(str(repo_root), base)
    absolute = {str(repo_root / rel): spans for rel, spans in ranges.items()}
    prefix = str(repo_root) + "/"
    symbols = []
    store = GraphStore(db, read_only=True)
    try:
        nodes = _innermost_changed(map_changes_to_nodes(store, absolute), ranges, repo_root)
        for node in sorted(nodes, key=lambda n: n.qualified_name):
            if node.is_test:
                continue
            callers = sorted({
                e.source_qualified for e in store.get_edges_by_target(node.qualified_name)
                if e.kind in ("CALLS", "REFERENCES", "INHERITS", "IMPLEMENTS")
            })
            caller_files = {c.split("::", 1)[0] for c in callers}
            tests = sorted({
                t["file_path"] for t in store.get_transitive_tests(node.qualified_name)
                if t.get("file_path")
            })
            rel_tests = [t[len(prefix):] for t in tests if t.startswith(prefix)]
            # A test that calls the symbol is counted under tests, not callers.
            rel_callers = {f[len(prefix):] for f in caller_files
                           if f.startswith(prefix) and f not in tests}
            # A file changed since the base was in front of whoever changed it.
            seen = activity["read"] | activity["edited"] | set(ranges)
            unread = sorted(rel_callers - seen)
            unrun = [t for t in rel_tests if not _test_was_run(t, activity["tests"])]
            symbols.append({
                "symbol": node.qualified_name[len(prefix):]
                if node.qualified_name.startswith(prefix) else node.qualified_name,
                "callers": sum(1 for c in callers if c.split("::", 1)[0] not in tests),
                "caller_files": len(rel_callers),
                "caller_files_read": len(rel_callers) - len(unread),
                "unread_caller_files": unread[:LIST_LIMIT],
                "tests": len(rel_tests),
                "tests_run": len(rel_tests) - len(unrun),
                "unrun_tests": unrun[:LIST_LIMIT],
            })
    finally:
        store.close()
    return {"session": activity["session"], "logged": activity["logged"],
            "symbols": symbols}


def _line(s: dict) -> str:
    return (f"{s['symbol']} changed: {s['callers']} caller(s) in {s['caller_files']} "
            f"file(s), {s['caller_files_read']} read; {s['tests']} test file(s), "
            f"{s['tests_run']} run")


def open_items(report: dict) -> list[dict]:
    return [s for s in report["symbols"] if s["unread_caller_files"] or s["unrun_tests"]]


def format_unchecked(report: dict) -> str:
    if report.get("error"):
        return f"Error: {report['error']}"
    if not report["symbols"]:
        return "No changed symbol since the base."
    lines = []
    if not report["logged"]:
        lines.append("No activity log here, so nothing counts as read or run. Start one "
                     "with `code-review-graph unchecked --start` or the enrich hook.")
    for s in report["symbols"]:
        lines.append(_line(s))
        lines.extend(f"  not read: {f}" for f in s["unread_caller_files"])
        lines.extend(f"  not run:  {t}" for t in s["unrun_tests"])
    return "\n".join(lines)


def hook_summary(report: dict, limit: int = 5) -> str:
    """One short note for the edit hook, or "" when nothing is unchecked."""
    items = open_items(report)
    if not items:
        return ""
    lines = ["code-review-graph: impact of your edits not checked yet:"]
    for s in items[:limit]:
        lines.append("- " + _line(s))
    if len(items) > limit:
        lines.append(f"- and {len(items) - limit} more (`code-review-graph unchecked`)")
    return "\n".join(lines)
