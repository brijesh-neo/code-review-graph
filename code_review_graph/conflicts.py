"""Conflict radar: changes in sibling Git worktrees that touch each other.

Git sees a conflict only when two branches edit the same lines. This finds
the ones it cannot see: one worktree changes a function while another adds
or changes code that calls it, inherits from it or tests it, usually in a
different file.

Every worktree's graph names a symbol by its repository-relative path plus
its name, so the symbols of two worktrees join on that key. For each pair,
both sides' changes are measured from the commit where their branches split
(``git merge-base``), working-tree edits included, and mapped to the
innermost symbols they touch. A finding is one of:

* ``both``: the two worktrees change the same symbol;
* ``uses``: a symbol this worktree changes calls, references, inherits or
  implements one the other worktree changes;
* ``used_by``: the same in the other direction.

A changed test that calls a symbol the other side changes is reported with
the edge ``TESTED_BY``.

Each worktree's own graph supplies its edges, so a graph that is behind its
files gives a partial answer. Read-only: nothing is built or written.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .changes import map_changes_to_nodes, parse_git_diff_ranges
from .constants import GIT_TIMEOUT
from .errors import GraphStoreError
from .graph import GraphNode, GraphStore

logger = logging.getLogger(__name__)

#: Edges from a changed symbol that make the target's change matter to it.
_USE_KINDS = frozenset({"CALLS", "REFERENCES", "INHERITS", "IMPLEMENTS"})


@dataclass
class _Side:
    root: Path
    branch: str
    store: GraphStore
    #: repository-relative key -> changed node
    changed: dict[str, GraphNode] = field(default_factory=dict)

    def key(self, qualified_name: str) -> Optional[str]:
        """The repository-relative key of *qualified_name*, or ``None`` for a
        bare or outside-the-repository name, which cannot match across
        worktrees."""
        prefix = str(self.root) + "/"
        if qualified_name.startswith(prefix):
            return qualified_name[len(prefix):]
        return None


def _git(root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git %s failed in %s: %s", " ".join(args), root, exc)
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def list_worktrees(root: Path) -> list[Path]:
    """Every checked-out worktree of *root*'s repository, *root*'s first."""
    output = _git(root, "worktree", "list", "--porcelain") or ""
    roots: list[Path] = []
    for line in output.splitlines():
        if not line.startswith("worktree "):
            continue
        try:
            candidate = Path(line.removeprefix("worktree ")).resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    me = root.resolve()
    return [me] + [r for r in roots if r != me]


def _innermost_changed(
    nodes: Iterable[GraphNode], ranges: dict[str, list[tuple[int, int]]], root: Path,
) -> list[GraphNode]:
    """The nodes a change touches outside the nodes nested in them.

    A one-line edit inside a method overlaps the method and its class; only
    the method changed. The class counts too when a changed line is in it but
    in none of its changed members (its header, a class attribute).
    """
    candidates = [n for n in nodes if n.kind != "File"]
    by_file: dict[str, list[GraphNode]] = {}
    for node in candidates:
        by_file.setdefault(node.file_path, []).append(node)
    kept: list[GraphNode] = []
    for file_path, file_nodes in by_file.items():
        try:
            rel = Path(file_path).relative_to(root).as_posix()
        except ValueError:
            rel = file_path
        changed_lines: set[int] = set()
        for start, end in ranges.get(rel, ranges.get(file_path, [])):
            changed_lines.update(range(start, end + 1))
        for node in file_nodes:
            mine = {
                line for line in changed_lines
                if node.line_start <= line <= node.line_end
            }
            for other in file_nodes:
                if other is node:
                    continue
                nested = (
                    node.line_start <= other.line_start
                    and other.line_end <= node.line_end
                    and (other.line_start, other.line_end)
                    != (node.line_start, node.line_end)
                )
                if nested:
                    mine -= set(range(other.line_start, other.line_end + 1))
            if mine:
                kept.append(node)
    return kept


def _open_side(root: Path, base: str) -> tuple[Optional[_Side], Optional[str]]:
    """The worktree's graph and changed symbols since *base*, or why not."""
    from .incremental import get_db_path

    db_path = get_db_path(root, read_only=True)
    if not db_path.is_file():
        return None, "no graph; run `code-review-graph build` there"
    try:
        store = GraphStore(db_path, read_only=True)
    except (GraphStoreError, sqlite3.Error, OSError) as exc:
        return None, f"graph could not be opened: {exc}"
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "?"
    side = _Side(root=root, branch=branch, store=store)
    ranges = parse_git_diff_ranges(str(root), base)
    absolute = {str(root / rel): spans for rel, spans in ranges.items()}
    nodes = map_changes_to_nodes(store, absolute)
    for node in _innermost_changed(nodes, ranges, root):
        key = side.key(node.qualified_name)
        if key is not None:
            side.changed[key] = node
    return side, None


def _uses(source: _Side, target: _Side) -> list[dict]:
    """Symbols *source* changes that use, or test, ones *target* changes."""
    found: list[dict] = []
    for key, node in sorted(source.changed.items()):
        for edge in source.store.get_edges_by_source(node.qualified_name):
            if edge.kind not in _USE_KINDS:
                continue
            target_key = source.key(edge.target_qualified)
            if target_key is None or target_key not in target.changed or target_key == key:
                continue
            kind = "TESTED_BY" if node.is_test and edge.kind == "CALLS" else edge.kind
            found.append({"symbol": key, "edge": kind, "changed_there": target_key})
    return _dedupe(found)


def _dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out = []
    for row in rows:
        marker = tuple(sorted(row.items()))
        if marker not in seen:
            seen.add(marker)
            out.append(row)
    return out


def find_conflicts(
    repo_root: Path,
    others: Optional[list[Path]] = None,
    base: Optional[str] = None,
) -> dict:
    """Compare *repo_root*'s changes with each sibling worktree's.

    *others* defaults to every other worktree of the repository. *base*
    defaults, per pair, to the commit where the two branches split.
    """
    repo_root = repo_root.resolve()
    siblings = (
        [p.resolve() for p in others] if others else list_worktrees(repo_root)[1:]
    )
    report: dict = {
        "worktree": str(repo_root),
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") or "?",
        "others": [],
    }
    my_head = _git(repo_root, "rev-parse", "HEAD")
    for other in siblings:
        entry: dict = {"worktree": str(other)}
        report["others"].append(entry)
        their_head = _git(other, "rev-parse", "HEAD")
        pair_base = base
        if pair_base is None and my_head and their_head:
            pair_base = _git(repo_root, "merge-base", my_head, their_head)
        if not pair_base:
            entry["skipped"] = "no common commit with this worktree"
            continue
        entry["base"] = pair_base
        mine, why = _open_side(repo_root, pair_base)
        if mine is None:
            report["skipped"] = why
            return report
        try:
            theirs, why = _open_side(other, pair_base)
            if theirs is None:
                entry["skipped"] = why
                continue
            try:
                entry["branch"] = theirs.branch
                entry["changed_here"] = len(mine.changed)
                entry["changed_there"] = len(theirs.changed)
                entry["both"] = sorted(set(mine.changed) & set(theirs.changed))
                entry["uses"] = _uses(mine, theirs)
                entry["used_by"] = _uses(theirs, mine)
            finally:
                theirs.store.close()
        finally:
            mine.store.close()
    return report


def _count(entry: dict) -> int:
    return len(entry.get("both", [])) + len(entry.get("uses", [])) + len(
        entry.get("used_by", [])
    )


_EDGE_WORDS = {
    "CALLS": "calls",
    "REFERENCES": "references",
    "INHERITS": "inherits",
    "IMPLEMENTS": "implements",
    "TESTED_BY": "tests",
}


def format_report(report: dict) -> str:
    lines = [f"Conflict radar for {report['worktree']} ({report['branch']})"]
    if report.get("skipped"):
        lines.append(f"  skipped: {report['skipped']}")
        return "\n".join(lines)
    if not report["others"]:
        lines.append("  No other worktrees.")
    for entry in report["others"]:
        name = entry["worktree"]
        if entry.get("skipped"):
            lines.append(f"\n{name}: skipped ({entry['skipped']})")
            continue
        lines.append(
            f"\n{name} ({entry['branch']}), since {entry['base'][:10]}: "
            f"{_count(entry)} finding(s); {entry['changed_here']} symbol(s) changed here, "
            f"{entry['changed_there']} there"
        )
        for key in entry["both"]:
            lines.append(f"  both change  {key}")
        for row in entry["uses"]:
            lines.append(
                f"  yours        {row['symbol']} {_EDGE_WORDS[row['edge']]} "
                f"{row['changed_there']}, which they change"
            )
        for row in entry["used_by"]:
            lines.append(
                f"  theirs       {row['symbol']} {_EDGE_WORDS[row['edge']]} "
                f"{row['changed_there']}, which you change"
            )
    return "\n".join(lines)


def run_conflicts(
    repo_root: Path,
    others: Optional[list[Path]] = None,
    base: Optional[str] = None,
    *,
    as_json: bool = False,
    exit_code: bool = False,
) -> int:
    """Print the report. Returns 1 with *exit_code* when anything was found."""
    report = find_conflicts(repo_root, others, base)
    print(json.dumps(report, indent=2) if as_json else format_report(report))
    found = any(_count(entry) for entry in report["others"])
    return 1 if exit_code and found else 0
