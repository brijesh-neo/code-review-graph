"""Compare two live graphs by structure: what a branch did to the code's shape.

A text diff shows lines. This shows edges: calls and imports added or
removed, functions that lost every test, new file-to-file dependencies and
flows whose criticality moved. Since schema v16 both graphs store paths
relative to their own checkout, so the stored strings compare as they are,
whatever directory each checkout lives in.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

from .relpaths import PATH_MODE_KEY, PATH_MODE_RELATIVE

logger = logging.getLogger(__name__)

#: Rows listed per section; the counts are always complete.
DEFAULT_LIMIT = 50
#: Criticality moves smaller than this are noise.
CRITICALITY_STEP = 0.05
#: Edge kinds the per-edge lists show. CONTAINS follows the nodes and only
#: repeats what the added and removed symbols already say.
_LISTED_KINDS = ("CALLS", "IMPORTS_FROM", "INHERITS", "IMPLEMENTS", "REFERENCES")

Edge = tuple[str, str, str]


class StructuralDiffError(Exception):
    """A graph that cannot be compared; the message says why."""


def _db_for(path: Path) -> Path:
    from .incremental import get_db_path

    if path.is_file():
        return path
    if path.is_dir():
        db = get_db_path(path, read_only=True)
        if db.is_file():
            return db
        raise StructuralDiffError(f"{path} has no graph; build one there first")
    raise StructuralDiffError(f"{path} does not exist")


def _show(name: str) -> str:
    return name[2:] if name.startswith("./") else name


class _Graph:
    """The parts of one graph the comparison needs, read once."""

    def __init__(self, db: Path) -> None:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise StructuralDiffError(f"{db} could not be opened: {exc}") from exc
        try:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (PATH_MODE_KEY,),
            ).fetchone()
            if not row or row[0] != PATH_MODE_RELATIVE:
                raise StructuralDiffError(
                    f"{db} stores absolute paths; run any command in its checkout "
                    "with this release to convert it"
                )
            self.nodes: dict[str, str] = dict(
                conn.execute("SELECT qualified_name, kind FROM nodes")
            )
            self.edges: set[Edge] = set(
                conn.execute("SELECT source_qualified, target_qualified, kind FROM edges")
            )
            self.flows: dict[str, float] = {
                name: crit for name, crit in conn.execute(
                    "SELECT n.qualified_name, f.criticality FROM flows f "
                    "JOIN nodes n ON n.id = f.entry_point_id"
                )
            }
        except sqlite3.Error as exc:
            raise StructuralDiffError(f"{db} is not a graph: {exc}") from exc
        finally:
            conn.close()

    def tested(self) -> set[str]:
        return {s for s, _t, k in self.edges if k == "TESTED_BY"}


def structural_diff(
    here: Path, other: Path, *, limit: int = DEFAULT_LIMIT,
) -> dict:
    """How *here*'s graph differs from *other*'s (the base)."""
    here_db, other_db = _db_for(here), _db_for(other)
    new, old = _Graph(here_db), _Graph(other_db)

    added = new.edges - old.edges
    removed = old.edges - new.edges

    def listed(edges: set[Edge]) -> list[dict]:
        rows = sorted(e for e in edges if e[2] in _LISTED_KINDS)
        return [
            {"source": _show(s), "target": _show(t), "kind": k} for s, t, k in rows[:limit]
        ]

    lost_tests = sorted(
        q for q in old.tested() - new.tested() if q in new.nodes
    )
    file_kinds = {"File"}
    new_deps = sorted(
        (s, t) for s, t, k in added
        if k == "IMPORTS_FROM" and new.nodes.get(s) in file_kinds
    )
    flows = []
    for name in sorted(set(new.flows) | set(old.flows)):
        before, after = old.flows.get(name), new.flows.get(name)
        if before is None or after is None or abs(after - before) >= CRITICALITY_STEP:
            flows.append({
                "entry": _show(name),
                "before": None if before is None else round(before, 3),
                "after": None if after is None else round(after, 3),
            })

    added_nodes = sorted(set(new.nodes) - set(old.nodes))
    removed_nodes = sorted(set(old.nodes) - set(new.nodes))
    return {
        "here": str(here_db),
        "base": str(other_db),
        "symbols_added": len(added_nodes),
        "symbols_removed": len(removed_nodes),
        "edges_added_by_kind": dict(sorted(Counter(k for *_x, k in added).items())),
        "edges_removed_by_kind": dict(sorted(Counter(k for *_x, k in removed).items())),
        "edges_added": listed(added),
        "edges_removed": listed(removed),
        "lost_all_tests": [_show(q) for q in lost_tests[:limit]],
        "lost_all_tests_count": len(lost_tests),
        "new_file_dependencies": [
            {"file": _show(s), "imports": _show(t)} for s, t in new_deps[:limit]
        ],
        "new_file_dependencies_count": len(new_deps),
        "flows_changed": flows[:limit],
        "flows_changed_count": len(flows),
        "limit": limit,
    }


def _kinds(counts: dict) -> str:
    return ", ".join(f"{n} {k}" for k, n in counts.items()) or "none"


def format_diff(report: dict) -> str:
    lines = [
        f"Structural diff of {report['here']}",
        f"  against {report['base']}",
        f"Symbols: +{report['symbols_added']} -{report['symbols_removed']}",
        f"Edges added: {_kinds(report['edges_added_by_kind'])}",
        f"Edges removed: {_kinds(report['edges_removed_by_kind'])}",
    ]

    def section(title: str, rows: list[str], total: int) -> None:
        if not total:
            return
        lines.append("")
        lines.append(f"{title} ({total}):")
        lines.extend(f"  {r}" for r in rows)
        if total > len(rows):
            lines.append(f"  ... and {total - len(rows)} more (raise --limit)")

    section(
        "Lost every test", report["lost_all_tests"], report["lost_all_tests_count"],
    )
    section(
        "New file dependencies",
        [f"{d['file']} -> {d['imports']}" for d in report["new_file_dependencies"]],
        report["new_file_dependencies_count"],
    )
    section(
        "Flow criticality moved",
        [
            f"{f['entry']}: {'new' if f['before'] is None else f['before']} -> "
            f"{'gone' if f['after'] is None else f['after']}"
            for f in report["flows_changed"]
        ],
        report["flows_changed_count"],
    )
    for key, title in (("edges_added", "Added"), ("edges_removed", "Removed")):
        rows = report[key]
        if rows:
            lines.append("")
            lines.append(f"{title} edges (calls, imports, inheritance, references):")
            lines.extend(f"  {r['source']} {r['kind']} {r['target']}" for r in rows)
    return "\n".join(lines)


def run_structural_diff(
    here: Path, other: Path, *, as_json: bool, limit: int,
) -> Optional[dict]:
    import json

    report = structural_diff(here, other, limit=limit)
    print(json.dumps(report, indent=2) if as_json else format_diff(report))
    return report
