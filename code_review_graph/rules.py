"""Architecture rules checked against the graph: ``check-rules``.

Rules live in ``.code-review-graph/rules.toml``::

    [[forbid]]
    name = "the UI does not reach the database"
    from = "src/ui/**"
    to = "src/db/**"
    kinds = ["IMPORTS_FROM", "CALLS"]   # optional; default below

    [no_cycles]
    between = "communities"             # optional

``from`` and ``to`` are globs on repository-relative paths (``*`` within one
directory, ``**`` across them) and may be lists. A ``forbid`` rule flags an
edge whose source file matches ``from`` and whose target file matches
``to``. ``no_cycles`` flags groups of communities that import each other in
a ring. Edges to names the graph never resolved to a file cannot be checked
and are skipped.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

RULES_FILE = "rules.toml"
DEFAULT_KINDS = ("IMPORTS_FROM", "CALLS", "INHERITS", "IMPLEMENTS", "REFERENCES")


class RulesError(Exception):
    """A rules file that cannot be read or used; the message says why."""


def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    pattern = pattern.strip("/")
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


@dataclass
class ForbidRule:
    name: str
    sources: list[re.Pattern[str]]
    targets: list[re.Pattern[str]]
    kinds: frozenset[str]

    def matches(self, src_file: str, dst_file: str, kind: str) -> bool:
        return (
            kind in self.kinds
            and any(p.match(src_file) for p in self.sources)
            and any(p.match(dst_file) for p in self.targets)
        )


def _patterns(value: Any, rule: str, key: str) -> list[re.Pattern[str]]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(v, str) for v in items):
        raise RulesError(f"rule {rule!r}: {key!r} must be a glob or a list of globs")
    return [_glob_regex(v) for v in items]


def load_rules(path: Path) -> tuple[list[ForbidRule], bool]:
    """The forbid rules and whether community cycles are checked."""
    if tomllib is None:
        raise RulesError("reading rules.toml on Python 3.10 needs the 'tomli' package")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RulesError(f"{path} could not be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RulesError(f"{path} is not valid TOML: {exc}") from exc
    rules = []
    for n, raw in enumerate(data.get("forbid", []), 1):
        if not isinstance(raw, dict):
            raise RulesError(f"forbid entry {n} must be a table")
        name = str(raw.get("name") or f"forbid #{n}")
        if "from" not in raw or "to" not in raw:
            raise RulesError(f"rule {name!r} needs both 'from' and 'to'")
        kinds = raw.get("kinds", list(DEFAULT_KINDS))
        if not isinstance(kinds, list) or not all(isinstance(k, str) for k in kinds):
            raise RulesError(f"rule {name!r}: 'kinds' must be a list of edge kinds")
        rules.append(ForbidRule(
            name=name,
            sources=_patterns(raw["from"], name, "from"),
            targets=_patterns(raw["to"], name, "to"),
            kinds=frozenset(k.upper() for k in kinds),
        ))
    cycles = data.get("no_cycles", {})
    if not isinstance(cycles, dict):
        raise RulesError("'no_cycles' must be a table")
    between = cycles.get("between")
    if between not in (None, "communities"):
        raise RulesError("no_cycles.between supports only \"communities\"")
    return rules, between == "communities"


def _file_of(qualified: str) -> Optional[str]:
    """``./a/b.py::f`` -> ``a/b.py``; ``None`` for a name never resolved."""
    if not qualified.startswith("./"):
        return None
    return qualified[2:].split("::", 1)[0]


def _strongly_connected(graph: dict[int, set[int]]) -> list[list[int]]:
    """Tarjan's algorithm, iterative; only groups of two or more."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    on_stack: set[int] = set()
    stack: list[int] = []
    out: list[list[int]] = []
    counter = 0
    for start in sorted(graph):
        if start in index:
            continue
        work = [(start, iter(sorted(graph.get(start, ()))))]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            node, children = work[-1]
            advanced = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(graph.get(child, ())))))
                    advanced = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                group = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    group.append(member)
                    if member == node:
                        break
                if len(group) > 1:
                    out.append(sorted(group))
    return out


def check_rules(
    repo_root: Path, *, changed_files: Optional[Iterable[str]] = None,
    rules_path: Optional[Path] = None,
) -> dict:
    """Every violation of the rules file. With *changed_files* (relative
    paths) only edges that start in those files are reported."""
    from .incremental import get_db_path
    from .relpaths import PATH_MODE_KEY, PATH_MODE_RELATIVE

    repo_root = repo_root.resolve()
    db = get_db_path(repo_root, read_only=True)
    rules_path = rules_path or db.parent / RULES_FILE
    if not rules_path.is_file():
        raise RulesError(f"no rules file at {rules_path}")
    rules, cycles = load_rules(rules_path)
    if not db.is_file():
        raise RulesError(f"{repo_root} has no graph; run `code-review-graph build`")
    only = None if changed_files is None else {Path(f).as_posix() for f in changed_files}

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (PATH_MODE_KEY,),
        ).fetchone()
        if not row or row[0] != PATH_MODE_RELATIVE:
            raise RulesError("the graph stores absolute paths; run `code-review-graph update`")
        edges = conn.execute(
            "SELECT source_qualified, target_qualified, kind, file_path, line FROM edges"
        ).fetchall()
        community_of: dict[str, int] = {}
        names: dict[int, str] = {}
        if cycles:
            community_of = {
                fp[2:]: cid for fp, cid in conn.execute(
                    "SELECT file_path, community_id FROM nodes "
                    "WHERE kind = 'File' AND community_id IS NOT NULL"
                )
            }
            names = dict(conn.execute("SELECT id, name FROM communities"))
    except sqlite3.Error as exc:
        raise RulesError(f"the graph could not be read: {exc}") from exc
    finally:
        conn.close()

    violations = []
    ring: dict[int, set[int]] = {}
    for source, target, kind, file_path, line in edges:
        src_file = _file_of(source) or (file_path[2:] if file_path.startswith("./") else None)
        dst_file = _file_of(target)
        if src_file is None or dst_file is None or src_file == dst_file:
            continue
        if only is not None and src_file not in only:
            continue
        for rule in rules:
            if rule.matches(src_file, dst_file, kind):
                violations.append({
                    "rule": rule.name, "kind": kind, "file": src_file, "line": line,
                    "source": source[2:], "target": target[2:],
                })
        if cycles and kind == "IMPORTS_FROM":
            a, b = community_of.get(src_file), community_of.get(dst_file)
            if a is not None and b is not None and a != b:
                ring.setdefault(a, set()).add(b)
    violations.sort(key=lambda v: (v["rule"], v["file"], v["line"] or 0, v["target"]))
    rings = [
        [names.get(c, f"community {c}") for c in group]
        for group in _strongly_connected(ring)
    ] if cycles else []
    return {"rules_file": str(rules_path), "violations": violations, "community_cycles": rings}


def format_rules_report(report: dict) -> str:
    v, rings = report["violations"], report["community_cycles"]
    if not v and not rings:
        return f"No rule is broken ({report['rules_file']})."
    lines = []
    for item in v:
        where = f"{item['file']}:{item['line']}" if item["line"] else item["file"]
        lines.append(f"{item['rule']}: {item['source']} {item['kind']} {item['target']} "
                     f"({where})")
    for group in rings:
        lines.append("Communities import each other in a ring: " + ", ".join(group))
    lines.append(f"{len(v)} rule violation(s), {len(rings)} community cycle(s).")
    return "\n".join(lines)
