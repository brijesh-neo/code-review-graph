"""Execution flow detection, tracing, and criticality scoring.

Detects entry points in the codebase (functions with no incoming CALLS edges,
framework-decorated handlers, and conventional name patterns), traces execution
paths via forward BFS through CALLS edges, scores each flow for criticality,
and persists results to the ``flows`` / ``flow_memberships`` tables.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import sqlite3
import stat
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .constants import SECURITY_KEYWORDS as _SECURITY_KEYWORDS
from .graph import FlowAdjacency, GraphNode, GraphStore, _sanitize_name
from .parser import fast_engine_enabled, normalize_file_path
from .parser import is_test_file as _is_test_file

logger = logging.getLogger(__name__)

_SQL_BATCH = 450


def _escape_like_literal(value: str) -> str:
    """Escape ``\\``, ``%``, and ``_`` for use in a SQL LIKE pattern."""
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _resolver() -> Callable[[Path], Path]:
    """``Path.resolve`` for many paths, resolving each directory once.

    The real path of ``dir/name`` is the real path of ``dir`` joined with
    ``name``, unless ``name`` is a link (resolved in full then) or a ``.`` or
    ``..`` step. Every changed file shares its directories with the others,
    so this spares most of the ``lstat`` calls, which were most of the time
    an update spent here. POSIX only; elsewhere each path is resolved as is.
    """
    if os.name != "posix":
        return Path.resolve
    directories: dict[str, str] = {}

    def resolve(path: Path) -> Path:
        directory, name = os.path.split(str(path))
        if name in ("", ".", "..") or not directory:
            return path.resolve()
        real = directories.get(directory)
        if real is None:
            real = directories[directory] = os.path.realpath(directory)
        joined = os.path.join(real, name)
        try:
            if stat.S_ISLNK(os.lstat(joined).st_mode):
                return path.resolve()
            os.stat(joined)
        except OSError as exc:
            # ``resolve`` reports a link loop on the way; let it.
            if exc.errno == errno.ELOOP:
                return path.resolve()
        return Path(joined)

    return resolve


def expand_changed_file_paths(
    store: GraphStore,
    changed_files: Iterable[str],
    repo_root: str | Path | None = None,
) -> list[str]:
    """Expand *changed_files* to every path form present in the graph.

    ``git diff --name-only`` yields repo-relative paths, while parsed nodes
    store absolute ``file_path`` values (and tests often use relative paths).
    Matching must accept both without treating ``%`` / ``_`` as wildcards
    (#569). When *repo_root* is provided, only exact absolute/relative
    variants are used — the ``LIKE`` basename fallback is skipped so a
    root-level ``util.py`` cannot match ``left/util.py``.
    """
    root = Path(repo_root).resolve() if repo_root is not None else None
    variants: list[str] = []
    seen: set[str] = set()
    relative_inputs: list[str] = []

    def _add(path: str) -> None:
        normalized = normalize_file_path(path)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append(normalized)

    resolve = _resolver()
    for raw in changed_files:
        text = normalize_file_path(raw)
        _add(text)
        path = Path(raw)
        if path.is_absolute():
            _add(str(path))
            try:
                resolved = resolve(path)
                _add(str(resolved))
                if root is not None:
                    _add(str(resolved.relative_to(root)))
            except (OSError, ValueError):
                pass
        else:
            relative_inputs.append(text)
            if root is not None:
                _add(str(root / path))
                try:
                    _add(str(resolve(root / path)))
                except OSError:
                    pass

    # Exact hits first (graph identities are POSIX-normalized; see #774).
    matched: list[str] = list(variants)
    matched_seen = set(variants)
    if variants:
        for i in range(0, len(variants), _SQL_BATCH):
            path_batch = variants[i:i + _SQL_BATCH]
            placeholders = ",".join("?" * len(path_batch))
            rows = store._conn.execute(
                f"SELECT DISTINCT file_path FROM nodes "  # nosec B608
                f"WHERE file_path IN ({placeholders})",
                path_batch,
            ).fetchall()
            for row in rows:
                fp = row[0]
                if fp not in matched_seen:
                    matched.append(fp)
                    matched_seen.add(fp)

    # LIKE fallback is only for relative inputs without a repo root, where we
    # cannot build exact absolute variants. When repo_root is known, skip it:
    # otherwise root-level ``util.py`` also matches ``left/util.py`` via
    # ``%/util.py``, and incremental clears the wrong file's flows.
    if root is None:
        for rel in relative_inputs:
            escaped = _escape_like_literal(rel)
            patterns = (
                escaped,                 # exact relative path
                f"%/{escaped}",          # POSIX separator anchor
                f"%\\\\{escaped}",       # legacy Windows separator anchor
            )
            for pattern in patterns:
                rows = store._conn.execute(
                    "SELECT DISTINCT file_path FROM nodes "
                    "WHERE file_path LIKE ? ESCAPE '\\'",
                    (pattern,),
                ).fetchall()
                for row in rows:
                    fp = row[0]
                    if fp not in matched_seen:
                        matched.append(fp)
                        matched_seen.add(fp)
    return matched


def _flows_touching_files(
    store: GraphStore,
    file_paths: list[str],
) -> tuple[list[int], list[int], set[str]]:
    """Return ``(node_ids, flow_ids, entry_point_qns)`` for *file_paths*."""
    if not file_paths:
        return [], [], set()

    node_ids = list(store.get_node_ids_by_files(file_paths))
    if not node_ids:
        return [], [], set()

    flow_ids = store.get_flow_ids_by_node_ids(set(node_ids))
    if not flow_ids:
        return node_ids, [], set()

    return node_ids, flow_ids, _flow_node_qns(store, flow_ids)


def _flow_node_qns(store: GraphStore, flow_ids: list[int]) -> set[str]:
    """Return the qualified names of every node in *flow_ids*, entry points included.

    Any of them may head a flow once these flows go: when a caller is deleted or
    stops calling, the next function down has no caller left, so a full rebuild
    traces it even though its own file did not change.
    """
    qns: set[str] = set()
    for i in range(0, len(flow_ids), _SQL_BATCH):
        batch = flow_ids[i:i + _SQL_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = store._conn.execute(
            f"SELECT n.qualified_name FROM flows f "  # nosec B608
            f"JOIN nodes n ON n.id = f.entry_point_id "
            f"WHERE f.id IN ({placeholders}) "
            f"UNION "
            f"SELECT n.qualified_name FROM flow_memberships fm "
            f"JOIN nodes n ON n.id = fm.node_id "
            f"WHERE fm.flow_id IN ({placeholders})",
            [*batch, *batch],
        ).fetchall()
        qns.update(r[0] for r in rows if r[0])
    return qns


def capture_affected_flow_entry_points(
    store: GraphStore,
    file_paths: list[str],
) -> set[str]:
    """Return the names that may head a flow, for flows touching *file_paths*.

    That is every node of those flows, not only their entry points; see
    :func:`_flow_node_qns`. Must run **before** node replacement: after
    ``remove_file_data`` the old node IDs are gone and membership JOINs can no
    longer find those flows.
    """
    _node_ids, _flow_ids, entry_qns = _flows_touching_files(store, file_paths)
    return entry_qns


def capture_flow_node_ids(store: GraphStore, file_paths: list[str]) -> dict[int, str]:
    """Map the IDs of flow nodes in *file_paths* to their qualified names.

    Must run **before** node replacement, which gives every node a new ID.
    """
    captured: dict[int, str] = {}
    for i in range(0, len(file_paths), _SQL_BATCH):
        batch = file_paths[i:i + _SQL_BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = store._conn.execute(
            f"SELECT id, qualified_name FROM nodes "  # nosec B608
            f"WHERE file_path IN ({placeholders}) AND ("
            f"id IN (SELECT node_id FROM flow_memberships) "
            f"OR id IN (SELECT entry_point_id FROM flows))",
            batch,
        ).fetchall()
        captured.update({row[0]: row[1] for row in rows})
    return captured


def remap_flow_node_ids(store: GraphStore, old_ids: dict[int, str]) -> int:
    """Point stored flows at the new rows of re-parsed nodes (#569).

    ``update --skip-flows``, the path the installed hooks run, never re-traces.
    Deleting the flows of a re-parsed file would lose them until the next full
    post-process, so they stay, keyed to the same symbols by qualified name.
    Nodes that no longer exist are left for :func:`purge_orphan_flow_data`.

    Returns the number of nodes remapped.
    """
    conn = store._conn
    new_ids: dict[int, int] = {}
    for old_id, qualified_name in old_ids.items():
        row = conn.execute(
            "SELECT id FROM nodes WHERE qualified_name = ?", (qualified_name,),
        ).fetchone()
        if row is not None and row[0] != old_id:
            new_ids[old_id] = row[0]
    if not new_ids:
        return 0

    if conn.in_transaction:
        logger.warning("Rolling back uncommitted transaction before BEGIN IMMEDIATE")
        conn.rollback()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for old_id, new_id in new_ids.items():
            conn.execute(
                "UPDATE flow_memberships SET node_id = ? WHERE node_id = ?",
                (new_id, old_id),
            )
            conn.execute(
                "UPDATE flows SET entry_point_id = ? WHERE entry_point_id = ?",
                (new_id, old_id),
            )
        for flow_id, path_json in conn.execute("SELECT id, path_json FROM flows").fetchall():
            path = json.loads(path_json)
            if any(node_id in new_ids for node_id in path):
                conn.execute(
                    "UPDATE flows SET path_json = ? WHERE id = ?",
                    (json.dumps([new_ids.get(n, n) for n in path]), flow_id),
                )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return len(new_ids)


def clear_flows_for_files(store: GraphStore, file_paths: list[str]) -> set[str]:
    """Delete flows that touch *file_paths* before those nodes are replaced.

    Returns the qualified names to re-trace after replacement if they are entry
    points then (stable across node ID churn); see :func:`_flow_node_qns`.
    """
    if not file_paths:
        return set()

    node_ids, flow_ids, entry_qns = _flows_touching_files(store, file_paths)
    if not node_ids and not flow_ids:
        return entry_qns

    conn = store._conn
    if conn.in_transaction:
        logger.warning("Rolling back uncommitted transaction before BEGIN IMMEDIATE")
        conn.rollback()
    conn.execute("BEGIN IMMEDIATE")
    try:
        for i in range(0, len(flow_ids), _SQL_BATCH):
            flow_batch = flow_ids[i:i + _SQL_BATCH]
            placeholders = ",".join("?" * len(flow_batch))
            conn.execute(
                f"DELETE FROM flow_memberships WHERE flow_id IN ({placeholders})",  # nosec B608
                flow_batch,
            )
            conn.execute(
                f"DELETE FROM flows WHERE id IN ({placeholders})",  # nosec B608
                flow_batch,
            )
        # Drop memberships that still point at nodes about to be deleted so
        # replacement cannot leave dangling flow_memberships rows.
        for i in range(0, len(node_ids), _SQL_BATCH):
            node_batch = node_ids[i:i + _SQL_BATCH]
            placeholders = ",".join("?" * len(node_batch))
            conn.execute(
                f"DELETE FROM flow_memberships WHERE node_id IN ({placeholders})",  # nosec B608
                node_batch,
            )
            conn.execute(
                f"DELETE FROM flows WHERE entry_point_id IN ({placeholders})",  # nosec B608
                node_batch,
            )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return entry_qns


def purge_orphan_flow_data(store: GraphStore) -> int:
    """Remove flow rows/memberships that reference missing nodes.

    Returns the number of orphan memberships deleted.
    """
    conn = store._conn
    if conn.in_transaction:
        logger.warning("Rolling back uncommitted transaction before BEGIN IMMEDIATE")
        conn.rollback()
    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            "DELETE FROM flow_memberships WHERE node_id NOT IN (SELECT id FROM nodes)"
        )
        deleted = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
        conn.execute(
            "DELETE FROM flows WHERE entry_point_id NOT IN (SELECT id FROM nodes)"
        )
        # Flows whose memberships were fully purged.
        conn.execute(
            "DELETE FROM flows WHERE id NOT IN ("
            "SELECT DISTINCT flow_id FROM flow_memberships)"
        )
        # No foreign key cascades: a flow dropped above for its missing entry
        # point still has rows for the members that are alive.
        cur = conn.execute(
            "DELETE FROM flow_memberships WHERE flow_id NOT IN (SELECT id FROM flows)"
        )
        deleted += cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else 0
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return deleted

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decorator patterns that indicate a function is a framework entry point.
_FRAMEWORK_DECORATOR_PATTERNS: list[re.Pattern[str]] = [
    # Python web frameworks
    re.compile(r"app\.(get|post|put|delete|patch|route|websocket|on_event)", re.IGNORECASE),
    re.compile(r"router\.(get|post|put|delete|patch|route)", re.IGNORECASE),
    re.compile(r"blueprint\.(route|before_request|after_request)", re.IGNORECASE),
    re.compile(r"(before|after)_(request|response)", re.IGNORECASE),
    # CLI frameworks
    re.compile(r"click\.(command|group)", re.IGNORECASE),
    re.compile(r"\w+\.(command|group)\b", re.IGNORECASE),  # Click subgroups: @mygroup.command()
    # Pydantic validators/serializers
    re.compile(r"(field|model)_(serializer|validator)", re.IGNORECASE),
    # Task queues
    re.compile(r"(celery\.)?(task|shared_task|periodic_task)", re.IGNORECASE),
    # Django
    re.compile(r"receiver", re.IGNORECASE),
    re.compile(r"api_view", re.IGNORECASE),
    re.compile(r"\baction\b", re.IGNORECASE),
    # Testing
    re.compile(r"pytest\.(fixture|mark)"),
    # JUnit / TestNG methods are invoked by the runner. The parser keeps an
    # annotation's package and arguments, so both are allowed around the name.
    re.compile(
        r"^(?:[\w.]+\.)?(?:Test|TestFactory|TestTemplate|ParameterizedTest|RepeatedTest"
        r"|BeforeEach|AfterEach|BeforeAll|AfterAll|Before|After|BeforeClass|AfterClass"
        r"|BeforeMethod|AfterMethod|BeforeSuite|AfterSuite|BeforeTest|AfterTest"
        r"|BeforeGroups|AfterGroups|DataProvider)(?:\s*\(.*)?$",
        re.DOTALL,
    ),
    re.compile(r"(override_settings|modify_settings)", re.IGNORECASE),
    # SQLAlchemy / event systems
    re.compile(r"(event\.)?listens_for", re.IGNORECASE),
    # Java Spring
    re.compile(r"(Get|Post|Put|Delete|Patch|RequestMapping)Mapping", re.IGNORECASE),
    re.compile(r"(Scheduled|EventListener|Bean|Configuration)", re.IGNORECASE),
    re.compile(r"KafkaListener", re.IGNORECASE),
    # Temporal Java callbacks are invoked by the workflow runtime.
    re.compile(r"(WorkflowMethod|ActivityMethod)", re.IGNORECASE),
    # JS/TS frameworks
    re.compile(r"(Component|Injectable|Controller|Module|Guard|Pipe)", re.IGNORECASE),
    re.compile(r"(Subscribe|Mutation|Query|Resolver)", re.IGNORECASE),
    # Express / Koa / Hono route handlers
    re.compile(r"(app|router)\.(get|post|put|delete|patch|use|all)\b"),
    # Kotlin coroutines / Android ViewModel
    re.compile(r"(HiltViewModel|AndroidEntryPoint|Inject)", re.IGNORECASE),
    # AI/agent frameworks (pydantic-ai, langchain, etc.)
    re.compile(r"\w+\.(tool|tool_plain|system_prompt|result_validator)\b", re.IGNORECASE),
    re.compile(r"^tool\b"),  # bare @tool (LangChain, etc.)
    # Middleware and exception handlers (Starlette, FastAPI, Sanic)
    re.compile(r"\w+\.(middleware|exception_handler|on_exception)\b", re.IGNORECASE),
    # Generic route decorator (Flask blueprints: @bp.route, @auth_bp.route, etc.)
    re.compile(r"\w+\.route\b", re.IGNORECASE),
]

# An override implements a supertype contract, so its caller is that contract:
# a supertype-typed reference, a framework, or the runtime for Object.hashCode and
# friends. The base is often outside the graph, so no hierarchy walk can rescue it.
# That stops it being dead, but it is no flow root: one with callers is mid-flow.
_OVERRIDE_DECORATOR = re.compile(r"^(?:java\.lang\.)?Override$")

# Name patterns that indicate conventional entry points.
_ENTRY_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^main$"),
    re.compile(r"^__main__$"),
    re.compile(r"^test_"),
    re.compile(r"^Test[A-Z]"),
    re.compile(r"^on_"),
    re.compile(r"^handle_"),
    # Lambda / serverless handler functions (wired via config, not code calls)
    re.compile(r"^handler$"),
    re.compile(r"^handle$"),
    re.compile(r"^lambda_handler$"),
    # Alembic migration entry points
    re.compile(r"^upgrade$"),
    re.compile(r"^downgrade$"),
    # FastAPI lifecycle / dependency injection
    re.compile(r"^lifespan$"),
    re.compile(r"^get_db$"),
    # Android Activity/Fragment lifecycle
    re.compile(r"^on(Create|Start|Resume|Pause|Stop|Destroy|Bind|Receive)"),
    # Servlet / JAX-RS
    re.compile(r"^do(Get|Post|Put|Delete)$"),
    # Python BaseHTTPRequestHandler
    re.compile(r"^do_(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)$"),
    re.compile(r"^log_message$"),
    # Express middleware signature
    re.compile(r"^(middleware|errorHandler)$"),
    # Angular lifecycle hooks
    re.compile(
        r"^ng(OnInit|OnChanges|OnDestroy|DoCheck"
        r"|AfterContentInit|AfterContentChecked|AfterViewInit|AfterViewChecked)$"
    ),
    # Angular Pipe / ControlValueAccessor / Guards / Resolvers
    re.compile(r"^(transform|writeValue|registerOnChange|registerOnTouched|setDisabledState)$"),
    re.compile(r"^(canActivate|canDeactivate|canActivateChild|canLoad|canMatch|resolve)$"),
    # React class component lifecycle
    re.compile(
        r"^(componentDidMount|componentDidUpdate|componentWillUnmount"
        r"|shouldComponentUpdate|render)$"
    ),
]

# Framework and language conventions that must not pollute other parsers.
_LANGUAGE_ENTRY_NAME_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "php": (
        re.compile(r"^(boot|register)$"),
        re.compile(r"^__invoke$"),
    ),
}


# The patterns above joined into one alternation each: a name matches the
# joined pattern exactly when it matches one of them, in one scan.
_ENTRY_NAME_RE = re.compile("|".join(f"(?:{p.pattern})" for p in _ENTRY_NAME_PATTERNS))
_LANGUAGE_ENTRY_NAME_RE: dict[str, re.Pattern[str]] = {
    language: re.compile("|".join(f"(?:{p.pattern})" for p in patterns))
    for language, patterns in _LANGUAGE_ENTRY_NAME_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# Entry-point detection
# ---------------------------------------------------------------------------


def _has_framework_decorator(node: GraphNode) -> bool:
    """Return True if *node* has a decorator matching a framework pattern."""
    decorators = node.extra.get("decorators")
    if not decorators:
        return False
    if isinstance(decorators, str):
        decorators = [decorators]
    for dec in decorators:
        for pat in _FRAMEWORK_DECORATOR_PATTERNS:
            if pat.search(dec):
                return True
    return False


def _has_override_decorator(node: GraphNode) -> bool:
    """Return True if *node* carries Java's ``@Override``."""
    decorators = node.extra.get("decorators")
    if not decorators:
        return False
    if isinstance(decorators, str):
        decorators = [decorators]
    return any(_OVERRIDE_DECORATOR.search(dec) for dec in decorators)


def _matches_entry_name(node: GraphNode) -> bool:
    """Return True if *node*'s name matches a conventional entry-point pattern."""
    if _ENTRY_NAME_RE.search(node.name):
        return True
    language = _LANGUAGE_ENTRY_NAME_RE.get(node.language)
    return language is not None and language.search(node.name) is not None


def detect_entry_points(
    store: GraphStore,
    include_tests: bool = False,
) -> list[GraphNode]:
    """Find functions that are entry points in the graph.

    An entry point is a Function/Test node that either:
    1. Has no incoming CALLS edges (true root), or
    2. Has a framework decorator (e.g. ``@app.get``), or
    3. Matches a conventional name pattern (``main``, ``test_*``, etc.).

    When *include_tests* is False (the default), Test nodes are excluded so
    that flow analysis focuses on production entry points.
    """
    if not include_tests and fast_engine_enabled():
        native = _native_entry_points(store)
        if native is not None:
            return native
    # Build a set of all qualified names that are CALLS targets. Exclude
    # edges sourced at File nodes so that script-/notebook-/top-level-only
    # callees (e.g. ``run_job()`` invoked from module scope, a top-level
    # ``<App />`` render) remain detectable as entry points.
    called_qnames = store.get_all_call_targets(include_file_sources=False)

    # Scan all nodes for entry-point candidates.
    candidate_nodes = store.get_nodes_by_kind(["Function", "Test"])

    # Ruby and Java test calls did not bind to the methods they call until
    # recently, so a method used only by its tests was an entry point. Keep it
    # one: being well tested must not hide a method from flow analysis.
    called_outside_tests = (
        store.get_call_targets_outside_tests()
        if any(node.language in ("ruby", "java") for node in candidate_nodes)
        else set()
    )

    entry_points: list[GraphNode] = []
    seen_qn: set[str] = set()

    # Stored file paths are absolute; the root turns them back into project
    # paths so ``tests/`` means the repository's own tests and not a parent
    # directory of the checkout. See :func:`code_review_graph.parser.is_test_file`.
    repo_root = store.get_repo_root()

    def is_called(node: GraphNode) -> bool:
        called = (
            called_outside_tests
            if node.language in ("ruby", "java")
            else called_qnames
        )
        return node.qualified_name in called

    for node in candidate_nodes:
        if (
            _is_entry_point(node, repo_root, include_tests, is_called)
            and node.qualified_name not in seen_qn
        ):
            entry_points.append(node)
            seen_qn.add(node.qualified_name)

    return entry_points


def _is_entry_point(
    node: GraphNode,
    repo_root: Optional[str],
    include_tests: bool,
    is_called: Callable[[GraphNode], bool],
) -> bool:
    """Whether :func:`detect_entry_points` keeps candidate *node*.

    *is_called* says whether a caller other than a File node (and, for Ruby
    and Java, other than test code) calls it.
    """
    if not include_tests and (
        node.is_test or _is_test_file(node.file_path, repo_root)
    ):
        return False
    if node.extra.get("verilog_kind"):
        return False
    # True root: no one calls this function.
    return (
        not is_called(node)
        or _has_framework_decorator(node)
        or _matches_entry_name(node)
    )


def _answer(called: bool) -> Callable[[GraphNode], bool]:
    """An ``is_called`` for :func:`_is_entry_point` that answers *called*."""
    return lambda _node: called


def _native_entry_points(
    store: GraphStore,
    among: Optional[tuple[list[str], list[int], list[str]]] = None,
) -> Optional[list[GraphNode]]:
    """``detect_entry_points(store)``, its reads done in one native pass.

    With *among* (file paths, node ids, qualified names), only the candidates
    in one of them, as :func:`_entry_points_among` keeps them. The test-file
    rule and the decorator and name patterns are this module's, applied to
    the same values. None when this ``crg_fast`` lacks the pass, SQLite would
    list candidates in an order it does not reproduce, or a row holds a value
    the pass does not read the way Python does.
    """
    try:
        import crg_fast
    except ImportError:
        return None
    stage_job = getattr(crg_fast, "stage_entry_scan", None)
    take = getattr(crg_fast, "take_entry_scan", None)
    if stage_job is None or take is None or not getattr(store, "_native_sql", False):
        return None
    order = _plan_order(store._conn, _CANDIDATE_SQL, _CANDIDATE_ORDERS, _CANDIDATE_PARAMS)
    if order is None:
        return None
    token = stage_job(order, among, store._codec_fields())
    try:
        row = store._conn.execute("SELECT crg_run(?)", (token,)).fetchone()
    except BaseException:
        crg_fast.discard_job(token)
        take(token)
        raise
    candidates = take(token)
    if int(row[0]) < 0 or candidates is None:
        return None
    repo_root = store.get_repo_root()
    test_files: dict[str, bool] = {}
    kept: list[int] = []
    for (
        node_id, kind, name, qualified_name, file_path, language, is_test, raw_extra,
        called_any, called_outside_tests,
    ) in candidates:
        if is_test:
            continue
        test_file = test_files.get(file_path)
        if test_file is None:
            test_file = test_files[file_path] = _is_test_file(file_path, repo_root)
        if test_file:
            continue
        probe = GraphNode(
            id=node_id, kind=kind, name=name, qualified_name=qualified_name,
            file_path=file_path, line_start=0, line_end=0, language=language or "",
            parent_name=None, params=None, return_type=None, is_test=False,
            file_hash=None,
            extra=json.loads(raw_extra) if raw_extra and raw_extra != "{}" else {},
        )
        called = (
            called_outside_tests if probe.language in ("ruby", "java") else called_any
        )
        # The test rule was applied above, as detect_entry_points applies it.
        if _is_entry_point(probe, repo_root, True, _answer(called)):
            kept.append(node_id)
    by_id: dict[int, GraphNode] = {}
    for i in range(0, len(kept), _SQL_BATCH):
        batch = kept[i:i + _SQL_BATCH]
        placeholders = ",".join("?" * len(batch))
        for node_row in store._conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})",  # nosec B608
            batch,
        ):
            node = store._row_to_node(node_row)
            by_id[node.id] = node
    if len(by_id) != len(kept):
        return None
    return [by_id[node_id] for node_id in kept]


def trace_flows_native(store: GraphStore, max_depth: int = 15) -> Optional[list[dict]]:
    """``trace_flows(store)`` for storing, with the reads and the BFS done natively.

    The flows carry what :func:`store_flows` writes (no ``files``). None when
    the native passes cannot run on this graph.
    """
    entry_points = detect_entry_points(store)
    if not entry_points:
        return []
    flows = _native_traced_flows(store, entry_points, max_depth)
    if flows is None:
        return None
    flows.sort(key=lambda f: f["criticality"], reverse=True)
    return flows


# The plans SQLite picks for the whole-graph reads of detect_entry_points and
# load_flow_adjacency, and the order each returns rows in. The changed-files
# path reads only the rows it needs, in that same order.
# The same text and parameters get_nodes_by_kind(["Function", "Test"]) runs.
_CANDIDATE_SQL = "SELECT * FROM nodes WHERE kind IN (?,?)"
_CANDIDATE_PARAMS = ("Function", "Test")
_CANDIDATE_ORDERS = {
    "SEARCH nodes USING INDEX idx_nodes_kind (kind=?)": "kind",
    "SCAN nodes": "id",
}
_CALLS_SQL = (
    "SELECT kind, source_qualified, target_qualified FROM edges "
    "WHERE kind IN ('CALLS', 'TESTED_BY')"
)
_CALLS_ORDERS = {
    "SEARCH edges USING COVERING INDEX idx_edges_composite (kind=?)":
        "target_qualified, file_path, line, id",
    "SCAN edges": "id",
}


def _plan_order(
    conn: Any, sql: str, orders: dict[str, str], params: tuple = (),
) -> Optional[str]:
    """The row order of *sql* under the plan SQLite picks, if one of *orders*."""
    try:
        plan = [row[3] for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)]
    except sqlite3.Error as exc:
        logger.debug("Reading the plan of %s failed: %s", sql, exc)
        return None
    if len(plan) != 1:
        return None
    return orders.get(plan[0])


class _LazyNodesByQn(dict):
    """``FlowAdjacency.nodes_by_qn`` read one node at a time."""

    def __init__(self, store: GraphStore) -> None:
        super().__init__()
        self._store = store
        self._known: dict[Any, Optional[GraphNode]] = {}

    def remember(self, node: GraphNode) -> None:
        self._known[node.qualified_name] = node

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._known:
            row = self._store._conn.execute(
                "SELECT * FROM nodes WHERE qualified_name = ?", (key,),
            ).fetchone()
            self._known[key] = None if row is None else self._store._row_to_node(row)
        node = self._known[key]
        return default if node is None else node

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None


class _LazyNodesById(dict):
    """``FlowAdjacency.nodes_by_id`` read one node at a time."""

    def __init__(self, store: GraphStore, by_qn: _LazyNodesByQn) -> None:
        super().__init__()
        self._store = store
        self._by_qn = by_qn
        self._known: dict[Any, Optional[GraphNode]] = {}

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._known:
            row = self._store._conn.execute(
                "SELECT * FROM nodes WHERE id = ?", (key,),
            ).fetchone()
            node = None if row is None else self._store._row_to_node(row)
            self._known[key] = node
            if node is not None:
                self._by_qn.remember(node)
        node = self._known[key]
        return default if node is None else node

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None


class _LazyCallsOut(dict):
    """``FlowAdjacency.calls_out`` read one caller at a time, in the order the
    whole-graph read lists each caller's targets."""

    def __init__(self, store: GraphStore, order: str) -> None:
        super().__init__()
        self._store = store
        # One of the fixed orders in _CALLS_ORDERS; no value reaches the SQL text.
        self._sql = (
            "SELECT target_qualified FROM edges "  # nosec B608
            f"WHERE kind = 'CALLS' AND source_qualified = ? ORDER BY {order}"
        )
        self._known: dict[Any, list[str]] = {}

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self._known:
            self._known[key] = [
                row[0] for row in self._store._conn.execute(self._sql, (key,))
            ]
        targets = self._known[key]
        return targets if targets else default

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None


class _LazyTestedBy(set):
    """``FlowAdjacency.has_tested_by`` read one name at a time."""

    def __init__(self, store: GraphStore) -> None:
        super().__init__()
        self._store = store
        self._known: dict[object, bool] = {}

    def __contains__(self, key: object) -> bool:
        if key not in self._known:
            self._known[key] = self._store._conn.execute(
                "SELECT 1 FROM edges WHERE kind = 'TESTED_BY' "
                "AND source_qualified = ? LIMIT 1",
                (key,),
            ).fetchone() is not None
        return self._known[key]


def _lazy_flow_adjacency(store: GraphStore) -> Optional[FlowAdjacency]:
    """A :class:`FlowAdjacency` that reads only what a trace asks for.

    It answers every lookup as :meth:`GraphStore.load_flow_adjacency` would.
    None when SQLite would read the whole graph in an order this does not
    reproduce.
    """
    order = _plan_order(store._conn, _CALLS_SQL, _CALLS_ORDERS)
    if order is None:
        return None
    by_qn = _LazyNodesByQn(store)
    return FlowAdjacency(
        calls_out=_LazyCallsOut(store, order),
        has_tested_by=_LazyTestedBy(store),
        nodes_by_qn=by_qn,
        nodes_by_id=_LazyNodesById(store, by_qn),
    )


def _entry_points_among(
    store: GraphStore,
    file_paths: Iterable[str],
    node_ids: Iterable[int],
    qualified_names: Iterable[str],
) -> Optional[list[GraphNode]]:
    """``detect_entry_points(store)`` narrowed to the given files, ids and names.

    Only the candidates in them are read, and each is checked for a caller on
    its own. None when SQLite would list candidates in an order this does not
    reproduce.
    """
    among = (sorted(set(file_paths)), sorted(set(node_ids)), sorted(set(qualified_names)))
    native = _native_entry_points(store, among)
    if native is not None:
        return native
    order = _plan_order(store._conn, _CANDIDATE_SQL, _CANDIDATE_ORDERS, _CANDIDATE_PARAMS)
    if order is None:
        return None
    conn = store._conn
    ids: set[int] = set()
    for column, values in (
        ("file_path", among[0]),
        ("id", among[1]),
        ("qualified_name", among[2]),
    ):
        for i in range(0, len(values), _SQL_BATCH):
            batch = values[i:i + _SQL_BATCH]
            placeholders = ",".join("?" * len(batch))
            # Only a fixed column name and placeholders reach the SQL text.
            ids.update(
                row[0] for row in conn.execute(
                    f"SELECT id FROM nodes WHERE kind IN ('Function', 'Test') "  # nosec B608
                    f"AND {column} IN ({placeholders})",
                    batch,
                )
            )
    wanted = sorted(ids)
    candidates: list[GraphNode] = []
    for i in range(0, len(wanted), _SQL_BATCH):
        batch = wanted[i:i + _SQL_BATCH]
        placeholders = ",".join("?" * len(batch))
        candidates.extend(
            store._row_to_node(row) for row in conn.execute(
                f"SELECT * FROM nodes WHERE id IN ({placeholders})",  # nosec B608
                batch,
            )
        )
    if order == "kind":
        candidates.sort(key=lambda node: (node.kind, node.id))
    else:
        candidates.sort(key=lambda node: node.id)

    def is_called(node: GraphNode) -> bool:
        caller_filter = (
            "(n.kind IS NULL OR (n.kind != 'File' AND n.is_test = 0))"
            if node.language in ("ruby", "java")
            else "(n.kind IS NULL OR n.kind != 'File')"
        )
        return conn.execute(
            "SELECT 1 FROM edges e "  # nosec B608
            "LEFT JOIN nodes n ON n.qualified_name = e.source_qualified "
            f"WHERE e.kind = 'CALLS' AND e.target_qualified = ? AND {caller_filter} "
            "LIMIT 1",
            (node.qualified_name,),
        ).fetchone() is not None

    repo_root = store.get_repo_root()
    return [
        node for node in candidates
        if _is_entry_point(node, repo_root, False, is_called)
    ]


# ---------------------------------------------------------------------------
# Flow tracing (BFS)
# ---------------------------------------------------------------------------


def _trace_single_flow(
    adj: FlowAdjacency,
    ep: GraphNode,
    max_depth: int = 15,
) -> Optional[dict]:
    """Trace a single execution flow from *ep* via forward BFS.

    Returns a flow dict (see :func:`trace_flows` for the schema) or ``None``
    if the flow is trivial (single-node, no outgoing CALLS that resolve).
    """
    path_ids: list[int] = [ep.id]
    path_qnames: list[str] = [ep.qualified_name]
    visited: set[str] = {ep.qualified_name}
    queue: deque[tuple[str, int]] = deque([(ep.qualified_name, 0)])

    actual_depth = 0
    nodes_by_qn = adj.nodes_by_qn
    calls_out = adj.calls_out

    while queue:
        current_qn, depth = queue.popleft()
        if depth > actual_depth:
            actual_depth = depth
        if depth >= max_depth:
            continue

        for target_qn in calls_out.get(current_qn, ()):
            if target_qn in visited:
                continue
            target_node = nodes_by_qn.get(target_qn)
            if target_node is None:
                continue
            visited.add(target_qn)
            path_ids.append(target_node.id)
            path_qnames.append(target_qn)
            queue.append((target_qn, depth + 1))

    # Skip trivial single-node flows.
    if len(path_ids) < 2:
        return None

    files = list({
        n.file_path
        for qn in path_qnames
        if (n := nodes_by_qn.get(qn)) is not None
    })

    flow: dict = {
        "name": _sanitize_name(ep.name),
        "entry_point": ep.qualified_name,
        "entry_point_id": ep.id,
        "path": path_ids,
        "depth": actual_depth,
        "node_count": len(path_ids),
        "file_count": len(files),
        "files": files,
        "criticality": 0.0,
    }
    flow["criticality"] = compute_criticality(flow, adj)
    return flow


def _native_traced_flows(
    store: GraphStore, entry_points: list[GraphNode], max_depth: int,
) -> Optional[list[dict]]:
    """``_trace_single_flow`` over a lazy adjacency, for each of *entry_points*, natively.

    The same reads in the same order, so the same flows. None when this
    ``crg_fast`` lacks the pass, SQLite would read calls in an order the lazy
    adjacency does not reproduce, or a row holds a value the pass does not
    read the way Python does.
    """
    try:
        import crg_fast
    except ImportError:
        return None
    stage_job = getattr(crg_fast, "stage_flow_trace", None)
    take = getattr(crg_fast, "take_flow_trace", None)
    if stage_job is None or take is None or not getattr(store, "_native_sql", False):
        return None
    order = _plan_order(store._conn, _CALLS_SQL, _CALLS_ORDERS)
    if order is None:
        return None
    entries = [(ep.id, ep.qualified_name, ep.name, ep.file_path) for ep in entry_points]
    token = stage_job(entries, max_depth, order, store._codec_fields())
    try:
        row = store._conn.execute("SELECT crg_run(?)", (token,)).fetchone()
    except BaseException:
        crg_fast.discard_job(token)
        take(token)
        raise
    traced = take(token)
    if int(row[0]) < 0 or traced is None:
        return None
    sensitive: dict[tuple[str, str], bool] = {}
    flows: list[dict] = []
    for ep, result in zip(entry_points, traced):
        if result is None:
            continue
        path, depth, file_count, scored, scored_files, external_count, tested_count = result
        security_hits = 0
        for qualified_name, name in scored:
            key = (name, qualified_name)
            hit = sensitive.get(key)
            if hit is None:
                hit = sensitive[key] = _security_sensitive(name, qualified_name)
            security_hits += hit
        flows.append({
            "name": _sanitize_name(ep.name),
            "entry_point": ep.qualified_name,
            "entry_point_id": ep.id,
            "path": path,
            "depth": depth,
            "node_count": len(path),
            "file_count": file_count,
            "criticality": _criticality_score(
                scored_files, external_count, security_hits, tested_count,
                len(scored), depth,
            ),
        })
    return flows


def trace_flows(
    store: GraphStore,
    max_depth: int = 15,
    include_tests: bool = False,
    adjacency: FlowAdjacency | None = None,
) -> list[dict]:
    """Trace execution flows from every entry point via forward BFS.

    Returns a list of flow dicts, each containing:
      - name: human-readable flow name (entry point name)
      - entry_point: qualified name of the entry point
      - entry_point_id: node database id of the entry point
      - path: ordered list of node IDs in the flow
      - depth: maximum BFS depth reached
      - node_count: number of distinct nodes in the path
      - file_count: number of distinct files touched
      - files: list of distinct file paths
      - criticality: computed criticality score (0.0-1.0)

    *adjacency* is a :meth:`GraphStore.load_flow_adjacency` result the caller
    already holds, which saves loading it again.
    """
    entry_points = detect_entry_points(store, include_tests=include_tests)
    if not entry_points:
        return []

    adj = adjacency if adjacency is not None else store.load_flow_adjacency()
    flows: list[dict] = []

    for ep in entry_points:
        flow = _trace_single_flow(adj, ep, max_depth)
        if flow is not None:
            flows.append(flow)

    # Sort by criticality descending.
    flows.sort(key=lambda f: f["criticality"], reverse=True)
    return flows


# ---------------------------------------------------------------------------
# Criticality scoring
# ---------------------------------------------------------------------------


def compute_criticality(flow: dict, adj: FlowAdjacency) -> float:
    """Score a flow from 0.0 to 1.0 based on multiple weighted factors.

    Weights:
      - File spread:         0.30
      - External calls:      0.20
      - Security sensitivity: 0.25
      - Test coverage gap:   0.15
      - Depth:               0.10
    """
    node_ids: list[int] = flow.get("path", [])
    if not node_ids:
        return 0.0

    nodes_by_id = adj.nodes_by_id
    nodes_by_qn = adj.nodes_by_qn
    calls_out = adj.calls_out
    has_tested_by = adj.has_tested_by

    nodes: list[GraphNode] = [
        n for nid in node_ids if (n := nodes_by_id.get(nid)) is not None
    ]
    if not nodes:
        return 0.0

    file_count = len({n.file_path for n in nodes})

    # Calls that target nodes NOT in the graph are considered external.
    external_count = 0
    for n in nodes:
        for target_qn in calls_out.get(n.qualified_name, ()):
            if target_qn not in nodes_by_qn:
                external_count += 1

    security_hits = sum(1 for n in nodes if _security_sensitive(n.name, n.qualified_name))
    tested_count = sum(1 for n in nodes if n.qualified_name in has_tested_by)
    return _criticality_score(
        file_count, external_count, security_hits, tested_count, len(nodes),
        flow.get("depth", 0),
    )


def _security_sensitive(name: str, qualified_name: str) -> bool:
    """Whether a security keyword is in a node's name or qualified name."""
    search = _SECURITY_KEYWORD_RE.search
    return search(name.lower()) is not None or search(qualified_name.lower()) is not None


# One alternation of the keywords answers "is any keyword a substring" in one scan.
_SECURITY_KEYWORD_RE = re.compile("|".join(sorted(map(re.escape, _SECURITY_KEYWORDS))))


def _criticality_score(
    file_count: int,
    external_count: int,
    security_hits: int,
    tested_count: int,
    node_count: int,
    depth: int,
) -> float:
    """The weighted score :func:`compute_criticality` gives these counts."""
    # --- File spread (0.0 - 1.0) ---
    # Normalize: 1 file => 0.0, 5+ files => 1.0
    file_spread = min((file_count - 1) / 4.0, 1.0) if file_count > 1 else 0.0

    # --- External calls (0.0 - 1.0) ---
    # Normalize: 0 => 0.0, 5+ => 1.0
    external_score = min(external_count / 5.0, 1.0)

    # --- Security sensitivity (0.0 - 1.0) ---
    security_score = min(security_hits / max(node_count, 1), 1.0)

    # --- Test coverage gap (0.0 - 1.0) ---
    coverage = tested_count / max(node_count, 1)
    test_gap = 1.0 - coverage

    # --- Depth (0.0 - 1.0) ---
    # Normalize: 0 => 0.0, 10+ => 1.0
    depth_score = min(depth / 10.0, 1.0)

    # --- Weighted sum ---
    criticality = (
        file_spread * 0.30
        + external_score * 0.20
        + security_score * 0.25
        + test_gap * 0.15
        + depth_score * 0.10
    )
    return round(min(max(criticality, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def store_flows(store: GraphStore, flows: list[dict]) -> int:
    """Clear existing flows and persist new ones.

    Returns the number of flows stored.
    """
    # NOTE: store_flows uses _conn directly because it performs
    # multi-statement batch writes (DELETE + INSERT loop) that are
    # tightly coupled to the DB transaction lifecycle.
    conn = store._conn

    if conn.in_transaction:
        logger.warning("Rolling back uncommitted transaction before BEGIN IMMEDIATE")
        conn.rollback()
    # Wrap the full DELETE + INSERT sequence in an explicit transaction
    # so partial writes cannot occur if an exception interrupts the loop.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM flow_memberships")
        conn.execute("DELETE FROM flows")

        count = 0
        membership_rows: list[tuple[int, int, int]] = []
        for flow in flows:
            node_ids = flow.get("path", [])
            path_json = json.dumps(node_ids)
            cursor = conn.execute(
                """INSERT INTO flows
                   (name, entry_point_id, depth, node_count, file_count,
                    criticality, path_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    flow["name"],
                    flow["entry_point_id"],
                    flow["depth"],
                    flow["node_count"],
                    flow["file_count"],
                    flow["criticality"],
                    path_json,
                ),
            )
            flow_id = cursor.lastrowid
            membership_rows.extend(
                (flow_id, node_id, position)
                for position, node_id in enumerate(node_ids)
            )
            count += 1
        if membership_rows:
            # The rows are integers only, so the path translation has nothing to do.
            sqlite3.Connection.executemany(
                conn,
                "INSERT OR IGNORE INTO flow_memberships (flow_id, node_id, position) "
                "VALUES (?, ?, ?)",
                membership_rows,
            )

        conn.execute("DELETE FROM metadata WHERE key = 'csharp_flows_dirty'")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return count


def incremental_trace_flows(
    store: GraphStore,
    changed_files: list[str],
    max_depth: int = 15,
    repo_root: str | Path | None = None,
    entry_point_qns: Iterable[str] | None = None,
) -> int:
    """Re-trace only flows that touch *changed_files*.  Much faster than full trace.

    1. Expand *changed_files* to absolute/relative forms present in the graph (#569).
    2. Purge orphan memberships left behind by node ID replacement.
    3. Find flow IDs whose memberships reference nodes in the expanded paths.
    4. Collect entry-point node IDs / qualified names before deleting those flows.
    5. Delete only the affected flows and their memberships.
    6. Re-detect entry points in expanded paths **or** matching the qualified
       names of nodes in those flows (stable across node replacement).
    7. BFS-trace each relevant entry point via :func:`_trace_single_flow`.
    8. INSERT the new flows (without clearing unrelated flows).

    Returns the number of re-traced flows that were stored.
    """
    if store.get_metadata("csharp_flows_dirty") == "1":
        # C# rebinding can change callers and entry points in files that were
        # never reparsed, so a changed-file set cannot bound the work. Retrace
        # in full until the binder reports a complete affected set.
        return store_flows(store, trace_flows(store, max_depth=max_depth))
    if not changed_files and not entry_point_qns:
        return 0

    # Graph identity uses POSIX separators (#774); bridge native-separator
    # caller paths before matching against stored file_path values.
    changed_files = [normalize_file_path(p) for p in changed_files]
    conn = store._conn
    expanded = expand_changed_file_paths(store, changed_files, repo_root=repo_root)
    changed_file_set = set(expanded)
    preserved_entry_qns = {qn for qn in (entry_point_qns or []) if qn}

    # Safety net for callers that replace nodes without clear_flows_for_files.
    purge_orphan_flow_data(store)

    # ------------------------------------------------------------------
    # 1. Find affected flow IDs (path-expanded)
    # ------------------------------------------------------------------
    affected_ids: list[int] = []
    if expanded:
        for i in range(0, len(expanded), _SQL_BATCH):
            path_batch = expanded[i:i + _SQL_BATCH]
            placeholders = ",".join("?" * len(path_batch))
            affected_rows = conn.execute(
                f"SELECT DISTINCT fm.flow_id FROM flow_memberships fm "  # nosec B608
                f"JOIN nodes n ON n.id = fm.node_id "
                f"WHERE n.file_path IN ({placeholders})",
                path_batch,
            ).fetchall()
            affected_ids.extend(r[0] for r in affected_rows)
    preserved_entry_qns |= _flow_node_qns(store, list(dict.fromkeys(affected_ids)))
    # A captured entry point's flow may have lost every row in the changed
    # files (the function it called was renamed). It is re-traced below, so
    # the old row has to go or the flow would be stored twice.
    preserved = sorted(preserved_entry_qns)
    for i in range(0, len(preserved), _SQL_BATCH):
        qn_batch = preserved[i:i + _SQL_BATCH]
        placeholders = ",".join("?" * len(qn_batch))
        affected_rows = conn.execute(
            f"SELECT f.id FROM flows f "  # nosec B608
            f"JOIN nodes n ON n.id = f.entry_point_id "
            f"WHERE n.qualified_name IN ({placeholders})",
            qn_batch,
        ).fetchall()
        affected_ids.extend(r[0] for r in affected_rows)
    affected_ids = list(dict.fromkeys(affected_ids))

    # ------------------------------------------------------------------
    # 2. Collect old entry-point node IDs / QNs before deletion
    # ------------------------------------------------------------------
    entry_point_ids: set[int] = set()
    if affected_ids:
        for i in range(0, len(affected_ids), _SQL_BATCH):
            id_batch = affected_ids[i:i + _SQL_BATCH]
            ep_placeholders = ",".join("?" * len(id_batch))
            ep_rows = conn.execute(
                f"SELECT f.entry_point_id, n.qualified_name FROM flows f "  # nosec B608
                f"LEFT JOIN nodes n ON n.id = f.entry_point_id "
                f"WHERE f.id IN ({ep_placeholders})",
                id_batch,
            ).fetchall()
            for row in ep_rows:
                if row[0] is not None:
                    entry_point_ids.add(row[0])
                if row[1]:
                    preserved_entry_qns.add(row[1])

    # ------------------------------------------------------------------
    # 3. Delete affected flows and their memberships
    # ------------------------------------------------------------------
    # Wrap in an explicit transaction so a crash mid-loop cannot leave
    # orphaned flow_memberships rows pointing at deleted flows.  See #258.
    if affected_ids:
        if conn.in_transaction:
            logger.warning(
                "Rolling back uncommitted transaction before BEGIN IMMEDIATE"
            )
            conn.rollback()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for fid in affected_ids:
                conn.execute(
                    "DELETE FROM flow_memberships WHERE flow_id = ?", (fid,),
                )
                conn.execute("DELETE FROM flows WHERE id = ?", (fid,))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # 4. Re-detect entry points and filter to relevant ones
    # ------------------------------------------------------------------
    narrowed = (
        _entry_points_among(store, changed_file_set, entry_point_ids, preserved_entry_qns)
        if fast_engine_enabled() else None
    )
    if narrowed is not None:
        relevant_eps = narrowed
    else:
        entry_points = detect_entry_points(store)
        relevant_eps = [
            ep for ep in entry_points
            if (
                ep.file_path in changed_file_set
                or ep.id in entry_point_ids
                or ep.qualified_name in preserved_entry_qns
            )
        ]

    # ------------------------------------------------------------------
    # 5. BFS-trace each relevant entry point
    # ------------------------------------------------------------------
    new_flows: list[dict] = []
    native_flows = (
        _native_traced_flows(store, relevant_eps, max_depth)
        if relevant_eps and fast_engine_enabled() else None
    )
    if native_flows is not None:
        new_flows = native_flows
    elif relevant_eps:
        adj = (_lazy_flow_adjacency(store) if fast_engine_enabled() else None) or (
            store.load_flow_adjacency()
        )
        for ep in relevant_eps:
            flow = _trace_single_flow(adj, ep, max_depth)
            if flow is not None:
                new_flows.append(flow)

    # ------------------------------------------------------------------
    # 6. INSERT new flows without clearing unrelated ones
    # ------------------------------------------------------------------
    count = 0
    if new_flows:
        store._begin_immediate()
        try:
            membership_rows: list[tuple[int, int, int]] = []
            for flow in new_flows:
                node_ids = flow.get("path", [])
                path_json = json.dumps(node_ids)
                cursor = conn.execute(
                    """INSERT INTO flows
                       (name, entry_point_id, depth, node_count, file_count,
                        criticality, path_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        flow["name"],
                        flow["entry_point_id"],
                        flow["depth"],
                        flow["node_count"],
                        flow["file_count"],
                        flow["criticality"],
                        path_json,
                    ),
                )
                flow_id = cursor.lastrowid
                membership_rows.extend(
                    (flow_id, node_id, position)
                    for position, node_id in enumerate(node_ids)
                )
                count += 1
            if membership_rows:
                # The rows are integers only, so the path translation has nothing to do.
                sqlite3.Connection.executemany(
                    conn,
                    "INSERT OR IGNORE INTO flow_memberships (flow_id, node_id, position) "
                    "VALUES (?, ?, ?)",
                    membership_rows,
                )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return count


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_flows(
    store: GraphStore,
    sort_by: str = "criticality",
    limit: int = 50,
) -> list[dict]:
    """Retrieve stored flows from the database.

    Args:
        store: The graph store.
        sort_by: Column to sort by (``criticality``, ``depth``, ``node_count``).
        limit: Maximum number of flows to return.
    """
    allowed_sort = {"criticality", "depth", "node_count", "file_count", "name"}
    if sort_by not in allowed_sort:
        sort_by = "criticality"

    order = "DESC" if sort_by in ("criticality", "depth", "node_count", "file_count") else "ASC"

    # NOTE: get_flows reads from the flows table which is managed by
    # the flows module; _conn access is documented coupling.
    rows = store._conn.execute(
        f"SELECT * FROM flows ORDER BY {sort_by} {order} LIMIT ?",  # nosec B608
        (limit,),
    ).fetchall()

    results: list[dict] = []
    for row in rows:
        results.append({
            "id": row["id"],
            "name": _sanitize_name(row["name"]),
            "entry_point_id": row["entry_point_id"],
            "depth": row["depth"],
            "node_count": row["node_count"],
            "file_count": row["file_count"],
            "criticality": row["criticality"],
            "path": json.loads(row["path_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })
    return results


def get_flow_by_id(store: GraphStore, flow_id: int) -> Optional[dict]:
    """Retrieve a single flow with full path details.

    Returns a dict with the flow metadata plus a ``steps`` list containing
    each node's name, kind, file, and line info.
    """
    # NOTE: get_flow_by_id reads from the flows table; see store_flows note.
    row = store._conn.execute(
        "SELECT * FROM flows WHERE id = ?", (flow_id,)
    ).fetchone()
    if row is None:
        return None

    path_ids: list[int] = json.loads(row["path_json"])

    # Build detailed step info.
    steps: list[dict] = []
    for nid in path_ids:
        node = store.get_node_by_id(nid)
        if node:
            steps.append({
                "node_id": node.id,
                "name": _sanitize_name(node.name),
                "kind": node.kind,
                "file": node.file_path,
                "line_start": node.line_start,
                "line_end": node.line_end,
                "qualified_name": _sanitize_name(node.qualified_name),
            })

    return {
        "id": row["id"],
        "name": _sanitize_name(row["name"]),
        "entry_point_id": row["entry_point_id"],
        "depth": row["depth"],
        "node_count": row["node_count"],
        "file_count": row["file_count"],
        "criticality": row["criticality"],
        "path": path_ids,
        "steps": steps,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_affected_flows(
    store: GraphStore,
    changed_files: list[str],
) -> dict:
    """Find flows that include nodes from the given changed files.

    Returns::

        {
            "affected_flows": [<flow dicts>],
            "total": <int>,
        }
    """
    if not changed_files:
        return {"affected_flows": [], "total": 0}

    # Find node IDs belonging to changed files.
    node_ids = store.get_node_ids_by_files(changed_files)

    if not node_ids:
        return {"affected_flows": [], "total": 0}

    # Find flow IDs that contain any of these nodes.
    flow_ids = store.get_flow_ids_by_node_ids(node_ids)

    if not flow_ids:
        return {"affected_flows": [], "total": 0}

    affected: list[dict] = []
    for fid in flow_ids:
        flow = get_flow_by_id(store, fid)
        if flow:
            affected.append(flow)

    # Sort by criticality descending.
    affected.sort(key=lambda f: f.get("criticality", 0), reverse=True)

    return {
        "affected_flows": affected,
        "total": len(affected),
    }
