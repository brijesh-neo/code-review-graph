"""The parser passes the native engine runs (``CRG_FAST_ENGINE=1``).

``parser.py`` stays the legacy engine, kept close to upstream; a pass that has
a native version hands a tree from the ``crg_fast`` module here. Each function
returns exactly what its legacy pass returns: the tree walks run in Rust, and
anything that reads other files or the import resolver is the legacy helper,
called back at the same point of the walk.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .parser import CodeParser, EdgeInfo


def is_fast_node(node) -> bool:
    """Whether *node* belongs to a tree the native engine parsed."""
    return type(node).__module__ == "crg_fast"


def module_property_return_annotations(
    parser: CodeParser, root,
) -> dict[tuple[str, str], str]:
    """``CodeParser._read_python_property_return_annotations`` on a native tree.

    The declaration scan of the receiver pass finds the same properties: it
    skips only error-free statement and expression subtrees, which hold no
    class or function definition.
    """
    import crg_fast

    def ignore(_node) -> None:
        return None

    class_names: set[str] = set()
    ambiguous: set[str] = set()
    properties: dict[tuple[str, str], str] = {}
    crg_fast.python_scan_declarations(
        root,
        SimpleNamespace(plain_import=ignore, from_import=ignore),
        parser._MAX_AST_DEPTH,
        class_names, ambiguous, properties, {}, set(),
    )
    if ambiguous:
        properties = {
            key: annotation
            for key, annotation in properties.items()
            if key[0] not in ambiguous
        }
    return properties


def collect_python_receiver_evidence(
    parser: CodeParser, root, file_path: str, source: bytes,
) -> tuple[dict[tuple[int, str], tuple[str, str]], list[EdgeInfo]]:
    """``CodeParser._collect_python_receiver_evidence`` on a native tree."""
    import crg_fast

    from .parser import (
        _PY_EXPR_RECEIVER_KEY,
        EdgeInfo,
        _PyReceiverContext,
        _python_function_local_names,
        _python_function_scope_bindings,
        _python_unreachable_call_positions,
    )

    module_bindings: dict[str, set[str]] = {}
    imported_names: set[str] = set()
    class_names: set[str] = set()
    ambiguous_property_classes: set[str] = set()
    symbol_binding_candidates: dict[str, set[tuple[str, str]]] = {}
    local_returns: dict[str, str] = {}
    property_returns: dict[tuple[str, str], str] = {}
    ambiguous_returns: set[str] = set()

    def plain_import(node) -> None:
        parser._python_plain_import_bindings(
            node, file_path, module_bindings, imported_names,
        )

    def from_import(node) -> None:
        import_origins: dict[str, str] = {}
        import_names: dict[str, str] = {}
        parser._python_from_import_bindings(
            node, file_path, module_bindings, imported_names,
            import_origins, import_names,
        )
        for name, origin in import_origins.items():
            imported_name = import_names.get(name)
            if imported_name is not None:
                symbol_binding_candidates.setdefault(name, set()).add(
                    (origin, imported_name),
                )

    crg_fast.python_scan_declarations(
        root,
        SimpleNamespace(plain_import=plain_import, from_import=from_import),
        parser._MAX_AST_DEPTH,
        class_names, ambiguous_property_classes, property_returns,
        local_returns, ambiguous_returns,
    )
    symbol_origins = {
        name: next(iter(bindings))[0]
        for name, bindings in symbol_binding_candidates.items()
        if len(bindings) == 1
    }
    symbol_names = {
        name: next(iter(bindings))[1]
        for name, bindings in symbol_binding_candidates.items()
        if len(bindings) == 1
    }
    for name in ambiguous_returns:
        local_returns.pop(name, None)
    if ambiguous_property_classes:
        property_returns = {
            key: annotation
            for key, annotation in property_returns.items()
            if key[0] not in ambiguous_property_classes
        }

    context = _PyReceiverContext(
        file_path=file_path,
        class_names=frozenset(class_names),
        imported_names=frozenset(imported_names),
        module_bindings=module_bindings,
        symbol_origins=symbol_origins,
        symbol_names=symbol_names,
        local_returns=local_returns,
        property_returns=property_returns,
    )

    module_env: dict[str, tuple[str, str]] = {}
    for name in imported_names | class_names:
        module_env[name] = ("class", name)
    for name, files in module_bindings.items():
        module_env[name] = (
            ("module", next(iter(files))) if len(files) == 1
            else ("unknown", "")
        )

    evidence: dict[tuple[int, str], tuple[str, str]] = {}
    property_read_edges: list[EdgeInfo] = []
    unreachable_positions = _python_unreachable_call_positions(source)
    function_scopes = _python_function_scope_bindings(source)

    def bind_targets(
        left, value, type_node, env, class_fields, current_class,
        instance_receiver_name, class_receiver_name,
    ) -> None:
        for name, path in parser._python_binding_targets(left):
            bound = parser._python_binding_evidence(
                value, type_node, path, context,
                env, class_fields, current_class, instance_receiver_name,
                class_receiver_name,
            )
            if bound is not None:
                env[name] = bound

    def expression_class_evidence(
        expression, env, class_fields, current_class,
        instance_receiver_name, class_receiver_name,
    ):
        return parser._python_expression_class_evidence(
            expression, env, class_fields, current_class, context,
            instance_receiver_name, class_receiver_name,
        )

    def property_read(
        owner_class: str, property_name: str, owner,
        current_function: Optional[str], current_class: Optional[str],
        class_receiver_name: Optional[str], line: int,
    ) -> None:
        definition = parser._python_property_definition(
            owner_class, property_name, context,
        )
        if definition is None:
            return
        if (
            owner is not None
            and owner.type == "identifier"
            and class_receiver_name is not None
            and owner.text.decode("utf-8", errors="replace") == class_receiver_name
        ):
            return
        target_file, target_class, _ = definition
        caller = (
            parser._qualify(current_function, file_path, current_class)
            if current_function
            else file_path
        )
        property_read_edges.append(EdgeInfo(
            kind="CALLS",
            source=caller,
            target=parser._qualify(property_name, target_file, target_class),
            file_path=file_path,
            line=line,
            extra={"python_property_read": True},
        ))

    callbacks = SimpleNamespace(
        class_fields=lambda node: parser._collect_python_class_fields(node, context),
        typed_parameters=lambda node: parser._collect_function_typed_parameters(
            node, "python",
        ),
        local_names=_python_function_local_names,
        bind_targets=bind_targets,
        expression_class_evidence=expression_class_evidence,
        property_read=property_read,
        expression_receiver_evidence=lambda node: (
            parser._python_expression_receiver_evidence(node, context)
        ),
    )
    expression_evidence = crg_fast.python_receiver_walk(
        root, module_env, callbacks, unreachable_positions, function_scopes,
        parser._MAX_AST_DEPTH, _PY_EXPR_RECEIVER_KEY, evidence,
    )
    for key, found in expression_evidence.items():
        # Two expression receivers on one line sharing a method name are
        # indistinguishable downstream, so they must agree to count.
        evidence[key] = parser._merge_python_evidence(found)
    return evidence, property_read_edges


# The languages whose ``_collect_typed_call_targets`` walk runs natively, and
# the node types ``_typed_bindings_from_node`` reads for each; it returns no
# bindings for any other type.
TYPED_CALL_LANGUAGES = frozenset({"python", "javascript", "typescript", "tsx"})
_TYPED_BINDING_TYPES = {
    "python": frozenset({"assignment", "typed_parameter", "typed_default_parameter"}),
    "javascript": frozenset({
        "required_parameter", "optional_parameter", "variable_declarator",
        "public_field_definition",
    }),
}
_TYPED_BINDING_TYPES["typescript"] = _TYPED_BINDING_TYPES["javascript"]
_TYPED_BINDING_TYPES["tsx"] = _TYPED_BINDING_TYPES["javascript"]


def collect_typed_call_targets(
    parser: CodeParser,
    root,
    language: str,
    file_path: str,
    import_map: dict[str, str],
    defined_names: set[str],
) -> dict[tuple[int, str, str], tuple[Optional[str], str, str]]:
    """``CodeParser._collect_typed_call_targets`` on a native tree."""
    import crg_fast

    class_types = set(parser._class_types.get(language, []))
    function_types = set(parser._function_types.get(language, []))
    call_types = set(parser._call_types.get(language, []))
    block_types = set() if language == "python" else {"statement_block"}
    targets: dict[tuple[int, str, str], tuple[Optional[str], str, str]] = {}
    declared_type_parameters: list[frozenset[str]] = []
    unresolved_kept = language in parser._UNRESOLVED_TYPE_LANGUAGES

    def typed_call(
        line: int, receiver: str, method: str, type_name: str, evidence: str,
    ) -> None:
        target = parser._resolve_typed_method_target(
            type_name, method, file_path, language, import_map, defined_names,
        )
        key = (line, receiver, method)
        if target:
            targets[key] = (target, type_name, evidence)
        elif evidence == "typed_receiver" and unresolved_kept:
            base_type = parser._base_type_name(type_name)
            if not base_type or base_type == "var":
                return
            if not declared_type_parameters:
                declared_type_parameters.append(crg_fast.type_parameter_names(root))
            if base_type not in declared_type_parameters[0]:
                targets[key] = (None, base_type, "typed_receiver_unresolved")

    callbacks = SimpleNamespace(
        class_typed_fields=lambda node: parser._collect_class_typed_fields(
            node, language, function_types, class_types,
        ),
        typed_parameters=lambda node: parser._collect_function_typed_parameters(
            node, language,
        ),
        typed_bindings=lambda node: parser._typed_bindings_from_node(node, language),
        typed_call=typed_call,
    )
    crg_fast.typed_call_walk(
        root, callbacks, parser._MAX_AST_DEPTH, class_types, function_types,
        call_types, block_types, _TYPED_BINDING_TYPES[language], import_map,
        defined_names,
    )
    return targets


def extract_from_tree(
    parser: CodeParser,
    root,
    source: bytes,
    language: str,
    file_path: str,
    nodes: list,
    edges: list,
    enclosing_class: Optional[str],
    enclosing_func: Optional[str],
    import_map: Optional[dict[str, str]],
    defined_names: Optional[set[str]],
    depth: int,
    go_receiver_bindings,
) -> None:
    """``CodeParser._extract_from_tree`` on a native tree."""
    import crg_fast

    type_sets = parser._node_type_sets(language)
    plans = parser.__dict__.setdefault("_fast_extract_plans", {})
    cached = plans.get(language)
    if cached is None or cached[0] is not type_sets:
        cached = plans[language] = (
            type_sets,
            crg_fast.ExtractPlan(language, *type_sets, parser._VALUE_REFERENCE_TYPES),
        )
    crg_fast.extract_walk(
        parser, cached[1], root, source, language, file_path, nodes, edges,
        enclosing_class, enclosing_func, import_map, defined_names, depth,
        go_receiver_bindings, parser._MAX_AST_DEPTH,
    )
