//! The tree walks of `CodeParser._collect_python_receiver_evidence`.
//!
//! The scope bookkeeping, the node dispatch and the pure-syntax checks run
//! here. Whatever reads other files or the import resolver (binding evidence,
//! class evidence, property definitions, import bindings) is still the Python
//! code, called back with the same arguments at the same point of the walk, so
//! the result is the one the Python walk builds.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};

use crate::{Node, TreeData};
use std::sync::Arc;

const STATEMENT_FREE_TYPES: &[&str] = &[
    "expression_statement",
    "return_statement",
    "raise_statement",
    "assert_statement",
    "delete_statement",
    "decorator",
    "parameters",
    "argument_list",
    "comment",
];

type TsNode<'a> = tree_sitter::Node<'a>;

pub(crate) struct Walk<'py, 'a> {
    pub(crate) py: Python<'py>,
    pub(crate) tree: &'a Arc<TreeData>,
    pub(crate) source: &'a [u8],
    pub(crate) callbacks: Bound<'py, PyAny>,
    pub(crate) max_depth: usize,
}

/// Python's `bytes.decode("utf-8", errors="replace")`.
fn decode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Python's `str.strip()`: `char::is_whitespace` plus the four separators
/// Python also counts as space.
fn py_strip(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\x1c'..='\x1f').contains(&c))
}

fn last_dotted(text: &str) -> &str {
    text.rsplit('.').next().unwrap_or(text)
}

impl<'py, 'a> Walk<'py, 'a> {
    pub(crate) fn text(&self, node: TsNode<'_>) -> String {
        let end = node.end_byte().min(self.source.len());
        let start = node.start_byte().min(end);
        decode(&self.source[start..end])
    }

    pub(crate) fn raw_text<'s>(&'s self, node: TsNode<'_>) -> &'s [u8] {
        let end = node.end_byte().min(self.source.len());
        let start = node.start_byte().min(end);
        &self.source[start..end]
    }

    pub(crate) fn wrap(&self, node: TsNode<'_>) -> PyResult<Bound<'py, Node>> {
        Bound::new(
            self.py,
            Node {
                raw: crate::RawNode(node.into_raw()),
                tree: Arc::clone(self.tree),
            },
        )
    }

    pub(crate) fn wrap_opt(&self, node: Option<TsNode<'_>>) -> PyResult<Bound<'py, PyAny>> {
        match node {
            Some(node) => Ok(self.wrap(node)?.into_any()),
            None => Ok(self.py.None().into_bound(self.py)),
        }
    }

    pub(crate) fn opt_str(&self, value: &Option<String>) -> Bound<'py, PyAny> {
        match value {
            Some(text) => PyString::new(self.py, text).into_any(),
            None => self.py.None().into_bound(self.py),
        }
    }

    pub(crate) fn children(node: TsNode<'a>) -> Vec<TsNode<'a>> {
        let mut cursor = node.walk();
        node.children(&mut cursor).collect()
    }

    /// `_python_decorator_names`.
    fn decorator_names(&self, node: TsNode<'_>) -> Vec<String> {
        let mut names = Vec::new();
        if let Some(parent) = node.parent() {
            if parent.kind() == "decorated_definition" {
                let mut cursor = parent.walk();
                for sibling in parent.children(&mut cursor) {
                    if sibling.kind() == "decorator" {
                        let text = self.text(sibling);
                        names.push(py_strip(text.trim_start_matches('@')).to_string());
                    }
                }
            }
        }
        names
    }

    fn first_parameter_name(&self, function: TsNode<'_>) -> Option<String> {
        let parameters = function.child_by_field_name("parameters")?;
        let mut cursor = parameters.walk();
        let first = parameters.named_children(&mut cursor).next()?;
        match first.kind() {
            "identifier" => Some(self.text(first)),
            "typed_parameter" | "typed_default_parameter" | "default_parameter" => {
                let mut cursor = first.walk();
                let name = first
                    .children(&mut cursor)
                    .find(|child| child.kind() == "identifier");
                name.map(|name| self.text(name))
            }
            _ => None,
        }
    }

    fn instance_receiver_parameter(&self, function: TsNode<'_>) -> Option<String> {
        if self
            .decorator_names(function)
            .iter()
            .any(|d| matches!(last_dotted(d), "classmethod" | "staticmethod"))
        {
            return None;
        }
        self.first_parameter_name(function)
    }

    fn class_receiver_parameter(&self, function: TsNode<'_>) -> Option<String> {
        if self
            .decorator_names(function)
            .iter()
            .any(|d| last_dotted(d) == "classmethod")
        {
            return self.first_parameter_name(function);
        }
        None
    }

    /// `_node_is_in_child`: byte ranges, as the Python helper compares them.
    fn node_is_in_child(node: TsNode<'_>, child: TsNode<'_>) -> bool {
        let (start, end) = (child.start_byte(), child.end_byte());
        let mut cursor = Some(node);
        while let Some(current) = cursor {
            if current.start_byte() == start && current.end_byte() == end {
                return true;
            }
            cursor = current.parent();
        }
        false
    }

    fn is_attribute_store_target(attribute: TsNode<'_>) -> bool {
        let mut parent = attribute.parent();
        while let Some(current) = parent {
            match current.kind() {
                "attribute" | "subscript" | "augmented_assignment" => return false,
                "assignment" | "for_statement" | "for_in_clause" => {
                    let target = current
                        .child_by_field_name("left")
                        .or_else(|| current.child_by_field_name("target"));
                    return match target {
                        Some(target) => {
                            target.start_byte() <= attribute.start_byte()
                                && attribute.end_byte() <= target.end_byte()
                        }
                        None => false,
                    };
                }
                "as_pattern" => {
                    return match current.child_by_field_name("alias") {
                        Some(alias) => Self::node_is_in_child(attribute, alias),
                        None => false,
                    };
                }
                "delete_statement" => return true,
                _ => {}
            }
            parent = current.parent();
        }
        false
    }

    /// The Python branch of `_get_member_call_receiver_method`.
    pub(crate) fn member_call_receiver_method(
        &self,
        node: TsNode<'_>,
    ) -> (Option<String>, Option<String>) {
        let callee = node.child_by_field_name("function").or_else(|| node.child(0));
        let Some(callee) = callee else {
            return (None, None);
        };
        if !matches!(callee.kind(), "attribute" | "member_expression") {
            return (None, None);
        }
        let object = callee.child_by_field_name("object");
        let property = callee
            .child_by_field_name("attribute")
            .or_else(|| callee.child_by_field_name("property"));
        let (Some(object), Some(property)) = (object, property) else {
            return (None, None);
        };
        let method = self.text(property);
        if matches!(object.kind(), "identifier" | "simple_identifier" | "self" | "this") {
            return (Some(self.text(object)), Some(method));
        }
        if matches!(object.kind(), "attribute" | "member_expression") {
            let root = object.child_by_field_name("object");
            let field = object
                .child_by_field_name("attribute")
                .or_else(|| object.child_by_field_name("property"));
            if let (Some(root), Some(field)) = (root, field) {
                if matches!(root.kind(), "identifier" | "self" | "this")
                    && matches!(self.raw_text(root), b"self" | b"this")
                {
                    return (Some(self.text(field)), Some(method));
                }
            }
        }
        (None, Some(method))
    }

    pub(crate) fn call(
        &self,
        name: &Bound<'py, PyString>,
        args: impl pyo3::call::PyCallArgs<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.callbacks.call_method1(name, args)
    }
}

struct Scope<'py> {
    env: Bound<'py, PyDict>,
    class_fields: Bound<'py, PyDict>,
    current_class: Option<String>,
    current_function: Option<String>,
    instance_receiver: Option<String>,
    class_receiver: Option<String>,
}

struct Outputs<'py> {
    evidence: Bound<'py, PyDict>,
    expression_evidence: Bound<'py, PyDict>,
    unreachable: Bound<'py, PyAny>,
    function_scopes: Bound<'py, PyAny>,
    expr_key: String,
}

pub(crate) fn truthy(value: &Option<String>) -> bool {
    value.as_deref().is_some_and(|text| !text.is_empty())
}

impl<'py, 'a> Walk<'py, 'a> {
    fn binding_args(&self, scope: &Scope<'py>) -> [Bound<'py, PyAny>; 5] {
        [
            scope.env.clone().into_any(),
            scope.class_fields.clone().into_any(),
            self.opt_str(&scope.current_class),
            self.opt_str(&scope.instance_receiver),
            self.opt_str(&scope.class_receiver),
        ]
    }

    fn bind(
        &self,
        left: Option<TsNode<'_>>,
        value: Option<TsNode<'_>>,
        type_node: Option<TsNode<'_>>,
        scope: &Scope<'py>,
    ) -> PyResult<()> {
        let [env, fields, class, inst, cls] = self.binding_args(scope);
        let args = PyTuple::new(
            self.py,
            [
                self.wrap_opt(left)?,
                self.wrap_opt(value)?,
                self.wrap_opt(type_node)?,
                env,
                fields,
                class,
                inst,
                cls,
            ],
        )?;
        self.call(intern!(self.py, "bind_targets"), args)?;
        Ok(())
    }

    fn class_evidence(
        &self,
        expression: Option<TsNode<'_>>,
        scope: &Scope<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [env, fields, class, inst, cls] = self.binding_args(scope);
        let args = PyTuple::new(
            self.py,
            [self.wrap_opt(expression)?, env, fields, class, inst, cls],
        )?;
        self.call(intern!(self.py, "expression_class_evidence"), args)
    }

    fn walk_children(
        &self,
        node: TsNode<'a>,
        scope: &Scope<'py>,
        depth: usize,
        skip_leaves: bool,
        out: &Outputs<'py>,
    ) -> PyResult<()> {
        for child in Self::children(node) {
            if skip_leaves && child.child_count() == 0 {
                continue;
            }
            self.walk(child, scope, depth + 1, out)?;
        }
        Ok(())
    }

    fn walk(
        &self,
        node: TsNode<'a>,
        scope: &Scope<'py>,
        depth: usize,
        out: &Outputs<'py>,
    ) -> PyResult<()> {
        if depth > self.max_depth {
            return Ok(());
        }
        let py = self.py;
        match node.kind() {
            "class_definition" => {
                let fields = self
                    .call(intern!(py, "class_fields"), (self.wrap(node)?,))?
                    .cast_into::<PyDict>()?;
                let nested_class = match node.child_by_field_name("name") {
                    Some(name) => Some(self.text(name)),
                    None => scope.current_class.clone(),
                };
                let class_env = scope.env.copy()?;
                class_env.update(fields.as_mapping())?;
                let inner = Scope {
                    env: class_env,
                    class_fields: fields,
                    current_class: nested_class,
                    current_function: scope.current_function.clone(),
                    instance_receiver: None,
                    class_receiver: scope.class_receiver.clone(),
                };
                return self.walk_children(node, &inner, depth, false, out);
            }
            "function_definition" => {
                let scoped = scope.env.copy()?;
                scoped.update(scope.class_fields.as_mapping())?;
                let start = node.start_position();
                let key = (start.row + 1, start.column);
                let default = (pyo3::types::PyFrozenSet::empty(py)?, py.None());
                let found = out
                    .function_scopes
                    .call_method1(intern!(py, "get"), (key, default))?;
                let parameter_names = found.get_item(0)?;
                let function_ast = found.get_item(1)?;
                for parameter in parameter_names.try_iter()? {
                    scoped.call_method1(intern!(py, "pop"), (parameter?, py.None()))?;
                }
                let typed = self
                    .call(intern!(py, "typed_parameters"), (self.wrap(node)?,))?
                    .cast_into::<PyDict>()?;
                let class_tag = intern!(py, "class");
                for (name, type_name) in typed.iter() {
                    scoped.set_item(name, (class_tag, type_name))?;
                }
                let function_name = match node.child_by_field_name("name") {
                    Some(name) => Some(self.text(name)),
                    None => scope.current_function.clone(),
                };
                let direct_method =
                    truthy(&scope.current_class) && scope.current_function.is_none();
                let local_names: Option<Bound<'py, PyAny>> = if !direct_method
                    && !function_ast.is_none()
                    && (scope.instance_receiver.is_some() || scope.class_receiver.is_some())
                {
                    Some(self.call(
                        intern!(py, "local_names"),
                        (function_ast, parameter_names),
                    )?)
                } else {
                    None
                };
                let hidden = |name: &Option<String>| -> PyResult<Option<String>> {
                    let Some(text) = name else { return Ok(None) };
                    if let Some(locals) = &local_names {
                        if locals.contains(text.as_str())? {
                            return Ok(None);
                        }
                    }
                    Ok(Some(text.clone()))
                };
                let (instance_receiver, class_receiver) =
                    if truthy(&scope.current_class) && scope.current_function.is_none() {
                        (
                            self.instance_receiver_parameter(node),
                            self.class_receiver_parameter(node),
                        )
                    } else {
                        (hidden(&scope.instance_receiver)?, hidden(&scope.class_receiver)?)
                    };
                let inner = Scope {
                    env: scoped,
                    class_fields: scope.class_fields.clone(),
                    current_class: scope.current_class.clone(),
                    current_function: function_name,
                    instance_receiver,
                    class_receiver,
                };
                return self.walk_children(node, &inner, depth, false, out);
            }
            "assignment" => {
                // The right-hand side is evaluated before the name is rebound.
                self.walk_children(node, scope, depth, false, out)?;
                return self.bind(
                    node.child_by_field_name("left"),
                    node.child_by_field_name("right"),
                    node.child_by_field_name("type"),
                    scope,
                );
            }
            "as_pattern" => {
                self.walk_children(node, scope, depth, false, out)?;
                let alias = node.child_by_field_name("alias");
                let value = node.child(0);
                // The Python walk's ``value is not alias`` compares two fresh
                // node objects, so it always holds.
                if let (Some(alias), Some(value)) = (alias, value) {
                    let target = if alias.kind() == "as_pattern_target" && alias.child_count() > 0
                    {
                        alias.child(0)
                    } else {
                        Some(alias)
                    };
                    self.bind(target, Some(value), None, scope)?;
                }
                return Ok(());
            }
            "attribute" => {
                let owner = node.child_by_field_name("object");
                let property = node.child_by_field_name("attribute");
                let owner_evidence = self.class_evidence(owner, scope)?;
                if !Self::is_attribute_store_target(node) {
                    let start = node.start_position();
                    let line = start.row + 1;
                    if !out.unreachable.contains((line, start.column))?
                        && !owner_evidence.is_none()
                        && owner_evidence.get_item(0)?.eq("class")?
                    {
                        if let Some(property) = property {
                            let args = PyTuple::new(
                                py,
                                [
                                    owner_evidence.get_item(1)?,
                                    PyString::new(py, &self.text(property)).into_any(),
                                    self.wrap_opt(owner)?,
                                    self.opt_str(&scope.current_function),
                                    self.opt_str(&scope.current_class),
                                    self.opt_str(&scope.class_receiver),
                                    line.into_pyobject(py)?.into_any(),
                                ],
                            )?;
                            self.call(intern!(py, "property_read"), args)?;
                        }
                    }
                }
            }
            "call" => {
                let (receiver, method) = self.member_call_receiver_method(node);
                let line = node.start_position().row + 1;
                if truthy(&receiver) && truthy(&method) {
                    let receiver = receiver.unwrap();
                    let found = match scope.env.get_item(receiver.as_str())? {
                        Some(found) => found,
                        None => (intern!(py, "unknown"), intern!(py, ""))
                            .into_pyobject(py)?
                            .into_any(),
                    };
                    out.evidence.set_item((line, receiver), found)?;
                } else if truthy(&method) {
                    let method = method.unwrap();
                    let mut constructed =
                        self.call(intern!(py, "expression_receiver_evidence"), (self.wrap(node)?,))?;
                    if constructed.is_none() {
                        let receiver_node = node
                            .child_by_field_name("function")
                            .filter(|callee| callee.kind() == "attribute")
                            .and_then(|callee| callee.child_by_field_name("object"));
                        constructed = self.class_evidence(receiver_node, scope)?;
                    }
                    if !constructed.is_none() {
                        let key = (line, format!("{}{}", out.expr_key, method));
                        let list = match out.expression_evidence.get_item(&key)? {
                            Some(list) => list.cast_into::<PyList>()?,
                            None => {
                                let list = PyList::empty(py);
                                out.expression_evidence.set_item(&key, &list)?;
                                list
                            }
                        };
                        list.append(constructed)?;
                    }
                }
            }
            _ => {}
        }
        // A leaf is no binding, attribute or call.
        self.walk_children(node, scope, depth, true, out)
    }
}

/// The receiver walk: fills ``evidence`` and returns the expression evidence
/// (key -> list of findings) for the Python side to merge.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn python_receiver_walk<'py>(
    py: Python<'py>,
    root: &Bound<'py, Node>,
    module_env: Bound<'py, PyDict>,
    callbacks: Bound<'py, PyAny>,
    unreachable: Bound<'py, PyAny>,
    function_scopes: Bound<'py, PyAny>,
    max_depth: usize,
    expr_key: String,
    evidence: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let node = root.get();
    let tree = &node.tree;
    let source = tree.source.bind(py).as_bytes();
    let walk = Walk { py, tree, source, callbacks, max_depth };
    let out = Outputs {
        evidence,
        expression_evidence: PyDict::new(py),
        unreachable,
        function_scopes,
        expr_key,
    };
    let scope = Scope {
        env: module_env,
        class_fields: PyDict::new(py),
        current_class: None,
        current_function: None,
        instance_receiver: None,
        class_receiver: None,
    };
    let ts_root = unsafe { tree_sitter::Node::from_raw(node.raw.0) };
    walk.walk(ts_root, &scope, 0, &out)?;
    Ok(out.expression_evidence)
}

struct Declarations<'py> {
    class_names: Bound<'py, PySet>,
    ambiguous_property_classes: Bound<'py, PySet>,
    property_returns: Bound<'py, PyDict>,
    local_returns: Bound<'py, PyDict>,
    ambiguous_returns: Bound<'py, PySet>,
}

impl<'py, 'a> Walk<'py, 'a> {
    fn scan(
        &self,
        node: TsNode<'a>,
        owner_class: Option<String>,
        in_function: bool,
        depth: usize,
        out: &Declarations<'py>,
    ) -> PyResult<()> {
        if depth > self.max_depth {
            return Ok(());
        }
        let py = self.py;
        let kind = node.kind();
        let mut owner_class = owner_class;
        let mut in_function = in_function;
        if kind == "class_definition" {
            if let Some(name) = node.child_by_field_name("name") {
                let name = self.text(name);
                if out.class_names.contains(name.as_str())? {
                    out.ambiguous_property_classes.add(name.as_str())?;
                }
                out.class_names.add(name.as_str())?;
                owner_class = Some(name);
            }
            in_function = false;
        } else if kind == "function_definition" && truthy(&owner_class) && !in_function {
            let name = node.child_by_field_name("name");
            let return_node = node.child_by_field_name("return_type");
            if let Some(name) = name {
                if self
                    .decorator_names(node)
                    .iter()
                    .any(|d| last_dotted(d) == "property")
                {
                    let annotation = return_node.map(|n| self.text(n)).unwrap_or_default();
                    out.property_returns.set_item(
                        (owner_class.as_deref().unwrap(), self.text(name)),
                        annotation,
                    )?;
                }
            }
        } else if kind == "function_definition" && owner_class.is_none() && !in_function {
            let name = node.child_by_field_name("name");
            let type_node = node.child_by_field_name("return_type");
            if let (Some(name), Some(type_node)) = (name, type_node) {
                let name = self.text(name);
                let annotation = self.text(type_node);
                if let Some(previous) = out.local_returns.get_item(name.as_str())? {
                    if !previous.eq(annotation.as_str())? {
                        // Two same-named defs disagreeing is no evidence.
                        out.ambiguous_returns.add(name.as_str())?;
                    }
                }
                out.local_returns.set_item(name, annotation)?;
            }
        } else if kind == "import_statement" {
            self.call(intern!(py, "plain_import"), (self.wrap(node)?,))?;
        } else if kind == "import_from_statement" {
            self.call(intern!(py, "from_import"), (self.wrap(node)?,))?;
        } else if STATEMENT_FREE_TYPES.contains(&kind) && !node.has_error() {
            return Ok(());
        }
        let child_in_function = in_function || kind == "function_definition";
        for child in Self::children(node) {
            if child.child_count() == 0 {
                continue; // a leaf declares and imports nothing
            }
            self.scan(child, owner_class.clone(), child_in_function, depth + 1, out)?;
        }
        Ok(())
    }
}

/// `scan_declarations`: fills the given containers in the Python walk's order.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn python_scan_declarations<'py>(
    py: Python<'py>,
    root: &Bound<'py, Node>,
    callbacks: Bound<'py, PyAny>,
    max_depth: usize,
    class_names: Bound<'py, PySet>,
    ambiguous_property_classes: Bound<'py, PySet>,
    property_returns: Bound<'py, PyDict>,
    local_returns: Bound<'py, PyDict>,
    ambiguous_returns: Bound<'py, PySet>,
) -> PyResult<()> {
    let node = root.get();
    let tree = &node.tree;
    let source = tree.source.bind(py).as_bytes();
    let walk = Walk { py, tree, source, callbacks, max_depth };
    let out = Declarations {
        class_names,
        ambiguous_property_classes,
        property_returns,
        local_returns,
        ambiguous_returns,
    };
    let ts_root = unsafe { tree_sitter::Node::from_raw(node.raw.0) };
    walk.scan(ts_root, None, false, 0, &out)
}
