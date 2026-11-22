//! The dispatch loop of `CodeParser._extract_from_tree` for Python,
//! JavaScript and TypeScript.
//!
//! The recursion and the node-type tests run here. Each extractor
//! (`_extract_classes`, `_extract_functions`, `_extract_calls`, ...) is the
//! legacy method, called on the same nodes, in the same order, with the same
//! arguments as the Python loop calls it.

use std::collections::HashSet;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use crate::Node;
use crate::python_receivers::Walk;

type TsNode<'a> = tree_sitter::Node<'a>;

/// The node-type tables of one language, read once per parser.
#[pyclass(frozen, module = "crg_fast")]
pub struct ExtractPlan {
    class_types: HashSet<String>,
    function_types: HashSet<String>,
    import_types: HashSet<String>,
    call_types: HashSet<String>,
    value_reference_types: HashSet<String>,
    javascript: bool,
    typescript: bool,
}

fn names(set: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    set.try_iter()?.map(|item| item?.extract::<String>()).collect()
}

#[pymethods]
impl ExtractPlan {
    #[new]
    fn new(
        language: &str,
        class_types: Bound<'_, PyAny>,
        function_types: Bound<'_, PyAny>,
        import_types: Bound<'_, PyAny>,
        call_types: Bound<'_, PyAny>,
        value_reference_types: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(ExtractPlan {
            class_types: names(&class_types)?,
            function_types: names(&function_types)?,
            import_types: names(&import_types)?,
            call_types: names(&call_types)?,
            value_reference_types: names(&value_reference_types)?,
            javascript: matches!(language, "javascript" | "typescript" | "tsx"),
            typescript: matches!(language, "typescript" | "tsx"),
        })
    }
}

/// The arguments every extractor shares, as Python objects.
struct Context<'py> {
    source: Bound<'py, PyAny>,
    language: Bound<'py, PyAny>,
    file_path: Bound<'py, PyAny>,
    nodes: Bound<'py, PyAny>,
    edges: Bound<'py, PyAny>,
    enclosing_class: Bound<'py, PyAny>,
    enclosing_func: Bound<'py, PyAny>,
    import_map: Bound<'py, PyAny>,
    defined_names: Bound<'py, PyAny>,
    go_receiver_bindings: Bound<'py, PyAny>,
}

impl<'py, 'a> Walk<'py, 'a> {
    fn method(
        &self,
        name: &Bound<'py, PyString>,
        args: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.callbacks.call_method1(name, PyTuple::new(self.py, args)?)
    }

    fn extract_walk(
        &self,
        root: TsNode<'a>,
        depth: usize,
        plan: &ExtractPlan,
        cx: &Context<'py>,
    ) -> PyResult<()> {
        if depth > self.max_depth {
            return Ok(());
        }
        let py = self.py;
        let depth_obj = depth.into_pyobject(py)?.into_any();
        for child in Self::children(root) {
            let kind = child.kind();
            let interesting = plan.class_types.contains(kind)
                || plan.function_types.contains(kind)
                || plan.import_types.contains(kind)
                || plan.call_types.contains(kind)
                || plan.value_reference_types.contains(kind)
                || (plan.javascript
                    && matches!(
                        kind,
                        "call_expression"
                            | "lexical_declaration"
                            | "variable_declaration"
                            | "public_field_definition"
                            | "expression_statement"
                            | "jsx_opening_element"
                            | "jsx_self_closing_element"
                    ))
                || (plan.typescript && kind == "type_identifier");
            // One Python node per child, as the legacy loop's ``child``; only
            // built for the nodes an extractor is called on.
            let wrapped = if interesting { Some(self.wrap(child)?.into_any()) } else { None };
            let node = || -> PyResult<Bound<'py, PyAny>> {
                Ok(wrapped.clone().expect("built for every extractor's node type"))
            };
            // name, source, language, file_path, nodes, edges, class, func,
            // import_map, defined_names, depth
            let scoped = |node: Bound<'py, PyAny>| {
                vec![
                    node,
                    cx.source.clone(),
                    cx.language.clone(),
                    cx.file_path.clone(),
                    cx.nodes.clone(),
                    cx.edges.clone(),
                    cx.enclosing_class.clone(),
                    cx.enclosing_func.clone(),
                    cx.import_map.clone(),
                    cx.defined_names.clone(),
                    depth_obj.clone(),
                ]
            };
            if plan.javascript && kind == "call_expression" {
                self.method(
                    intern!(py, "_extract_js_module_call"),
                    vec![node()?, cx.file_path.clone(), cx.language.clone(), cx.edges.clone()],
                )?;
            }
            if plan.javascript
                && matches!(kind, "lexical_declaration" | "variable_declaration")
                && self
                    .method(intern!(py, "_extract_js_var_functions"), scoped(node()?))?
                    .is_truthy()?
            {
                continue;
            }
            if plan.class_types.contains(kind)
                && self
                    .method(
                        intern!(py, "_extract_classes"),
                        vec![
                            node()?,
                            cx.source.clone(),
                            cx.language.clone(),
                            cx.file_path.clone(),
                            cx.nodes.clone(),
                            cx.edges.clone(),
                            cx.enclosing_class.clone(),
                            cx.import_map.clone(),
                            cx.defined_names.clone(),
                            depth_obj.clone(),
                        ],
                    )?
                    .is_truthy()?
            {
                continue;
            }
            if plan.javascript
                && kind == "public_field_definition"
                && self
                    .method(intern!(py, "_extract_js_field_function"), scoped(node()?))?
                    .is_truthy()?
            {
                continue;
            }
            if plan.javascript
                && kind == "expression_statement"
                && self
                    .method(intern!(py, "_extract_js_member_functions"), scoped(node()?))?
                    .is_truthy()?
            {
                continue;
            }
            if plan.function_types.contains(kind)
                && self
                    .method(
                        intern!(py, "_extract_functions"),
                        vec![
                            node()?,
                            cx.source.clone(),
                            cx.language.clone(),
                            cx.file_path.clone(),
                            cx.nodes.clone(),
                            cx.edges.clone(),
                            cx.enclosing_class.clone(),
                            cx.import_map.clone(),
                            cx.defined_names.clone(),
                            depth_obj.clone(),
                            cx.enclosing_func.clone(),
                        ],
                    )?
                    .is_truthy()?
            {
                continue;
            }
            if plan.import_types.contains(kind)
                && self
                    .method(
                        intern!(py, "_extract_imports"),
                        vec![
                            node()?,
                            cx.language.clone(),
                            cx.source.clone(),
                            cx.file_path.clone(),
                            cx.edges.clone(),
                        ],
                    )?
                    .is_truthy()?
            {
                continue;
            }
            if plan.call_types.contains(kind) {
                let mut args = scoped(node()?);
                args.push(cx.go_receiver_bindings.clone());
                if self.method(intern!(py, "_extract_calls"), args)?.is_truthy()? {
                    continue;
                }
            }
            let referenced = |node: Bound<'py, PyAny>| {
                vec![
                    node,
                    cx.language.clone(),
                    cx.file_path.clone(),
                    cx.edges.clone(),
                    cx.enclosing_class.clone(),
                    cx.enclosing_func.clone(),
                    cx.import_map.clone(),
                    cx.defined_names.clone(),
                ]
            };
            if plan.javascript
                && matches!(kind, "jsx_opening_element" | "jsx_self_closing_element")
            {
                self.method(intern!(py, "_extract_jsx_component_call"), referenced(node()?))?;
            }
            if plan.typescript && kind == "type_identifier" {
                self.method(intern!(py, "_extract_ts_type_reference"), referenced(node()?))?;
            }
            // ``_extract_value_references`` returns at once for any other type.
            if plan.value_reference_types.contains(kind) {
                self.method(
                    intern!(py, "_extract_value_references"),
                    vec![
                        node()?,
                        PyString::new(py, kind).into_any(),
                        cx.source.clone(),
                        cx.language.clone(),
                        cx.file_path.clone(),
                        cx.edges.clone(),
                        cx.enclosing_class.clone(),
                        cx.enclosing_func.clone(),
                        cx.import_map.clone(),
                        cx.defined_names.clone(),
                    ],
                )?;
            }
            if child.child_count() > 0 {
                self.extract_walk(child, depth + 1, plan, cx)?;
            }
        }
        Ok(())
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn extract_walk<'py>(
    py: Python<'py>,
    parser: Bound<'py, PyAny>,
    plan: &Bound<'py, ExtractPlan>,
    root: &Bound<'py, Node>,
    source: Bound<'py, PyAny>,
    language: Bound<'py, PyAny>,
    file_path: Bound<'py, PyAny>,
    nodes: Bound<'py, PyAny>,
    edges: Bound<'py, PyAny>,
    enclosing_class: Bound<'py, PyAny>,
    enclosing_func: Bound<'py, PyAny>,
    import_map: Bound<'py, PyAny>,
    defined_names: Bound<'py, PyAny>,
    depth: usize,
    go_receiver_bindings: Bound<'py, PyAny>,
    max_depth: usize,
) -> PyResult<()> {
    let node = root.get();
    let tree = &node.tree;
    let bytes = tree.source.bind(py).as_bytes();
    let walk = Walk { py, tree, source: bytes, callbacks: parser, max_depth };
    let cx = Context {
        source,
        language,
        file_path,
        nodes,
        edges,
        enclosing_class,
        enclosing_func,
        import_map,
        defined_names,
        go_receiver_bindings,
    };
    let ts_root = unsafe { tree_sitter::Node::from_raw(node.raw.0) };
    walk.extract_walk(ts_root, depth, plan.get(), &cx)
}
