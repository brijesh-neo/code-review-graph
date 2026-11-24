//! The tree walk of `CodeParser._collect_typed_call_targets` for Python,
//! JavaScript and TypeScript.
//!
//! Scopes, binding order and call dispatch run here; the typed-field,
//! typed-parameter and typed-binding readers and the target resolution are the
//! legacy helpers, called back where the Python walk calls them.

use std::collections::HashSet;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PySet};

use crate::Node;
use crate::python_receivers::{truthy, Walk};

type TsNode<'a> = tree_sitter::Node<'a>;

struct Kinds {
    class_types: HashSet<String>,
    function_types: HashSet<String>,
    call_types: HashSet<String>,
    block_types: HashSet<String>,
    binding_types: HashSet<String>,
}

struct Lookups<'py> {
    import_map: Bound<'py, PyAny>,
    defined_names: Bound<'py, PyAny>,
}

/// Python's `str.isupper()` on the first character of *text*.
fn starts_upper(text: &str) -> bool {
    text.chars().next().is_some_and(|c| c.is_uppercase())
}

impl<'py, 'a> Walk<'py, 'a> {
    fn typed_walk(
        &self,
        node: TsNode<'a>,
        bindings: &Bound<'py, PyDict>,
        class_fields: &Bound<'py, PyDict>,
        depth: usize,
        kinds: &Kinds,
        lookups: &Lookups<'py>,
    ) -> PyResult<()> {
        if depth > self.max_depth {
            return Ok(());
        }
        let py = self.py;
        let kind = node.kind();
        if kinds.class_types.contains(kind) {
            let fields = self
                .call(intern!(py, "class_typed_fields"), (self.wrap(node)?,))?
                .cast_into::<PyDict>()?;
            let class_bindings = bindings.copy()?;
            class_bindings.update(fields.as_mapping())?;
            for child in Self::children(node) {
                self.typed_walk(child, &class_bindings, &fields, depth + 1, kinds, lookups)?;
            }
            return Ok(());
        }
        if kinds.function_types.contains(kind) {
            let scoped = bindings.copy()?;
            scoped.update(class_fields.as_mapping())?;
            let typed = self
                .call(intern!(py, "typed_parameters"), (self.wrap(node)?,))?
                .cast_into::<PyDict>()?;
            scoped.update(typed.as_mapping())?;
            for child in Self::children(node) {
                self.typed_walk(child, &scoped, class_fields, depth + 1, kinds, lookups)?;
            }
            return Ok(());
        }
        if kinds.block_types.contains(kind) {
            let scoped = bindings.copy()?;
            for child in Self::children(node) {
                self.typed_walk(child, &scoped, class_fields, depth + 1, kinds, lookups)?;
            }
            return Ok(());
        }
        // ``_typed_bindings_from_node`` returns nothing for any other type.
        if kinds.binding_types.contains(kind) {
            let new_bindings = self
                .call(intern!(py, "typed_bindings"), (self.wrap(node)?,))?
                .cast_into::<PyDict>()?;
            if !new_bindings.is_empty() {
                // Initializers are evaluated before their declaration becomes
                // visible, so visit children with the previous environment.
                for child in Self::children(node) {
                    self.typed_walk(child, bindings, class_fields, depth + 1, kinds, lookups)?;
                }
                bindings.update(new_bindings.as_mapping())?;
                return Ok(());
            }
        }
        if kinds.call_types.contains(kind) {
            let (receiver, method) = self.member_call_receiver_method(node);
            if truthy(&receiver) && truthy(&method) {
                let receiver = receiver.unwrap();
                let method = method.unwrap();
                let mut type_name = bindings.get_item(receiver.as_str())?;
                let mut evidence = "typed_receiver";
                if type_name.is_none()
                    && starts_upper(&receiver)
                    && (lookups.import_map.contains(receiver.as_str())?
                        || lookups.defined_names.contains(receiver.as_str())?)
                {
                    type_name = Some(pyo3::types::PyString::new(py, &receiver).into_any());
                    evidence = "class_receiver";
                }
                if let Some(type_name) = type_name {
                    if type_name.is_truthy()? {
                        let line = node.start_position().row + 1;
                        self.call(
                            intern!(py, "typed_call"),
                            (line, receiver, method, type_name, evidence),
                        )?;
                    }
                }
            }
        }
        // A leaf (a name, a keyword, punctuation) binds and calls nothing.
        for child in Self::children(node) {
            if child.child_count() > 0 {
                self.typed_walk(child, bindings, class_fields, depth + 1, kinds, lookups)?;
            }
        }
        Ok(())
    }
}

fn names(set: &Bound<'_, PyAny>) -> PyResult<HashSet<String>> {
    set.try_iter()?.map(|item| item?.extract::<String>()).collect()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn typed_call_walk<'py>(
    py: Python<'py>,
    root: &Bound<'py, Node>,
    callbacks: Bound<'py, PyAny>,
    max_depth: usize,
    class_types: Bound<'py, PyAny>,
    function_types: Bound<'py, PyAny>,
    call_types: Bound<'py, PyAny>,
    block_types: Bound<'py, PyAny>,
    binding_types: Bound<'py, PyAny>,
    import_map: Bound<'py, PyAny>,
    defined_names: Bound<'py, PyAny>,
) -> PyResult<()> {
    let node = root.get();
    let tree = &node.tree;
    let source = tree.source.bind(py).as_bytes();
    let walk = Walk { py, tree, source, callbacks, max_depth };
    let kinds = Kinds {
        class_types: names(&class_types)?,
        function_types: names(&function_types)?,
        call_types: names(&call_types)?,
        block_types: names(&block_types)?,
        binding_types: names(&binding_types)?,
    };
    let lookups = Lookups { import_map, defined_names };
    let ts_root = unsafe { tree_sitter::Node::from_raw(node.raw.0) };
    walk.typed_walk(ts_root, &PyDict::new(py), &PyDict::new(py), 0, &kinds, &lookups)
}

/// The `type_parameters()` closure: every generic parameter name in the tree.
#[pyfunction]
pub fn type_parameter_names<'py>(
    py: Python<'py>,
    root: &Bound<'py, Node>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    let node = root.get();
    let source = node.tree.source.bind(py).as_bytes();
    let names = PySet::empty(py)?;
    let mut stack = vec![unsafe { tree_sitter::Node::from_raw(node.raw.0) }];
    while let Some(current) = stack.pop() {
        if current.kind() == "type_parameter" {
            let mut cursor = current.walk();
            let name = current
                .named_children(&mut cursor)
                .find(|child| matches!(child.kind(), "type_identifier" | "identifier"));
            if let Some(name) = name {
                let end = name.end_byte().min(source.len());
                let start = name.start_byte().min(end);
                names.add(String::from_utf8_lossy(&source[start..end]).as_ref())?;
            }
            continue;
        }
        let mut cursor = current.walk();
        stack.extend(current.children(&mut cursor));
    }
    PyFrozenSet::new(py, names.iter())
}
