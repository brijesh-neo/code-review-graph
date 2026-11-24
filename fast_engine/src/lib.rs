//! Native engine for code-review-graph (`CRG_FAST_ENGINE=1`).
//!
//! Stage one: a tree-sitter `Parser`, `Tree` and `Node` with the interface the
//! Python parser reads from py-tree-sitter, so every Python pass runs unchanged
//! on a tree parsed here. Passes then move into Rust one at a time and read the
//! same tree, so node ids agree between the ported and the unported passes.
//!
//! The grammar is the one tree-sitter-language-pack ships: its capsule holds
//! the `TSLanguage` pointer, so both engines parse with identical grammars.

use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule, PyList, PyString, PyTuple};
use tree_sitter::ffi;

mod python_receivers;
mod typed_calls;
mod extract;
mod store;
mod sqlite_ext;
mod certainty;
mod restore;
mod identities;
mod node_rows;

const CAPSULE_NAME: &CStr = c"tree_sitter.Language";

/// Interned node type names for one grammar, indexed by symbol id.
struct Kinds {
    names: Vec<Py<PyString>>,
}

struct TreeData {
    tree: tree_sitter::Tree,
    source: Py<PyBytes>,
    kinds: Arc<Kinds>,
}

/// A `TSNode` is plain data pointing into its tree; the `Arc<TreeData>` held
/// next to it keeps that tree alive, and nothing here mutates it.
#[derive(Clone, Copy)]
struct RawNode(ffi::TSNode);
unsafe impl Send for RawNode {}
unsafe impl Sync for RawNode {}

#[pyclass(frozen, module = "crg_fast")]
struct Parser {
    parser: Mutex<tree_sitter::Parser>,
    kinds: Arc<Kinds>,
}

#[pymethods]
impl Parser {
    #[new]
    fn new(py: Python<'_>, capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let ptr = capsule.pointer_checked(Some(CAPSULE_NAME))?;
        let language = unsafe { tree_sitter::Language::from_raw(ptr.as_ptr().cast()) };
        let count = language.node_kind_count();
        let names = (0..count)
            .map(|id| {
                let name = language.node_kind_for_id(id as u16).unwrap_or("");
                PyString::intern(py, name).unbind()
            })
            .collect();
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&language)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Parser {
            parser: Mutex::new(parser),
            kinds: Arc::new(Kinds { names }),
        })
    }

    fn parse(&self, source: Bound<'_, PyBytes>) -> PyResult<Tree> {
        let tree = {
            let mut parser = self.parser.lock().unwrap();
            parser.parse(source.as_bytes(), None)
        };
        let tree = tree.ok_or_else(|| PyValueError::new_err("parsing failed"))?;
        Ok(Tree {
            data: Arc::new(TreeData {
                tree,
                source: source.unbind(),
                kinds: Arc::clone(&self.kinds),
            }),
        })
    }
}

#[pyclass(frozen, module = "crg_fast")]
struct Tree {
    data: Arc<TreeData>,
}

#[pymethods]
impl Tree {
    #[getter]
    fn root_node(&self) -> Node {
        Node {
            raw: RawNode(self.data.tree.root_node().into_raw()),
            tree: Arc::clone(&self.data),
        }
    }
}

#[pyclass(frozen, module = "crg_fast")]
struct Node {
    raw: RawNode,
    tree: Arc<TreeData>,
}

impl Node {
    #[inline]
    fn ts(&self) -> tree_sitter::Node<'_> {
        unsafe { tree_sitter::Node::from_raw(self.raw.0) }
    }

    #[inline]
    fn wrap(&self, node: tree_sitter::Node<'_>) -> Node {
        Node {
            raw: RawNode(node.into_raw()),
            tree: Arc::clone(&self.tree),
        }
    }

    fn wrap_opt(&self, node: Option<tree_sitter::Node<'_>>) -> Option<Node> {
        node.map(|n| self.wrap(n))
    }

    fn collect_children<'py>(&self, py: Python<'py>, named_only: bool) -> PyResult<Bound<'py, PyList>> {
        let node = self.ts();
        let list = PyList::empty(py);
        let mut cursor = node.walk();
        if cursor.goto_first_child() {
            loop {
                let child = cursor.node();
                if !named_only || child.is_named() {
                    list.append(self.wrap(child))?;
                }
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
        }
        Ok(list)
    }
}

#[pymethods]
impl Node {
    #[getter(r#type)]
    fn kind(&self, py: Python<'_>) -> Py<PyString> {
        let id = self.ts().kind_id() as usize;
        match self.tree.kinds.names.get(id) {
            Some(name) => name.clone_ref(py),
            None => PyString::new(py, self.ts().kind()).unbind(),
        }
    }

    #[getter]
    fn children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.collect_children(py, false)
    }

    #[getter]
    fn named_children<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.collect_children(py, true)
    }

    #[getter]
    fn child_count(&self) -> u32 {
        self.ts().child_count() as u32
    }

    #[getter]
    fn named_child_count(&self) -> u32 {
        self.ts().named_child_count() as u32
    }

    fn child(&self, index: i64) -> PyResult<Option<Node>> {
        let node = self.ts();
        if index < 0 {
            return Err(PyIndexError::new_err("child index out of range"));
        }
        Ok(self.wrap_opt(node.child(index as u32)))
    }

    fn child_by_field_name(&self, name: &str) -> Option<Node> {
        self.wrap_opt(self.ts().child_by_field_name(name))
    }

    fn children_by_field_name<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyList>> {
        let node = self.ts();
        let list = PyList::empty(py);
        let language = node.language();
        if let Some(field) = language.field_id_for_name(name) {
            let mut cursor = node.walk();
            for child in node.children_by_field_id(field, &mut cursor) {
                list.append(self.wrap(child))?;
            }
        }
        Ok(list)
    }

    fn descendant_for_point_range(
        &self,
        start: (usize, usize),
        end: (usize, usize),
    ) -> Option<Node> {
        let start = tree_sitter::Point::new(start.0, start.1);
        let end = tree_sitter::Point::new(end.0, end.1);
        self.wrap_opt(self.ts().descendant_for_point_range(start, end))
    }

    #[getter]
    fn text<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let source = self.tree.source.bind(py).as_bytes();
        let node = self.ts();
        let end = node.end_byte().min(source.len());
        let start = node.start_byte().min(end);
        PyBytes::new(py, &source[start..end])
    }

    #[getter]
    fn start_byte(&self) -> usize {
        self.ts().start_byte()
    }

    #[getter]
    fn end_byte(&self) -> usize {
        self.ts().end_byte()
    }

    #[getter]
    fn start_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let point = self.ts().start_position();
        PyTuple::new(py, [point.row, point.column])
    }

    #[getter]
    fn end_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let point = self.ts().end_position();
        PyTuple::new(py, [point.row, point.column])
    }

    #[getter]
    fn parent(&self) -> Option<Node> {
        self.wrap_opt(self.ts().parent())
    }

    #[getter]
    fn prev_sibling(&self) -> Option<Node> {
        self.wrap_opt(self.ts().prev_sibling())
    }

    #[getter]
    fn next_sibling(&self) -> Option<Node> {
        self.wrap_opt(self.ts().next_sibling())
    }

    #[getter]
    fn prev_named_sibling(&self) -> Option<Node> {
        self.wrap_opt(self.ts().prev_named_sibling())
    }

    #[getter]
    fn next_named_sibling(&self) -> Option<Node> {
        self.wrap_opt(self.ts().next_named_sibling())
    }

    #[getter]
    fn is_named(&self) -> bool {
        self.ts().is_named()
    }

    #[getter]
    fn is_missing(&self) -> bool {
        self.ts().is_missing()
    }

    #[getter]
    fn is_error(&self) -> bool {
        self.ts().is_error()
    }

    #[getter]
    fn has_error(&self) -> bool {
        self.ts().has_error()
    }

    #[getter]
    fn id(&self) -> usize {
        self.ts().id()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.cast::<Node>() {
            Ok(other) => self.ts() == other.get().ts(),
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        self.ts().id() as u64
    }

    fn __repr__(&self) -> String {
        let node = self.ts();
        let start = node.start_position();
        let end = node.end_position();
        format!(
            "<Node type={}, start_point=({}, {}), end_point=({}, {})>",
            node.kind(), start.row, start.column, end.row, end.column,
        )
    }
}

#[pymodule]
fn crg_fast(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Parser>()?;
    module.add_class::<Tree>()?;
    module.add_class::<Node>()?;
    module.add_function(wrap_pyfunction!(python_receivers::python_receiver_walk, module)?)?;
    module.add_function(wrap_pyfunction!(python_receivers::python_scan_declarations, module)?)?;
    module.add_function(wrap_pyfunction!(typed_calls::typed_call_walk, module)?)?;
    module.add_function(wrap_pyfunction!(extract::extract_walk, module)?)?;
    module.add_class::<extract::ExtractPlan>()?;
    module.add_function(wrap_pyfunction!(typed_calls::type_parameter_names, module)?)?;
    module.add_function(wrap_pyfunction!(store::edge_rows, module)?)?;
    module.add_class::<store::PathStripper>()?;
    module.add_class::<restore::RowRestorer>()?;
    module.add_function(wrap_pyfunction!(identities::settle_edges, module)?)?;
    module.add_function(wrap_pyfunction!(node_rows::node_rows, module)?)?;
    module.add_function(wrap_pyfunction!(sqlite_ext::sqlite_extension_loaded, module)?)?;
    module.add_function(wrap_pyfunction!(sqlite_ext::discard_job, module)?)?;
    module.add_function(wrap_pyfunction!(certainty::stage_certainty_refresh, module)?)?;
    Ok(())
}
