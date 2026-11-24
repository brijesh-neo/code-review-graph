//! `PathCodec.row_factory` natively: every row a relative graph returns.
//!
//! Each row read from `graph.db` passes through the codec's row factory,
//! which gives stored values (`./b.py::f`) back their root. A build reads
//! several hundred thousand rows, so the factory's Python body is a cost of
//! its own. This one does the same with the same objects:
//!
//! * which columns are restored comes from the codec's own `_spec`, cached
//!   against the cursor description exactly as `_last_spec` is;
//! * `./x` becomes `root + x[1:]` here; JSON text with no `".` or `"!` is
//!   left as it is, as `restore_json` leaves it;
//! * every other value that may change goes to the codec's own `restore`.

use std::sync::{Arc, Mutex};

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

type Spec = Arc<Vec<usize>>;

/// `PathCodec.row_factory` for one codec.
#[pyclass(module = "crg_fast", frozen)]
pub struct RowRestorer {
    root: String,
    restore: Py<PyAny>,
    spec_of: Py<PyAny>,
    row_type: Py<PyAny>,
    last: Mutex<Option<(Py<PyAny>, Spec)>>,
}

fn is_json_first(c: char) -> bool {
    c == '{' || c == '[' || c == '"'
}

impl RowRestorer {
    fn spec(&self, py: Python<'_>, cursor: &Bound<'_, PyAny>) -> PyResult<Spec> {
        let desc = cursor.getattr(intern!(py, "description"))?;
        if !desc.is_none() {
            if let Some((last, spec)) = self.last.lock().unwrap().as_ref() {
                if last.is(&desc) {
                    return Ok(spec.clone());
                }
            }
        }
        let indexes: Vec<usize> = self.spec_of.bind(py).call1((cursor,))?.extract()?;
        let spec = Arc::new(indexes);
        *self.last.lock().unwrap() = Some((desc.unbind(), spec.clone()));
        Ok(spec)
    }

    /// The restored value, or None when *value* comes back unchanged.
    fn restored<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Ok(text) = value.cast::<PyString>() else { return Ok(None) };
        let Ok(s) = text.to_str() else {
            return self.restore.bind(py).call1((value,)).map(Some);
        };
        let Some(first) = s.chars().next() else { return Ok(None) };
        if let Some(rest) = s.strip_prefix("./") {
            let mut out = String::with_capacity(self.root.len() + 1 + rest.len());
            out.push_str(&self.root);
            out.push('/');
            out.push_str(rest);
            return Ok(Some(PyString::new(py, &out).into_any()));
        }
        let may_change = match first {
            '.' | '!' => true,
            c if is_json_first(c) => s.contains("\".") || s.contains("\"!"),
            _ => false,
        };
        if !may_change {
            return Ok(None);
        }
        self.restore.bind(py).call1((value,)).map(Some)
    }
}

#[pymethods]
impl RowRestorer {
    #[new]
    fn new(
        root: String,
        restore: Py<PyAny>,
        spec_of: Py<PyAny>,
        row_type: Py<PyAny>,
    ) -> Self {
        RowRestorer { root, restore, spec_of, row_type, last: Mutex::new(None) }
    }

    fn __call__<'py>(
        &self,
        py: Python<'py>,
        cursor: Bound<'py, PyAny>,
        row: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec = self.spec(py, &cursor)?;
        let mut values: Option<Vec<Bound<'py, PyAny>>> = None;
        for &index in spec.iter() {
            let value = row.get_item(index)?;
            if let Some(restored) = self.restored(py, &value)? {
                values.get_or_insert_with(|| row.iter().collect())[index] = restored;
            }
        }
        let row = match values {
            None => row,
            Some(values) => PyTuple::new(py, values)?,
        };
        self.row_type.bind(py).call1((cursor, row))
    }
}
