//! Rows for the graph store, built and path-translated natively.
//!
//! `GraphStore._insert_edges` turns a file's parsed edges into `edges` rows and
//! `PathCodec.strip_params` turns every string into its stored form. Both run on
//! the main thread for every edge of a build, while the parse workers wait for
//! it. This module does the same work with the same Python objects:
//!
//! * the call-site collapse goes through a Python `dict` keyed by a Python
//!   tuple, so hashing and equality (`1 == 1.0`, `None`) are Python's own;
//! * `float()`, `str()` and `json.dumps` are the Python callables themselves;
//! * a stored form is worked out here only for plain path and name strings.
//!   JSON text, and any string Rust cannot read as UTF-8, goes back to the
//!   codec's own Python method.

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};

const REL_PREFIX: &str = "./";
const ROOT_TOKEN: &str = ".";
const ESCAPE: char = '!';
/// Cached stored forms before the cache starts over (`_STRIP_CACHE_LIMIT`).
const CACHE_LIMIT: usize = 200_000;
/// JSON text up to this many characters is cached too (`_CACHED_JSON_MAX`).
const CACHED_JSON_MAX: usize = 256;

fn is_json_first(c: char) -> bool {
    c == '{' || c == '[' || c == '"'
}

/// `_looks_relative`: would *value*, read back, be taken as in-repo?
fn looks_relative(value: &str) -> bool {
    let body = value.trim_start_matches(ESCAPE);
    body == ROOT_TOKEN || body.starts_with(REL_PREFIX) || body.starts_with("../")
}

/// `PathCodec.strip` for one codec, with the stored-form cache of
/// `strip_params`.
#[pyclass(module = "crg_fast")]
pub struct PathStripper {
    prefixes: Vec<String>,
    roots: Vec<String>,
    leads: Vec<char>,
    /// The codec's own `strip`, for JSON text and unreadable strings.
    strip: Py<PyAny>,
    cache: Py<PyDict>,
}

impl PathStripper {
    fn first_is_lead(&self, first: char) -> bool {
        is_json_first(first) || self.leads.contains(&first)
    }

    /// The stored form of a plain (non-JSON) string, or None when unchanged.
    fn strip_plain(&self, value: &str) -> Option<String> {
        for prefix in &self.prefixes {
            if let Some(rest) = value.strip_prefix(prefix.as_str()) {
                return Some(format!("{REL_PREFIX}{rest}"));
            }
        }
        if self.roots.iter().any(|root| root == value) {
            return Some(ROOT_TOKEN.to_string());
        }
        if looks_relative(value) {
            return Some(format!("{ESCAPE}{value}"));
        }
        None
    }

    /// The stored form of one parameter value (`strip_params` per element).
    fn stored<'py>(&self, py: Python<'py>, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let Ok(text) = value.cast_exact::<PyString>() else {
            if value.is_instance_of::<PyString>() {
                // A str subclass: the plain path, never cached.
                let lead = value
                    .cast::<PyString>()?
                    .to_str()
                    .ok()
                    .and_then(|s| s.chars().next())
                    .map(|c| self.first_is_lead(c));
                if lead == Some(false) {
                    return Ok(value);
                }
                return self.strip.bind(py).call1((value,));
            }
            return Ok(value);
        };
        let cache = self.cache.bind(py);
        if let Some(stored) = cache.get_item(text)? {
            return Ok(stored);
        }
        let (stored, cacheable) = match text.to_str() {
            Ok(s) => match s.chars().next() {
                None => (value.clone(), true),
                Some(first) if is_json_first(first) => (
                    self.strip.bind(py).call1((text,))?,
                    s.chars().count() <= CACHED_JSON_MAX,
                ),
                Some(first) if !self.leads.contains(&first) => (value.clone(), true),
                Some(_) => match self.strip_plain(s) {
                    Some(out) => (PyString::new(py, &out).into_any(), true),
                    None => (value.clone(), true),
                },
            },
            // Lone surrogates: Python's own strip, and not cached.
            Err(_) => (self.strip.bind(py).call1((text,))?, false),
        };
        if cacheable {
            if cache.len() > CACHE_LIMIT {
                cache.clear();
            }
            cache.set_item(text, &stored)?;
        }
        Ok(stored)
    }

    fn stored_row<'py>(&self, py: Python<'py>, row: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
        let mut out: Option<Vec<Bound<'py, PyAny>>> = None;
        for (index, value) in row.iter().enumerate() {
            let stored = self.stored(py, value.clone())?;
            if !stored.is(&value) {
                let values = out.get_or_insert_with(|| row.iter().collect());
                values[index] = stored;
            }
        }
        match out {
            None => Ok(row.clone()),
            Some(values) => PyTuple::new(py, values),
        }
    }
}

#[pymethods]
impl PathStripper {
    #[new]
    fn new(
        py: Python<'_>,
        prefixes: Vec<String>,
        roots: Vec<String>,
        leads: Vec<String>,
        strip: Py<PyAny>,
    ) -> PyResult<Self> {
        let mut lead_chars = Vec::new();
        for lead in leads {
            let mut chars = lead.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => lead_chars.push(c),
                _ => return Err(PyTypeError::new_err("each lead must be one character")),
            }
        }
        Ok(PathStripper {
            prefixes,
            roots,
            leads: lead_chars,
            strip,
            cache: PyDict::new(py).unbind(),
        })
    }

    /// `PathCodec.strip_params` over a list of row tuples.
    fn stored_rows<'py>(&self, py: Python<'py>, rows: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for row in rows.iter() {
            out.append(self.stored_row(py, row.cast::<PyTuple>()?)?)?;
        }
        Ok(out)
    }
}

/// `GraphStore._insert_edges` rows for *edges*: one per call site, the last
/// edge at a site winning, in first-seen order. With a *stripper* each row is
/// in its stored form, ready for the plain `sqlite3` `executemany`.
#[pyfunction]
#[pyo3(signature = (edges, now, dumps, stripper=None))]
pub fn edge_rows<'py>(
    py: Python<'py>,
    edges: &Bound<'py, PyList>,
    now: Bound<'py, PyAny>,
    dumps: &Bound<'py, PyAny>,
    stripper: Option<PyRef<'py, PathStripper>>,
) -> PyResult<Bound<'py, PyList>> {
    let by_site = PyDict::new(py);
    let float_type = py.get_type::<PyFloat>();
    let str_type = py.get_type::<PyString>();
    let empty_extra = intern!(py, "{}");
    let default_tier = intern!(py, "EXTRACTED");
    let default_confidence = PyFloat::new(py, 1.0);
    let none = py.None().into_bound(py);
    for edge in edges.iter() {
        let kind = edge.getattr(intern!(py, "kind"))?;
        let source = edge.getattr(intern!(py, "source"))?;
        let target = edge.getattr(intern!(py, "target"))?;
        let file_path = edge.getattr(intern!(py, "file_path"))?;
        let line = edge.getattr(intern!(py, "line"))?;
        let raw_extra = edge.getattr(intern!(py, "extra"))?;
        let (confidence, tier, extra, offset) = if raw_extra.is_truthy()? {
            let get = raw_extra.getattr(intern!(py, "get"))?;
            let confidence = float_type
                .call1((get.call1((intern!(py, "confidence"), &default_confidence))?,))?;
            let tier = str_type
                .call1((get.call1((intern!(py, "confidence_tier"), default_tier))?,))?;
            let extra = dumps.call1((&raw_extra,))?;
            let offset = get.call1((intern!(py, "source_offset"),))?;
            (confidence, tier, extra, offset)
        } else {
            // ``{}``: every lookup takes its default.
            (
                float_type.call1((&default_confidence,))?,
                str_type.call1((default_tier,))?,
                empty_extra.clone().into_any(),
                none.clone(),
            )
        };
        let site = PyTuple::new(
            py,
            [&kind, &source, &target, &file_path, &line, &offset],
        )?;
        let row = PyTuple::new(
            py,
            [kind, source, target, file_path, line, extra, confidence, tier, now.clone()],
        )?;
        by_site.set_item(site, row)?;
    }
    let out = PyList::empty(py);
    for row in by_site.values().iter() {
        let row = row.cast_into::<PyTuple>()?;
        match &stripper {
            Some(stripper) => out.append(stripper.stored_row(py, &row)?)?,
            None => out.append(row)?,
        }
    }
    Ok(out)
}
