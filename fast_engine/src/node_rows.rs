//! `GraphStore._replace_file_data`'s `nodes` rows, built natively.
//!
//! Each parsed node becomes one row: its JSON `extra`, the `symbol` of its
//! qualified name, the indexed docstring summary and the `name_tokens`
//! camelCase splits. This does the same with the same objects. `json.dumps`
//! and `int()` are the Python callables; a node with any attribute that is
//! not plain `str` (or `None`, `dict`, where those are allowed) is built by
//! the Python `_node_row`, and so is a docstring summary that is not ASCII,
//! whose whitespace is Python's to define.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyInt, PyString, PyTuple};

use crate::store::PathStripper;

/// `MAX_INDEXED_DOCSTRING_CHARS`.
const DOCSTRING_MAX: usize = 400;
/// `MAX_INDEXED_TOKEN_CHARS`.
const TOKENS_MAX: usize = 200;

/// `_CAMEL_BOUNDARY_RE.split(run)` on one `[0-9A-Za-z]+` run.
fn camel_parts(run: &[u8]) -> Vec<&[u8]> {
    let mut parts = Vec::new();
    let mut start = 0;
    for i in 1..run.len() {
        let (prev, cur) = (run[i - 1], run[i]);
        let lower_to_upper =
            (prev.is_ascii_lowercase() || prev.is_ascii_digit()) && cur.is_ascii_uppercase();
        let acronym_end = prev.is_ascii_uppercase()
            && cur.is_ascii_uppercase()
            && run.get(i + 1).is_some_and(|next| next.is_ascii_lowercase());
        if lower_to_upper || acronym_end {
            parts.push(&run[start..i]);
            start = i;
        }
    }
    parts.push(&run[start..]);
    parts
}

/// `_index_tokens(*values)`.
fn index_tokens(values: &[&str]) -> String {
    let mut out: Vec<&[u8]> = Vec::new();
    let mut seen: Vec<Vec<u8>> = Vec::new();
    for value in values {
        for run in value.as_bytes().split(|b| !b.is_ascii_alphanumeric()) {
            if run.is_empty() {
                continue;
            }
            let parts = camel_parts(run);
            if parts.len() < 2 {
                continue;
            }
            for part in parts {
                let lowered = part.to_ascii_lowercase();
                if !seen.contains(&lowered) {
                    seen.push(lowered);
                    out.push(part);
                }
            }
        }
    }
    let mut joined = out.join(&b' ');
    joined.truncate(TOKENS_MAX);
    // Every byte is ASCII, so this cannot fail.
    String::from_utf8(joined).unwrap_or_default()
}

/// `" ".join(raw.split())[:400] or None` for ASCII *raw*.
fn ascii_docstring(raw: &str) -> Option<String> {
    // ``str.split()`` also splits at the ASCII separators \x1c-\x1f.
    let words = raw
        .split(|c: char| c.is_ascii_whitespace() || c == '\x0b' || ('\x1c'..='\x1f').contains(&c))
        .filter(|word| !word.is_empty());
    let mut out = words.collect::<Vec<_>>().join(" ");
    out.truncate(DOCSTRING_MAX);
    (!out.is_empty()).then_some(out)
}

fn plain_str<'a, 'py>(value: &'a Bound<'py, PyAny>) -> Option<&'a Bound<'py, PyString>> {
    if value.is_exact_instance_of::<PyString>() {
        value.cast::<PyString>().ok()
    } else {
        None
    }
}

/// A plain `str` or `None` as `Some(text or "")`, else None.
fn optional_text(value: &Bound<'_, PyAny>) -> Option<String> {
    if value.is_none() {
        return Some(String::new());
    }
    plain_str(value).and_then(|text| text.to_str().ok().map(str::to_owned))
}

/// `nodes` rows for one file: `_node_row(node, qualified)` for each pair.
#[pyfunction]
#[pyo3(signature = (nodes, identities, fhash, now, stem, dumps, indexed_docstring, node_row, stripper=None))]
#[allow(clippy::too_many_arguments)]
pub fn node_rows<'py>(
    py: Python<'py>,
    nodes: &Bound<'py, PyList>,
    identities: &Bound<'py, PyList>,
    fhash: Bound<'py, PyAny>,
    now: Bound<'py, PyAny>,
    stem: &Bound<'py, PyString>,
    dumps: &Bound<'py, PyAny>,
    indexed_docstring: &Bound<'py, PyAny>,
    node_row: &Bound<'py, PyAny>,
    stripper: Option<PyRef<'py, PathStripper>>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    let stem = stem.to_str().ok();
    let int_type = py.get_type::<PyInt>();
    let empty_extra = intern!(py, "{}");
    let none = py.None().into_bound(py);
    for (node, qualified) in nodes.iter().zip(identities.iter()) {
        let row = match stem {
            Some(stem) => native_row(
                py, &node, &qualified, &fhash, &now, stem, dumps, indexed_docstring,
                &int_type, empty_extra, &none,
            )?,
            None => None,
        };
        let row = match row {
            Some(row) => row,
            None => node_row.call1((&node, &qualified))?.cast_into::<PyTuple>()?,
        };
        match &stripper {
            Some(stripper) => out.append(stripper.stored_row(py, &row)?)?,
            None => out.append(row)?,
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn native_row<'py>(
    py: Python<'py>,
    node: &Bound<'py, PyAny>,
    qualified: &Bound<'py, PyAny>,
    fhash: &Bound<'py, PyAny>,
    now: &Bound<'py, PyAny>,
    stem: &str,
    dumps: &Bound<'py, PyAny>,
    indexed_docstring: &Bound<'py, PyAny>,
    int_type: &Bound<'py, pyo3::types::PyType>,
    empty_extra: &Bound<'py, PyString>,
    none: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let kind = node.getattr(intern!(py, "kind"))?;
    let name = node.getattr(intern!(py, "name"))?;
    let file_path = node.getattr(intern!(py, "file_path"))?;
    let parent_name = node.getattr(intern!(py, "parent_name"))?;
    let identity_name = node.getattr(intern!(py, "identity_name"))?;
    let extra = node.getattr(intern!(py, "extra"))?;
    let (Some(kind_s), Some(name_s), Some(path_s)) = (
        plain_str(&kind).and_then(|t| t.to_str().ok()),
        plain_str(&name).and_then(|t| t.to_str().ok()),
        plain_str(&file_path).and_then(|t| t.to_str().ok()),
    ) else {
        return Ok(None);
    };
    let (Some(parent_s), Some(identity_s)) = (optional_text(&parent_name), optional_text(&identity_name))
    else {
        return Ok(None);
    };
    if !extra.is_exact_instance_of::<PyDict>() {
        return Ok(None);
    }
    let extra_dict = extra.cast::<PyDict>()?;

    // ``_symbol_of(qualified_name_of(node))``.
    let qualified_name = if kind_s == "File" {
        path_s.to_owned()
    } else {
        let identity = if identity_s.is_empty() { name_s } else { identity_s.as_str() };
        if parent_s.is_empty() {
            format!("{path_s}::{identity}")
        } else {
            format!("{path_s}::{parent_s}.{identity}")
        }
    };
    let symbol = match qualified_name.split_once("::") {
        Some((_, symbol)) => symbol,
        None => qualified_name.as_str(),
    };

    let (extra_text, docstring) = if extra_dict.is_empty() {
        (empty_extra.clone().into_any(), none.clone())
    } else {
        let docstring = match extra_dict.get_item(intern!(py, "docstring"))? {
            Some(raw) => match plain_str(&raw).and_then(|t| t.to_str().ok()) {
                Some(text) if text.is_ascii() => match ascii_docstring(text) {
                    Some(summary) => PyString::new(py, &summary).into_any(),
                    None => none.clone(),
                },
                _ if raw.is_instance_of::<PyString>() => indexed_docstring.call1((&extra,))?,
                _ => none.clone(),
            },
            None => none.clone(),
        };
        (crate::json::dumps_or(py, &extra, dumps)?, docstring)
    };

    let token_name = if kind_s == "File" { "" } else { name_s };
    let tokens = index_tokens(&[token_name, &parent_s, stem]);

    let values = [
        kind,
        name,
        qualified.clone(),
        file_path,
        node.getattr(intern!(py, "line_start"))?,
        node.getattr(intern!(py, "line_end"))?,
        node.getattr(intern!(py, "language"))?,
        parent_name,
        node.getattr(intern!(py, "params"))?,
        node.getattr(intern!(py, "return_type"))?,
        node.getattr(intern!(py, "modifiers"))?,
        int_type.call1((node.getattr(intern!(py, "is_test"))?,))?,
        fhash.clone(),
        extra_text,
        PyString::new(py, symbol).into_any(),
        docstring,
        PyString::new(py, &tokens).into_any(),
        now.clone(),
    ];
    Ok(Some(PyTuple::new(py, values)?))
}
