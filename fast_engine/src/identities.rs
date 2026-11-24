//! `GraphStore._aligned_edges`' identity pass natively, for the edges it keeps.
//!
//! Each stored edge has its endpoints checked against the collision
//! identities already in the graph (`a.py::f:S…`). Almost every endpoint is
//! the file's own, has no `::`, or has one candidate that is itself, and then
//! the edge is returned unchanged. That check runs here. An edge with any
//! other endpoint goes to the Python `_resolve_edge_identities`, with the
//! same per-file candidate cache, so it gets exactly the Python answer.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

/// Whether *endpoint* comes back unchanged, or None when Python must decide.
fn unchanged<'py>(
    endpoint: &Bound<'py, PyAny>,
    local_prefix: &str,
    cache: &Bound<'py, PyDict>,
    candidates_of: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    if !endpoint.is_exact_instance_of::<PyString>() {
        return Ok(false);
    }
    let text = endpoint.cast::<PyString>()?;
    let Ok(s) = text.to_str() else { return Ok(false) };
    if s.starts_with(local_prefix) || !s.contains("::") {
        return Ok(true);
    }
    let candidates = match cache.get_item(endpoint)? {
        Some(found) => found,
        None => {
            let found = candidates_of.call1((endpoint,))?;
            cache.set_item(endpoint, &found)?;
            found
        }
    };
    let Ok(list) = candidates.cast::<PyList>() else { return Ok(false) };
    if list.len() != 1 {
        return Ok(false);
    }
    let only = list.get_item(0)?;
    Ok(only.is_exact_instance_of::<PyString>() && only.eq(endpoint)?)
}

/// `[resolve(edge, local_prefix, cache) for edge in edges]`, settling the
/// edges that come back unchanged without the call.
#[pyfunction]
pub fn settle_edges<'py>(
    py: Python<'py>,
    edges: &Bound<'py, PyList>,
    local_prefix: &Bound<'py, PyString>,
    cache: &Bound<'py, PyDict>,
    candidates_of: &Bound<'py, PyAny>,
    resolve: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    // A prefix that is not UTF-8 is compared by Python, edge by edge.
    let prefix = local_prefix.to_str().ok();
    for edge in edges.iter() {
        let settled = match prefix {
            Some(prefix) => {
                let source = edge.getattr(intern!(py, "source"))?;
                let target = edge.getattr(intern!(py, "target"))?;
                unchanged(&source, prefix, cache, candidates_of)?
                    && unchanged(&target, prefix, cache, candidates_of)?
            }
            None => false,
        };
        if settled {
            out.append(edge)?;
        } else {
            out.append(resolve.call1((&edge, local_prefix, cache))?)?;
        }
    }
    Ok(out)
}
