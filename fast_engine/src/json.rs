//! `json.dumps` with its defaults, for the plain values node and edge extras hold.
//!
//! Every stored node and edge carries its `extra` dict as JSON text, so a build
//! calls `json.dumps` tens of thousands of times. This writes the same text
//! the C encoder writes with the default arguments (`ensure_ascii`, `", "` and
//! `": "` separators, `NaN`/`Infinity` allowed, keys in insertion order) for
//! exact `dict`, `list`, `tuple`, `str`, `int`, `float`, `bool` and `None`
//! values with `str` keys. Anything else in the value (a subclass, another
//! key type, a string Rust cannot read as UTF-8, nesting deep enough to be a
//! cycle) sends the whole value to the Python `dumps`, so its text or its
//! error is Python's own.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

/// Deeper values go to Python, which also reports a cycle as Python does.
const MAX_DEPTH: usize = 64;
const HEX: &[u8; 16] = b"0123456789abcdef";

fn push_u16(out: &mut String, unit: u32) {
    out.push_str("\\u");
    for shift in [12, 8, 4, 0] {
        out.push(HEX[((unit >> shift) & 0xf) as usize] as char);
    }
}

/// `c_encode_basestring_ascii`: printable ASCII as is, the rest escaped.
fn push_str(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            ' '..='~' => out.push(c),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => {
                let code = c as u32;
                if code >= 0x10000 {
                    let v = code - 0x10000;
                    push_u16(out, 0xd800 | ((v >> 10) & 0x3ff));
                    push_u16(out, 0xdc00 | (v & 0x3ff));
                } else {
                    push_u16(out, code);
                }
            }
        }
    }
    out.push('"');
}

/// Append *value* as JSON; `false` when Python must encode it.
fn push_value(out: &mut String, value: &Bound<'_, PyAny>, depth: usize) -> PyResult<bool> {
    if value.is_none() {
        out.push_str("null");
    } else if let Ok(text) = value.cast_exact::<PyString>() {
        match text.to_str() {
            Ok(text) => push_str(out, text),
            Err(_) => return Ok(false),
        }
    } else if let Ok(flag) = value.cast_exact::<PyBool>() {
        out.push_str(if flag.is_true() { "true" } else { "false" });
    } else if value.is_exact_instance_of::<PyInt>() {
        match value.extract::<i64>() {
            Ok(number) => out.push_str(&number.to_string()),
            Err(_) => out.push_str(value.repr()?.to_str()?),
        }
    } else if let Ok(number) = value.cast_exact::<PyFloat>() {
        let v = number.value();
        if v.is_nan() {
            out.push_str("NaN");
        } else if v == f64::INFINITY {
            out.push_str("Infinity");
        } else if v == f64::NEG_INFINITY {
            out.push_str("-Infinity");
        } else {
            out.push_str(number.repr()?.to_str()?);
        }
    } else if depth >= MAX_DEPTH {
        return Ok(false);
    } else if let Ok(dict) = value.cast_exact::<PyDict>() {
        out.push('{');
        for (index, (key, item)) in dict.iter().enumerate() {
            let Ok(key) = key.cast_exact::<PyString>() else { return Ok(false) };
            let Ok(key) = key.to_str() else { return Ok(false) };
            if index > 0 {
                out.push_str(", ");
            }
            push_str(out, key);
            out.push_str(": ");
            if !push_value(out, &item, depth + 1)? {
                return Ok(false);
            }
        }
        out.push('}');
    } else if let Ok(list) = value.cast_exact::<PyList>() {
        out.push('[');
        for (index, item) in list.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            if !push_value(out, &item, depth + 1)? {
                return Ok(false);
            }
        }
        out.push(']');
    } else if let Ok(tuple) = value.cast_exact::<PyTuple>() {
        out.push('[');
        for (index, item) in tuple.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            if !push_value(out, &item, depth + 1)? {
                return Ok(false);
            }
        }
        out.push(']');
    } else {
        return Ok(false);
    }
    Ok(true)
}

/// `dumps(value)`, written here when the value is plain.
pub(crate) fn dumps_or<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    dumps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut out = String::new();
    if push_value(&mut out, value, 0)? {
        Ok(PyString::new(py, &out).into_any())
    } else {
        dumps.call1((value,))
    }
}

/// `json.dumps(value)` for plain values, else `fallback(value)`.
#[pyfunction]
pub fn json_dumps<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    fallback: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    dumps_or(py, value, fallback)
}
