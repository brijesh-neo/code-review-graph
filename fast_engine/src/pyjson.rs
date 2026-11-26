//! `json.loads` and `json.dumps` as Python runs them, for the native resolvers.
//!
//! A resolver reads `edges.extra`, changes a few keys and writes it back with
//! `json.dumps`. The text written must be byte for byte what Python writes, so:
//!
//! * a dict keeps insertion order, and a repeated key keeps its first position
//!   with the last value (`dict` assignment);
//! * numbers stay int or float as `json.loads` returns them, and a float is
//!   written back as `repr(float)`;
//! * `==` is Python's: `True == 1 == 1.0`, dicts compared without order;
//! * strings are written with `ensure_ascii` and the default separators.
//!
//! Anything this module is not sure to read exactly as Python does (NaN and
//! Infinity, integers outside i64, lone surrogates) makes `parse` return None,
//! and the caller hands the whole job back to Python.

use std::fmt::Write as _;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
    Dict(Dict),
}

/// An insertion-ordered dict with string keys.
#[derive(Clone, Debug, Default)]
pub struct Dict {
    pub items: Vec<(String, Value)>,
}

impl Dict {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.items.iter().any(|(k, _)| k == key)
    }

    /// `d[key] = value`: replaced in place, or appended.
    pub fn set(&mut self, key: &str, value: Value) {
        if let Some(slot) = self.items.iter_mut().find(|(k, _)| k == key) {
            slot.1 = value;
        } else {
            self.items.push((key.to_string(), value));
        }
    }

    /// `d.pop(key, None)`.
    pub fn pop(&mut self, key: &str) -> Option<Value> {
        let index = self.items.iter().position(|(k, _)| k == key)?;
        Some(self.items.remove(index).1)
    }

    /// `d.get(key)` when it is a str.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Value::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Python truthiness.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::List(l) => !l.is_empty(),
            Value::Dict(d) => !d.items.is_empty(),
        }
    }

    fn number(&self) -> Option<f64> {
        match self {
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn int_like(&self) -> Option<i64> {
        match self {
            Value::Bool(b) => Some(*b as i64),
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }
}

/// Python `==`.
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Null, Value::Null) => true,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Dict(x), Value::Dict(y)) => dict_eq(x, y),
        _ => {
            if let (Some(x), Some(y)) = (a.int_like(), b.int_like()) {
                return x == y;
            }
            match (a.number(), b.number()) {
                // An int beside a float: exact when the int fits a double.
                (Some(x), Some(y)) => {
                    let exact = |v: &Value| match v {
                        Value::Int(i) => i.unsigned_abs() <= (1u64 << 53),
                        _ => true,
                    };
                    if exact(a) && exact(b) {
                        x == y
                    } else {
                        // Rare enough to answer slowly and exactly.
                        let (int, float) = match (a, b) {
                            (Value::Int(i), _) => (*i, y),
                            (_, Value::Int(i)) => (*i, x),
                            _ => return x == y,
                        };
                        float.fract() == 0.0
                            && float >= -9.223372036854775808e18
                            && float < 9.223372036854775808e18
                            && float as i64 == int
                    }
                }
                _ => false,
            }
        }
    }
}

pub fn dict_eq(x: &Dict, y: &Dict) -> bool {
    x.items.len() == y.items.len()
        && x.items.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
}

// -- parse -----------------------------------------------------------------

struct Reader<'a> {
    bytes: &'a [u8],
    text: &'a str,
    pos: usize,
}

const MAX_DEPTH: usize = 200;

impl<'a> Reader<'a> {
    fn ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn value(&mut self, depth: usize) -> Option<Value> {
        if depth > MAX_DEPTH {
            return None;
        }
        self.ws();
        match *self.bytes.get(self.pos)? {
            b'{' => {
                self.pos += 1;
                let mut dict = Dict::default();
                self.ws();
                if self.bytes.get(self.pos) == Some(&b'}') {
                    self.pos += 1;
                    return Some(Value::Dict(dict));
                }
                loop {
                    self.ws();
                    if self.bytes.get(self.pos) != Some(&b'"') {
                        return None;
                    }
                    let key = self.string()?;
                    self.ws();
                    if self.bytes.get(self.pos) != Some(&b':') {
                        return None;
                    }
                    self.pos += 1;
                    let value = self.value(depth + 1)?;
                    dict.set(&key, value);
                    self.ws();
                    match self.bytes.get(self.pos)? {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            return Some(Value::Dict(dict));
                        }
                        _ => return None,
                    }
                }
            }
            b'[' => {
                self.pos += 1;
                let mut list = Vec::new();
                self.ws();
                if self.bytes.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Some(Value::List(list));
                }
                loop {
                    list.push(self.value(depth + 1)?);
                    self.ws();
                    match self.bytes.get(self.pos)? {
                        b',' => self.pos += 1,
                        b']' => {
                            self.pos += 1;
                            return Some(Value::List(list));
                        }
                        _ => return None,
                    }
                }
            }
            b'"' => self.string().map(Value::Str),
            b't' => self.word("true", Value::Bool(true)),
            b'f' => self.word("false", Value::Bool(false)),
            b'n' => self.word("null", Value::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn word(&mut self, word: &str, value: Value) -> Option<Value> {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            Some(value)
        } else {
            None
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Option<Value> {
        let start = self.pos;
        if self.bytes[self.pos] == b'-' {
            self.pos += 1;
        }
        match self.bytes.get(self.pos)? {
            b'0' => self.pos += 1,
            b'1'..=b'9' => {
                self.digits();
            }
            _ => return None,
        }
        let mut float = false;
        if self.bytes.get(self.pos) == Some(&b'.')
            && self.bytes.get(self.pos + 1).is_some_and(u8::is_ascii_digit)
        {
            self.pos += 1;
            self.digits();
            float = true;
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            let mark = self.pos;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                // Python's scanner stops before the `e`; the next token then fails.
                self.pos = mark;
            } else {
                float = true;
            }
        }
        let text = &self.text[start..self.pos];
        if float {
            let value: f64 = text.parse().ok()?;
            value.is_finite().then_some(Value::Float(value))
        } else {
            text.parse::<i64>().ok().map(Value::Int)
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let chunk = self.text.get(self.pos..self.pos + 4)?;
        if !chunk.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        self.pos += 4;
        u32::from_str_radix(chunk, 16).ok()
    }

    fn string(&mut self) -> Option<String> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&self.text[start..self.pos]);
            match *self.bytes.get(self.pos)? {
                b'"' => {
                    self.pos += 1;
                    return Some(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let escape = *self.bytes.get(self.pos)?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            let code = if (0xd800..0xdc00).contains(&unit) {
                                if !self.text[self.pos..].starts_with("\\u") {
                                    return None;
                                }
                                self.pos += 2;
                                let low = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&low) {
                                    return None;
                                }
                                0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00)
                            } else {
                                unit
                            };
                            out.push(char::from_u32(code)?);
                        }
                        _ => return None,
                    }
                }
                // A control character: strict json.loads refuses it.
                _ => return None,
            }
        }
    }
}

/// `json.loads(text)`, or None when Python might read it otherwise.
pub fn parse(text: &str) -> Option<Value> {
    let mut reader = Reader { bytes: text.as_bytes(), text, pos: 0 };
    let value = reader.value(0)?;
    reader.ws();
    (reader.pos == text.len()).then_some(value)
}

// -- dumps -----------------------------------------------------------------

const HEX: &[u8; 16] = b"0123456789abcdef";

fn push_u16(out: &mut String, unit: u32) {
    out.push_str("\\u");
    for shift in [12, 8, 4, 0] {
        out.push(HEX[((unit >> shift) & 0xf) as usize] as char);
    }
}

/// `c_encode_basestring_ascii`.
pub fn push_str(out: &mut String, text: &str) {
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

/// `repr(float)` for a finite float.
pub fn push_float(out: &mut String, value: f64) {
    // Rust's Debug picks the same shortest digits and the same switch to
    // exponent form (below 1e-4, from 1e16); only the exponent is spelt
    // differently: `1e-7` and `1e16` against Python's `1e-07` and `1e+16`.
    let text = format!("{value:?}");
    match text.split_once('e') {
        None => out.push_str(&text),
        Some((mantissa, exponent)) => {
            out.push_str(mantissa);
            out.push('e');
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exponent),
            };
            out.push(sign);
            if digits.len() < 2 {
                out.push('0');
            }
            out.push_str(digits);
        }
    }
}

fn push_value(out: &mut String, value: &Value, sort_keys: bool) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Value::Float(f) => push_float(out, *f),
        Value::Str(s) => push_str(out, s),
        Value::List(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                push_value(out, item, sort_keys);
            }
            out.push(']');
        }
        Value::Dict(dict) => push_dict(out, dict, sort_keys),
    }
}

fn push_dict(out: &mut String, dict: &Dict, sort_keys: bool) {
    out.push('{');
    let mut items: Vec<&(String, Value)> = dict.items.iter().collect();
    if sort_keys {
        // UTF-8 byte order is code point order, as Python sorts str.
        items.sort_by(|a, b| a.0.cmp(&b.0));
    }
    for (index, (key, item)) in items.into_iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        push_str(out, key);
        out.push_str(": ");
        push_value(out, item, sort_keys);
    }
    out.push('}');
}

/// `json.dumps(dict, sort_keys=sort_keys)`.
pub fn dumps_dict(dict: &Dict, sort_keys: bool) -> String {
    let mut out = String::new();
    push_dict(&mut out, dict, sort_keys);
    out
}
