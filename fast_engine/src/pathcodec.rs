//! `relpaths.PathCodec` restore and strip, as plain string functions.
//!
//! A native pass reads stored values straight from `graph.db`, while the
//! Python logic it replaces sees them through the codec's row factory, and
//! writes through `strip_params`. The pass restores each value it reads and
//! strips each value it writes with the same rules, so its logic runs on the
//! values Python sees and its writes are the bytes Python writes:
//!
//! * plain values: `./x` and `.` gain the root, `!`-escaped relative-looking
//!   values lose one `!`; on the way in the root is stripped, or a
//!   relative-looking value escaped;
//! * JSON text: the regexes `_JSON_STORED` and `PathCodec._json_inbound`,
//!   applied by hand at each position where a JSON string may open.

use std::borrow::Cow;

const ESCAPE: u8 = b'!';

/// `str.isspace` for one character (the lookbehind's `\s`).
pub fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn is_json_first(c: u8) -> bool {
    c == b'{' || c == b'[' || c == b'"'
}

/// `_looks_relative`.
pub fn looks_relative(value: &str) -> bool {
    let body = value.trim_start_matches('!');
    body == "." || body.starts_with("./") || body.starts_with("../")
}

/// May a JSON string open at byte *i* (`(?:^|(?<=[{\[,:\s]))"`)?
fn string_opens(text: &str, i: usize) -> bool {
    if i == 0 {
        return true;
    }
    match text[..i].chars().next_back() {
        Some(c) => matches!(c, '{' | '[' | ',' | ':') || py_isspace(c),
        None => true,
    }
}

/// Is `\.\.?/|\."` at byte *i*?
fn relative_ahead(bytes: &[u8], i: usize) -> Option<usize> {
    let rest = &bytes[i.min(bytes.len())..];
    if rest.starts_with(b"./") {
        Some(2)
    } else if rest.starts_with(b"../") {
        Some(3)
    } else if rest.starts_with(b".\"") {
        Some(2)
    } else {
        None
    }
}

pub struct Codec {
    pub root: String,
    json_root_out: String,
    prefixes: Vec<String>,
    roots: Vec<String>,
    json_roots: Vec<String>,
    leads: Vec<char>,
}

impl Codec {
    pub fn new(
        root: String,
        json_root_out: String,
        prefixes: Vec<String>,
        roots: Vec<String>,
        json_roots: Vec<String>,
        leads: Vec<char>,
    ) -> Codec {
        Codec { root, json_root_out, prefixes, roots, json_roots, leads }
    }

    // -- outbound ----------------------------------------------------------

    /// `PathCodec.restore` for a str.
    pub fn restore<'a>(&self, value: &'a str) -> Cow<'a, str> {
        let Some(&first) = value.as_bytes().first() else { return Cow::Borrowed(value) };
        if first == b'.' {
            if value == "." {
                return Cow::Owned(self.root.clone());
            }
            if let Some(rest) = value.strip_prefix("./") {
                let mut out = String::with_capacity(self.root.len() + 1 + rest.len());
                out.push_str(&self.root);
                out.push('/');
                out.push_str(rest);
                return Cow::Owned(out);
            }
        } else if first == ESCAPE && looks_relative(value) {
            return Cow::Borrowed(&value[1..]);
        } else if is_json_first(first) {
            return self.restore_json(value);
        }
        Cow::Borrowed(value)
    }

    /// `PathCodec.restore_json`: `_JSON_STORED.sub(...)`.
    pub fn restore_json<'a>(&self, value: &'a str) -> Cow<'a, str> {
        if !value.contains("\".") && !value.contains("\"!") {
            return Cow::Borrowed(value);
        }
        let bytes = value.as_bytes();
        let mut out = String::with_capacity(value.len() + 64);
        let mut copied = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'"' || !string_opens(value, i) {
                i += 1;
                continue;
            }
            let after = i + 1;
            let mut j = after;
            while bytes.get(j) == Some(&ESCAPE) {
                j += 1;
            }
            if j > after && relative_ahead(bytes, j).is_some() {
                // `"` + one `!` fewer; the lookahead is not consumed.
                out.push_str(&value[copied..i]);
                out.push('"');
                out.push_str(&value[after + 1..j]);
                copied = j;
                i = j;
                continue;
            }
            if bytes.get(after) == Some(&b'.')
                && matches!(bytes.get(after + 1), Some(b'/' | b'"'))
            {
                out.push_str(&value[copied..i]);
                out.push('"');
                out.push_str(&self.json_root_out);
                out.push(bytes[after + 1] as char);
                copied = after + 2;
                i = after + 2;
                continue;
            }
            i += 1;
        }
        if copied == 0 {
            return Cow::Borrowed(value);
        }
        out.push_str(&value[copied..]);
        Cow::Owned(out)
    }

    // -- inbound -----------------------------------------------------------

    /// `PathCodec.strip` for a str.
    pub fn strip<'a>(&self, value: &'a str) -> Cow<'a, str> {
        let Some(first) = value.chars().next() else { return Cow::Borrowed(value) };
        if first.is_ascii() && is_json_first(first as u8) {
            return self.strip_json(value);
        }
        if !self.leads.contains(&first) {
            return Cow::Borrowed(value);
        }
        for prefix in &self.prefixes {
            if let Some(rest) = value.strip_prefix(prefix.as_str()) {
                return Cow::Owned(format!("./{rest}"));
            }
        }
        if self.roots.iter().any(|root| root == value) {
            return Cow::Borrowed(".");
        }
        if looks_relative(value) {
            return Cow::Owned(format!("!{value}"));
        }
        Cow::Borrowed(value)
    }

    /// `PathCodec.strip_json`: `_json_inbound.sub(...)`.
    pub fn strip_json<'a>(&self, value: &'a str) -> Cow<'a, str> {
        if !value.contains("\".")
            && !value.contains("\"!")
            && !self.json_roots.iter().any(|root| value.contains(root.as_str()))
        {
            return Cow::Borrowed(value);
        }
        let bytes = value.as_bytes();
        let mut out = String::with_capacity(value.len());
        let mut copied = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'"' || !string_opens(value, i) {
                i += 1;
                continue;
            }
            let after = i + 1;
            let mut j = after;
            while bytes.get(j) == Some(&ESCAPE) {
                j += 1;
            }
            if let Some(len) = relative_ahead(bytes, j) {
                // `"` + `!` + the relative text, which the match consumes.
                let end = j + len;
                out.push_str(&value[copied..i]);
                out.push_str("\"!");
                out.push_str(&value[after..end]);
                copied = end;
                i = end;
                continue;
            }
            let mut matched = None;
            for root in &self.json_roots {
                if value[after..].starts_with(root.as_str()) {
                    let end = after + root.len();
                    if let Some(&c @ (b'/' | b'"')) = bytes.get(end) {
                        matched = Some((end + 1, c));
                        break;
                    }
                }
            }
            if let Some((end, c)) = matched {
                out.push_str(&value[copied..i]);
                out.push_str(if c == b'/' { "\"./" } else { "\".\"" });
                copied = end;
                i = end;
                continue;
            }
            i += 1;
        }
        if copied == 0 {
            return Cow::Borrowed(value);
        }
        out.push_str(&value[copied..]);
        Cow::Owned(out)
    }
}

/// Restore and strip for a graph that may store absolute paths (no codec).
pub struct Paths(pub Option<Codec>);

impl Paths {
    pub fn restore<'a>(&self, value: &'a str) -> Cow<'a, str> {
        match &self.0 {
            Some(codec) => codec.restore(value),
            None => Cow::Borrowed(value),
        }
    }

    pub fn strip<'a>(&self, value: &'a str) -> Cow<'a, str> {
        match &self.0 {
            Some(codec) => codec.strip(value),
            None => Cow::Borrowed(value),
        }
    }
}
