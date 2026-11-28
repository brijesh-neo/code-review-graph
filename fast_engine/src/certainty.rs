//! `GraphStore.refresh_target_resolution`'s four UPDATEs in one native pass.
//!
//! The SQL rewrites `target_resolution` for CALLS and REFERENCES, then
//! `confidence_tier` for them and for TESTED_BY, then `confidence` for all
//! three, each statement reading what the one before it wrote. Here every
//! row goes through the same three steps in the same order, and only the
//! columns that change are written.
//!
//! The pass works on stored values exactly as the SQL does:
//!
//! * `=` against `nodes.qualified_name` is a byte comparison (BINARY);
//! * `LIKE` folds ASCII case only and reads `_` as any one character;
//! * `json_valid` and `json_extract` are SQLite's own, run on the same text.
//!
//! When a row holds anything those rules are not written for here (a column
//! that is not TEXT, text that is not UTF-8, a `confidence` that is not a
//! number), the pass writes nothing and returns `None`, and the caller runs
//! the SQL instead.

use std::collections::HashSet;
use std::ffi::c_int;

use libsqlite3_sys as ffi;
use pyo3::prelude::*;

use crate::sqlite_ext::{Stmt, errmsg, stage};

const UNRESOLVED_TIER: &str = "UNRESOLVED";
const AMBIGUOUS_TIER: &str = "AMBIGUOUS";
const INFERRED_TIER: &str = "INFERRED";
const EXTRACTED_TIER: &str = "EXTRACTED";

/// Returned by the job when the caller must run the SQL instead.
const FALL_BACK: i64 = -1;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// CALLS or REFERENCES: classified by their target.
    Target,
    /// TESTED_BY: unbound when its source names no node.
    TestedBy,
}

#[derive(Clone, PartialEq, Debug)]
enum Number {
    Null,
    Int(i64),
    Real(f64),
}

/// `a IS NOT b` for two numbers SQLite compares numerically, or None when
/// the comparison is not one written for here.
fn is_not(a: &Number, b: &Number) -> Option<bool> {
    const EXACT: i64 = 1 << 53;
    Some(match (a, b) {
        (Number::Null, Number::Null) => false,
        (Number::Null, _) | (_, Number::Null) => true,
        (Number::Int(x), Number::Int(y)) => x != y,
        (Number::Real(x), Number::Real(y)) => x != y,
        (Number::Int(x), Number::Real(y)) | (Number::Real(y), Number::Int(x)) => {
            if x.abs() > EXACT {
                return None;
            }
            (*x as f64) != *y
        }
    })
}

/// One `extra` value folded for SQLite's `LIKE '%needle%'` (no ESCAPE):
/// ASCII letters lowered, every character one slot, so `_` in a needle takes
/// exactly one character, and the slots holding `"`, where every needle here
/// starts.
#[derive(Default)]
struct Folded {
    chars: Vec<u32>,
    quotes: Vec<usize>,
}

impl Folded {
    fn load(&mut self, text: &str) {
        self.chars.clear();
        self.quotes.clear();
        for c in text.chars() {
            if c == '"' {
                self.quotes.push(self.chars.len());
            }
            self.chars.push(c.to_ascii_lowercase() as u32);
        }
    }

    /// *needle* is lowercase ASCII starting with `"`.
    fn contains(&self, needle: &[u8]) -> bool {
        self.quotes.iter().any(|&start| {
            self.chars.len() - start >= needle.len()
                && needle
                    .iter()
                    .zip(&self.chars[start..])
                    .all(|(&want, &have)| want == b'_' || u32::from(want) == have)
        })
    }
}

/// A needle for `Folded::contains`, or None when it is not one written for
/// here.
fn needle(key: &str) -> Option<Vec<u8>> {
    let quoted = format!("\"{key}\"");
    let ascii = quoted.is_ascii() && !quoted.contains('%') && !quoted.contains('\\');
    ascii.then(|| quoted.to_ascii_lowercase().into_bytes())
}

/// A column as UTF-8 text borrowed from the current row: Ok(None) for NULL,
/// Err(()) for anything else. Valid until the statement steps again.
unsafe fn text_column<'row>(
    stmt: *mut ffi::sqlite3_stmt,
    index: c_int,
) -> Result<Option<&'row str>, ()> {
    unsafe {
        match ffi::sqlite3_column_type(stmt, index) {
            ffi::SQLITE_NULL => Ok(None),
            ffi::SQLITE_TEXT => {
                let ptr = ffi::sqlite3_column_text(stmt, index);
                let len = ffi::sqlite3_column_bytes(stmt, index) as usize;
                if len == 0 {
                    return Ok(Some(""));
                }
                let bytes = std::slice::from_raw_parts(ptr, len);
                std::str::from_utf8(bytes).map(Some).map_err(|_| ())
            }
            _ => Err(()),
        }
    }
}

fn number_column(stmt: *mut ffi::sqlite3_stmt, index: c_int) -> Result<Number, ()> {
    unsafe {
        match ffi::sqlite3_column_type(stmt, index) {
            ffi::SQLITE_NULL => Ok(Number::Null),
            ffi::SQLITE_INTEGER => Ok(Number::Int(ffi::sqlite3_column_int64(stmt, index))),
            ffi::SQLITE_FLOAT => Ok(Number::Real(ffi::sqlite3_column_double(stmt, index))),
            _ => Err(()),
        }
    }
}

fn step(stmt: &Stmt, db: *mut ffi::sqlite3) -> Result<bool, String> {
    match unsafe { ffi::sqlite3_step(stmt.raw) } {
        ffi::SQLITE_ROW => Ok(true),
        ffi::SQLITE_DONE => Ok(false),
        _ => Err(errmsg(db)),
    }
}

struct Settings {
    resolver_keys: Vec<Vec<u8>>,
    ambiguous: Vec<u8>,
    confidence: Vec<u8>,
    scores: [f64; 4],
}

/// The columns one row needs written; None leaves a column as it is.
struct RowChange {
    id: i64,
    resolution: Option<&'static str>,
    tier: Option<Option<String>>,
    confidence: Option<Number>,
}

fn qualified_names(db: *mut ffi::sqlite3) -> Result<Option<HashSet<Vec<u8>>>, String> {
    let stmt = Stmt::prepare(db, "SELECT qualified_name FROM nodes")?;
    let mut names = HashSet::new();
    while step(&stmt, db)? {
        unsafe {
            if ffi::sqlite3_column_type(stmt.raw, 0) != ffi::SQLITE_TEXT {
                return Ok(None);
            }
            let ptr = ffi::sqlite3_column_blob(stmt.raw, 0).cast::<u8>();
            let len = ffi::sqlite3_column_bytes(stmt.raw, 0) as usize;
            let bytes = if len == 0 { &[][..] } else { std::slice::from_raw_parts(ptr, len) };
            names.insert(bytes.to_vec());
        }
    }
    Ok(Some(names))
}

/// The `$.confidence` of *extra* as SQLite reads it, or None (fall back).
fn json_confidence(lookup: &Stmt, db: *mut ffi::sqlite3, extra: &str) -> Result<Option<Number>, String> {
    lookup.bind_text(1, extra)?;
    let found = step(lookup, db);
    let value = match found {
        Ok(true) => number_column(lookup.raw, 0).ok(),
        Ok(false) => Some(Number::Null),
        Err(message) => {
            unsafe { ffi::sqlite3_reset(lookup.raw) };
            return Err(message);
        }
    };
    unsafe { ffi::sqlite3_reset(lookup.raw) };
    Ok(value)
}

fn plan(
    db: *mut ffi::sqlite3,
    settings: &Settings,
    scope_floor: Option<i64>,
) -> Result<Option<(Vec<RowChange>, i64)>, String> {
    let Some(names) = qualified_names(db)? else { return Ok(None) };
    let lookup = Stmt::prepare(
        db,
        "SELECT CASE WHEN json_valid(?1) THEN json_extract(?1, '$.confidence') END",
    )?;
    let mut sql = String::from(
        "SELECT id, kind, source_qualified, target_qualified, extra, confidence_tier, \
         target_resolution, confidence FROM edges \
         WHERE kind IN ('CALLS', 'REFERENCES', 'TESTED_BY')",
    );
    // Only the rows the certainty capture says can change, in id order.
    if scope_floor.is_some() {
        sql = format!(
            "SELECT * FROM ({sql} AND id > ?1 \
             UNION SELECT id, kind, source_qualified, target_qualified, extra, confidence_tier, \
             target_resolution, confidence FROM edges \
             WHERE id IN (SELECT id FROM temp.crg_cert_edge) \
             AND kind IN ('CALLS', 'REFERENCES', 'TESTED_BY') \
             UNION SELECT id, kind, source_qualified, target_qualified, extra, confidence_tier, \
             target_resolution, confidence FROM edges \
             WHERE target_qualified IN (SELECT qn FROM temp.crg_cert_qn) \
             AND kind IN ('CALLS', 'REFERENCES', 'TESTED_BY') \
             UNION SELECT id, kind, source_qualified, target_qualified, extra, confidence_tier, \
             target_resolution, confidence FROM edges \
             WHERE source_qualified IN (SELECT qn FROM temp.crg_cert_qn) \
             AND kind IN ('CALLS', 'REFERENCES', 'TESTED_BY')) ORDER BY id"
        );
    }
    let scan = Stmt::prepare(db, &sql)?;
    if let Some(floor) = scope_floor {
        scan.bind_i64(1, floor)?;
    }
    let [score_unresolved, score_ambiguous, score_inferred, score_extracted] = settings.scores;
    let mut changes = Vec::new();
    let mut resolution_changes: i64 = 0;
    let mut folded = Folded::default();
    while step(&scan, db)? {
        let raw = scan.raw;
        let id = unsafe { ffi::sqlite3_column_int64(raw, 0) };
        let columns = unsafe {
            (
                text_column(raw, 1),
                text_column(raw, 2),
                text_column(raw, 3),
                text_column(raw, 4),
                text_column(raw, 5),
                text_column(raw, 6),
            )
        };
        let (Ok(Some(kind)), Ok(source), Ok(target), Ok(extra), Ok(tier), Ok(resolution)) =
            columns
        else {
            return Ok(None);
        };
        let Ok(confidence) = number_column(raw, 7) else { return Ok(None) };
        let kind = if kind == "TESTED_BY" { Kind::TestedBy } else { Kind::Target };
        let mut change = RowChange { id, resolution: None, tier: None, confidence: None };

        // 1. target_resolution (CALLS, REFERENCES).
        let unbound = match kind {
            Kind::Target => {
                let direct = target.is_some_and(|t| names.contains(t.as_bytes()));
                let new = if direct { "direct" } else { "unresolved" };
                if resolution != Some(new) {
                    change.resolution = Some(new);
                    resolution_changes += 1;
                }
                !direct
            }
            Kind::TestedBy => !source.is_some_and(|s| names.contains(s.as_bytes())),
        };

        // 2. confidence_tier. A NULL extra makes every LIKE NULL, never true.
        if let Some(extra) = extra {
            folded.load(extra);
        }
        let has = |needle: &[u8]| extra.is_some() && folded.contains(needle);
        let new_tier: Option<&str> = if unbound && has(&settings.ambiguous) {
            Some(AMBIGUOUS_TIER)
        } else if unbound {
            Some(UNRESOLVED_TIER)
        } else if settings.resolver_keys.iter().any(|key| has(key)) {
            Some(INFERRED_TIER)
        } else if matches!(tier, Some(AMBIGUOUS_TIER) | Some(UNRESOLVED_TIER)) {
            Some(EXTRACTED_TIER)
        } else {
            tier
        };
        if new_tier != tier {
            change.tier = Some(new_tier.map(str::to_owned));
        }

        // 3. confidence, from the tier just written.
        let new_confidence = match new_tier {
            Some(UNRESOLVED_TIER) => Number::Real(score_unresolved),
            Some(AMBIGUOUS_TIER) => Number::Real(score_ambiguous),
            _ => {
                let from_extra = match extra {
                    Some(e) if has(&settings.confidence) => {
                        match json_confidence(&lookup, db, e)? {
                            Some(value) => value,
                            None => return Ok(None),
                        }
                    }
                    _ => Number::Null,
                };
                match from_extra {
                    Number::Null if new_tier == Some(INFERRED_TIER) => {
                        Number::Real(score_inferred)
                    }
                    Number::Null => Number::Real(score_extracted),
                    value => value,
                }
            }
        };
        match is_not(&confidence, &new_confidence) {
            None => return Ok(None),
            Some(true) => change.confidence = Some(new_confidence),
            Some(false) => {}
        }
        if change.resolution.is_some() || change.tier.is_some() || change.confidence.is_some() {
            changes.push(change);
        }
    }
    Ok(Some((changes, resolution_changes)))
}

/// One UPDATE per changed row, setting only the columns that change.
fn write(db: *mut ffi::sqlite3, changes: &[RowChange]) -> Result<(), String> {
    const COLUMNS: [&str; 3] = ["target_resolution", "confidence_tier", "confidence"];
    let mut statements: [Option<Stmt>; 8] = Default::default();
    for change in changes {
        let mask = usize::from(change.resolution.is_some())
            | usize::from(change.tier.is_some()) << 1
            | usize::from(change.confidence.is_some()) << 2;
        let stmt = match &mut statements[mask] {
            Some(stmt) => stmt,
            slot => {
                let set: Vec<String> = (0..3)
                    .filter(|bit| mask & (1 << bit) != 0)
                    .map(|bit| format!("{} = ?{}", COLUMNS[bit], bit + 1))
                    .collect();
                let sql = format!("UPDATE edges SET {} WHERE id = ?4", set.join(", "));
                slot.insert(Stmt::prepare(db, &sql)?)
            }
        };
        if let Some(value) = change.resolution {
            stmt.bind_text(1, value)?;
        }
        match &change.tier {
            Some(Some(text)) => stmt.bind_text(2, text)?,
            Some(None) => stmt.bind_null(2)?,
            None => {}
        }
        match change.confidence {
            Some(Number::Null) => stmt.bind_null(3)?,
            Some(Number::Int(int)) => stmt.bind_i64(3, int)?,
            Some(Number::Real(real)) => stmt.bind_f64(3, real)?,
            None => {}
        }
        stmt.bind_i64(4, change.id)?;
        stmt.run()?;
    }
    Ok(())
}

/// Stages the refresh. The job returns how many rows had their
/// `target_resolution` changed, or -1 when the caller must run the SQL.
///
/// *scores* is (unresolved, ambiguous, inferred, extracted).
#[pyfunction]
#[pyo3(signature = (resolver_keys, scores, scope_floor=None))]
pub fn stage_certainty_refresh(
    resolver_keys: Vec<String>,
    scores: (f64, f64, f64, f64),
    scope_floor: Option<i64>,
) -> PyResult<i64> {
    let keys: Option<Vec<Vec<u8>>> = resolver_keys.iter().map(|key| needle(key)).collect();
    let Some(keys) = keys else {
        return stage(Box::new(|_| Ok(FALL_BACK)));
    };
    let settings = Settings {
        resolver_keys: keys,
        ambiguous: needle("ambiguous_targets").unwrap_or_default(),
        confidence: needle("confidence").unwrap_or_default(),
        scores: [scores.0, scores.1, scores.2, scores.3],
    };
    stage(Box::new(move |db| {
        let Some((changes, resolution_changes)) = plan(db, &settings, scope_floor)? else {
            return Ok(FALL_BACK);
        };
        write(db, &changes)?;
        Ok(resolution_changes)
    }))
}
