//! The reads of `detect_entry_points` (and of `_entry_points_among`), as one
//! native pass on the caller's connection.
//!
//! The candidate rows come back in the order `_plan_order` says SQLite lists
//! them, each with whether a caller that is not a File node calls it, and
//! whether one that is also not test code does: what `get_all_call_targets(
//! include_file_sources=False)` and `get_call_targets_outside_tests` answer,
//! taken from the same LEFT JOIN on stored values. Values come back restored
//! as the codec's row factory restores them; filters are stripped as
//! `strip_params` strips them. The test-file rule, the decorator and name
//! patterns stay in Python. Nothing is written.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::resolve::{CodecFields, Fail, HashMap, HashSet, Rows, paths_from, run_job};
use crate::sqlite_ext::stage;

/// The two candidate orders `_CANDIDATE_ORDERS` names; nothing else reaches
/// the SQL text.
const ORDERS: [(&str, &str); 2] = [("kind", "kind, id"), ("id", "id")];

/// id, kind, name, qualified name, file path, language, is_test, extra, called
/// by a non-File caller, called by a non-File caller outside test code.
type Candidate = (
    i64,
    String,
    String,
    String,
    String,
    Option<String>,
    bool,
    Option<String>,
    bool,
    bool,
);

/// file paths, node ids, qualified names: a candidate in any of them is kept.
type Among = (Vec<String>, Vec<i64>, Vec<String>);

type Slot = Arc<Mutex<Option<Vec<Candidate>>>>;

static RESULTS: Mutex<Option<HashMap<i64, Slot>>> = Mutex::new(None);

/// Stages reading the entry-point candidates in *order* ("kind" or "id"),
/// all of them or only those *among* the given files, ids and names. The job
/// returns the number read, or -1 when a row holds a value this pass does not
/// read the way Python does; the rows are then taken with `take_entry_scan`.
#[pyfunction]
#[pyo3(signature = (order, among, codec))]
pub fn stage_entry_scan(order: String, among: Option<Among>, codec: CodecFields) -> PyResult<i64> {
    let Some((_, order_sql)) = ORDERS.iter().find(|(name, _)| *name == order).copied() else {
        return Err(pyo3::exceptions::PyValueError::new_err("invalid candidate order"));
    };
    let paths = paths_from(codec);
    let slot: Slot = Arc::new(Mutex::new(None));
    let filled = Arc::clone(&slot);
    let token = stage(Box::new(move |db| {
        run_job(|| {
            // Bound as strip_params binds them, compared with stored values
            // as SQLite's `IN (...)` compares them.
            let among = among.as_ref().map(|(files, ids, names)| {
                let files: HashSet<String> =
                    files.iter().map(|f| paths.strip(f).into_owned()).collect();
                let ids: HashSet<i64> = ids.iter().copied().collect();
                let names: HashSet<String> =
                    names.iter().map(|n| paths.strip(n).into_owned()).collect();
                (files, ids, names)
            });
            let rows = Rows::new(
                db,
                &format!(
                    "SELECT id, kind, name, qualified_name, file_path, language, is_test, \
                     extra FROM nodes WHERE kind IN ('Function', 'Test') ORDER BY {order_sql}"
                ),
                &paths,
            )?;
            let mut candidates: Vec<Candidate> = Vec::new();
            // Stored qualified name -> position in candidates.
            let mut at: HashMap<String, usize> = HashMap::default();
            while rows.step()? {
                let id = rows.int(0)?;
                let stored_qn = rows.stored(3)?.ok_or(Fail::Back)?.to_owned();
                if let Some((files, ids, names)) = &among {
                    let stored_file = rows.stored(4)?.ok_or(Fail::Back)?;
                    if !(files.contains(stored_file)
                        || ids.contains(&id)
                        || names.contains(&stored_qn))
                    {
                        continue;
                    }
                }
                at.insert(stored_qn, candidates.len());
                candidates.push((
                    id,
                    rows.req(1)?,
                    rows.req(2)?,
                    rows.req(3)?,
                    rows.req(4)?,
                    rows.text(5)?,
                    rows.truthy(6)?,
                    rows.text(7)?,
                    false,
                    false,
                ));
            }
            if !candidates.is_empty() {
                // n.kind IS NULL OR n.kind != 'File', and the same with
                // n.is_test = 0, per stored qualified name. SQLite works out
                // both, so NULL compares as it does in the JOIN.
                let callers = Rows::new(
                    db,
                    "SELECT qualified_name, kind IS NULL OR kind != 'File', \
                     kind IS NULL OR (kind != 'File' AND is_test = 0) FROM nodes",
                    &paths,
                )?;
                let mut caller_ok: HashMap<String, (bool, bool)> = HashMap::default();
                while callers.step()? {
                    let qn = callers.stored(0)?.ok_or(Fail::Back)?.to_owned();
                    caller_ok.insert(qn, (callers.truthy(1)?, callers.truthy(2)?));
                }
                let calls = Rows::new(
                    db,
                    "SELECT source_qualified, target_qualified FROM edges WHERE kind = 'CALLS'",
                    &paths,
                )?;
                while calls.step()? {
                    let Some(target) = calls.stored(1)? else { continue };
                    let Some(&position) = at.get(target) else { continue };
                    // No matching node: the JOIN's n.kind IS NULL holds.
                    let (any, outside_tests) = match calls.stored(0)? {
                        Some(source) => caller_ok.get(source).copied().unwrap_or((true, true)),
                        None => (true, true),
                    };
                    let candidate = &mut candidates[position];
                    candidate.8 |= any;
                    candidate.9 |= outside_tests;
                }
            }
            let count = candidates.len() as i64;
            *filled.lock().map_err(|_| Fail::Back)? = Some(candidates);
            Ok(count)
        })
    }))?;
    RESULTS.lock().unwrap().get_or_insert_with(HashMap::default).insert(token, slot);
    Ok(token)
}

/// The candidates a finished `stage_entry_scan` job read; None when the job
/// did not finish.
#[pyfunction]
pub fn take_entry_scan(token: i64) -> Option<Vec<Candidate>> {
    let slot = RESULTS.lock().unwrap().as_mut().and_then(|all| all.remove(&token))?;
    let taken = slot.lock().ok()?.take();
    taken
}
