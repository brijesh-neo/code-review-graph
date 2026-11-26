//! `_trace_single_flow` and the counts `compute_criticality` takes, as one
//! native pass on the caller's connection.
//!
//! Every lookup reads what the Python adjacency reads (`_LazyNodesByQn`,
//! `_LazyCallsOut`, `_LazyTestedBy`, `_LazyNodesById`): the same SQL, the
//! same row order, values restored as the codec's row factory restores them
//! and names bound as `strip_params` strips them. The float score and the
//! keyword match stay in Python, so the stored criticality comes from the
//! same code. Nothing is written.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::pathcodec::Paths;
use crate::resolve::{CodecFields, Fail, HashMap, HashSet, R, Rows, paths_from, run_job};
use crate::sqlite_ext::stage;

/// The two orders `_CALLS_ORDERS` allows; nothing else reaches the SQL text.
const ORDERS: [&str; 2] = ["target_qualified, file_path, line, id", "id"];

/// A node `compute_criticality` scores: qualified name as the row holds it, name.
type Scored = (String, String);
/// path ids, depth, file_count, then what `compute_criticality` reads: the
/// scored nodes, their file count, external_count, tested_count.
type Traced = (Vec<i64>, i64, i64, Vec<Scored>, i64, i64, i64);

static RESULTS: Mutex<Option<HashMap<i64, Arc<Mutex<Option<Vec<Option<Traced>>>>>>>> =
    Mutex::new(None);

#[derive(Clone)]
struct Node {
    id: i64,
    qualified_name: String,
    name: String,
    file_path: Option<String>,
}

struct Reader<'p> {
    paths: &'p Paths,
    by_qn: Rows<'p>,
    by_id: Rows<'p>,
    calls: Rows<'p>,
    tested: Rows<'p>,
    nodes: HashMap<Option<String>, Option<Node>>,
    calls_out: HashMap<String, Vec<Option<String>>>,
    tested_by: HashMap<String, bool>,
}

impl<'p> Reader<'p> {
    fn node(&mut self, qn: &Option<String>) -> R<Option<Node>> {
        if let Some(found) = self.nodes.get(qn) {
            return Ok(found.clone());
        }
        let found = match qn {
            // ``WHERE qualified_name = NULL`` matches nothing.
            None => None,
            Some(name) => {
                self.by_qn.reset();
                self.by_qn.bind_text(1, &self.paths.strip(name))?;
                if self.by_qn.step()? {
                    Some(Node {
                        id: self.by_qn.int(0)?,
                        qualified_name: self.by_qn.req(1)?,
                        name: self.by_qn.req(2)?,
                        file_path: self.by_qn.text(3)?,
                    })
                } else {
                    None
                }
            }
        };
        self.nodes.insert(qn.clone(), found.clone());
        Ok(found)
    }

    /// `_LazyNodesById.get`: only the entry point needs it, since every other
    /// path node was read by qualified name and ids and qualified names are
    /// both unique.
    fn node_by_id(&mut self, id: i64) -> R<Option<Node>> {
        self.by_id.reset();
        self.by_id.bind_int(1, id)?;
        if !self.by_id.step()? {
            return Ok(None);
        }
        Ok(Some(Node {
            id: self.by_id.int(0)?,
            qualified_name: self.by_id.req(1)?,
            name: self.by_id.req(2)?,
            file_path: self.by_id.text(3)?,
        }))
    }

    fn targets(&mut self, qn: &str) -> R<Vec<Option<String>>> {
        if let Some(found) = self.calls_out.get(qn) {
            return Ok(found.clone());
        }
        self.calls.reset();
        self.calls.bind_text(1, &self.paths.strip(qn))?;
        let mut targets = Vec::new();
        while self.calls.step()? {
            targets.push(self.calls.text(0)?);
        }
        self.calls_out.insert(qn.to_string(), targets.clone());
        Ok(targets)
    }

    fn has_tested_by(&mut self, qn: &str) -> R<bool> {
        if let Some(&found) = self.tested_by.get(qn) {
            return Ok(found);
        }
        self.tested.reset();
        self.tested.bind_text(1, &self.paths.strip(qn))?;
        let found = self.tested.step()?;
        self.tested_by.insert(qn.to_string(), found);
        Ok(found)
    }

    /// `_trace_single_flow` up to the score: None for a trivial flow.
    fn trace(&mut self, entry: &Node, max_depth: i64) -> R<Option<Traced>> {
        let mut path: Vec<Node> = vec![entry.clone()];
        let mut visited: HashSet<String> = HashSet::default();
        visited.insert(entry.qualified_name.clone());
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((entry.qualified_name.clone(), 0i64));
        let mut actual_depth = 0i64;
        while let Some((current, depth)) = queue.pop_front() {
            actual_depth = actual_depth.max(depth);
            if depth >= max_depth {
                continue;
            }
            for target in self.targets(&current)? {
                let Some(name) = &target else {
                    // ``None in visited`` is False and its lookup finds nothing.
                    continue;
                };
                if visited.contains(name) {
                    continue;
                }
                let Some(node) = self.node(&target)? else { continue };
                visited.insert(name.clone());
                path.push(node);
                queue.push_back((name.clone(), depth + 1));
            }
        }
        if path.len() < 2 {
            return Ok(None);
        }
        let ids: Vec<i64> = path.iter().map(|n| n.id).collect();
        // ``files``: each path qualified name looked up by name, the entry
        // point included; one with no row adds nothing.
        let by_name = self.node(&Some(entry.qualified_name.clone()))?;
        let mut files: HashSet<Option<String>> =
            path[1..].iter().map(|n| n.file_path.clone()).collect();
        if let Some(found) = &by_name {
            files.insert(found.file_path.clone());
        }
        // ``compute_criticality``: each path id looked up by id.
        let mut scored_nodes: Vec<Node> = Vec::with_capacity(path.len());
        if let Some(found) = self.node_by_id(entry.id)? {
            scored_nodes.push(found);
        }
        scored_nodes.extend(path.drain(1..));
        let scored_files: HashSet<Option<&String>> =
            scored_nodes.iter().map(|n| n.file_path.as_ref()).collect();
        let mut external = 0i64;
        let mut tested = 0i64;
        let mut scored = Vec::with_capacity(scored_nodes.len());
        for node in &scored_nodes {
            for target in self.targets(&node.qualified_name)? {
                if self.node(&target)?.is_none() {
                    external += 1;
                }
            }
            if self.has_tested_by(&node.qualified_name)? {
                tested += 1;
            }
            scored.push((node.qualified_name.clone(), node.name.clone()));
        }
        Ok(Some((
            ids,
            actual_depth,
            files.len() as i64,
            scored,
            scored_files.len() as i64,
            external,
            tested,
        )))
    }
}

/// Stages tracing one flow from each entry point, given as (id, qualified
/// name, name, file path). The job returns the number traced, or -1 when a
/// row holds a value this pass does not read the way Python does; the flows
/// are then taken with `take_flow_trace`.
#[pyfunction]
pub fn stage_flow_trace(
    entries: Vec<(i64, String, String, Option<String>)>,
    max_depth: i64,
    order: String,
    codec: CodecFields,
) -> PyResult<i64> {
    let Some(order) = ORDERS.iter().find(|allowed| **allowed == order).copied() else {
        return Err(pyo3::exceptions::PyValueError::new_err("invalid calls order"));
    };
    let paths = paths_from(codec);
    let slot: Arc<Mutex<Option<Vec<Option<Traced>>>>> = Arc::new(Mutex::new(None));
    let filled = Arc::clone(&slot);
    let token = stage(Box::new(move |db| {
        run_job(|| {
            let calls_sql = format!(
                "SELECT target_qualified FROM edges \
                 WHERE kind = 'CALLS' AND source_qualified = ? ORDER BY {order}"
            );
            let mut reader = Reader {
                paths: &paths,
                by_qn: Rows::new(
                    db,
                    "SELECT id, qualified_name, name, file_path FROM nodes \
                     WHERE qualified_name = ?",
                    &paths,
                )?,
                by_id: Rows::new(
                    db,
                    "SELECT id, qualified_name, name, file_path FROM nodes WHERE id = ?",
                    &paths,
                )?,
                calls: Rows::new(db, &calls_sql, &paths)?,
                tested: Rows::new(
                    db,
                    "SELECT 1 FROM edges WHERE kind = 'TESTED_BY' \
                     AND source_qualified = ? LIMIT 1",
                    &paths,
                )?,
                nodes: HashMap::default(),
                calls_out: HashMap::default(),
                tested_by: HashMap::default(),
            };
            let mut flows = Vec::with_capacity(entries.len());
            for (id, qualified_name, name, file_path) in &entries {
                let entry = Node {
                    id: *id,
                    qualified_name: qualified_name.clone(),
                    name: name.clone(),
                    file_path: file_path.clone(),
                };
                flows.push(reader.trace(&entry, max_depth)?);
            }
            let count = flows.len() as i64;
            *filled.lock().map_err(|_| Fail::Back)? = Some(flows);
            Ok(count)
        })
    }))?;
    RESULTS.lock().unwrap().get_or_insert_with(HashMap::default).insert(token, slot);
    Ok(token)
}

/// The flows a finished `stage_flow_trace` job traced, one per entry point
/// (None for a trivial flow); None when the job did not finish.
#[pyfunction]
pub fn take_flow_trace(token: i64) -> Option<Vec<Option<Traced>>> {
    let slot = RESULTS.lock().unwrap().as_mut().and_then(|all| all.remove(&token))?;
    let taken = slot.lock().ok()?.take();
    taken
}
