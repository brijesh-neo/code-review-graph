//! `detect_communities` + `store_communities` for the Leiden detector, with
//! every step but igraph's Leiden itself done natively.
//!
//! Python's igraph still runs each Leiden call, seeded as before, on the
//! graph this module hands it: the same vertices (the rows
//! `get_all_nodes(exclude_files=True)` reads, in that order), the same edge
//! list and weights. Everything around those calls is `_detect_leiden`,
//! `_reassign_test_nodes`, `_split_oversized`, `_dedupe_community_names`
//! and `store_communities`: read here once, kept in a state between the
//! steps, and written on Python's connection. Anything the file-based job
//! does not read the way Python does (a non-ASCII name) hands back before
//! the first Leiden call.

use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use libsqlite3_sys as ffi;
use pyo3::prelude::*;

use crate::communities::{
    Counts, Node, community_name, extract_keywords, naming_members, read_nodes, round4, to_slug,
};
use crate::pathcodec::Paths;
use crate::resolve::{CodecFields, Fail, HashMap, HashSet, R, Rows, paths_from, run_job};
use crate::sqlite_ext::{Stmt, stage};

/// One edge row as the Leiden steps read it.
struct Edge {
    weight: f64,
    tested_by: bool,
    /// The node whose qualified name the endpoint is.
    source: Option<usize>,
    target: Option<usize>,
    /// For TESTED_BY: the node `names_to_qns` maps the endpoint to, if any.
    source_by_name: Option<usize>,
    target_by_name: Option<usize>,
}

struct Community {
    id: Option<usize>,
    name: String,
    level: i64,
    cohesion: f64,
    size: usize,
    dominant_language: String,
    description: String,
    members: Vec<usize>,
}

struct State {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    communities: Vec<Community>,
}

static STATES: Mutex<Option<HashMap<i64, State>>> = Mutex::new(None);
static NEXT_STATE: AtomicI64 = AtomicI64::new(1);

fn with_state<T>(id: i64, f: impl FnOnce(&mut State) -> Option<T>) -> Option<T> {
    let mut guard = STATES.lock().unwrap();
    let state = guard.as_mut()?.get_mut(&id)?;
    f(state)
}

fn read(db: *mut ffi::sqlite3, paths: &Paths, weights: &HashMap<String, f64>, default: f64) -> R<State> {
    let nodes = read_nodes(db, paths)?;
    let mut by_qn: HashMap<&str, usize> = HashMap::default();
    // names_to_qns: a name held by two nodes maps to nothing.
    let mut by_name: HashMap<&str, Option<usize>> = HashMap::default();
    for (at, node) in nodes.iter().enumerate() {
        if by_qn.insert(node.qualified_name.as_str(), at).is_some() {
            return Err(Fail::Back);
        }
        by_name
            .entry(node.name.as_str())
            .and_modify(|slot| *slot = None)
            .or_insert(Some(at));
    }
    let mut edges = Vec::new();
    let rows = Rows::new(db, "SELECT kind, source_qualified, target_qualified FROM edges", paths)?;
    while rows.step()? {
        let kind = rows.stored(0)?.map(|raw| paths.restore(raw).into_owned());
        let weight = kind.as_ref().and_then(|k| weights.get(k).copied()).unwrap_or(default);
        let tested_by = kind.as_deref() == Some("TESTED_BY");
        let mut ends = [(None, None), (None, None)];
        for (slot, column) in ends.iter_mut().zip([1, 2]) {
            if let Some(raw) = rows.stored(column)? {
                let endpoint = paths.restore(raw);
                slot.0 = by_qn.get(endpoint.as_ref()).copied();
                if tested_by {
                    slot.1 = by_name.get(endpoint.as_ref()).copied().flatten();
                }
            }
        }
        edges.push(Edge {
            weight,
            tested_by,
            source: ends[0].0,
            target: ends[1].0,
            source_by_name: ends[0].1,
            target_by_name: ends[1].1,
        });
    }
    drop(by_qn);
    drop(by_name);
    Ok(State { nodes, edges, communities: Vec::new() })
}

/// Stages reading the graph into a Leiden state. Returns the job token and
/// the state id; the job returns 0, or -1 when Python must run instead.
#[pyfunction]
pub fn stage_leiden_read(
    weights: Vec<(String, f64)>,
    default_weight: f64,
    codec: CodecFields,
) -> PyResult<(i64, i64)> {
    let paths = paths_from(codec);
    let weights: HashMap<String, f64> = weights.into_iter().collect();
    let id = NEXT_STATE.fetch_add(1, Ordering::Relaxed);
    let token = stage(Box::new(move |db| {
        run_job(|| {
            let state = read(db, &paths, &weights, default_weight)?;
            STATES.lock().unwrap().get_or_insert_with(HashMap::default).insert(id, state);
            Ok(0)
        })
    }))?;
    Ok((token, id))
}

/// Drops a state; a missing one is ignored.
#[pyfunction]
pub fn discard_leiden(id: i64) {
    if let Some(states) = STATES.lock().unwrap().as_mut() {
        states.remove(&id);
    }
}

/// `_detect_leiden`'s graph: vertex count, deduplicated edges and weights.
/// None when it has no vertex or no edge (Python's fallbacks run then).
#[pyfunction]
pub fn leiden_graph(id: i64) -> Option<(usize, Vec<(usize, usize)>, Vec<f64>)> {
    with_state(id, |state| {
        let mut seen: HashSet<(usize, usize)> = HashSet::default();
        let mut pairs = Vec::new();
        let mut weights = Vec::new();
        for edge in &state.edges {
            if let (Some(s), Some(t)) = (edge.source, edge.target) {
                if s != t {
                    let pair = (s.min(t), s.max(t));
                    if seen.insert(pair) {
                        pairs.push(pair);
                        weights.push(edge.weight);
                    }
                }
            }
        }
        if state.nodes.is_empty() || pairs.is_empty() {
            return None;
        }
        Some((state.nodes.len(), pairs, weights))
    })
}

/// `_reassign_test_nodes` over local vertices: `local_of` maps a node to its
/// vertex, `node_of` a vertex to its node.
fn reassign(
    clusters: &[Vec<usize>],
    state: &State,
    local_of: &dyn Fn(usize) -> Option<usize>,
    node_of: &dyn Fn(usize) -> usize,
    vertex_count: usize,
) -> Vec<Vec<usize>> {
    let mut cluster_of: Vec<Option<usize>> = vec![None; vertex_count];
    for (cluster_id, cluster) in clusters.iter().enumerate() {
        for &vertex in cluster {
            cluster_of[vertex] = Some(cluster_id);
        }
    }
    let resolve = |exact: Option<usize>, by_name: Option<usize>| -> Option<usize> {
        if let Some(local) = exact.and_then(local_of) {
            return Some(local);
        }
        by_name.and_then(local_of)
    };
    let mut subjects: HashMap<usize, Vec<usize>> = HashMap::default();
    for edge in &state.edges {
        if !edge.tested_by {
            continue;
        }
        let (Some(source), Some(target)) = (
            resolve(edge.source, edge.source_by_name),
            resolve(edge.target, edge.target_by_name),
        ) else {
            continue;
        };
        let source_is_test = state.nodes[node_of(source)].is_test_node();
        let target_is_test = state.nodes[node_of(target)].is_test_node();
        if source_is_test == target_is_test {
            continue;
        }
        let (test, subject) = if source_is_test { (source, target) } else { (target, source) };
        let list = subjects.entry(test).or_default();
        if !list.contains(&subject) {
            list.push(subject);
        }
    }
    let mut target_of: HashMap<usize, usize> = HashMap::default();
    for (test, list) in &subjects {
        let Some(current) = cluster_of[*test] else {
            continue;
        };
        let mut votes: HashMap<usize, usize> = HashMap::default();
        for &subject in list {
            if let Some(cluster) = cluster_of[subject] {
                *votes.entry(cluster).or_default() += 1;
            }
        }
        let Some(&highest) = votes.values().max() else {
            continue;
        };
        let tied: Vec<usize> =
            votes.iter().filter(|entry| *entry.1 == highest).map(|entry| *entry.0).collect();
        let target = if tied.contains(&current) {
            current
        } else {
            *tied.iter().min().unwrap()
        };
        target_of.insert(*test, target);
    }
    let mut reassigned: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
    for (current, cluster) in clusters.iter().enumerate() {
        for &vertex in cluster {
            let target = target_of.get(&vertex).copied().unwrap_or(current);
            reassigned[target].push(vertex);
        }
    }
    reassigned
}

/// `_compute_cohesion_batch`: the last group holding a node wins.
fn cohesions(groups: &[Vec<usize>], state: &State) -> Vec<f64> {
    let mut group_of: Vec<Option<usize>> = vec![None; state.nodes.len()];
    for (at, members) in groups.iter().enumerate() {
        for &m in members {
            group_of[m] = Some(at);
        }
    }
    let mut internal = vec![0u64; groups.len()];
    let mut external = vec![0u64; groups.len()];
    for edge in &state.edges {
        let sc = edge.source.and_then(|s| group_of[s]);
        let tc = edge.target.and_then(|t| group_of[t]);
        match (sc, tc) {
            (None, None) => {}
            (Some(s), Some(t)) if s == t => internal[s] += 1,
            _ => {
                if let Some(s) = sc {
                    external[s] += 1;
                }
                if let Some(t) = tc {
                    external[t] += 1;
                }
            }
        }
    }
    (0..groups.len())
        .map(|i| {
            let total = internal[i] + external[i];
            if total > 0 { internal[i] as f64 / total as f64 } else { 0.0 }
        })
        .collect()
}

fn dominant_language(members: &[&Node]) -> String {
    let mut languages = Counts::new();
    for member in members {
        if !member.language.is_empty() {
            languages.add(&member.language);
        }
    }
    languages.top().map(|(l, _)| l.to_string()).unwrap_or_default()
}

/// `_split_oversized`'s threshold.
fn split_threshold(communities: &[Community]) -> Option<usize> {
    let total: usize = communities.iter().map(|c| c.size).sum();
    if total == 0 { None } else { Some((total / 4).max(10)) }
}

/// The subgraph `_split_oversized` runs Leiden on: its nodes, edges, weights.
fn split_subgraph(community: &Community, state: &State) -> (Vec<usize>, Vec<(usize, usize)>, Vec<f64>) {
    let mut members = community.members.clone();
    members.sort_unstable();
    members.dedup();
    let mut local: Vec<Option<usize>> = vec![None; state.nodes.len()];
    for (at, &m) in members.iter().enumerate() {
        local[m] = Some(at);
    }
    let mut weights_by_pair: HashMap<(usize, usize), f64> = HashMap::default();
    for edge in &state.edges {
        let (Some(s), Some(t)) = (edge.source, edge.target) else {
            continue;
        };
        let (Some(si), Some(ti)) = (local[s], local[t]) else {
            continue;
        };
        if si != ti {
            let pair = (si.min(ti), si.max(ti));
            let slot = weights_by_pair.entry(pair).or_insert(0.0);
            if edge.weight > *slot {
                *slot = edge.weight;
            }
        }
    }
    let mut pairs: Vec<(usize, usize)> = weights_by_pair.keys().copied().collect();
    pairs.sort_unstable();
    let weights = pairs.iter().map(|p| weights_by_pair[p]).collect();
    (members, pairs, weights)
}

/// Builds `_detect_leiden`'s communities from Leiden's *clusters* and
/// returns, for each community `_split_oversized` runs Leiden on, its
/// position, vertex count, edges and weights. None for a bad cluster list.
#[pyfunction]
pub fn leiden_communities(
    id: i64,
    clusters: Vec<Vec<usize>>,
    min_size: usize,
) -> Option<Vec<(usize, usize, Vec<(usize, usize)>, Vec<f64>)>> {
    with_state(id, |state| {
        let count = state.nodes.len();
        if clusters.iter().flatten().any(|&v| v >= count) {
            return None;
        }
        let clusters = reassign(&clusters, state, &|n| Some(n), &|v| v, count);
        let pending: Vec<Vec<usize>> =
            clusters.into_iter().filter(|c| c.len() >= min_size).collect();
        let scores = cohesions(&pending, state);
        let mut communities = Vec::with_capacity(pending.len());
        for (members, cohesion) in pending.into_iter().zip(scores) {
            let member_nodes: Vec<&Node> = members.iter().map(|&m| &state.nodes[m]).collect();
            communities.push(Community {
                id: None,
                name: community_name(&member_nodes),
                level: 0,
                cohesion: round4(cohesion),
                size: members.len(),
                dominant_language: dominant_language(&member_nodes),
                description: format!("Community of {} nodes", members.len()),
                members,
            });
        }
        let mut splits = Vec::new();
        if let Some(threshold) = split_threshold(&communities) {
            for (position, community) in communities.iter().enumerate() {
                let distinct: HashSet<usize> = community.members.iter().copied().collect();
                if distinct.len() <= threshold || distinct.len() < 10 {
                    continue;
                }
                let (members, pairs, weights) = split_subgraph(community, state);
                if pairs.is_empty() {
                    continue;
                }
                splits.push((position, members.len(), pairs, weights));
            }
        }
        state.communities = communities;
        Some(splits)
    })
}

/// `_split_oversized` from the Leiden *splits*, then `_dedupe_community_names`.
fn finish(state: &mut State, splits: Vec<(usize, Option<Vec<Vec<usize>>>)>) -> R<Vec<Community>> {
    let mut by_position: HashMap<usize, Option<Vec<Vec<usize>>>> = splits.into_iter().collect();
    let communities = std::mem::take(&mut state.communities);
    let mut result: Vec<Community> = Vec::with_capacity(communities.len());
    let mut next_id = 1;
    for (position, community) in communities.into_iter().enumerate() {
        let Some(Some(clusters)) = by_position.remove(&position) else {
            result.push(community);
            continue;
        };
        let (members, _, _) = split_subgraph(&community, state);
        if clusters.iter().flatten().any(|&v| v >= members.len()) {
            return Err(Fail::Back);
        }
        let mut local: Vec<Option<usize>> = vec![None; state.nodes.len()];
        for (at, &m) in members.iter().enumerate() {
            local[m] = Some(at);
        }
        let reassigned = reassign(&clusters, state, &|n| local[n], &|v| members[v], members.len());
        let subs: Vec<Vec<usize>> = reassigned
            .into_iter()
            .filter(|c| !c.is_empty())
            .map(|c| c.into_iter().map(|v| members[v]).collect())
            .collect();
        if subs.len() <= 1 {
            result.push(community);
            continue;
        }
        let scores = cohesions(&subs, state);
        for (sub, cohesion) in subs.into_iter().zip(scores) {
            let sub_nodes: Vec<&Node> = sub.iter().map(|&m| &state.nodes[m]).collect();
            let mut name = community_name(&sub_nodes);
            if matches!(name.as_str(), "" | "empty" | "cluster") {
                name = format!("{}-{}", community.name, next_id);
            }
            let size = sub.iter().copied().collect::<HashSet<usize>>().len();
            result.push(Community {
                id: Some(next_id),
                name,
                level: community.level + 1,
                cohesion,
                size,
                dominant_language: community.dominant_language.clone(),
                description: format!("Split from {}", community.name),
                members: sub,
            });
            next_id += 1;
        }
    }
    dedupe(&mut result, &state.nodes);
    Ok(result)
}

/// `_dedupe_community_names`, ordering duplicates by size, then id (or position).
fn dedupe(communities: &mut [Community], nodes: &[Node]) {
    let mut by_name: Vec<(String, Vec<usize>)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::default();
    for (position, community) in communities.iter().enumerate() {
        match index.get(&community.name) {
            Some(&slot) => by_name[slot].1.push(position),
            None => {
                index.insert(community.name.clone(), by_name.len());
                by_name.push((community.name.clone(), vec![position]));
            }
        }
    }
    let mut taken: std::collections::HashSet<String> = communities
        .iter()
        .filter(|c| !c.name.is_empty())
        .map(|c| c.name.clone())
        .collect();
    for (base_name, mut duplicates) in by_name {
        if base_name.is_empty() || duplicates.len() <= 1 {
            continue;
        }
        duplicates.sort_by_key(|&p| {
            let c = &communities[p];
            (std::cmp::Reverse(c.size), c.id.unwrap_or(p), p)
        });
        let base_words: Vec<&str> = base_name.split('-').collect();
        for &position in &duplicates[1..] {
            let member_nodes: Vec<&Node> =
                communities[position].members.iter().map(|&m| &nodes[m]).collect();
            let mut candidate_name = String::new();
            for keyword in extract_keywords(&naming_members(&member_nodes)) {
                let suffix = to_slug(&keyword);
                if suffix.is_empty() || base_words.contains(&suffix.as_str()) {
                    continue;
                }
                let candidate = format!("{base_name}-{suffix}");
                if !taken.contains(&candidate) {
                    candidate_name = candidate;
                    break;
                }
            }
            if candidate_name.is_empty() {
                let mut number = 2;
                candidate_name = format!("{base_name}-{number}");
                while taken.contains(&candidate_name) {
                    number += 1;
                    candidate_name = format!("{base_name}-{number}");
                }
            }
            taken.insert(candidate_name.clone());
            communities[position].name = candidate_name;
        }
    }
}

/// `store_communities` for communities the Leiden detector made.
/// Every node's id, stored qualified name and community id, read before
/// anything is written so a value this cannot read fails with nothing written.
pub(crate) fn read_assignments(
    db: *mut ffi::sqlite3,
    paths: &Paths,
) -> R<Vec<(i64, String, Option<i64>)>> {
    let rows = Rows::new(
        db,
        "SELECT id, qualified_name, community_id IS NULL, IFNULL(community_id, 0) FROM nodes",
        paths,
    )?;
    let mut current = Vec::new();
    while rows.step()? {
        let id = rows.int(0)?;
        let qn = rows.stored(1)?.ok_or(Fail::Back)?.to_owned();
        let community = if rows.truthy(2)? { None } else { Some(rows.int(3)?) };
        current.push((id, qn, community));
    }
    Ok(current)
}

/// What `UPDATE nodes SET community_id = NULL` followed by one
/// `UPDATE nodes SET community_id = ? WHERE qualified_name = ?` per member
/// leaves, written only where a node's value changes. *wanted* maps a stored
/// qualified name to its community, the last assignment winning.
pub(crate) fn write_assignments(
    db: *mut ffi::sqlite3,
    current: &[(i64, String, Option<i64>)],
    wanted: &HashMap<String, i64>,
) -> R<()> {
    let update = Stmt::prepare(db, "UPDATE nodes SET community_id = ? WHERE id = ?")?;
    for (id, qn, community) in current {
        let target = wanted.get(qn).copied();
        if target == *community {
            continue;
        }
        match target {
            Some(value) => update.bind_i64(1, value)?,
            None => update.bind_null(1)?,
        }
        update.bind_i64(2, *id)?;
        update.run()?;
    }
    Ok(())
}

fn store(db: *mut ffi::sqlite3, paths: &Paths, nodes: &[Node], communities: &[Community]) -> R<i64> {
    let current = read_assignments(db, paths)?;
    Stmt::prepare(db, "DELETE FROM communities")?.run()?;
    let mut wanted: HashMap<String, i64> = HashMap::default();
    let insert = Stmt::prepare(
        db,
        "INSERT INTO communities \
         (name, level, cohesion, size, dominant_language, description) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )?;
    for community in communities {
        insert.bind_text(1, &paths.strip(&community.name))?;
        insert.bind_i64(2, community.level)?;
        insert.bind_f64(3, community.cohesion)?;
        insert.bind_i64(4, community.size as i64)?;
        insert.bind_text(5, &paths.strip(&community.dominant_language))?;
        insert.bind_text(6, &paths.strip(&community.description))?;
        insert.run()?;
        let id = unsafe { ffi::sqlite3_last_insert_rowid(db) };
        for &m in &community.members {
            wanted.insert(paths.strip(&nodes[m].qualified_name).into_owned(), id);
        }
    }
    write_assignments(db, &current, &wanted)?;
    // Metadata values are not translated.
    if communities.is_empty() {
        let delete = Stmt::prepare(db, "DELETE FROM metadata WHERE key = ?")?;
        delete.bind_text(1, "community_detector")?;
        delete.run()?;
    } else {
        let record = Stmt::prepare(db, "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)")?;
        record.bind_text(1, "community_detector")?;
        record.bind_text(2, "leiden")?;
        record.run()?;
    }
    Ok(communities.len() as i64)
}

/// Stages splitting, naming and storing the communities of state *id*, with
/// Leiden's clusters for each split position (None: keep it whole). The job
/// returns the number stored, or -1 having written nothing.
#[pyfunction]
pub fn stage_leiden_store(
    id: i64,
    splits: Vec<(usize, Option<Vec<Vec<usize>>>)>,
    codec: CodecFields,
) -> PyResult<i64> {
    let paths = paths_from(codec);
    stage(Box::new(move |db| {
        run_job(|| {
            let Some(mut state) = STATES.lock().unwrap().as_mut().and_then(|s| s.remove(&id))
            else {
                return Err(Fail::Back);
            };
            let communities = finish(&mut state, splits)?;
            store(db, &paths, &state.nodes, &communities)
        })
    }))
}
