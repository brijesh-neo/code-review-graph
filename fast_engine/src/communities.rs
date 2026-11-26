//! `detect_communities` + `store_communities` for the file-based detector
//! (no igraph), as one job on Python's connection.
//!
//! Reads the rows `store.get_all_nodes(exclude_files=True)` and the community
//! edge query read, restored as the codec's row factory restores them, and
//! writes the statements `store_communities` runs with the values
//! `strip_params` strips. Anything Python might treat differently (a
//! non-ASCII symbol name, whose lowercasing and whitespace rules this does
//! not copy, or an unexpected column type) hands back before any write.

use std::borrow::Cow;
use libsqlite3_sys as ffi;
use pyo3::prelude::*;

use crate::pathcodec::{Paths, py_isspace};
use crate::resolve::{CodecFields, Fail, HashMap, R, Rows, paths_from, run_job};
use crate::sqlite_ext::{Stmt, stage};

const SLUG_MAX_LEN: usize = 30;

const COMMON_WORDS: &[&str] = &[
    "get", "set", "self", "init", "new", "create", "update", "delete",
    "add", "remove", "make", "build", "from", "to", "for", "with",
    "the", "and", "test", "main", "run", "do", "is", "has", "on",
    "of", "in", "at", "by", "my", "this", "that", "all", "none",
    "should", "when", "then", "given", "return", "returns", "raise",
    "raises", "expect", "expected", "assert", "tests", "be", "it", "if",
    "not",
];

pub(crate) struct Node {
    pub(crate) qualified_name: String,
    pub(crate) kind: String,
    pub(crate) name: String,
    pub(crate) file_path: String,
    pub(crate) language: String,
    pub(crate) is_test: bool,
}

impl Node {
    pub(crate) fn is_test_node(&self) -> bool {
        self.kind == "Test" || self.is_test
    }
}

/// Keys in first-seen order with their counts, as `Counter` keeps them.
pub(crate) struct Counts<'a> {
    order: Vec<(&'a str, usize)>,
    index: HashMap<&'a str, usize>,
}

impl<'a> Counts<'a> {
    pub(crate) fn new() -> Self {
        Counts { order: Vec::new(), index: HashMap::default() }
    }

    pub(crate) fn add(&mut self, key: &'a str) {
        match self.index.get(key) {
            Some(&at) => self.order[at].1 += 1,
            None => {
                self.index.insert(key, self.order.len());
                self.order.push((key, 1));
            }
        }
    }

    /// `most_common(1)[0]`: the first key with the highest count.
    pub(crate) fn top(&self) -> Option<(&'a str, usize)> {
        let mut best: Option<(&'a str, usize)> = None;
        for &(key, count) in &self.order {
            if best.map_or(true, |(_, top)| count > top) {
                best = Some((key, count));
            }
        }
        best
    }

    /// `most_common(n)` keys: by count, ties in first-seen order.
    fn most_common(&self, n: usize) -> Vec<&'a str> {
        let mut sorted: Vec<&(&'a str, usize)> = self.order.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(n).map(|(key, _)| *key).collect()
    }
}

/// `_split_name` for ASCII text.
fn split_name(name: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut previous: Option<char> = None;
    for c in name.chars() {
        if c == '_' || c == '-' || c == '.' || py_isspace(c) {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
            previous = Some(c);
            continue;
        }
        if c.is_ascii_uppercase() && previous.is_some_and(|p| p.is_ascii_lowercase()) {
            if !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
        }
        word.push(c);
        previous = Some(c);
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// `_to_slug`.
pub(crate) fn to_slug(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut in_gap = false;
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            normalized.push(c);
            in_gap = false;
        } else if !in_gap {
            normalized.push(' ');
            in_gap = true;
        }
    }
    let slug = split_name(&normalized)
        .iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join("-");
    if slug.len() <= SLUG_MAX_LEN {
        return slug;
    }
    match slug[..=SLUG_MAX_LEN].rfind('-') {
        Some(boundary) if boundary > 0 => slug[..boundary].to_string(),
        _ => slug[..SLUG_MAX_LEN].to_string(),
    }
}

/// `_extract_keywords`. Names were checked to be ASCII when read.
pub(crate) fn extract_keywords(members: &[&Node]) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    for member in members {
        if matches!(member.kind.as_str(), "Function" | "Class" | "Test" | "Type") {
            for word in split_name(&member.name) {
                let lower = word.to_ascii_lowercase();
                if !COMMON_WORDS.contains(&lower.as_str()) && lower.len() > 1 {
                    words.push(lower);
                }
            }
        }
    }
    let mut counts = Counts::new();
    for word in &words {
        counts.add(word);
    }
    counts.most_common(5).into_iter().map(str::to_string).collect()
}

/// `_naming_members`.
pub(crate) fn naming_members<'a>(members: &[&'a Node]) -> Vec<&'a Node> {
    let production: Vec<&Node> =
        members.iter().copied().filter(|m| !m.is_test_node()).collect();
    if production.is_empty() { members.to_vec() } else { production }
}

/// `"\\"`-normalised path split on `/`, as `replace("\\", "/").split("/")`.
fn path_segments(path: &str) -> Vec<Cow<'_, str>> {
    if path.contains('\\') {
        path.replace('\\', "/").split('/').map(|s| Cow::Owned(s.to_string())).collect()
    } else {
        path.split('/').map(Cow::Borrowed).collect()
    }
}

/// `rsplit(".", 1)[0]`.
fn stem(segment: &str) -> &str {
    match segment.rfind('.') {
        Some(at) => &segment[..at],
        None => segment,
    }
}

/// `_extract_file_prefix`.
fn file_prefix(members: &[&Node]) -> String {
    if members.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = members
        .iter()
        .map(|m| {
            let segments = path_segments(&m.file_path);
            if segments.len() >= 2 {
                segments[segments.len() - 2].to_string()
            } else {
                stem(&segments[segments.len() - 1]).to_string()
            }
        })
        .collect();
    let mut counts = Counts::new();
    for part in &parts {
        counts.add(part);
    }
    counts.top().map(|(top, _)| to_slug(top)).unwrap_or_default()
}

/// `_generate_community_name`.
pub(crate) fn community_name(members: &[&Node]) -> String {
    if members.is_empty() {
        return "empty".to_string();
    }
    let naming = naming_members(members);
    let prefix = file_prefix(&naming);
    let mut classes = Counts::new();
    for member in &naming {
        if member.kind == "Class" {
            classes.add(&member.name);
        }
    }
    if let Some((top_class, top_count)) = classes.top() {
        if (top_count as f64) > (naming.len() as f64) * 0.4 {
            if !prefix.is_empty() {
                return format!("{}-{}", prefix, to_slug(top_class));
            }
            return to_slug(top_class);
        }
    }
    let keyword = extract_keywords(&naming).into_iter().next().unwrap_or_default();
    match (prefix.is_empty(), keyword.is_empty()) {
        (false, false) => format!("{prefix}-{keyword}"),
        (false, true) => prefix,
        (true, false) => keyword,
        (true, true) => "cluster".to_string(),
    }
}

/// `round(value, 4)`: both round the exact binary value half to even.
pub(crate) fn round4(value: f64) -> f64 {
    format!("{value:.4}").parse().unwrap_or(value)
}

struct Community {
    name: String,
    size: usize,
    cohesion: f64,
    dominant_language: String,
    description: String,
    members: Vec<usize>,
}

pub(crate) fn read_nodes(db: *mut ffi::sqlite3, paths: &Paths) -> R<Vec<Node>> {
    let rows = Rows::new(db, "SELECT * FROM nodes WHERE kind != 'File'", paths)?;
    let qn_at = rows.column_index("qualified_name")?;
    let kind_at = rows.column_index("kind")?;
    let name_at = rows.column_index("name")?;
    let file_at = rows.column_index("file_path")?;
    let language_at = rows.column_index("language")?;
    let test_at = rows.column_index("is_test")?;
    let mut nodes = Vec::new();
    while rows.step()? {
        let name = rows.req(name_at)?;
        if !name.is_ascii() {
            return Err(Fail::Back);
        }
        nodes.push(Node {
            qualified_name: rows.req(qn_at)?,
            kind: rows.req(kind_at)?,
            name,
            file_path: rows.req(file_at)?,
            language: rows.text(language_at)?.unwrap_or_default(),
            is_test: rows.truthy(test_at)?,
        });
    }
    Ok(nodes)
}

/// `_detect_file_based`, then `_dedupe_community_names`.
fn detect(db: *mut ffi::sqlite3, paths: &Paths, nodes: &[Node], min_size: usize) -> R<Vec<Community>> {
    let dir_parts: Vec<Vec<Cow<'_, str>>> = nodes
        .iter()
        .map(|n| {
            let mut segments = path_segments(&n.file_path);
            segments.pop();
            segments.retain(|s| !s.is_empty());
            segments
        })
        .collect();
    let mut prefix_len = 0;
    if let Some(shortest) = dir_parts.iter().map(Vec::len).min() {
        for i in 0..shortest {
            let seg = &dir_parts[0][i];
            if dir_parts.iter().all(|p| &p[i] == seg) {
                prefix_len = i + 1;
            } else {
                break;
            }
        }
    }
    let file_stems: Vec<String> = nodes
        .iter()
        .map(|n| {
            let segments = path_segments(&n.file_path);
            stem(&segments[segments.len() - 1]).to_string()
        })
        .collect();
    let group_at_depth = |depth: usize| -> Vec<(String, Vec<usize>)> {
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::default();
        for (at, parts) in dir_parts.iter().enumerate() {
            let remainder = &parts[prefix_len.min(parts.len())..];
            let key = if remainder.is_empty() {
                file_stems[at].clone()
            } else {
                remainder[..depth.min(remainder.len())].join("/")
            };
            match index.get(&key) {
                Some(&slot) => groups[slot].1.push(at),
                None => {
                    index.insert(key.clone(), groups.len());
                    groups.push((key, vec![at]));
                }
            }
        }
        groups
    };
    let max_depth = dir_parts.iter().map(|p| p.len().saturating_sub(prefix_len)).max().unwrap_or(0);
    let mut best = group_at_depth(1);
    for depth in 1..=max_depth {
        let groups = if depth == 1 { std::mem::take(&mut best) } else { group_at_depth(depth) };
        let qualifying = groups.iter().filter(|(_, m)| m.len() >= min_size).count();
        best = groups;
        if qualifying >= 10 {
            break;
        }
    }
    let pending: Vec<(String, Vec<usize>)> =
        best.into_iter().filter(|(_, m)| m.len() >= min_size).collect();

    // _compute_cohesion_batch: the last community holding a name wins.
    let mut qn_to_idx: HashMap<&str, usize> = HashMap::default();
    for (idx, (_, members)) in pending.iter().enumerate() {
        for &m in members {
            qn_to_idx.insert(nodes[m].qualified_name.as_str(), idx);
        }
    }
    let mut internal = vec![0u64; pending.len()];
    let mut external = vec![0u64; pending.len()];
    if !pending.is_empty() {
        let edges = Rows::new(db, "SELECT kind, source_qualified, target_qualified FROM edges", paths)?;
        while edges.step()? {
            let sc = match edges.stored(1)? {
                Some(raw) => qn_to_idx.get(paths.restore(raw).as_ref()).copied(),
                None => None,
            };
            let tc = match edges.stored(2)? {
                Some(raw) => qn_to_idx.get(paths.restore(raw).as_ref()).copied(),
                None => None,
            };
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
    }

    let mut communities = Vec::with_capacity(pending.len());
    for (i, (dir_path, members)) in pending.into_iter().enumerate() {
        let total = internal[i] + external[i];
        let cohesion = if total > 0 { internal[i] as f64 / total as f64 } else { 0.0 };
        let member_nodes: Vec<&Node> = members.iter().map(|&m| &nodes[m]).collect();
        let mut languages = Counts::new();
        for member in &member_nodes {
            if !member.language.is_empty() {
                languages.add(&member.language);
            }
        }
        let dominant_language = languages.top().map(|(l, _)| l.to_string()).unwrap_or_default();
        communities.push(Community {
            name: community_name(&member_nodes),
            size: members.len(),
            cohesion: round4(cohesion),
            dominant_language,
            description: format!("Directory-based community: {dir_path}"),
            members,
        });
    }
    dedupe_names(&mut communities, nodes);
    Ok(communities)
}

/// `_dedupe_community_names`.
fn dedupe_names(communities: &mut [Community], nodes: &[Node]) {
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
        duplicates.sort_by(|&a, &b| {
            communities[b].size.cmp(&communities[a].size).then(a.cmp(&b))
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

/// `store_communities` for communities the file detector made.
fn store(db: *mut ffi::sqlite3, paths: &Paths, nodes: &[Node], communities: &[Community]) -> R<i64> {
    Stmt::prepare(db, "DELETE FROM communities")?.run()?;
    Stmt::prepare(db, "UPDATE nodes SET community_id = NULL")?.run()?;
    let insert = Stmt::prepare(
        db,
        "INSERT INTO communities \
         (name, level, cohesion, size, dominant_language, description) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )?;
    let assign = Stmt::prepare(db, "UPDATE nodes SET community_id = ? WHERE qualified_name = ?")?;
    for community in communities {
        insert.bind_text(1, &paths.strip(&community.name))?;
        insert.bind_i64(2, 0)?;
        insert.bind_f64(3, community.cohesion)?;
        insert.bind_i64(4, community.size as i64)?;
        insert.bind_text(5, &paths.strip(&community.dominant_language))?;
        insert.bind_text(6, &paths.strip(&community.description))?;
        insert.run()?;
        let id = unsafe { ffi::sqlite3_last_insert_rowid(db) };
        for &m in &community.members {
            assign.bind_i64(1, id)?;
            assign.bind_text(2, &paths.strip(&nodes[m].qualified_name))?;
            assign.run()?;
        }
    }
    // Metadata values are not translated.
    if communities.is_empty() {
        let delete = Stmt::prepare(db, "DELETE FROM metadata WHERE key = ?")?;
        delete.bind_text(1, "community_detector")?;
        delete.run()?;
    } else {
        let record = Stmt::prepare(db, "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)")?;
        record.bind_text(1, "community_detector")?;
        record.bind_text(2, "file")?;
        record.run()?;
    }
    Ok(communities.len() as i64)
}

/// Stages file-based community detection and storage. The job returns the
/// number of communities stored, or -1 having written nothing.
#[pyfunction]
pub fn stage_file_communities(min_size: i64, codec: CodecFields) -> PyResult<i64> {
    let paths = paths_from(codec);
    let min_size = min_size.max(0) as usize;
    stage(Box::new(move |db| {
        run_job(|| {
            let nodes = read_nodes(db, &paths)?;
            let communities = detect(db, &paths, &nodes, min_size)?;
            store(db, &paths, &nodes, &communities)
        })
    }))
}
