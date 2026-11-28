//! `GraphStore._resolve_bare_endpoints` as one native pass on Python's own
//! connection (`CRG_FAST_ENGINE=1`).
//!
//! The pass reads the same rows with the same SQL, in the same order, and
//! sees each value as the codec's row factory hands it to Python. It builds
//! the evidence `_get_bare_endpoint_context` builds and the
//! `inheritance.InheritanceIndex`, decides every edge as the Python loop
//! does, and writes the same `UPDATE`s with each value stripped as
//! `strip_params` strips it. Anything Python would read differently, or
//! raise on, makes the job return -1 before it writes: the caller then runs
//! the Python pass.

use std::cell::RefCell;
use std::collections::{HashMap as StdMap, HashSet as StdSet};
use std::ffi::c_int;
use std::hash::{BuildHasherDefault, Hasher};

use libsqlite3_sys as ffi;
use pyo3::prelude::*;

use crate::pathcodec::{Codec, Paths, py_isspace};
use crate::pyjson::{self, Dict, Value};
use crate::sqlite_ext::{Stmt, errmsg, stage};

const FALL_BACK: i64 = -1;

/// FxHash (rustc's): SipHash dominated the pass, and no input here is
/// adversarial. Nothing reads these maps in iteration order where it shows.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        const K: u64 = 0xf135_7aea_2e62_a9c5;
        let mut hash = self.0;
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let word = u64::from_le_bytes(chunk.try_into().unwrap());
            hash = (hash.rotate_left(5) ^ word).wrapping_mul(K);
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            hash = (hash.rotate_left(5) ^ u64::from_le_bytes(word)).wrapping_mul(K);
        }
        self.0 = hash;
    }

    fn write_u8(&mut self, value: u8) {
        self.0 = (self.0.rotate_left(5) ^ value as u64).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }

    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

pub type HashMap<K, V> = StdMap<K, V, BuildHasherDefault<Fx>>;
pub type HashSet<T> = StdSet<T, BuildHasherDefault<Fx>>;

fn set_of(item: String) -> HashSet<String> {
    let mut set = HashSet::default();
    set.insert(item);
    set
}

const SELF_RECEIVERS: &[&str] = &["this", "$this", "self"];
const SUPER_RECEIVERS: &[&str] = &["super", "base"];
const TYPED_RECEIVER_LANGUAGES: &[&str] = &["csharp", "java", "kotlin", "tsx", "typescript"];
const UNTYPED_RECEIVER_LANGUAGES: &[&str] = &["rust"];
const SINGLE_SUPERCLASS: &[&str] = &["java", "csharp", "typescript", "tsx", "javascript", "php"];
const MAX_DEPTH: usize = 32;
const IMPORT_SCOPE_KEY: &str = "import_scope";
const IMPORT_SCOPE_PACKAGE: &str = "package";
const IMPORT_SCOPE_TREE: &str = "tree";
const IMPORT_SCOPE_MAX_ANCESTORS: usize = 12;
const CLAIMED_ENDPOINT_KEY: &str = "claimed_endpoint";
const INHERITED_ENDPOINT_KEY: &str = "inherited_endpoint";
const INHERITED_THROUGH_KEY: &str = "inherited_through";

/// Why the pass stops: Python would see something else, or SQLite failed.
pub enum Fail {
    Back,
    Sql(String),
}

impl From<String> for Fail {
    fn from(message: String) -> Fail {
        Fail::Sql(message)
    }
}

pub type R<T> = Result<T, Fail>;

/// One SELECT, its rows read as the codec's row factory restores them.
pub struct Rows<'p> {
    stmt: Stmt,
    db: *mut ffi::sqlite3,
    paths: &'p Paths,
}

impl<'p> Rows<'p> {
    pub fn new(db: *mut ffi::sqlite3, sql: &str, paths: &'p Paths) -> R<Rows<'p>> {
        Ok(Rows { stmt: Stmt::prepare(db, sql)?, db, paths })
    }

    pub fn bind_text(&self, index: c_int, value: &str) -> R<()> {
        Ok(self.stmt.bind_text(index, value)?)
    }

    pub fn bind_int(&self, index: c_int, value: i64) -> R<()> {
        Ok(self.stmt.bind_i64(index, value)?)
    }

    /// Ready the statement to run again with new bindings.
    pub fn reset(&self) {
        unsafe {
            ffi::sqlite3_reset(self.stmt.raw);
        }
    }

    pub fn step(&self) -> R<bool> {
        match unsafe { ffi::sqlite3_step(self.stmt.raw) } {
            ffi::SQLITE_ROW => Ok(true),
            ffi::SQLITE_DONE => Ok(false),
            _ => Err(Fail::Sql(errmsg(self.db))),
        }
    }

    /// A TEXT or NULL column, restored; any other type hands back.
    pub fn text(&self, index: c_int) -> R<Option<String>> {
        unsafe {
            match ffi::sqlite3_column_type(self.stmt.raw, index) {
                ffi::SQLITE_NULL => Ok(None),
                ffi::SQLITE_TEXT => {
                    let ptr = ffi::sqlite3_column_text(self.stmt.raw, index);
                    let len = ffi::sqlite3_column_bytes(self.stmt.raw, index) as usize;
                    if len == 0 {
                        return Ok(Some(String::new()));
                    }
                    let bytes = std::slice::from_raw_parts(ptr, len);
                    let text = std::str::from_utf8(bytes).map_err(|_| Fail::Back)?;
                    Ok(Some(self.paths.restore(text).into_owned()))
                }
                _ => Err(Fail::Back),
            }
        }
    }

    /// The index of the result column called *name*.
    pub fn column_index(&self, name: &str) -> R<c_int> {
        unsafe {
            let count = ffi::sqlite3_column_count(self.stmt.raw);
            for index in 0..count {
                let ptr = ffi::sqlite3_column_name(self.stmt.raw, index);
                if !ptr.is_null() && std::ffi::CStr::from_ptr(ptr).to_bytes() == name.as_bytes() {
                    return Ok(index);
                }
            }
        }
        Err(Fail::Back)
    }

    /// A TEXT or NULL column as stored, not restored; valid until the next step.
    pub fn stored(&self, index: c_int) -> R<Option<&str>> {
        unsafe {
            match ffi::sqlite3_column_type(self.stmt.raw, index) {
                ffi::SQLITE_NULL => Ok(None),
                ffi::SQLITE_TEXT => {
                    let ptr = ffi::sqlite3_column_text(self.stmt.raw, index);
                    let len = ffi::sqlite3_column_bytes(self.stmt.raw, index) as usize;
                    if len == 0 {
                        return Ok(Some(""));
                    }
                    let bytes = std::slice::from_raw_parts(ptr, len);
                    std::str::from_utf8(bytes).map(Some).map_err(|_| Fail::Back)
                }
                _ => Err(Fail::Back),
            }
        }
    }

    /// `bool(row[index])` for a NULL, INTEGER or REAL column.
    pub fn truthy(&self, index: c_int) -> R<bool> {
        unsafe {
            match ffi::sqlite3_column_type(self.stmt.raw, index) {
                ffi::SQLITE_NULL => Ok(false),
                ffi::SQLITE_INTEGER => Ok(ffi::sqlite3_column_int64(self.stmt.raw, index) != 0),
                ffi::SQLITE_FLOAT => Ok(ffi::sqlite3_column_double(self.stmt.raw, index) != 0.0),
                _ => Err(Fail::Back),
            }
        }
    }

    /// A column that must hold text.
    pub fn req(&self, index: c_int) -> R<String> {
        self.text(index)?.ok_or(Fail::Back)
    }

    pub fn int(&self, index: c_int) -> R<i64> {
        unsafe {
            if ffi::sqlite3_column_type(self.stmt.raw, index) != ffi::SQLITE_INTEGER {
                return Err(Fail::Back);
            }
            Ok(ffi::sqlite3_column_int64(self.stmt.raw, index))
        }
    }
}

/// `json.loads(raw)`, handing back where this parser and Python's may differ.
fn loads(raw: &str) -> R<Value> {
    pyjson::parse(raw).ok_or(Fail::Back)
}

/// `_load_extra` / `_load_edge_extra`: a dict, or empty.
pub fn load_dict(raw: Option<&str>) -> R<Dict> {
    let text = match raw {
        Some(text) if !text.is_empty() => text,
        _ => "{}",
    };
    match loads(text)? {
        Value::Dict(dict) => Ok(dict),
        _ => Ok(Dict::default()),
    }
}

/// `value in frozenset(...)`: a list or dict raises TypeError in Python.
fn member(value: Option<&Value>, set: &[&str]) -> R<bool> {
    match value {
        Some(Value::List(_) | Value::Dict(_)) => Err(Fail::Back),
        Some(Value::Str(text)) => Ok(set.contains(&text.as_str())),
        _ => Ok(false),
    }
}

fn rpartition_head<'a>(text: &'a str, sep: char) -> &'a str {
    text.rsplit_once(sep).map_or("", |(head, _)| head)
}

fn rpartition_tail<'a>(text: &'a str, sep: char) -> &'a str {
    text.rsplit_once(sep).map_or(text, |(_, tail)| tail)
}

/// `PurePosixPath(p).is_absolute() or PureWindowsPath(p).is_absolute()`.
fn is_absolute_path(path: &str) -> R<bool> {
    if path.starts_with('/') {
        return Ok(true);
    }
    if path.starts_with('\\') {
        // A UNC share or a rooted path: not worth spelling out here.
        return Err(Fail::Back);
    }
    let mut chars = path.chars();
    let (_, second, third) = (chars.next(), chars.next(), chars.next());
    Ok(second == Some(':') && matches!(third, Some('\\' | '/')))
}

/// `_parent_dir`.
fn parent_dir(path: &str) -> Option<&str> {
    let (head, _) = path.rsplit_once('/')?;
    (!head.is_empty()).then_some(head)
}

/// `import_scope_ancestors`: (directory, is the file's own directory).
fn import_scope_ancestors(file_path: &str) -> Vec<(&str, bool)> {
    let Some(parent) = parent_dir(file_path) else { return Vec::new() };
    let mut out = vec![(parent, true)];
    let mut current = parent;
    for _ in 0..IMPORT_SCOPE_MAX_ANCESTORS {
        match parent_dir(current) {
            Some(up) => {
                out.push((up, false));
                current = up;
            }
            None => break,
        }
    }
    out
}

/// The dotted module names `_python_module_file_index` derives from a path.
fn module_names(file_path: &str) -> R<Vec<String>> {
    let mut segments: Vec<&str> =
        file_path.split('/').filter(|part| !part.is_empty() && *part != ".").collect();
    let Some(last) = segments.pop() else { return Ok(Vec::new()) };
    // `with_suffix("")`.
    let stem = match last.rfind('.') {
        Some(i) if i > 0 && i + 1 < last.len() => &last[..i],
        _ => last,
    };
    if stem.is_empty() || stem == "." || stem == ".." {
        return Err(Fail::Back);
    }
    if stem != "__init__" {
        segments.push(stem);
    }
    Ok((0..segments.len()).map(|start| segments[start..].join(".")).collect())
}

// -- Evidence ---------------------------------------------------------------

pub struct Context {
    /// bare name -> [(qualified name, defining file)], in row order.
    node_lookup: HashMap<String, Vec<(String, String)>>,
    parent_lookup: HashMap<String, Option<String>>,
    import_targets: HashMap<String, HashSet<String>>,
    ambiguous_imports: HashMap<String, HashSet<String>>,
    ruby_nodes: HashSet<String>,
    language_lookup: HashMap<String, String>,
    raw_imports: HashMap<String, HashMap<String, String>>,
}

impl Context {
    pub fn build(db: *mut ffi::sqlite3, paths: &Paths) -> R<Context> {
        let mut node_lookup: HashMap<String, Vec<(String, String)>> = HashMap::default();
        let mut parent_lookup = HashMap::default();
        let mut ruby_nodes = HashSet::default();
        let mut language_lookup = HashMap::default();
        let rows = Rows::new(
            db,
            "SELECT name, qualified_name, file_path, parent_name, kind, language \
             FROM nodes WHERE kind IN ('Function', 'Test', 'Class')",
            paths,
        )?;
        while rows.step()? {
            let name = rows.req(0)?;
            let qualified = rows.req(1)?;
            let file_path = rows.req(2)?;
            let parent = rows.text(3)?;
            let kind = rows.req(4)?;
            let language = rows.text(5)?;
            node_lookup.entry(name).or_default().push((qualified.clone(), file_path));
            if language.as_deref() == Some("ruby") {
                ruby_nodes.insert(qualified.clone());
            }
            if kind != "Class" {
                language_lookup.insert(qualified.clone(), language.unwrap_or_default());
            }
            parent_lookup.insert(qualified, parent);
        }
        let (import_targets, ambiguous_imports, raw_imports) = import_evidence(db, paths)?;
        Ok(Context {
            node_lookup,
            parent_lookup,
            import_targets,
            ambiguous_imports,
            ruby_nodes,
            language_lookup,
            raw_imports,
        })
    }

    /// `parent_lookup.get(q)`, None for a missing key and for NULL alike.
    fn parent(&self, qualified: &str) -> Option<&str> {
        self.parent_lookup.get(qualified).and_then(|p| p.as_deref())
    }
}

type Evidence = (
    HashMap<String, HashSet<String>>,
    HashMap<String, HashSet<String>>,
    HashMap<String, HashMap<String, String>>,
);

/// `GraphStore._import_evidence`.
fn import_evidence(db: *mut ffi::sqlite3, paths: &Paths) -> R<Evidence> {
    let mut import_targets: HashMap<String, HashSet<String>> = HashMap::default();
    let mut ambiguous: HashMap<String, HashSet<String>> = HashMap::default();
    let mut scoped_dirs: HashMap<String, String> = HashMap::default();
    let mut raw_imports: HashMap<String, HashMap<String, String>> = HashMap::default();
    let rows = Rows::new(
        db,
        "SELECT DISTINCT file_path, target_qualified, extra FROM edges \
         WHERE kind = 'IMPORTS_FROM'",
        paths,
    )?;
    while rows.step()? {
        let file_path = rows.req(0)?;
        let target = rows.req(1)?;
        let extra = rows.text(2)?;
        let target_file = target.split_once("::").map_or(target.as_str(), |(head, _)| head);
        import_targets.entry(file_path.clone()).or_default().insert(target_file.to_string());
        if !is_absolute_path(target_file)? {
            let short = rpartition_tail(rpartition_tail(target_file, '\\'), '.');
            if !short.is_empty() {
                raw_imports
                    .entry(file_path.clone())
                    .or_default()
                    .insert(short.to_string(), target_file.to_string());
            }
        }
        let Some(raw) = extra.as_deref() else { continue };
        if raw.is_empty() || raw == "{}" {
            continue;
        }
        if raw.contains(IMPORT_SCOPE_KEY) {
            if let Value::Dict(payload) = loads(raw)? {
                if let Some(scope) = payload.get_str(IMPORT_SCOPE_KEY) {
                    scoped_dirs.insert(target.clone(), scope.to_string());
                }
            }
        }
        if raw.contains("ambiguous") {
            if let Value::Dict(payload) = loads(raw)? {
                if payload.get_str("import_resolution") == Some("ambiguous") {
                    if let Some(Value::List(candidates)) = payload.get("import_candidates") {
                        let set = ambiguous.entry(file_path.clone()).or_default();
                        for candidate in candidates {
                            if let Value::Str(text) = candidate {
                                set.insert(text.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    drop(rows);

    // C# namespaces back to the files that declare them.
    let mut namespace_files: HashMap<String, HashSet<String>> = HashMap::default();
    let rows = Rows::new(
        db,
        "SELECT file_path, extra FROM nodes \
         WHERE kind = 'File' AND extra LIKE '%\"csharp_namespaces\"%'",
        paths,
    )?;
    while rows.step()? {
        let file_path = rows.text(0)?;
        let extra = rows.text(1)?;
        let Value::Dict(payload) = loads(extra.as_deref().filter(|e| !e.is_empty()).unwrap_or("{}"))?
        else {
            continue;
        };
        let Some(Value::List(declared)) = payload.get("csharp_namespaces") else { continue };
        for ns in declared {
            if let Value::Str(ns) = ns {
                if !ns.is_empty() {
                    let file_path = file_path.clone().ok_or(Fail::Back)?;
                    namespace_files.entry(ns.clone()).or_default().insert(file_path);
                }
            }
        }
    }
    drop(rows);
    if !namespace_files.is_empty() {
        for imported in import_targets.values_mut() {
            let mut expanded = HashSet::default();
            for target in imported.iter() {
                if let Some(files) = namespace_files.get(target) {
                    expanded.extend(files.iter().cloned());
                }
            }
            imported.extend(expanded);
        }
    }

    if !scoped_dirs.is_empty() {
        let mut members: HashMap<String, HashSet<String>> = HashMap::default();
        let rows = Rows::new(db, "SELECT file_path FROM nodes WHERE kind = 'File'", paths)?;
        while rows.step()? {
            let Some(file_path) = rows.text(0)? else { continue };
            for (directory, own) in import_scope_ancestors(&file_path) {
                let admitted = match scoped_dirs.get(directory).map(String::as_str) {
                    Some(IMPORT_SCOPE_TREE) => true,
                    Some(IMPORT_SCOPE_PACKAGE) => own,
                    _ => false,
                };
                if admitted {
                    members.entry(directory.to_string()).or_default().insert(file_path.clone());
                }
            }
        }
        drop(rows);
        for imported in import_targets.values_mut() {
            let mut grown = HashSet::default();
            for target in imported.iter() {
                if let Some(files) = members.get(target) {
                    grown.extend(files.iter().cloned());
                }
            }
            imported.extend(grown);
        }
    }

    // `_python_module_file_index`.
    let mut modules: HashMap<String, Option<String>> = HashMap::default();
    let rows = Rows::new(db, "SELECT DISTINCT file_path FROM nodes WHERE file_path LIKE '%.py'", paths)?;
    while rows.step()? {
        let file_path = rows.req(0)?;
        for module in module_names(&file_path)? {
            match modules.get_mut(&module) {
                None => {
                    modules.insert(module, Some(file_path.clone()));
                }
                Some(slot) => {
                    if slot.as_deref() != Some(file_path.as_str()) {
                        *slot = None;
                    }
                }
            }
        }
    }
    drop(rows);
    // A module once claimed by two files stays None, as `setdefault` keeps it.
    if modules.values().any(Option::is_some) {
        for imported in import_targets.values_mut() {
            let mut expanded = HashSet::default();
            for target in imported.iter() {
                if let Some(Some(file)) = modules.get(target) {
                    expanded.insert(file.clone());
                }
            }
            imported.extend(expanded);
        }
    }
    Ok((import_targets, ambiguous, raw_imports))
}

// -- inheritance.InheritanceIndex ---------------------------------------------

struct ClassInfo {
    name: String,
    file: String,
    language: String,
    interface: bool,
}

fn is_interface(language: &str, extra: &Dict) -> bool {
    match language {
        "java" => extra.get_str("java_kind") == Some("interface"),
        "typescript" | "tsx" => matches!(
            extra.get_str("typescript_kind"),
            Some("interface_declaration" | "type_alias_declaration")
        ),
        "php" => matches!(extra.get_str("php_kind"), Some("interface" | "trait")),
        _ => false,
    }
}

/// `_strip_type`.
fn strip_type(raw: &str) -> &str {
    let mut text = raw.trim_matches(py_isspace);
    for stop in ['<', '(', '['] {
        text = text.split_once(stop).map_or(text, |(head, _)| head);
    }
    text.trim_matches(py_isspace)
}

/// `_only`.
fn only<'a>(candidates: impl IntoIterator<Item = &'a String>) -> Option<String> {
    let mut first: Option<&String> = None;
    for candidate in candidates {
        match first {
            None => first = Some(candidate),
            Some(seen) if seen != candidate => return None,
            _ => {}
        }
    }
    first.cloned()
}

pub struct Inheritance<'c> {
    imports: &'c HashMap<String, HashSet<String>>,
    raw_imports: &'c HashMap<String, HashMap<String, String>>,
    classes: HashMap<String, ClassInfo>,
    by_name: HashMap<String, Vec<String>>,
    java_fqcn: HashMap<String, Option<String>>,
    java_package: HashMap<String, String>,
    php_namespaces: HashMap<String, HashSet<String>>,
    php_fqn: HashMap<String, Option<String>>,
    methods: HashMap<(String, String), String>,
    raw_parents: HashMap<String, Vec<(String, Dict)>>,
    parents: RefCell<HashMap<String, (Vec<String>, bool)>>,
    inherited: RefCell<HashMap<(String, String), Option<String>>>,
}

impl<'c> Inheritance<'c> {
    pub fn build(
        db: *mut ffi::sqlite3,
        paths: &Paths,
        imports: &'c HashMap<String, HashSet<String>>,
        raw_imports: &'c HashMap<String, HashMap<String, String>>,
    ) -> R<Inheritance<'c>> {
        let mut classes: HashMap<String, ClassInfo> = HashMap::default();
        // Insertion order of `_classes`, for the PHP pass below.
        let mut class_order: Vec<String> = Vec::new();
        let mut by_name: HashMap<String, Vec<String>> = HashMap::default();
        let mut java_fqcn: HashMap<String, Option<String>> = HashMap::default();
        let mut java_package: HashMap<String, String> = HashMap::default();
        let rows = Rows::new(
            db,
            "SELECT qualified_name, name, file_path, parent_name, language, extra \
             FROM nodes WHERE kind = 'Class'",
            paths,
        )?;
        while rows.step()? {
            let qualified = rows.req(0)?;
            let name = rows.req(1)?;
            let file_path = rows.req(2)?;
            let parent = rows.text(3)?;
            let language = rows.text(4)?.unwrap_or_default();
            let extra = load_dict(rows.text(5)?.as_deref())?;
            let interface = is_interface(&language, &extra);
            if language == "java" {
                if let Some(package) = extra.get_str("java_package").filter(|p| !p.is_empty()) {
                    java_package.entry(file_path.clone()).or_insert_with(|| package.to_string());
                    let outer = match parent.as_deref() {
                        Some(p) if !p.is_empty() => format!("{p}."),
                        _ => String::new(),
                    };
                    let fqcn = format!("{package}.{outer}{name}");
                    let value = if java_fqcn.contains_key(&fqcn) { None } else { Some(qualified.clone()) };
                    java_fqcn.insert(fqcn, value);
                }
            }
            by_name.entry(name.clone()).or_default().push(qualified.clone());
            if !classes.contains_key(&qualified) {
                class_order.push(qualified.clone());
            }
            classes.insert(qualified, ClassInfo { name, file: file_path, language, interface });
        }
        drop(rows);

        let mut php_namespaces: HashMap<String, HashSet<String>> = HashMap::default();
        let rows = Rows::new(
            db,
            "SELECT file_path, extra FROM nodes \
             WHERE kind = 'File' AND extra LIKE '%\"php_namespaces\"%'",
            paths,
        )?;
        while rows.step()? {
            let file_path = rows.text(0)?;
            let extra = load_dict(rows.text(1)?.as_deref())?;
            if let Some(Value::List(declared)) = extra.get("php_namespaces") {
                let set = declared
                    .iter()
                    .filter_map(|ns| ns.as_str().filter(|ns| !ns.is_empty()).map(str::to_string))
                    .collect();
                php_namespaces.insert(file_path.ok_or(Fail::Back)?, set);
            }
        }
        drop(rows);
        let mut php_fqn: HashMap<String, Option<String>> = HashMap::default();
        for qualified in &class_order {
            let info = &classes[qualified];
            if info.language != "php" {
                continue;
            }
            for namespace in php_namespaces.get(&info.file).into_iter().flatten() {
                let fqn = format!("{namespace}\\{}", info.name);
                let value = if php_fqn.contains_key(&fqn) { None } else { Some(qualified.clone()) };
                php_fqn.insert(fqn, value);
            }
        }

        let mut methods: HashMap<(String, String), String> = HashMap::default();
        let rows = Rows::new(
            db,
            "SELECT qualified_name, name FROM nodes \
             WHERE kind IN ('Function', 'Test') AND parent_name IS NOT NULL",
            paths,
        )?;
        while rows.step()? {
            let qualified = rows.req(0)?;
            let name = rows.req(1)?;
            let owner = rpartition_head(&qualified, '.');
            if classes.contains_key(owner) {
                methods.entry((owner.to_string(), name)).or_insert_with(|| qualified.clone());
            }
        }
        drop(rows);

        let mut raw_parents: HashMap<String, Vec<(String, Dict)>> = HashMap::default();
        let rows = Rows::new(
            db,
            "SELECT source_qualified, target_qualified, extra FROM edges \
             WHERE kind = 'INHERITS'",
            paths,
        )?;
        while rows.step()? {
            let Some(source) = rows.text(0)? else { continue };
            if !classes.contains_key(&source) {
                continue;
            }
            let target = rows.req(1)?;
            let extra = load_dict(rows.text(2)?.as_deref())?;
            raw_parents.entry(source).or_default().push((target, extra));
        }
        drop(rows);

        Ok(Inheritance {
            imports,
            raw_imports,
            classes,
            by_name,
            java_fqcn,
            java_package,
            php_namespaces,
            php_fqn,
            methods,
            raw_parents,
            parents: RefCell::new(HashMap::default()),
            inherited: RefCell::new(HashMap::default()),
        })
    }

    pub fn is_class(&self, qualified: &str) -> bool {
        self.classes.contains_key(qualified)
    }

    pub fn java_package(&self, file_path: &str) -> Option<&str> {
        self.java_package.get(file_path).map(String::as_str)
    }

    pub fn method_of(&self, class_name: &str, method: &str) -> Option<String> {
        self.methods.get(&(class_name.to_string(), method.to_string())).cloned()
    }

    fn method_truthy(&self, class_name: &str, method: &str) -> Option<String> {
        self.method_of(class_name, method).filter(|found| !found.is_empty())
    }

    fn namespaces_of_class(&self, qualified: &str) -> Option<&HashSet<String>> {
        self.php_namespaces.get(&self.classes[qualified].file)
    }

    pub fn resolve_type(&self, raw: &str, context_file: &str, language: &str) -> Option<String> {
        let text = strip_type(raw);
        let (absolute, namespace, name) = if language == "php" {
            let absolute = text.starts_with('\\');
            let body = text.trim_matches('\\');
            match body.rsplit_once('\\') {
                Some((ns, name)) => (absolute, ns, name),
                None => (absolute, "", body),
            }
        } else {
            (false, "", rpartition_tail(text, '.'))
        };
        let candidates = self.by_name.get(name).filter(|c| !c.is_empty())?;
        if !namespace.is_empty() {
            let mut wanted: HashSet<String> = HashSet::default();
            if absolute {
                wanted.insert(namespace.to_string());
            } else {
                let (head, rest) = namespace.split_once('\\').unwrap_or((namespace, ""));
                for ns in self.php_namespaces.get(context_file).into_iter().flatten() {
                    wanted.insert(format!("{ns}\\{namespace}"));
                }
                for target in self.imports.get(context_file).into_iter().flatten() {
                    if rpartition_tail(target, '\\') == head && target.contains('\\') {
                        wanted.insert(if rest.is_empty() {
                            target.clone()
                        } else {
                            format!("{target}\\{rest}")
                        });
                    }
                }
            }
            return only(candidates.iter().filter(|c| {
                self.namespaces_of_class(c).is_some_and(|set| set.iter().any(|ns| wanted.contains(ns)))
            }));
        }
        let same_file: Vec<&String> =
            candidates.iter().filter(|c| self.classes[*c].file == context_file).collect();
        if !same_file.is_empty() {
            return only(same_file);
        }
        let full_name = self.raw_imports.get(context_file).and_then(|m| m.get(name));
        if let Some(full_name) = full_name {
            if language == "java" {
                return self.java_fqcn.get(full_name).cloned().flatten();
            }
            if language == "php" {
                return self.php_fqn.get(full_name.trim_matches('\\')).cloned().flatten();
            }
        }
        if let Some(imported_files) = self.imports.get(context_file) {
            let imported: Vec<&String> = candidates
                .iter()
                .filter(|c| imported_files.contains(&self.classes[*c].file))
                .collect();
            if !imported.is_empty() {
                return only(imported);
            }
        }
        if language == "java" {
            if let Some(package) = self.java_package.get(context_file).filter(|p| !p.is_empty()) {
                return only(candidates.iter().filter(|c| {
                    self.java_package.get(&self.classes[*c].file) == Some(package)
                }));
            }
        }
        if language == "php" {
            if let Some(namespaces) = self.php_namespaces.get(context_file).filter(|n| !n.is_empty()) {
                return only(candidates.iter().filter(|c| {
                    self.namespaces_of_class(c)
                        .is_some_and(|set| set.iter().any(|ns| namespaces.contains(ns)))
                }));
            }
        }
        None
    }

    pub fn inherited_method(&self, class_name: &str, method: &str) -> Option<String> {
        let key = (class_name.to_string(), method.to_string());
        if let Some(found) = self.inherited.borrow().get(&key) {
            return found.clone();
        }
        let found = self.walk(class_name, method);
        self.inherited.borrow_mut().insert(key, found.clone());
        found
    }

    fn resolve_parent(&self, child: &str, raw: &str, extra: &Dict) -> Option<String> {
        if self.classes.contains_key(raw) {
            return Some(raw.to_string());
        }
        let info = &self.classes[child];
        if info.language == "java" {
            if let Some(java_type) = extra.get_str("java_type") {
                let stripped = strip_type(java_type);
                if stripped.contains('.') {
                    return self.java_fqcn.get(stripped).cloned().flatten();
                }
            }
        }
        self.resolve_type(raw, &info.file, &info.language)
    }

    fn parents_of(&self, class_name: &str) -> (Vec<String>, bool) {
        if let Some(found) = self.parents.borrow().get(class_name) {
            return found.clone();
        }
        let child_is_interface = self.classes[class_name].interface;
        let mut known: Vec<String> = Vec::new();
        let mut unknown = false;
        for (raw, extra) in self.raw_parents.get(class_name).into_iter().flatten() {
            match self.resolve_parent(class_name, raw, extra) {
                None => unknown = true,
                Some(parent) if parent == class_name => {}
                Some(parent) if self.classes[&parent].interface && !child_is_interface => {}
                Some(parent) => {
                    if !known.contains(&parent) {
                        known.push(parent);
                    }
                }
            }
        }
        let result = (known, unknown);
        self.parents.borrow_mut().insert(class_name.to_string(), result.clone());
        result
    }

    fn walk(&self, class_name: &str, method: &str) -> Option<String> {
        let mut current = class_name.to_string();
        let mut seen: HashSet<String> = set_of(current.clone());
        for _ in 0..MAX_DEPTH {
            let (parents, unknown) = self.parents_of(&current);
            if parents.len() > 1 {
                return self.only_declaration(&current, method);
            }
            let Some(parent) = parents.into_iter().next() else { return None };
            if unknown && !SINGLE_SUPERCLASS.contains(&self.classes[&current].language.as_str()) {
                return None;
            }
            if !seen.insert(parent.clone()) {
                return None;
            }
            if let Some(found) = self.method_truthy(&parent, method) {
                return Some(found);
            }
            current = parent;
        }
        None
    }

    fn only_declaration(&self, class_name: &str, method: &str) -> Option<String> {
        let mut found: HashSet<String> = HashSet::default();
        let mut seen: HashSet<String> = set_of(class_name.to_string());
        let mut frontier = vec![class_name.to_string()];
        let mut emptied = false;
        for _ in 0..MAX_DEPTH {
            if frontier.is_empty() {
                emptied = true;
                break;
            }
            let mut next_frontier = Vec::new();
            for current in &frontier {
                let (parents, unknown) = self.parents_of(current);
                if unknown {
                    return None;
                }
                for parent in parents {
                    if !seen.insert(parent.clone()) {
                        continue;
                    }
                    if let Some(declared) = self.method_truthy(&parent, method) {
                        found.insert(declared);
                    }
                    next_frontier.push(parent);
                }
            }
            frontier = next_frontier;
        }
        if !emptied {
            return None;
        }
        if found.len() == 1 { found.into_iter().next() } else { None }
    }
}

// -- The bare pass ------------------------------------------------------------

type Candidates<'a> = Vec<(&'a str, &'a str)>;

/// `_receiver_backed_candidates`.
fn receiver_backed<'a>(candidates: Candidates<'a>, extra: &Dict, context: &Context) -> Candidates<'a> {
    match extra.get_str("receiver_binding") {
        Some("module") => {
            let Some(module_file) = extra.get_str("receiver_module").filter(|m| !m.is_empty()) else {
                return Vec::new();
            };
            candidates
                .into_iter()
                .filter(|(q, f)| *f == module_file && context.parent(q).is_none())
                .collect()
        }
        Some("class") => {
            let Some(class_name) = extra.get_str("receiver_class").filter(|c| !c.is_empty()) else {
                return Vec::new();
            };
            let class_file = match extra.get("receiver_class_file") {
                None | Some(Value::Null) => None,
                Some(Value::Str(file)) => Some(file.as_str()),
                Some(_) => return Vec::new(),
            };
            let mut lookup: Vec<String> = Vec::new();
            if let Some(Value::List(scopes)) = extra.get("receiver_scopes") {
                for scope in scopes {
                    if let Some(scope) = scope.as_str().filter(|s| !s.is_empty()) {
                        lookup.push(format!("{scope}.{class_name}"));
                    }
                }
            }
            lookup.push(class_name.to_string());
            for path in &lookup {
                let matched: Candidates = candidates
                    .iter()
                    .filter(|(q, f)| {
                        context.parent(q) == Some(path.as_str())
                            && class_file.is_none_or(|cf| cf.is_empty() || *f == cf)
                    })
                    .copied()
                    .collect();
                if !matched.is_empty() {
                    return matched;
                }
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

/// `_class_backed_candidates`.
fn class_backed(
    extra: &Dict,
    bare_name: &str,
    caller_class: &str,
    caller_language: &str,
    context_file: &str,
    inheritance: &Inheritance,
) -> R<Option<(Vec<String>, Option<String>)>> {
    let receiver = extra.get("receiver");
    let owner: String;
    if extra.get_str("receiver_resolution") == Some("typed_receiver_unresolved") {
        let resolved = match extra.get_str("receiver_type") {
            Some(t) if !t.is_empty() => inheritance.resolve_type(t, context_file, caller_language),
            _ => None,
        };
        match resolved {
            Some(found) => owner = found,
            None => return Ok(Some((Vec::new(), None))),
        }
    } else if (member(receiver, SELF_RECEIVERS)? || member(receiver, SUPER_RECEIVERS)?)
        && !extra.contains("receiver_binding")
    {
        if caller_class.is_empty() {
            return Ok(None);
        }
        owner = caller_class.to_string();
    } else if caller_language == "java"
        && let Some(Value::Str(text)) = receiver
        && java_upper(text)?
        && !extra.contains("receiver_type")
    {
        match inheritance.resolve_type(text, context_file, "java") {
            Some(found) => owner = found,
            None => return Ok(None),
        }
    } else {
        return Ok(None);
    }

    let is_super = member(receiver, SUPER_RECEIVERS)?;
    if !is_super {
        if let Some(own) = inheritance.method_of(&owner, bare_name) {
            return Ok(Some((vec![own], None)));
        }
    }
    if let Some(inherited) = inheritance.inherited_method(&owner, bare_name) {
        return Ok(Some((vec![inherited], Some(owner))));
    }
    if member(receiver, SELF_RECEIVERS)? || is_super {
        return Ok(None);
    }
    Ok(Some((Vec::new(), None)))
}

/// `receiver[:1].isupper()`, handing back for a non-ASCII first character.
fn java_upper(text: &str) -> R<bool> {
    match text.chars().next() {
        None => Ok(false),
        Some(c) if c.is_ascii() => Ok(c.is_ascii_uppercase()),
        Some(_) => Err(Fail::Back),
    }
}

/// `_receiver_is_untyped`.
fn receiver_is_untyped(extra: &Dict, caller_language: &str) -> bool {
    UNTYPED_RECEIVER_LANGUAGES.contains(&caller_language)
        && matches!(extra.get_str("receiver"), Some(r) if !r.is_empty()
            && !SELF_RECEIVERS.contains(&r) && !SUPER_RECEIVERS.contains(&r))
        && !extra.contains("receiver_binding")
        && !extra.contains("receiver_type")
}

/// `_own_method_excluded`.
fn own_method_excluded(
    extra: &Dict,
    bare_name: &str,
    caller_class: &str,
    caller_language: &str,
    inheritance: &Inheritance,
) -> Option<String> {
    let receiver = extra.get_str("receiver")?;
    if caller_class.is_empty()
        || SELF_RECEIVERS.contains(&receiver)
        || SUPER_RECEIVERS.contains(&receiver)
        || !TYPED_RECEIVER_LANGUAGES.contains(&caller_language)
        || extra.contains("receiver_binding")
        || extra.contains("receiver_type")
    {
        return None;
    }
    inheritance.method_of(caller_class, bare_name)
}

/// One pass's mutations: (endpoint, extra, id), and the resolved count.
type Plan = (Vec<(String, String, i64)>, i64);

fn plan_pass(
    db: *mut ffi::sqlite3,
    paths: &Paths,
    context: &Context,
    inheritance: &Inheritance,
    kind: &str,
    target_endpoint: bool,
    scope_floor: Option<i64>,
) -> R<Plan> {
    let (raw_key, base_sql, endpoint_index, caller_index): (&str, &str, c_int, c_int) =
        if target_endpoint {
            (
                "bare_call_target",
                "SELECT id, source_qualified, target_qualified, file_path, extra \
                 FROM edges WHERE kind = ? AND (target_qualified NOT LIKE '%::%' \
                 OR extra LIKE '%\"bare_call_target\"%')",
                2,
                1,
            )
        } else {
            (
                "bare_tested_by_source",
                "SELECT id, source_qualified, target_qualified, file_path, extra \
                 FROM edges WHERE kind = ? AND (source_qualified NOT LIKE '%::%' \
                 OR extra LIKE '%\"bare_tested_by_source\"%')",
                1,
                2,
            )
        };
    // `_resolve_bare_endpoints(scoped=True)`: only the edges the last change
    // capture can have moved; the rest are already at the answer.
    let mut sql = base_sql.to_string();
    if scope_floor.is_some() {
        let (endpoint_column, caller_column) = if target_endpoint {
            ("target_qualified", "source_qualified")
        } else {
            ("source_qualified", "target_qualified")
        };
        sql.push_str(&format!(
            " AND (id > ? \
             OR id IN (SELECT id FROM temp.crg_dirty_edge) \
             OR file_path IN (SELECT path FROM temp.crg_dirty_file) \
             OR {endpoint_column} IN (SELECT name FROM temp.crg_dirty_name) \
             OR {caller_column} IN (SELECT qn FROM temp.crg_dirty_qn) \
             OR (extra LIKE '%\"{raw_key}\"%' AND CASE WHEN json_valid(extra) \
             THEN json_extract(extra, '$.{raw_key}') END \
             IN (SELECT name FROM temp.crg_dirty_name)))"
        ));
    }
    let rows = Rows::new(db, &sql, paths)?;
    rows.bind_text(1, kind)?;
    if let Some(floor) = scope_floor {
        rows.bind_int(2, floor)?;
    }
    let empty_set: HashSet<String> = HashSet::default();
    let mut updates = Vec::new();
    let mut resolved = 0i64;
    while rows.step()? {
        let id = rows.int(0)?;
        let endpoint_value = rows.text(endpoint_index)?;
        let raw_extra = rows.text(4)?;
        let edge_extra: Dict;
        let bare_name: String;
        let receiver_constant: bool;
        let class_found: Vec<String>;
        let mut candidates: Candidates;
        match raw_extra.as_deref() {
            None | Some("") | Some("{}") => {
                let Some(name) = endpoint_value.clone() else { continue };
                match context.node_lookup.get(&name) {
                    Some(found) if !found.is_empty() => {
                        candidates = found.iter().map(|(q, f)| (q.as_str(), f.as_str())).collect()
                    }
                    _ => continue,
                }
                bare_name = name;
                edge_extra = Dict::default();
                receiver_constant = false;
            }
            Some(raw) => {
                edge_extra = match loads(raw)? {
                    Value::Dict(dict) => dict,
                    _ => Dict::default(),
                };
                if !edge_extra.contains(raw_key)
                    && (edge_extra.contains("ambiguous_targets")
                        || edge_extra.contains("unresolved_targets")
                        || edge_extra.contains(CLAIMED_ENDPOINT_KEY))
                {
                    continue;
                }
                if edge_extra.contains(INHERITED_ENDPOINT_KEY) {
                    continue;
                }
                let name = match edge_extra.get(raw_key) {
                    Some(Value::Str(name)) => name.clone(),
                    Some(_) => continue,
                    None => match &endpoint_value {
                        Some(name) => name.clone(),
                        None => continue,
                    },
                };
                receiver_constant = matches!(edge_extra.get("receiver_constant"), Some(Value::Bool(true)));
                let lookup_name = if receiver_constant && name == "new" { "initialize" } else { &name };
                candidates = context
                    .node_lookup
                    .get(lookup_name)
                    .map(|found| found.iter().map(|(q, f)| (q.as_str(), f.as_str())).collect())
                    .unwrap_or_default();
                if receiver_constant {
                    candidates.retain(|(q, _)| context.ruby_nodes.contains(*q));
                }
                if edge_extra.contains("receiver_binding") {
                    candidates = receiver_backed(candidates, &edge_extra, context);
                }
                bare_name = name;
            }
        }

        let context_file = rows.req(3)?;
        let caller = rows.req(caller_index)?;
        let caller_language = context.language_lookup.get(&caller).map_or("", String::as_str);
        let mut caller_class = match context.parent(&caller) {
            Some(parent) if !parent.is_empty() => rpartition_head(&caller, '.'),
            _ => "",
        };
        if !inheritance.is_class(caller_class) {
            caller_class = "";
        }
        let backed = class_backed(
            &edge_extra, &bare_name, caller_class, caller_language, &context_file, inheritance,
        )?;
        let mut supported: Vec<&str>;
        let ambiguity_supported: Vec<&str>;
        let inherited_through: Option<String>;
        if let Some((found, through)) = backed {
            class_found = found;
            candidates = class_found.iter().map(|q| (q.as_str(), "")).collect();
            supported = class_found.iter().map(String::as_str).collect();
            ambiguity_supported = Vec::new();
            inherited_through = through;
        } else {
            inherited_through = None;
            let imported_files = context.import_targets.get(&context_file).unwrap_or(&empty_set);
            let receiver_file = if edge_extra.get_str("receiver_binding") == Some("module") {
                edge_extra.get_str("receiver_module")
            } else {
                None
            };
            let receiver_class_file = edge_extra.get_str("receiver_class_file");
            let java_package =
                if caller_language == "java" { inheritance.java_package(&context_file) } else { None };
            supported = candidates
                .iter()
                .filter(|(q, f)| {
                    *f == context_file
                        || imported_files.contains(*f)
                        || receiver_file == Some(*f)
                        || receiver_class_file == Some(*f)
                        || (java_package.is_some()
                            && inheritance.is_class(q)
                            && inheritance.java_package(f) == java_package)
                        || receiver_constant
                })
                .map(|(q, _)| *q)
                .collect();
            if receiver_is_untyped(&edge_extra, caller_language) {
                supported.retain(|q| context.parent(q).is_none_or(str::is_empty));
            }
            let own = own_method_excluded(
                &edge_extra, &bare_name, caller_class, caller_language, inheritance,
            );
            if let Some(own) = own {
                if supported.len() == 1 && supported[0] == own {
                    supported.clear();
                }
            }
            let ambiguous_files = context.ambiguous_imports.get(&context_file).unwrap_or(&empty_set);
            ambiguity_supported = candidates
                .iter()
                .filter(|(_, f)| ambiguous_files.contains(*f))
                .map(|(q, _)| *q)
                .collect();
        }
        let managed = edge_extra.contains(raw_key);
        if supported.is_empty() && !managed && ambiguity_supported.is_empty() {
            continue;
        }
        let mut desired = edge_extra.clone();
        desired.set(raw_key, Value::Str(bare_name.clone()));
        match &inherited_through {
            Some(through) if !through.is_empty() && supported.len() == 1 => {
                desired.set(INHERITED_THROUGH_KEY, Value::Str(through.clone()));
            }
            _ => {
                desired.pop(INHERITED_THROUGH_KEY);
            }
        }
        let desired_endpoint: String;
        if supported.len() == 1 {
            desired_endpoint = supported[0].to_string();
            for key in [
                "ambiguous_targets",
                "ambiguous_target_count",
                "ambiguous_targets_truncated",
                "unresolved_targets",
                "unresolved_target_count",
                "unresolved_targets_truncated",
            ] {
                desired.pop(key);
            }
        } else {
            desired_endpoint = bare_name.clone();
            let (resolution, mut chosen, other) = if supported.len() > 1 {
                ("ambiguous", supported.clone(), "unresolved")
            } else if !ambiguity_supported.is_empty() {
                if ambiguity_supported.len() > 1 {
                    ("ambiguous", ambiguity_supported.clone(), "unresolved")
                } else {
                    ("unresolved", ambiguity_supported.clone(), "ambiguous")
                }
            } else {
                ("unresolved", candidates.iter().map(|(q, _)| *q).collect(), "ambiguous")
            };
            for suffix in ["_targets", "_target_count", "_targets_truncated"] {
                desired.pop(&format!("{other}{suffix}"));
            }
            chosen.sort();
            let count = chosen.len();
            let listed = chosen.into_iter().take(20).map(|q| Value::Str(q.to_string())).collect();
            desired.set(&format!("{resolution}_targets"), Value::List(listed));
            desired.set(&format!("{resolution}_target_count"), Value::Int(count as i64));
            desired.set(&format!("{resolution}_targets_truncated"), Value::Bool(count > 20));
        }
        let same_endpoint = endpoint_value.as_deref() == Some(desired_endpoint.as_str());
        if same_endpoint && pyjson::dict_eq(&edge_extra, &desired) {
            continue;
        }
        if supported.len() == 1 && !same_endpoint {
            resolved += 1;
        }
        updates.push((desired_endpoint, pyjson::dumps_dict(&desired, true), id));
    }
    Ok((updates, resolved))
}

fn write_pass(
    db: *mut ffi::sqlite3,
    paths: &Paths,
    target_endpoint: bool,
    updates: &[(String, String, i64)],
) -> Result<(), String> {
    let sql = if target_endpoint {
        "UPDATE edges SET target_qualified = ?, extra = ? WHERE id = ?"
    } else {
        "UPDATE edges SET source_qualified = ?, extra = ? WHERE id = ?"
    };
    let stmt = Stmt::prepare(db, sql)?;
    for (endpoint, extra, id) in updates {
        stmt.bind_text(1, &paths.strip(endpoint))?;
        stmt.bind_text(2, &paths.strip(extra))?;
        stmt.bind_i64(3, *id)?;
        stmt.run()?;
    }
    Ok(())
}

/// The codec fields Python passes: (root, json_root_out, prefixes, roots,
/// json_roots, leads), or None for a graph that stores absolute paths.
pub type CodecFields = Option<(String, String, Vec<String>, Vec<String>, Vec<String>, Vec<String>)>;

pub fn paths_from(fields: CodecFields) -> Paths {
    Paths(fields.map(|(root, json_root_out, prefixes, roots, json_roots, leads)| {
        let leads = leads.iter().filter_map(|lead| lead.chars().next()).collect();
        Codec::new(root, json_root_out, prefixes, roots, json_roots, leads)
    }))
}

pub fn run_job<F>(job: F) -> Result<i64, String>
where
    F: FnOnce() -> R<i64>,
{
    match job() {
        Ok(count) => Ok(count),
        Err(Fail::Back) => Ok(FALL_BACK),
        Err(Fail::Sql(message)) => Err(message),
    }
}

/// Stages `_resolve_bare_endpoints(kind, endpoint)`. The job returns the
/// resolved count, or -1 having written nothing.
#[pyfunction]
#[pyo3(signature = (kind, endpoint, codec, scope_floor=None))]
pub fn stage_bare_resolve(
    kind: String,
    endpoint: String,
    codec: CodecFields,
    scope_floor: Option<i64>,
) -> PyResult<i64> {
    let target_endpoint = match endpoint.as_str() {
        "target_qualified" => true,
        "source_qualified" => false,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("invalid endpoint")),
    };
    let paths = paths_from(codec);
    stage(Box::new(move |db| {
        run_job(|| {
            let context = Context::build(db, &paths)?;
            let inheritance = Inheritance::build(
                db, &paths, &context.import_targets, &context.raw_imports,
            )?;
            let (updates, resolved) =
                plan_pass(
                    db, &paths, &context, &inheritance, &kind, target_endpoint, scope_floor,
                )?;
            write_pass(db, &paths, target_endpoint, &updates)?;
            Ok(resolved)
        })
    }))
}

/// Stages the CALLS pass then the TESTED_BY pass of `_resolve_bare_endpoints`
/// on one evidence context. The CALLS pass writes only CALLS edges, and the
/// context reads nodes and IMPORTS_FROM and INHERITS edges, so the second pass
/// sees the context a job of its own would build. The job returns the CALLS
/// count shifted 32 bits left plus the TESTED_BY count, or -1 having written
/// nothing.
#[pyfunction]
#[pyo3(signature = (codec, scope_floor=None))]
pub fn stage_bare_resolve_pair(codec: CodecFields, scope_floor: Option<i64>) -> PyResult<i64> {
    let paths = paths_from(codec);
    stage(Box::new(move |db| {
        run_job(|| {
            let context = Context::build(db, &paths)?;
            let inheritance = Inheritance::build(
                db, &paths, &context.import_targets, &context.raw_imports,
            )?;
            let (updates, calls) = plan_pass(
                db, &paths, &context, &inheritance, "CALLS", true, scope_floor,
            )?;
            write_pass(db, &paths, true, &updates)?;
            let (updates, tested) = plan_pass(
                db, &paths, &context, &inheritance, "TESTED_BY", false, scope_floor,
            )?;
            write_pass(db, &paths, false, &updates)?;
            if !(0..1 << 31).contains(&calls) || !(0..1 << 31).contains(&tested) {
                return Err(Fail::Back);
            }
            Ok((calls << 32) | tested)
        })
    }))
}

// -- _resolve_inherited_endpoints ------------------------------------------------

const RESOLVER_BOUND_KEYS: &[&str] = &[
    "bare_call_target",
    "bare_tested_by_source",
    "cpp_scoped_target",
    INHERITED_ENDPOINT_KEY,
    "rescript_open_module",
    "scoped_resolved",
    "spring_resolved",
    "temporal_resolved",
];
const TIER_EXTRACTED: &str = "EXTRACTED";
const SCORE_EXTRACTED: f64 = 1.0;

/// `_is_path_qualified`.
fn is_path_qualified(name: &str) -> R<bool> {
    is_absolute_path(name.split_once("::").map_or(name, |(head, _)| head))
}

/// `_plain_symbol_name`: the text after the last `::` or `.` that
/// `re.split(r"::|\.")` finds scanning left to right.
fn plain_symbol_name(qualified: &str) -> &str {
    let symbol = qualified.split_once("::").map_or("", |(_, symbol)| symbol);
    let bytes = symbol.as_bytes();
    let (mut i, mut last) = (0, 0);
    while i < bytes.len() {
        if bytes[i] == b':' && bytes.get(i + 1) == Some(&b':') {
            i += 2;
            last = i;
        } else if bytes[i] == b'.' {
            i += 1;
            last = i;
        } else {
            i += 1;
        }
    }
    &symbol[last..]
}

/// `_load_edge_extra`, parsed or handed back.
fn edge_dict(raw: Option<&str>) -> R<Dict> {
    load_dict(raw)
}

struct InheritedWork {
    target_column: bool,
    id: i64,
    endpoint: Option<String>,
    extra: Dict,
    written: String,
    owner: String,
    method: String,
}

enum Change {
    Move { target_column: bool, endpoint: String, extra: String, id: i64 },
    Revert { target_column: bool, endpoint: String, extra: String, tier: String, score: f64, id: i64 },
}

/// The rows of *kind* (`?1`) the certainty capture holds, with `{column}`
/// read as `endpoint`: written since the edge id floor (`?{floor}`), rewritten
/// since, or naming a name a node gained or lost. Whether a row is unbound
/// turns on nothing else.
fn captured_unbound_sql(column: &str, floor: u8) -> String {
    format!(
        "SELECT u.id, u.endpoint, u.extra FROM (\
         SELECT id, {column} AS endpoint, extra FROM edges \
         WHERE kind = ?1 AND id > ?{floor} \
         UNION SELECT id, {column}, extra FROM edges \
         WHERE kind = ?1 AND id IN (SELECT id FROM temp.crg_cert_edge) \
         UNION SELECT id, {column}, extra FROM edges \
         WHERE kind = ?1 AND {column} IN (SELECT qn FROM temp.crg_cert_qn)) u \
         WHERE u.endpoint LIKE '%::%' AND NOT EXISTS (\
         SELECT 1 FROM nodes n WHERE n.qualified_name = u.endpoint)"
    )
}

/// Every row whose `extra` matched `LIKE '%endpoint"%'` when the update's
/// refresh began, kept by `refresh_target_resolution` in
/// `temp.crg_endpoint_rows`: a superset of both LIKEs the passes use.
const ENDPOINT_ROWS: &str = "SELECT id FROM temp.crg_endpoint_rows";

fn plan_inherited(db: *mut ffi::sqlite3, paths: &Paths, scope: Option<i64>) -> R<Vec<Change>> {
    let mut work: Vec<InheritedWork> = Vec::new();
    for (kind, target_column) in [("CALLS", true), ("TESTED_BY", false)] {
        let column = if target_column { "target_qualified" } else { "source_qualified" };
        let sql = match scope {
            None => format!(
                "SELECT e.id, e.{column} AS endpoint, e.extra FROM edges e \
                 WHERE e.kind = ?1 AND (e.extra LIKE ?2 OR e.extra LIKE ?3 OR (\
                 e.{column} LIKE '%::%' AND NOT EXISTS (\
                 SELECT 1 FROM nodes n WHERE n.qualified_name = e.{column})))"
            ),
            // A row is read for its claimed or inherited endpoint, or for
            // being unbound; the update's unbound rows are the captured ones.
            Some(_) => format!(
                "SELECT e.id, e.{column} AS endpoint, e.extra FROM edges e \
                 WHERE e.kind = ?1 AND e.id IN ({ENDPOINT_ROWS}) \
                 AND (e.extra LIKE ?2 OR e.extra LIKE ?3) \
                 UNION {} ORDER BY 1",
                captured_unbound_sql(column, 4),
            ),
        };
        let rows = Rows::new(db, &sql, paths)?;
        rows.bind_text(1, kind)?;
        rows.bind_text(2, "%\"claimed_endpoint\"%")?;
        rows.bind_text(3, "%\"inherited_endpoint\"%")?;
        if let Some(floor) = scope {
            rows.bind_int(4, floor)?;
        }
        while rows.step()? {
            let id = rows.int(0)?;
            let endpoint = rows.text(1)?;
            let extra = edge_dict(rows.text(2)?.as_deref())?;
            let written = match extra.get(INHERITED_ENDPOINT_KEY) {
                Some(Value::Str(managed)) => managed.clone(),
                _ if RESOLVER_BOUND_KEYS.iter().any(|key| extra.contains(key))
                    || extra.contains("ambiguous_targets")
                    || extra.contains("unresolved_targets") =>
                {
                    continue;
                }
                _ => match extra.get(CLAIMED_ENDPOINT_KEY) {
                    Some(Value::Str(claimed)) => claimed.clone(),
                    _ => endpoint.clone().ok_or(Fail::Back)?,
                },
            };
            if !is_path_qualified(&written)? {
                continue;
            }
            let (file_part, symbol) = written.split_once("::").unwrap_or((&written, ""));
            let Some((class_part, method)) = symbol.rsplit_once('.') else { continue };
            if class_part.is_empty() || method.is_empty() {
                continue;
            }
            let owner = format!("{file_part}::{class_part}");
            let method = method.to_string();
            work.push(InheritedWork {
                target_column,
                id,
                endpoint,
                extra,
                written,
                owner,
                method,
            });
        }
    }
    if work.is_empty() {
        return Ok(Vec::new());
    }
    let mut with_parents: HashSet<String> = HashSet::default();
    let rows = Rows::new(db, "SELECT DISTINCT source_qualified FROM edges WHERE kind = 'INHERITS'", paths)?;
    while rows.step()? {
        if let Some(source) = rows.text(0)? {
            with_parents.insert(source);
        }
    }
    drop(rows);
    work.retain(|w| with_parents.contains(&w.owner) || w.extra.contains(INHERITED_ENDPOINT_KEY));
    if work.is_empty() {
        return Ok(Vec::new());
    }
    let (imports, _, raw_imports) = import_evidence(db, paths)?;
    let inheritance = Inheritance::build(db, paths, &imports, &raw_imports)?;
    let mut moves = Vec::new();
    let mut reverts = Vec::new();
    for w in work {
        let mut target = None;
        if inheritance.is_class(&w.owner) && inheritance.method_of(&w.owner, &w.method).is_none() {
            target = inheritance.inherited_method(&w.owner, &w.method);
        }
        if let Some(target) = target {
            let mut desired = w.extra.clone();
            desired.set(INHERITED_ENDPOINT_KEY, Value::Str(w.written.clone()));
            desired.pop(CLAIMED_ENDPOINT_KEY);
            if w.endpoint.as_deref() != Some(target.as_str()) || !pyjson::dict_eq(&desired, &w.extra) {
                moves.push(Change::Move {
                    target_column: w.target_column,
                    endpoint: target,
                    extra: pyjson::dumps_dict(&desired, false),
                    id: w.id,
                });
            }
        } else if w.extra.contains(INHERITED_ENDPOINT_KEY) {
            let mut restored = w.extra.clone();
            restored.pop(INHERITED_ENDPOINT_KEY);
            let tier = match restored.get("confidence_tier") {
                None => TIER_EXTRACTED.to_string(),
                Some(Value::Str(tier)) => tier.clone(),
                Some(_) => return Err(Fail::Back),
            };
            let score = match restored.get("confidence") {
                None => SCORE_EXTRACTED,
                Some(Value::Float(score)) => *score,
                Some(Value::Int(score)) if score.unsigned_abs() < (1u64 << 53) => *score as f64,
                Some(_) => return Err(Fail::Back),
            };
            reverts.push(Change::Revert {
                target_column: w.target_column,
                endpoint: w.written,
                extra: pyjson::dumps_dict(&restored, false),
                tier,
                score,
                id: w.id,
            });
        }
    }
    moves.extend(reverts);
    Ok(moves)
}

fn write_changes(db: *mut ffi::sqlite3, paths: &Paths, changes: &[Change]) -> Result<(), String> {
    let mut statements: HashMap<&'static str, Stmt> = HashMap::default();
    for change in changes {
        match change {
            Change::Move { target_column, endpoint, extra, id } => {
                let sql = if *target_column {
                    "UPDATE edges SET target_qualified = ?, extra = ? WHERE id = ?"
                } else {
                    "UPDATE edges SET source_qualified = ?, extra = ? WHERE id = ?"
                };
                if !statements.contains_key(sql) {
                    statements.insert(sql, Stmt::prepare(db, sql)?);
                }
                let stmt = &statements[sql];
                stmt.bind_text(1, &paths.strip(endpoint))?;
                stmt.bind_text(2, &paths.strip(extra))?;
                stmt.bind_i64(3, *id)?;
                stmt.run()?;
            }
            Change::Revert { target_column, endpoint, extra, tier, score, id } => {
                let sql = if *target_column {
                    "UPDATE edges SET target_qualified = ?, extra = ?, \
                     confidence_tier = ?, confidence = ? WHERE id = ?"
                } else {
                    "UPDATE edges SET source_qualified = ?, extra = ?, \
                     confidence_tier = ?, confidence = ? WHERE id = ?"
                };
                if !statements.contains_key(sql) {
                    statements.insert(sql, Stmt::prepare(db, sql)?);
                }
                let stmt = &statements[sql];
                stmt.bind_text(1, &paths.strip(endpoint))?;
                stmt.bind_text(2, &paths.strip(extra))?;
                stmt.bind_text(3, &paths.strip(tier))?;
                stmt.bind_f64(4, *score)?;
                stmt.bind_i64(5, *id)?;
                stmt.run()?;
            }
        }
    }
    Ok(())
}

/// Stages `_resolve_inherited_endpoints`: edges changed, or -1 having
/// written nothing.
/// With *scope_floor* it reads the rows `temp.crg_endpoint_rows` and the
/// certainty capture hold.
#[pyfunction]
#[pyo3(signature = (codec, scope_floor=None))]
pub fn stage_inherited_resolve(codec: CodecFields, scope_floor: Option<i64>) -> PyResult<i64> {
    let paths = paths_from(codec);
    stage(Box::new(move |db| {
        run_job(|| {
            let changes = plan_inherited(db, &paths, scope_floor)?;
            write_changes(db, &paths, &changes)?;
            Ok(changes.len() as i64)
        })
    }))
}

// -- _settle_unbound_endpoints ----------------------------------------------------

/// Does a node have this qualified name? (`_node_exists`, bound stripped.)
struct NodeExists {
    stmt: Stmt,
}

impl NodeExists {
    fn new(db: *mut ffi::sqlite3) -> R<NodeExists> {
        Ok(NodeExists {
            stmt: Stmt::prepare(db, "SELECT 1 FROM nodes WHERE qualified_name = ? LIMIT 1")?,
        })
    }

    fn check(&self, db: *mut ffi::sqlite3, paths: &Paths, name: Option<&str>) -> R<bool> {
        let Some(name) = name else { return Ok(false) };
        unsafe {
            ffi::sqlite3_reset(self.stmt.raw);
        }
        self.stmt.bind_text(1, &paths.strip(name))?;
        let found = match unsafe { ffi::sqlite3_step(self.stmt.raw) } {
            ffi::SQLITE_ROW => true,
            ffi::SQLITE_DONE => false,
            _ => return Err(Fail::Sql(errmsg(db))),
        };
        unsafe {
            ffi::sqlite3_reset(self.stmt.raw);
        }
        Ok(found)
    }
}

fn plan_unbound(db: *mut ffi::sqlite3, paths: &Paths, scope: Option<i64>) -> R<Vec<Change>> {
    let exists = NodeExists::new(db)?;
    let mut targets: Vec<Change> = Vec::new();
    let mut sources: Vec<Change> = Vec::new();
    for (kind, target_column, managed_key) in [
        ("CALLS", true, "bare_call_target"),
        ("REFERENCES", true, "bare_call_target"),
        ("TESTED_BY", false, "bare_tested_by_source"),
    ] {
        let column = if target_column { "target_qualified" } else { "source_qualified" };
        let updates = if target_column { &mut targets } else { &mut sources };
        let push = |updates: &mut Vec<Change>, endpoint: String, extra: &Dict, id: i64| {
            updates.push(Change::Move {
                target_column,
                endpoint,
                extra: pyjson::dumps_dict(extra, false),
                id,
            });
        };
        let unbound_sql = match scope {
            None => format!(
                "SELECT e.id, e.{column} AS endpoint, e.extra FROM edges e \
                 WHERE e.kind = ?1 AND e.{column} LIKE '%::%' AND NOT EXISTS (\
                 SELECT 1 FROM nodes n WHERE n.qualified_name = e.{column})"
            ),
            Some(_) => format!("{} ORDER BY 1", captured_unbound_sql(column, 2)),
        };
        let rows = Rows::new(db, &unbound_sql, paths)?;
        rows.bind_text(1, kind)?;
        if let Some(floor) = scope {
            rows.bind_int(2, floor)?;
        }
        while rows.step()? {
            let id = rows.int(0)?;
            let endpoint = rows.req(1)?;
            if !is_path_qualified(&endpoint)? {
                continue;
            }
            let mut extra = edge_dict(rows.text(2)?.as_deref())?;
            if let Some(managed) = extra.get_str(managed_key).filter(|m| !m.is_empty()) {
                let managed = managed.to_string();
                push(updates, managed, &extra, id);
                continue;
            }
            let plain = plain_symbol_name(&endpoint).to_string();
            if plain.is_empty() {
                continue;
            }
            extra.set(CLAIMED_ENDPOINT_KEY, Value::Str(endpoint));
            push(updates, plain, &extra, id);
        }
        drop(rows);
        let claimed_sql = match scope {
            None => format!(
                "SELECT e.id, e.{column} AS endpoint, e.extra FROM edges e \
                 WHERE e.kind = ? AND e.extra LIKE ?"
            ),
            // Rows the inherited pass rewrote are in the capture.
            Some(_) => format!(
                "SELECT e.id, e.{column} AS endpoint, e.extra FROM edges e \
                 WHERE e.kind = ?1 AND e.extra LIKE ?2 AND (e.id IN ({ENDPOINT_ROWS}) \
                 OR e.id > ?3 OR e.id IN (SELECT id FROM temp.crg_cert_edge)) ORDER BY e.id"
            ),
        };
        let rows = Rows::new(db, &claimed_sql, paths)?;
        rows.bind_text(1, kind)?;
        rows.bind_text(2, "%\"claimed_endpoint\"%")?;
        if let Some(floor) = scope {
            rows.bind_int(3, floor)?;
        }
        while rows.step()? {
            let id = rows.int(0)?;
            let endpoint = rows.text(1)?;
            let mut extra = edge_dict(rows.text(2)?.as_deref())?;
            let Some(claimed) = extra.get_str(CLAIMED_ENDPOINT_KEY).map(str::to_string) else {
                continue;
            };
            let bound = if exists.check(db, paths, Some(&claimed))? {
                claimed
            } else if exists.check(db, paths, endpoint.as_deref())? {
                endpoint.clone().unwrap_or_default()
            } else {
                continue;
            };
            extra.pop(CLAIMED_ENDPOINT_KEY);
            push(updates, bound, &extra, id);
        }
    }
    targets.extend(sources);
    Ok(targets)
}

/// Stages `_settle_unbound_endpoints`: edges changed, or -1 having written
/// nothing.
/// With *scope_floor* the unbound scans read only what the certainty
/// capture holds, and the claimed scans what `temp.crg_endpoint_rows` holds.
#[pyfunction]
#[pyo3(signature = (codec, scope_floor=None))]
pub fn stage_unbound_settle(codec: CodecFields, scope_floor: Option<i64>) -> PyResult<i64> {
    let paths = paths_from(codec);
    stage(Box::new(move |db| {
        run_job(|| {
            let changes = plan_unbound(db, &paths, scope_floor)?;
            write_changes(db, &paths, &changes)?;
            Ok(changes.len() as i64)
        })
    }))
}
