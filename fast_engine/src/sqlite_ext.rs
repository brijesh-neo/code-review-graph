//! Native work on the caller's own SQLite connection.
//!
//! `crg_fast` is also a SQLite loadable extension (`sqlite3_crgfast_init`).
//! Python loads it into its `sqlite3` connection, so every call here goes
//! through that SQLite's own API table: one SQLite in the process, its file
//! locks intact. No connection handle ever reaches Python. Python stages a
//! job (`stage_*`), gets a token, and runs `SELECT crg_run(token)` on its
//! connection; the job then runs with the handle SQLite passes to the call,
//! inside whatever transaction Python has open.

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use libsqlite3_sys as ffi;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub type Job = Box<dyn FnOnce(*mut ffi::sqlite3) -> Result<i64, String> + Send>;

static JOBS: Mutex<Option<HashMap<i64, Job>>> = Mutex::new(None);
static NEXT: AtomicU64 = AtomicU64::new(1);
static LOADED: AtomicU64 = AtomicU64::new(0);

pub fn stage(job: Job) -> PyResult<i64> {
    if LOADED.load(Ordering::Acquire) == 0 {
        return Err(PyRuntimeError::new_err("the crg_fast SQLite extension is not loaded"));
    }
    let token = NEXT.fetch_add(1, Ordering::Relaxed) as i64;
    JOBS.lock().unwrap().get_or_insert_with(HashMap::new).insert(token, job);
    Ok(token)
}

/// Whether some connection in this process loaded the extension.
#[pyfunction]
pub fn sqlite_extension_loaded() -> bool {
    LOADED.load(Ordering::Acquire) != 0
}

/// Drops a staged job that will not run (the caller's statement failed).
#[pyfunction]
pub fn discard_job(token: i64) {
    if let Some(jobs) = JOBS.lock().unwrap().as_mut() {
        jobs.remove(&token);
    }
}

unsafe extern "C" fn crg_run(
    ctx: *mut ffi::sqlite3_context,
    argc: c_int,
    argv: *mut *mut ffi::sqlite3_value,
) {
    unsafe {
        if argc != 1 {
            ffi::sqlite3_result_error(ctx, c"crg_run takes one token".as_ptr(), -1);
            return;
        }
        let token = ffi::sqlite3_value_int64(*argv);
        let job = JOBS.lock().unwrap().as_mut().and_then(|jobs| jobs.remove(&token));
        let Some(job) = job else {
            ffi::sqlite3_result_error(ctx, c"crg_run: no such job".as_ptr(), -1);
            return;
        };
        let db = ffi::sqlite3_context_db_handle(ctx);
        match job(db) {
            Ok(value) => ffi::sqlite3_result_int64(ctx, value),
            Err(message) => {
                let bytes = message.as_bytes();
                ffi::sqlite3_result_error(ctx, bytes.as_ptr().cast(), bytes.len() as c_int);
            }
        }
    }
}

/// The entry point SQLite derives from the file name `crg_fast.*`.
///
/// # Safety
/// Called by SQLite's `load_extension` with a valid connection and API table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_crgfast_init(
    db: *mut ffi::sqlite3,
    _err: *mut *mut c_char,
    api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    unsafe {
        if ffi::rusqlite_extension_init2(api).is_err() {
            return ffi::SQLITE_ERROR;
        }
        let rc = ffi::sqlite3_create_function_v2(
            db,
            c"crg_run".as_ptr(),
            1,
            ffi::SQLITE_UTF8,
            std::ptr::null_mut::<c_void>(),
            Some(crg_run),
            None,
            None,
            None,
        );
        if rc == ffi::SQLITE_OK {
            LOADED.store(1, Ordering::Release);
        }
        rc
    }
}

/// The connection's last error text.
pub fn errmsg(db: *mut ffi::sqlite3) -> String {
    unsafe { CStr::from_ptr(ffi::sqlite3_errmsg(db)).to_string_lossy().into_owned() }
}

/// One prepared statement, finalized on drop.
pub struct Stmt {
    pub raw: *mut ffi::sqlite3_stmt,
    db: *mut ffi::sqlite3,
}

impl Stmt {
    pub fn prepare(db: *mut ffi::sqlite3, sql: &str) -> Result<Stmt, String> {
        let mut raw = std::ptr::null_mut();
        let rc = unsafe {
            ffi::sqlite3_prepare_v2(db, sql.as_ptr().cast(), sql.len() as c_int, &mut raw,
                                    std::ptr::null_mut())
        };
        if rc != ffi::SQLITE_OK {
            return Err(errmsg(db));
        }
        Ok(Stmt { raw, db })
    }

    pub fn bind_text(&self, index: c_int, value: &str) -> Result<(), String> {
        // SQLITE_TRANSIENT: SQLite copies the text.
        let rc = unsafe {
            ffi::sqlite3_bind_text(self.raw, index, value.as_ptr().cast(), value.len() as c_int,
                                   ffi::SQLITE_TRANSIENT())
        };
        self.check(rc)
    }

    pub fn bind_i64(&self, index: c_int, value: i64) -> Result<(), String> {
        self.check(unsafe { ffi::sqlite3_bind_int64(self.raw, index, value) })
    }

    pub fn bind_f64(&self, index: c_int, value: f64) -> Result<(), String> {
        self.check(unsafe { ffi::sqlite3_bind_double(self.raw, index, value) })
    }

    pub fn bind_null(&self, index: c_int) -> Result<(), String> {
        self.check(unsafe { ffi::sqlite3_bind_null(self.raw, index) })
    }

    /// Steps a statement that returns no rows, then resets it.
    pub fn run(&self) -> Result<(), String> {
        let rc = unsafe { ffi::sqlite3_step(self.raw) };
        unsafe { ffi::sqlite3_reset(self.raw) };
        if rc != ffi::SQLITE_DONE {
            return Err(errmsg(self.db));
        }
        Ok(())
    }

    fn check(&self, rc: c_int) -> Result<(), String> {
        if rc == ffi::SQLITE_OK { Ok(()) } else { Err(errmsg(self.db)) }
    }
}

impl Drop for Stmt {
    fn drop(&mut self) {
        unsafe { ffi::sqlite3_finalize(self.raw) };
    }
}
