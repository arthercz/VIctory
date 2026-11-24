//! Continuous performance capture (diagnostics): with
//! `VICTORY_PROFILE=<file.jsonl>` every submission, command list and noted
//! command is appended to the file as one JSON object per line, so a whole
//! run can be analysed afterwards (`tools/profile_report.py`). Profiling
//! implies per-list execution and timing, like `VICTORY_GPU_TIMING`.

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Seconds since the first event, for the `t` field.
pub(crate) fn now() -> f64 {
    start().elapsed().as_secs_f64()
}

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PROFILE").is_some())
}

fn writer() -> &'static Mutex<Option<std::io::BufWriter<std::fs::File>>> {
    static W: OnceLock<Mutex<Option<std::io::BufWriter<std::fs::File>>>> = OnceLock::new();
    W.get_or_init(|| {
        let f = std::env::var_os("VICTORY_PROFILE").and_then(|p| {
            std::fs::File::create(p)
                .map_err(|e| tracing::warn!(error = %e, "VICTORY_PROFILE file not created"))
                .ok()
        });
        Mutex::new(f.map(std::io::BufWriter::new))
    })
}

/// Appends one JSON object (already encoded) and flushes, so a crash keeps
/// everything up to the fault.
pub(crate) fn event(json: String) {
    if !on() {
        return;
    }
    let mut w = writer().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(w) = w.as_mut() {
        let _ = writeln!(w, "{json}");
        let _ = w.flush();
    }
}

/// A JSON string literal (escapes quotes, backslashes, control chars).
pub(crate) fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A JSON array of strings.
pub(crate) fn strings<'a>(items: impl Iterator<Item = &'a str>) -> String {
    let mut out = String::from("[");
    for (i, s) in items.enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&quote(s));
    }
    out.push(']');
    out
}
