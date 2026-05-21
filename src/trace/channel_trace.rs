//! Per-channel send trace compatible with `hwsim/tools/visualize_df_sim.py`.
//!
//! Environment:
//! - `STEP_PERF_TRACE=1` or `text` — human-readable lines on stderr + optional file
//! - `STEP_PERF_TRACE=jsonl` — `{"tick", "dir":"send", "ch", "msg":{...}}` per line
//! - `STEP_PERF_TRACE_FILE=<path>` — append trace lines (truncate on first open)

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use dam::types::DAMType;

use crate::primitives::elem::Elem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceMode {
    Off,
    Text,
    Jsonl,
}

static TRACE_MODE: OnceLock<TraceMode> = OnceLock::new();
static TRACE_FILE: OnceLock<Option<String>> = OnceLock::new();
static TRACE_FILE_MX: Mutex<()> = Mutex::new(());
static TRACE_CH_MAP: OnceLock<Mutex<HashMap<(u32, u32), u64>>> = OnceLock::new();
static TRACE_NEXT_CH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Edges collected for optional graph export: (producer_id, stream_idx, ch).
static TRACE_EDGES: OnceLock<Mutex<Vec<(u32, u32, u64)>>> = OnceLock::new();

pub fn trace_mode() -> TraceMode {
    *TRACE_MODE.get_or_init(|| {
        let Ok(v) = std::env::var("STEP_PERF_TRACE") else {
            return TraceMode::Off;
        };
        let v = v.trim();
        if v.is_empty() || v == "0" {
            return TraceMode::Off;
        }
        match v.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "text" => TraceMode::Text,
            "jsonl" => TraceMode::Jsonl,
            _ => TraceMode::Text,
        }
    })
}

fn trace_file_path() -> Option<&'static str> {
    TRACE_FILE
        .get_or_init(|| {
            std::env::var("STEP_PERF_TRACE_FILE")
                .ok()
                .filter(|s| !s.trim().is_empty())
        })
        .as_deref()
}

fn ch_map() -> &'static Mutex<HashMap<(u32, u32), u64>> {
    TRACE_CH_MAP.get_or_init(|| Mutex::new(HashMap::new()))
}

fn edges() -> &'static Mutex<Vec<(u32, u32, u64)>> {
    TRACE_EDGES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Snapshot of all `(producer_id, stream_idx) → trace_ch` registered during graph build.
pub fn ch_map_snapshot() -> HashMap<(u32, u32), u64> {
    ch_map().lock().unwrap().clone()
}

/// Reset channel registry between simulations (same process).
pub fn reset_trace_registry() {
    ch_map().lock().unwrap().clear();
    edges().lock().unwrap().clear();
    TRACE_NEXT_CH.store(1, std::sync::atomic::Ordering::SeqCst);
}

/// Stable logical channel id for `(producer_op_id, stream_idx)` — matches trace + graph JSON.
pub fn alloc_trace_ch(producer_id: u32, stream_idx: u32) -> u64 {
    let key = (producer_id, stream_idx);
    let mut map = ch_map().lock().unwrap();
    if let Some(&ch) = map.get(&key) {
        return ch;
    }
    let ch = TRACE_NEXT_CH.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    map.insert(key, ch);
    edges().lock().unwrap().push((producer_id, stream_idx, ch));
    ch
}

fn ensure_trace_file() {
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        if let Some(path) = trace_file_path() {
            let _ = std::fs::File::create(path);
        }
    });
}

fn emit_line(line: &str) {
    let _lk = TRACE_FILE_MX.lock().unwrap();
    if trace_file_path().is_some() {
        ensure_trace_file();
        use std::fs::OpenOptions;
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(trace_file_path().unwrap()) {
            let _ = writeln!(f, "{}", line);
            let _ = f.flush();
        }
    }
    eprintln!("{}", line);
}

fn elem_tile_json(st: u32, end: bool) -> String {
    format!(r#"{{"kind":"Tile","st":{},"end":{}}}"#, st, end)
}

fn elem_msg_json<T: std::fmt::Debug>(data: &Elem<T>) -> String {
    match data {
        Elem::Val(_) => elem_tile_json(0, false),
        Elem::ValStop(_, st) => elem_tile_json(*st, true),
    }
}

/// Trace one send on a logical channel (cycle = simulation tick).
pub fn trace_send_payload(trace_ch: u64, cycle: u64, msg: &str) {
    let mode = trace_mode();
    if mode == TraceMode::Off {
        return;
    }
    match mode {
        TraceMode::Text => {
            emit_line(&format!(
                "[step-perf-trace] cycle={} ch={} send {}",
                cycle, trace_ch, msg
            ));
        }
        TraceMode::Jsonl => {
            emit_line(&format!(
                r#"{{"tick":{},"dir":"send","ch":{},"msg":{}}}"#,
                cycle, trace_ch, msg
            ));
        }
        TraceMode::Off => {}
    }
}

/// Trace one send when the channel payload is [`Elem`].
pub fn trace_send_elem<T: std::fmt::Debug>(trace_ch: u64, cycle: u64, data: &Elem<T>) {
    trace_send_payload(trace_ch, cycle, &elem_msg_json(data));
}

/// Optional per-payload trace body (HBM / control channels return `None`).
pub trait TraceChannelPayload {
    fn trace_msg_json(&self) -> Option<String> {
        None
    }
}

impl<V: std::fmt::Debug> TraceChannelPayload for Elem<V> {
    fn trace_msg_json(&self) -> Option<String> {
        Some(elem_msg_json(self))
    }
}

