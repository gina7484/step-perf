use std::io::Write;
use std::sync::Mutex;

static AW_TRACE_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);
static AW_TRACE_INIT: std::sync::Once = std::sync::Once::new();

fn ensure_init() {
    AW_TRACE_INIT.call_once(|| {
        if let Ok(path) = std::env::var("STEP_PERF_AW_TRACE_FILE") {
            let path = path.trim();
            if !path.is_empty() {
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(path)
                    .unwrap_or_else(|e| panic!("cannot open AW trace file {path}: {e}"));
                *AW_TRACE_FILE.lock().unwrap() = Some(file);
            }
        }
    });
}

pub fn trace_aw_write(id: u32, cycle: u64, tensor_shape_tiled: &[usize], st: u32, end: bool) {
    ensure_init();
    let mut guard = AW_TRACE_FILE.lock().unwrap();
    if let Some(ref mut f) = *guard {
        let shape_json: String = format!(
            "[{}]",
            tensor_shape_tiled
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let _ = writeln!(
            f,
            r#"{{"stage":"step_perf","kind":"aw_write","id":{},"file_path_id":{},"cycle":{},"tensor_shape_tiled":{},"st":{},"end":{}}}"#,
            id, id, cycle, shape_json, st, end,
        );
    }
}
