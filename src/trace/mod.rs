//! Channel trace for step_perf (dfsim-compatible JSONL).

pub mod channel_trace;
pub mod graph_export;
pub mod tracing_sender;

pub use channel_trace::{trace_mode, trace_send_elem, TraceChannelPayload, TraceMode};
pub use graph_export::{reset_trace_registry, write_graph_json_if_requested};
pub use tracing_sender::TracingSender;
