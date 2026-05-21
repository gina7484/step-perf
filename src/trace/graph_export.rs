//! Export `step_perf_graph.json` using the same `(producer_id, stream_idx) → ch` map as the trace.

use std::collections::HashMap;

use crate::proto_driver::proto_headers::graph_proto::{operation::OpType, Operation, ProgramGraph};

use super::channel_trace::{alloc_trace_ch, ch_map_snapshot};

fn opt_stream(v: Option<u32>) -> u32 {
    v.unwrap_or(0)
}

fn push_input(inputs: &mut Vec<(u32, u32)>, src_id: u32, stream_idx: Option<u32>) {
    // Operator id 0 is valid (e.g. LinearOffChipLoad_0 → Broadcast_58). Unset inputs are
    // omitted per-op in `extract_inputs`, not filtered here.
    inputs.push((src_id, opt_stream(stream_idx)));
}

fn extract_inputs(op: &Operation) -> Vec<(u32, u32)> {
    let mut inputs = Vec::new();
    let Some(ref ot) = op.op_type else {
        return inputs;
    };
    match ot {
        OpType::Binarymap(b) => {
            push_input(&mut inputs, b.input_id1, b.stream_idx1);
            push_input(&mut inputs, b.input_id2, b.stream_idx2);
        }
        OpType::Unarymap(u) => push_input(&mut inputs, u.input_id, u.stream_idx),
        OpType::BinarymapAccum(b) => {
            push_input(&mut inputs, b.input_id1, b.stream_idx1);
            push_input(&mut inputs, b.input_id2, b.stream_idx2);
        }
        OpType::Accum(a) => push_input(&mut inputs, a.input_id, a.stream_idx),
        OpType::Bufferize(b) => push_input(&mut inputs, b.input_id, b.stream_idx),
        OpType::Streamify(s) => push_input(&mut inputs, s.input_id, s.stream_idx),
        OpType::DynStreamify(d) => {
            push_input(&mut inputs, d.input_id, d.input_stream_idx);
            push_input(&mut inputs, d.ref_id, d.ref_stream_idx);
        }
        OpType::Broadcast(b) => push_input(&mut inputs, b.input_id, b.stream_idx),
        OpType::FlatPartition(p) => {
            push_input(&mut inputs, p.input_id, p.input_stream_idx);
            push_input(&mut inputs, p.control_id, p.control_stream_idx);
        }
        OpType::Parallelize(p) => push_input(&mut inputs, p.input_id, p.input_stream_idx),
        OpType::FlatReassemble(r) => {
            for (i, &src_id) in r.input_id_list.iter().enumerate() {
                let sid = r
                    .input_stream_idx_list
                    .get(i)
                    .copied()
                    .unwrap_or(0);
                let sid = if sid < 0 { 0 } else { sid as u32 };
                push_input(&mut inputs, src_id, Some(sid));
            }
            push_input(&mut inputs, r.control_id, r.control_stream_idx);
        }
        OpType::Promote(p) => push_input(&mut inputs, p.input_id, p.stream_idx),
        OpType::PromoteOuter(p) => push_input(&mut inputs, p.input_id, p.stream_idx),
        OpType::Flatten(f) => push_input(&mut inputs, f.input_id, f.stream_idx),
        OpType::RepeatStatic(r) => push_input(&mut inputs, r.input_id, r.stream_idx),
        OpType::RepeatRef(r) => {
            push_input(&mut inputs, r.input_id, r.input_stream_idx);
            push_input(&mut inputs, r.ref_id, r.ref_stream_idx);
        }
        OpType::ExpandRef(e) => {
            push_input(&mut inputs, e.input_id, e.stream_idx);
            push_input(&mut inputs, e.ref_id, e.ref_stream_idx);
        }
        OpType::LinearOffChipLoadRef(l) => {
            push_input(&mut inputs, l.ref_id, l.ref_stream_idx);
        }
        OpType::OffChipStore(s) => push_input(&mut inputs, s.input_id, s.stream_idx),
        OpType::DynOffChipStore(s) => push_input(&mut inputs, s.input_id, s.stream_idx),
        OpType::RandomOffChipStore(s) => {
            push_input(&mut inputs, s.wdata_id, s.wdata_stream_idx);
            push_input(&mut inputs, s.waddr_id, s.waddr_stream_idx);
        }
        OpType::RandomOffChipLoad(l) => {
            push_input(&mut inputs, l.raddr_id, l.raddr_stream_idx);
        }
        OpType::PrinterContext(p) => push_input(&mut inputs, p.input_id, p.stream_idx),
        OpType::ConsumerContext(c) => push_input(&mut inputs, c.input_id, c.stream_idx),
        OpType::EagerMerge(e) => {
            for (i, &src_id) in e.input_id_list.iter().enumerate() {
                let sid = e.input_stream_idx_list.get(i).copied().unwrap_or(0);
                let sid = if sid < 0 { 0 } else { sid as u32 };
                push_input(&mut inputs, src_id, Some(sid));
            }
        }
        OpType::StaticReassemble(r) => {
            for (i, &src_id) in r.input_id_list.iter().enumerate() {
                let sid = r.input_stream_idx_list.get(i).copied().unwrap_or(0);
                let sid = if sid < 0 { 0 } else { sid as u32 };
                push_input(&mut inputs, src_id, Some(sid));
            }
        }
        OpType::ReshapePadStream(r) => push_input(&mut inputs, r.input_id, r.stream_idx),
        OpType::RetileStreamify(r) => push_input(&mut inputs, r.input_id, r.stream_idx),
        OpType::FlatmapFilterRowStreamify(f) => push_input(&mut inputs, f.input_id, f.stream_idx),
        OpType::FlatmapCounter(_) | OpType::SelectGen(_) | OpType::MetadataGen(_) => {}
        OpType::LinearOffChipLoad(_) | OpType::DynLinearOffChipLoad(_) => {}
        OpType::ExpertAddrGen(_)
        | OpType::CacheReadAddrGen(_)
        | OpType::FilterLastTile(_)
        | OpType::Reshape(_) => {}
    }
    inputs
}

fn count_outputs(op: &Operation) -> u32 {
    let Some(ref ot) = op.op_type else {
        return 1;
    };
    match ot {
        OpType::Broadcast(b) => b.num_consumers.max(1),
        OpType::FlatPartition(p) => p.num_consumers.max(1),
        OpType::Parallelize(p) => p.num_consumers.max(1),
        _ => (op.out_stream_shape.len().max(1)) as u32,
    }
}

fn op_display(op: &Operation) -> String {
    match &op.op_type {
        Some(OpType::Binarymap(_)) => "BinaryMap",
        Some(OpType::Unarymap(_)) => "UnaryMap",
        Some(OpType::BinarymapAccum(_)) => "BinaryMapAccum",
        Some(OpType::Accum(_)) => "Accum",
        Some(OpType::Bufferize(_)) => "Bufferize",
        Some(OpType::Streamify(_)) => "Streamify",
        Some(OpType::DynStreamify(_)) => "DynStreamify",
        Some(OpType::Broadcast(_)) => "Broadcast",
        Some(OpType::FlatPartition(_)) => "FlatPartition",
        Some(OpType::Parallelize(_)) => "Parallelize",
        Some(OpType::FlatReassemble(_)) => "FlatReassemble",
        Some(OpType::Promote(_)) => "Promote",
        Some(OpType::PromoteOuter(_)) => "PromoteOuter",
        Some(OpType::Flatten(_)) => "Flatten",
        Some(OpType::RepeatStatic(_)) => "RepeatStatic",
        Some(OpType::RepeatRef(_)) => "RepeatRef",
        Some(OpType::ExpandRef(_)) => "ExpandRef",
        Some(OpType::LinearOffChipLoad(_)) => "LinearOffChipLoad",
        Some(OpType::LinearOffChipLoadRef(_)) => "LinearOffChipLoadRef",
        Some(OpType::DynLinearOffChipLoad(_)) => "DynLinearOffChipLoad",
        Some(OpType::OffChipStore(_)) => "OffChipStore",
        Some(OpType::DynOffChipStore(_)) => "DynOffChipStore",
        Some(OpType::RandomOffChipStore(_)) => "RandomOffChipStore",
        Some(OpType::RandomOffChipLoad(_)) => "RandomOffChipLoad",
        Some(OpType::PrinterContext(_)) => "PrinterContext",
        Some(OpType::ConsumerContext(_)) => "ConsumerContext",
        Some(OpType::EagerMerge(_)) => "EagerMerge",
        Some(OpType::SelectGen(_)) => "SelectGen",
        Some(OpType::MetadataGen(_)) => "MetadataGen",
        Some(OpType::ReshapePadStream(_)) => "ReshapePadStream",
        Some(OpType::RetileStreamify(_)) => "RetileStreamify",
        Some(OpType::FlatmapFilterRowStreamify(_)) => "FlatmapFilterRowStreamify",
        Some(OpType::FlatmapCounter(_)) => "FlatmapCounter",
        Some(OpType::ExpertAddrGen(_)) => "ExpertAddrGen",
        Some(OpType::CacheReadAddrGen(_)) => "CacheReadAddrGen",
        Some(OpType::FilterLastTile(_)) => "FilterLastTile",
        Some(OpType::StaticReassemble(_)) => "StaticReassemble",
        Some(OpType::Reshape(_)) => "Reshape",
        None => "Unknown",
    }
    .to_string()
}

/// Clear channel registry (call once per simulation before `build_from_proto`).
pub fn reset_trace_registry() {
    super::channel_trace::reset_trace_registry();
}

/// If `STEP_PERF_GRAPH_JSON` is set, write graph nodes with trace channel ids.
pub fn write_graph_json_if_requested(graph: &ProgramGraph) -> std::io::Result<()> {
    let Ok(path) = std::env::var("STEP_PERF_GRAPH_JSON") else {
        return Ok(());
    };
    let path = path.trim();
    if path.is_empty() {
        return Ok(());
    }

    let ch_map = ch_map_snapshot();
    let ops_by_id: HashMap<u32, &Operation> = graph.operators.iter().map(|o| (o.id, o)).collect();

    let mut nodes = Vec::new();
    for op in &graph.operators {
        let op_id = op.id;
        let mut in_channels = Vec::new();
        for (src_id, stream_idx) in extract_inputs(op) {
            if ops_by_id.contains_key(&src_id) {
                if let Some(&ch) = ch_map.get(&(src_id, stream_idx)) {
                    in_channels.push(ch);
                }
            }
        }

        let n_out = count_outputs(op);
        let mut out_pairs: Vec<(u32, u64)> = ch_map
            .iter()
            .filter(|((pid, _s), _)| *pid == op_id)
            .map(|((_, s), c)| (*s, *c))
            .collect();
        out_pairs.sort_by_key(|(s, _)| *s);
        let out_channels: Vec<u64> = out_pairs.into_iter().map(|(_, c)| c).collect();

        let mut node = serde_json::json!({
            "graph_node_id": op_id,
            "operator": op_display(op),
            "name": op.name,
            "in_channels": in_channels,
            "out_channels": out_channels,
        });

        if matches!(
            &op.op_type,
            Some(OpType::OffChipStore(_))
                | Some(OpType::RandomOffChipStore(_))
                | Some(OpType::DynOffChipStore(_))
        ) {
            node["file_path_id"] = serde_json::json!(op_id);
            if let Some(OpType::OffChipStore(s)) = &op.op_type {
                if !s.tensor_shape_tiled.is_empty() {
                    node["tensor_shape_tiled"] = serde_json::json!(
                        s.tensor_shape_tiled.iter().map(|x| x.to_string()).collect::<Vec<_>>()
                    );
                }
            }
        }

        nodes.push(node);
    }

    let doc = serde_json::json!({
        "nodes": nodes,
        "trace_hint": "Channel ids match step_perf_trace.jsonl (`ch` field). Built after build_from_proto.",
        "n_channels_registered": ch_map.len(),
    });
    std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap())?;
    eprintln!(
        "[step_perf] wrote graph JSON ({} nodes, {} channels) → {}",
        graph.operators.len(),
        ch_map.len(),
        path
    );
    Ok(())
}

/// Ensure a channel id exists (used by tests); normal sim uses [`alloc_trace_ch`].
#[allow(dead_code)]
pub fn ensure_trace_ch(producer_id: u32, stream_idx: u32) -> u64 {
    alloc_trace_ch(producer_id, stream_idx)
}
