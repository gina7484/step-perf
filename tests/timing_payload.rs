//! A functionally serialized graph must also run in timing mode.
use step_perf::proto_driver::{parse_proto, configs::SimConfig, proto_headers::graph_proto::*};
use step_perf::ramulator::hbm_context::HBMConfig;

fn float_type() -> Option<DataType> {
    Some(DataType { r#type: Some(data_type::Type::F32(F32::default())) })
}

#[test]
fn timing_mode_ignores_float_payload_but_preserves_routing_masks() {
    let dir = std::env::temp_dir().join(format!("step-perf-timing-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data_path = dir.join("data.npy");
    // Valid NPY v1 header for one float32 tile.
    let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 1), }".to_string();
    while (10 + header.len() + 1) % 64 != 0 { header.push(' '); }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&7.0f32.to_le_bytes());
    std::fs::write(&data_path, bytes).unwrap();
    let mask_path = dir.join("mask.npy");
    let mut header = "{'descr': '<i8', 'fortran_order': False, 'shape': (1, 1), }".to_string();
    while (10 + header.len() + 1) % 64 != 0 { header.push(' '); }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1i64.to_le_bytes());
    std::fs::write(&mask_path, bytes).unwrap();
    let mask_type = Some(DataType { r#type: Some(data_type::Type::MultiHot(MultiHot { width: 1 })) });
    let op = |id, op_type| Operation { id, op_type: Some(op_type), ..Default::default() };
    let graph = ProgramGraph { operators: vec![
        op(1, operation::OpType::LinearOffChipLoad(LinearOffChipLoad {
            tensor_shape_tiled: vec![1, 1], stride: vec![1], out_shape_tiled: vec![1],
            tile_row: 1, tile_col: 1, n_byte: 4, par_dispatch: 1,
            npy_path: Some(data_path.to_string_lossy().into()), dtype: float_type(), ..Default::default()
        })),
        op(2, operation::OpType::SelectGen(SelectGen {
            npy_path: mask_path.to_string_lossy().into(), is_multihot: true,
            dtype: mask_type.clone(), tensor_shape_tiled: vec![1],
        })),
        op(3, operation::OpType::Shuffle(Shuffle {
            input_id: 1, index_id: 2, rank: 1, input_dtype: float_type(), index_dtype: mask_type.clone(), ..Default::default()
        })),
        op(4, operation::OpType::Accum(Accum {
            input_id: 3, stream_idx: Some(0), rank: 1, tile_row: 0, tile_col: 1, compute_bw: 1,
            func: Some(AccumFunc { accum_fn: Some(accum_func::AccumFn::RetileRow(RetileRow {})) }),
            init_func: Some(InitFunc { init_fn: Some(init_func::InitFn::Empty(Empty {})) }),
            dtype_a: float_type(), dtype_b: float_type(), ..Default::default()
        })),
        op(5, operation::OpType::ConsumerContext(ConsumerContext { input_id: 4, dtype: float_type(), ..Default::default() })),
        op(6, operation::OpType::ConsumerContext(ConsumerContext { input_id: 3, stream_idx: Some(1), dtype: mask_type })),
    ], ..Default::default() };
    let hbm = HBMConfig { addr_offset: 4, channel_num: 1, per_channel_latency: 1, per_channel_init_interval: 1, per_channel_outstanding: 4, per_channel_start_up_time: 0 };
    for functional_sim in [false, true] {
        let (passed, cycles, _) = parse_proto(graph.clone(), false, hbm.clone(), SimConfig {
            channel_depth: Some(2), functional_sim, mock_bf16: false, config_dict: Default::default(),
        }, None, None);
        assert!(passed, "functional_sim={functional_sim}");
        assert!(cycles > 0);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
