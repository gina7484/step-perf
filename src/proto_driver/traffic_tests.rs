use super::proto_headers::graph_proto::{
    DataType, LinearOffChipLoad, OffChipStore, Operation, F32,
};
use super::*;

#[test]
fn graph_traffic_counts_full_requests_and_hbm_bypass() {
    for simulate_ramulator in [true, false] {
        let dtype = Some(DataType {
            r#type: Some(Type::F32(F32::default())),
        });
        let graph = ProgramGraph {
            name: "traffic_test".to_string(),
            operators: vec![
                Operation {
                    id: 0,
                    dtype_bytes: 4,
                    op_type: Some(OpType::LinearOffChipLoad(LinearOffChipLoad {
                        tensor_shape_tiled: vec![1, 3],
                        stride: vec![3, 1],
                        out_shape_tiled: vec![1, 3],
                        tile_row: 2,
                        tile_col: 9,
                        dtype: dtype.clone(),
                        par_dispatch: 3,
                        simulate_ramulator,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
                Operation {
                    id: 1,
                    dtype_bytes: 4,
                    op_type: Some(OpType::OffChipStore(OffChipStore {
                        input_id: 0,
                        tensor_shape_tiled: vec![1, 3],
                        tile_row: 2,
                        tile_col: 9,
                        dtype,
                        par_dispatch: 3,
                        ..Default::default()
                    })),
                    ..Default::default()
                },
            ],
        };
        let hbm_config = HBMConfig {
            addr_offset: 32,
            channel_num: 2,
            per_channel_latency: 3,
            per_channel_init_interval: 2,
            per_channel_outstanding: 1,
            per_channel_start_up_time: 5,
        };
        let sim_config = SimConfig {
            channel_depth: Some(1),
            config_dict: HashMap::new(),
        };
        let mut builder = ProgramBuilder::default();
        let stats = build_from_proto(
            graph.clone(),
            &mut ChannelMapCollection::default(),
            &mut builder,
            &hbm_config,
            &sim_config,
            None,
        );
        let executed = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        assert!(executed.passed());

        // Three tiles, two rows per tile, and two 32-byte requests per 36-byte row.
        let write_bytes = 3 * 2 * 2 * 32;
        let read_bytes = if simulate_ramulator { write_bytes } else { 0 };
        assert_eq!(stats.read_bytes(), read_bytes);
        assert_eq!(stats.write_bytes(), write_bytes);
        assert_eq!(stats.total_bytes(), read_bytes + write_bytes);

        let (passed, cycles, _, returned_stats) =
            parse_proto(graph, false, hbm_config, sim_config, None, None);
        assert!(passed);
        assert!(cycles > 0);
        assert_eq!(returned_stats.read_bytes(), read_bytes);
        assert_eq!(returned_stats.write_bytes(), write_bytes);
        assert_eq!(returned_stats.total_bytes(), read_bytes + write_bytes);
    }
}
