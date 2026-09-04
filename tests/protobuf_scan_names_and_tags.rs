use prost::Message;
use step_perf::proto_driver::proto_headers::graph_proto::{
    operation::OpType, NathanScan, Operation, Scan,
};

fn encoded(op_type: OpType) -> Vec<u8> {
    Operation {
        name: "scan".to_string(),
        id: 7,
        group_id: 0,
        out_stream_shape: vec![],
        op_type: Some(op_type),
        parallel_factor: 1,
    }
    .encode_to_vec()
}

#[test]
fn protobuf_scan_names_and_tags() {
    let legacy = encoded(OpType::NathanScan(NathanScan::default()));
    let gina = encoded(OpType::Scan(Scan::default()));

    // oneof tag 49, wire type 2 => varint key 0x018a => 8a 03
    assert!(legacy.windows(2).any(|bytes| bytes == [0x8a, 0x03]));
    // oneof tag 55, wire type 2 => varint key 0x01ba => ba 03
    assert!(gina.windows(2).any(|bytes| bytes == [0xba, 0x03]));

    let decoded_legacy = Operation::decode(legacy.as_slice()).unwrap();
    let decoded_gina = Operation::decode(gina.as_slice()).unwrap();
    assert!(matches!(decoded_legacy.op_type, Some(OpType::NathanScan(_))));
    assert!(matches!(decoded_gina.op_type, Some(OpType::Scan(_))));
}
