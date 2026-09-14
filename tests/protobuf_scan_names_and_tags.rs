use prost::Message;
use step_perf::proto_driver::proto_headers::graph_proto::{
    operation::OpType, Operation, Scan,
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
    let gina = encoded(OpType::Scan(Scan::default()));

    // oneof tag 55, wire type 2 => varint key 0x01ba => ba 03
    assert!(gina.windows(2).any(|bytes| bytes == [0xba, 0x03]));

    let decoded_gina = Operation::decode(gina.as_slice()).unwrap();
    assert!(matches!(decoded_gina.op_type, Some(OpType::Scan(_))));
}

#[test]
fn retired_scan_wire_tag_is_unknown() {
    // Retired oneof field 49, containing an empty length-delimited message.
    // Old graphs must not be interpreted as the rank-delimited Scan (tag 55).
    let decoded = Operation::decode([0x8a, 0x03, 0x00].as_slice()).unwrap();
    assert!(decoded.op_type.is_none());
}
