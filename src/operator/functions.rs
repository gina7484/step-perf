use crate::memory::data::DataSizeInfo;
use crate::utils::calculation::div_ceil;

pub fn matmul(
    in1: DataSizeInfo,
    in2: DataSizeInfo,
    flop_per_cycle: u64,
    read_from_mu: bool,
) -> (u64, DataSizeInfo) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    assert_eq!(in1.shape[1], in2.shape[0]);
    assert_eq!(in1.bytes_per_elem, in2.bytes_per_elem);

    let m = in1.shape[0];
    let k = in1.shape[1];
    let n = in2.shape[1];

    (
        div_ceil((2 * m * k * n) as u64, flop_per_cycle),
        DataSizeInfo {
            shape: vec![m, n],
            bytes_per_elem: in1.bytes_per_elem,
            read_from_mu: read_from_mu,
        },
    )
}
