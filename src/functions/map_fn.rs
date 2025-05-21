use crate::memory::data::Tile;
use crate::utils::calculation::div_ceil;

/// matmul
/// - `write_back_mu`: Whether the output is written to a memory unit. <br/>
///     - If yes, the `read_from_mu` field of output tile should be set to this value
///     so that the next unit receiving the tile knows it's reading in a tile that was
///     stored in a memory unit and add load latency accordingly
/// - `weight_transposed`: Set this field to true if weight is stored in a transposed
///     way to optimize memory access
pub fn matmul(
    in1: &Tile,
    in2: &Tile,
    flop_per_cycle: u64,
    write_back_mu: bool,
    weight_transposed: bool,
) -> (u64, Tile) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    if !weight_transposed {
        assert_eq!(in1.shape[1], in2.shape[0]); // reduction dim has to be the same (K)
    } else {
        assert_eq!(in1.shape[1], in2.shape[1]); // reduction dim has to be the same (K)
    }
    assert_eq!(in1.bytes_per_elem, in2.bytes_per_elem);

    let m = in1.shape[0];
    let k = in1.shape[1];
    let n = if !weight_transposed {
        in2.shape[1] // in2: [K,N]
    } else {
        in2.shape[0] // in2: [N,K]
    };

    (
        div_ceil((2 * m * k * n) as u64, flop_per_cycle),
        Tile {
            shape: vec![m, n],
            bytes_per_elem: in1.bytes_per_elem,
            read_from_mu: write_back_mu,
        },
    )
}
