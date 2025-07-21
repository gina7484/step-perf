use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;

/// matmul
/// - `write_back_mu`: Whether the output is written to a memory unit. <br/>
///     - If yes, the `read_from_mu` field of output tile should be set to this value
///     so that the next unit receiving the tile knows it's reading in a tile that was
///     stored in a memory unit and add load latency accordingly
/// - `weight_transposed`: Set this field to true if weight is stored in a transposed
///     way to optimize memory access
pub fn matmul<T: ndarray::LinalgScalar>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    accumulator: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    weight_transposed: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    if !weight_transposed {
        assert_eq!(in1.shape[1], in2.shape[0]); // reduction dim has to be the same (K)
        assert_eq!(accumulator.shape[0], in1.shape[0]); // accumulator shape check (M)
        assert_eq!(accumulator.shape[1], in2.shape[1]); // accumulator shape check (N)
    } else {
        assert_eq!(in1.shape[1], in2.shape[1]); // reduction dim has to be the same (K)
        assert_eq!(accumulator.shape[0], in1.shape[0]); // accumulator shape check (M)
        assert_eq!(accumulator.shape[1], in2.shape[0]); // accumulator shape check (N)
    }
    assert_eq!(in1.bytes_per_elem, in2.bytes_per_elem); // has to be represented in the same data type

    // offset is propagated from the input tile
    let offset = in1.offset;

    let m = in1.shape[0];
    let k = in1.shape[1];
    let n = if !weight_transposed {
        in2.shape[1] // in2: [K,N]
    } else {
        in2.shape[0] // in2: [N,K]
    };

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            let map_arr = match weight_transposed {
                true => arr1.dot(&arr2.t()),
                false => arr1.dot(arr2),
            };
            let out_arr = match &accumulator.underlying {
                Some(arr) => arr + map_arr,
                None => {
                    panic!("Accumulator tile must have an underlying array for matmul operation")
                }
            };
            (
                div_ceil((2 * m * k * n) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        (_, _) => (
            div_ceil((2 * m * k * n) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![m, n], in1.bytes_per_elem, write_back_mu, offset),
        ),
    }
}

/// - `write_back_mu`: Whether the output is written to a memory unit. <br/>
///     - If yes, the `read_from_mu` field of output tile should be set to this value
///     so that the next unit receiving the tile knows it's reading in a tile that was
///     stored in a memory unit and add load latency accordingly
/// - `weight_transposed`: Set this field to true if weight is stored in a transposed
///     way to optimize memory access
pub fn dyn_matmul<T: ndarray::LinalgScalar>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    accumulator: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    weight_transposed: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    if !weight_transposed {
        assert_eq!(in1.shape[1], in2.shape[0]); // reduction dim has to be the same (K)
        if accumulator.shape[0] != 0 && accumulator.shape[1] != 0 {
            // check if accumulator is not empty
            assert_eq!(accumulator.shape[0], in1.shape[0]); // accumulator shape check (M)
            assert_eq!(accumulator.shape[1], in2.shape[1]); // accumulator shape check (N)
        }
    } else {
        assert_eq!(in1.shape[1], in2.shape[1]); // reduction dim has to be the same (K)
        if accumulator.shape[0] != 0 && accumulator.shape[1] != 0 {
            // check if accumulator is not empty
            assert_eq!(accumulator.shape[0], in1.shape[0]); // accumulator shape check (M)
            assert_eq!(accumulator.shape[1], in2.shape[0]); // accumulator shape check (N)
        }
    }
    assert_eq!(in1.bytes_per_elem, in2.bytes_per_elem); // has to be represented in the same data type

    // offset is propagated from the input tile
    let offset = in1.offset;

    let m = in1.shape[0];
    let k = in1.shape[1];
    let n = if !weight_transposed {
        in2.shape[1] // in2: [K,N]
    } else {
        in2.shape[0] // in2: [N,K]
    };

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            let map_arr = match weight_transposed {
                true => arr1.dot(&arr2.t()),
                false => arr1.dot(arr2),
            };
            let out_arr = match &accumulator.underlying {
                Some(acc_arr) => {
                    if accumulator.shape[0] == 0 || accumulator.shape[1] == 0 {
                        map_arr
                        // This can happen if the accumulator is a dynamically sized tile.
                        // In this case, we use the first input tile as the accumulator.
                        // (i.e., allocation happend dynamically when the first input comes in)
                    } else {
                        acc_arr + map_arr
                    }
                }
                None => {
                    panic!("Accumulator tile must have an underlying array for matmul operation")
                }
            };
            (
                div_ceil((2 * m * k * n) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        (_, _) => (
            div_ceil((2 * m * k * n) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![m, n], in1.bytes_per_elem, write_back_mu, offset),
        ),
    }
}
