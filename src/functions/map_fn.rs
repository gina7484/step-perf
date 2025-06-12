use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;

/// matmul
/// - `write_back_mu`: Whether the output is written to a memory unit. <br/>
///     - If yes, the `read_from_mu` field of output tile should be set to this value
///     so that the next unit receiving the tile knows it's reading in a tile that was
///     stored in a memory unit and add load latency accordingly
/// - `weight_transposed`: Set this field to true if weight is stored in a transposed
///     way to optimize memory access
use std::fmt::Debug;

pub fn matmul<T: Debug + ndarray::LinalgScalar>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    weight_transposed: bool,
) -> (u64, Tile<T>) {
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

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            // println!("in1: {:?}", arr1);
            // println!("in2: {:?}", arr2);
            let out_arr = match weight_transposed {
                true => arr1.dot(&arr2.t()),
                false => arr1.dot(arr2),
            };
            // println!("out_arr: {:?}", out_arr);

            (
                div_ceil((2 * m * k * n) as u64, flop_per_cycle),
                Tile::new(out_arr.to_shared(), in1.bytes_per_elem, write_back_mu),
            )
        }
        (_, _) => (
            div_ceil((2 * m * k * n) as u64, flop_per_cycle),
            Tile::new_blank(vec![m, n], in1.bytes_per_elem, write_back_mu),
        ),
    }
}


pub fn retile_col<T: Debug + ndarray::LinalgScalar>(
    in_data: &Tile<T>,
    accumulator: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(accumulator.shape.len(), 2);
    let in_arr = in_data.underlying.clone().unwrap();
    let cur_arr = accumulator.underlying.clone().unwrap();

    (0, 
    ndarray::concatenate(ndarray::Axis(1), &[cur_arr.view(), in_arr.view()])
        .map(|arr| Tile::new(arr.to_shared(), in_data.bytes_per_elem, in_data.read_from_mu))
        .unwrap_or_else(|_| {
            panic!("Failed to concatenate input data and accumulator data")
        }))

}