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

    (
        0,
        ndarray::concatenate(ndarray::Axis(1), &[cur_arr.view(), in_arr.view()])
            .map(|arr| {
                Tile::new(
                    arr.to_shared(),
                    in_data.bytes_per_elem,
                    in_data.read_from_mu,
                )
            })
            .unwrap_or_else(|_| panic!("Failed to concatenate input data and accumulator data")),
    )
}

pub fn retile_row<T: Debug + ndarray::LinalgScalar>(
    in_data: &Tile<T>,
    accumulator: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(accumulator.shape.len(), 2);

    let offset = accumulator.offset;

    let in_arr = in_data.underlying.clone().unwrap();
    let cur_arr = accumulator.underlying.clone().unwrap();

    (
        0, // TODO: Add cycles it took for grouping smaller tiles into larger tiles
        ndarray::concatenate(ndarray::Axis(0), &[cur_arr.view(), in_arr.view()])
            .map(|arr| {
                Tile::new_padded(
                    arr.to_shared(),
                    in_data.bytes_per_elem,
                    in_data.read_from_mu,
                    offset + 1,
                )
            })
            .unwrap_or_else(|_| panic!("Failed to concatenate input data and accumulator data")),
    )
}

pub fn mul<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    let in1_shape_0 = in1.shape[0];
    let in1_shape_1 = in1.shape[1];
    let in2_shape_0 = in2.shape[0];
    let in2_shape_1 = in2.shape[1];
    assert!((in1_shape_0 == in2_shape_0) || (in1_shape_0 == 1) || (in2_shape_0 == 1));
    assert!((in1_shape_1 == in2_shape_1) || (in1_shape_1 == 1) || (in2_shape_1 == 1));

    let out_shape_0 = in1_shape_0.max(in2_shape_0);
    let out_shape_1 = in1_shape_1.max(in2_shape_1);

    let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
    for i in 0..out_shape_0 {
        for j in 0..out_shape_1 {
            let i0 = i.min(in1_shape_0 - 1);
            let j0 = j.min(in1_shape_1 - 1);
            let val1 = in1.underlying.as_ref().unwrap().get((i0, j0)).unwrap();
            let i1 = i.min(in2_shape_0 - 1);
            let j1 = j.min(in2_shape_1 - 1);
            let val2 = in2.underlying.as_ref().unwrap().get((i1, j1)).unwrap();
            let out_val = val1.mul(*val2);
            out_arr[[i, j]] = out_val;
        }
    }
    (
        div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
        Tile::new(out_arr.to_shared(), in1.bytes_per_elem, write_back_mu),
    )
}

pub fn add<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    let in1_shape_0 = in1.shape[0];
    let in1_shape_1 = in1.shape[1];
    let in2_shape_0 = in2.shape[0];
    let in2_shape_1 = in2.shape[1];
    assert!((in1_shape_0 == in2_shape_0) || (in1_shape_0 == 1) || (in2_shape_0 == 1));
    assert!((in1_shape_1 == in2_shape_1) || (in1_shape_1 == 1) || (in2_shape_1 == 1));

    let out_shape_0 = in1_shape_0.max(in2_shape_0);
    let out_shape_1 = in1_shape_1.max(in2_shape_1);

    let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
    for i in 0..out_shape_0 {
        for j in 0..out_shape_1 {
            let i0 = i.min(in1_shape_0 - 1);
            let j0 = j.min(in1_shape_1 - 1);
            let val1 = in1.underlying.as_ref().unwrap().get((i0, j0)).unwrap();
            let i1 = i.min(in2_shape_0 - 1);
            let j1 = j.min(in2_shape_1 - 1);
            let val2 = in2.underlying.as_ref().unwrap().get((i1, j1)).unwrap();
            let out_val = val1.add(*val2);
            out_arr[[i, j]] = out_val;
        }
    }
    (
        div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
        Tile::new(out_arr.to_shared(), in1.bytes_per_elem, write_back_mu),
    )
}

// SiLU(x)= x / (1 + e^-x)
pub fn silu<T: Debug + ndarray::LinalgScalar + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new(
                arr.mapv(|x| x / (T::one() + (-x).exp())).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
            ),
        ),
        None => panic!("Input tile does not contain underlying data"),
    }
}
