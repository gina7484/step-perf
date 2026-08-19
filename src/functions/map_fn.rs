use ndarray::Array2;

use crate::primitives::select::{MultiHotN, SelectAdapter};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use dam::types::DAMType;

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
            // println!("in1: {:?}", arr1);
            // println!("in2: {:?}", arr2);
            let out_arr = match weight_transposed {
                true => arr1.dot(&arr2.t()),
                false => arr1.dot(arr2),
            };
            // println!("out_arr: {:?}", out_arr);

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

pub fn div<T: Debug + ndarray::LinalgScalar + Default>(
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

    let offset = if in1_shape_0 == in2_shape_0 {
        in1.offset.max(in2.offset)
    } else if in1_shape_0 == 1 {
        in2.offset
    } else {
        // in2_shape_0 == 1
        in1.offset
    };

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
            for i in 0..out_shape_0 {
                for j in 0..out_shape_1 {
                    let i0 = i.min(in1_shape_0 - 1);
                    let j0 = j.min(in1_shape_1 - 1);
                    let val1 = arr1.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = arr2.get((i1, j1)).unwrap();
                    let out_val = val1.div(*val2);
                    out_arr[[i, j]] = out_val;
                }
            }
            (
                div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        (_, _) => (
            div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![out_shape_0, out_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

pub fn sub<T: Debug + ndarray::LinalgScalar + Default>(
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

    let offset = if in1_shape_0 == in2_shape_0 {
        in1.offset.max(in2.offset)
    } else if in1_shape_0 == 1 {
        in2.offset
    } else {
        // in2_shape_0 == 1
        in1.offset
    };

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
            for i in 0..out_shape_0 {
                for j in 0..out_shape_1 {
                    let i0 = i.min(in1_shape_0 - 1);
                    let j0 = j.min(in1_shape_1 - 1);
                    let val1 = arr1.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = arr2.get((i1, j1)).unwrap();
                    let out_val = val1.sub(*val2);
                    out_arr[[i, j]] = out_val;
                }
            }
            (
                div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        (_, _) => (
            div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![out_shape_0, out_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
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

    let offset = if in1_shape_0 == in2_shape_0 {
        in1.offset.max(in2.offset)
    } else if in1_shape_0 == 1 {
        in2.offset
    } else {
        // in2_shape_0 == 1
        in1.offset
    };

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
            for i in 0..out_shape_0 {
                for j in 0..out_shape_1 {
                    let i0 = i.min(in1_shape_0 - 1);
                    let j0 = j.min(in1_shape_1 - 1);
                    let val1 = arr1.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = arr2.get((i1, j1)).unwrap();
                    let out_val = val1.mul(*val2);
                    out_arr[[i, j]] = out_val;
                }
            }
            (
                div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        (_, _) => (
            div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![out_shape_0, out_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
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

    let offset = if in1_shape_0 == in2_shape_0 {
        in1.offset.max(in2.offset)
    } else if in1_shape_0 == 1 {
        in2.offset
    } else {
        // in2_shape_0 == 1
        in1.offset
    };

    match (&in1.underlying, &in2.underlying) {
        (Some(arr1), Some(arr2)) => {
            let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
            for i in 0..out_shape_0 {
                for j in 0..out_shape_1 {
                    let i0 = i.min(in1_shape_0 - 1);
                    let j0 = j.min(in1_shape_1 - 1);
                    let val1 = arr1.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = arr2.get((i1, j1)).unwrap();
                    let out_val = val1.add(*val2);
                    out_arr[[i, j]] = out_val;
                }
            }
            (
                div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        (_, _) => (
            div_ceil((out_shape_0 * out_shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![out_shape_0, out_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

pub fn add_constant<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    constant: T,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    let in1_shape_0 = in1.shape[0];
    let in1_shape_1 = in1.shape[1];

    match &in1.underlying {
        Some(arr1) => {
            // Multiply all elements by the constant
            let out_arr = arr1.mapv(|x| x + constant);
            (
                div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle),
                Tile::new(out_arr.to_shared(), in1.bytes_per_elem, write_back_mu),
            )
        }
        None => (
            div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle),
            Tile::new_blank(
                vec![in1_shape_0, in1_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
            ),
        ),
    }
}

pub fn sub_constant<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    constant: T,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    let in1_shape_0 = in1.shape[0];
    let in1_shape_1 = in1.shape[1];

    match &in1.underlying {
        Some(arr1) => {
            // Multiply all elements by the constant
            let out_arr = arr1.mapv(|x| x - constant);
            (
                div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle),
                Tile::new(out_arr.to_shared(), in1.bytes_per_elem, write_back_mu),
            )
        }
        None => (
            div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle),
            Tile::new_blank(
                vec![in1_shape_0, in1_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
            ),
        ),
    }
}
// SiLU(x)= x / (1 + e^-x)
// We will count this as 8 FLOPs per element
pub fn silu<T: Debug + ndarray::LinalgScalar + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 8) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| x / (T::one() + (-x).exp())).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 8) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// exp(x) (~ 4 FLOPs per element)
pub fn exp<T: Debug + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| x.exp()).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// pow2(x) = 2^x (~ 4 FLOPs per element)
pub fn pow2<T: Debug + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| T::from(2.0).unwrap().powf(x)).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// tanh(x) (~ 4 FLOPs per element)
pub fn tanh<T: Debug + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| x.tanh()).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// pow(x, c) = x^c (~ 4 FLOPs per element)
pub fn pow<T: Debug + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    exponent: T,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| x.powf(exponent)).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// rsqrt(x) = 1/sqrt(x) (~ 4 FLOPs per element)
pub fn rsqrt<T: Debug + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| T::one() / x.sqrt()).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 4) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

pub fn row_wise_sum<T: Debug + num_traits::Num + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => {
            // Perform row-wise sum: sum each row to get a [shape_0, 1] array
            let row_sums = arr.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1));
            (
                div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    row_sums.to_shared(),
                    in_data.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, 1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

pub fn row_wise_max<T: Debug + PartialOrd + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => {
            // Perform row-wise max: reduce each row to get a [shape_0, 1] array.
            // A compare-and-select costs the same as an add in the roofline below,
            // so the cycle count matches `row_wise_sum`.
            let row_maxes: Vec<T> = arr
                .rows()
                .into_iter()
                .map(|row| {
                    row.iter()
                        .copied()
                        .reduce(|a, b| if b > a { b } else { a })
                        .expect("row_wise_max requires each row to be non-empty")
                })
                .collect();
            let row_maxes = Array2::from_shape_vec((shape_0, 1), row_maxes)
                .expect("row_wise_max: failed to reshape row maxima to [rows, 1]");
            (
                div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    row_maxes.to_shared(),
                    in_data.bytes_per_elem,
                    write_back_mu,
                    offset,
                ),
            )
        }
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, 1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

pub fn set_offset<T: Debug + ndarray::LinalgScalar + Default>(
    in_data: &Tile<T>,
    offset: &Tile<u64>,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset_val = offset.underlying.as_ref().unwrap()[[0, 0]];

    match &in_data.underlying {
        Some(arr) => (
            1,
            Tile::new_padded(
                arr.to_owned().into_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset_val as usize,
            ),
        ),
        None => (
            1,
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset_val as usize,
            ),
        ),
    }
}

pub fn mask_row<
    T: Debug + Default + Clone + TryInto<usize>,
    D: num_traits::Float + Debug + Default + Clone,
>(
    in_data: &Tile<T>,
    write_back_mu: bool,
    row: usize,
    col: usize,
    dtype_bytes: usize,
) -> (u64, Tile<D>) {
    assert_eq!(in_data.shape, vec![1, 1]);

    match &in_data.underlying {
        Some(arr) => {
            // Extract the index value from in_data
            let val = arr[[0, 0]].clone();
            let i = val
                .try_into()
                .unwrap_or_else(|_| panic!("Failed to convert value to usize"));
            // Create a zero-filled array of shape [row, col]
            let mut out_arr = Array2::<D>::default((row, col));

            // Set the i-th row to 1.0
            if i < row {
                for j in 0..col {
                    out_arr[[i, j]] = D::one();
                }
            }

            (
                1,
                Tile::new(out_arr.to_shared(), dtype_bytes, write_back_mu),
            )
        }
        None => (
            1,
            Tile::new_blank(vec![row, col], dtype_bytes, write_back_mu),
        ),
    }
}

pub fn row_wise_append<T: Debug + Default + Clone>(
    in_data: &Tile<T>,
    data_to_append: &Tile<T>,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(data_to_append.shape.len(), 2);
    assert_eq!(in_data.shape[1], data_to_append.shape[1]);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let offset = in_data.offset;
    assert!(
        offset + data_to_append.shape[0] <= shape_0,
        "should have enough space to append new rows"
    );

    match (&in_data.underlying, &data_to_append.underlying) {
        (Some(arr), Some(arr_to_append)) => {
            // Create a mutable copy of the original array
            let mut result = arr.to_owned();

            // Copy data from arr_to_append into the slice starting at offset
            let mut slice =
                result.slice_mut(ndarray::s![offset..offset + data_to_append.shape[0], ..]);
            slice.assign(arr_to_append);

            (
                1,
                Tile::new_padded(
                    result.into_shared(),
                    in_data.bytes_per_elem,
                    write_back_mu,
                    (offset + data_to_append.shape[0]) as usize,
                ),
            )
        }
        _ => (
            1,
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                (offset + data_to_append.shape[0]) as usize,
            ),
        ),
    }
}

pub fn cache_write_addr_gen(
    idx: &Tile<u64>,
    len: &Tile<u64>,
    offset_per_idx: u64,
    comp_bw: u64,
    write_back_mu: bool,
) -> (u64, Tile<u64>) {
    let idx_val = idx.underlying.as_ref().unwrap()[[0, 0]];
    let len_val = len.underlying.as_ref().unwrap()[[0, 0]];
    let addr = idx_val * offset_per_idx + len_val;

    (
        1,
        Tile::new(
            Array2::from_shape_vec((1, 1), vec![addr])
                .unwrap()
                .to_shared(),
            8,
            write_back_mu,
        ),
    )
}

pub fn is_equal_scalar<T: Default + Debug + Clone + PartialEq + Copy + From<u64>>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    write_back_mu: bool,
) -> (u64, MultiHotN) {
    // Check if shapes match first
    assert_eq!(in1.shape, vec![1, 1]);
    assert_eq!(in2.shape, vec![1, 1]);

    let in1_val = in1.underlying.as_ref().unwrap()[[0, 0]];
    let in2_val = in2.underlying.as_ref().unwrap()[[0, 0]];

    // Compare underlying data
    let is_equal = in1_val == in2_val;

    // Return [1, 0] if equal, [0, 1] if not equal
    if is_equal {
        (1, MultiHotN::new(vec![true, false], write_back_mu)) // 1
    } else {
        (1, MultiHotN::new(vec![false, true], write_back_mu)) // 0
    }
}

pub fn mul_constant<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    constant: T,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    let in1_shape_0 = in1.shape[0];
    let in1_shape_1 = in1.shape[1];

    match &in1.underlying {
        Some(arr1) => {
            // Multiply all elements by the constant
            let out_arr = arr1.mapv(|x| x * constant);
            (
                div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle),
                Tile::new(out_arr.to_shared(), in1.bytes_per_elem, write_back_mu),
            )
        }
        None => (
            div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle),
            Tile::new_blank(
                vec![in1_shape_0, in1_shape_1],
                in1.bytes_per_elem,
                write_back_mu,
            ),
        ),
    }
}

pub fn broadcast_rows<T: Debug + Clone + Default>(
    in_data: &Tile<T>,
    row_size: usize,
    _flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(
        in_data.shape[0], 1,
        "BroadcastRows input must have a single row, got shape {:?}",
        in_data.shape
    );

    let shape_1 = in_data.shape[1];

    match &in_data.underlying {
        Some(arr) => {
            let mut out_arr = Array2::<T>::default((row_size, shape_1));
            for i in 0..row_size {
                for j in 0..shape_1 {
                    out_arr[[i, j]] = arr[[0, j]].clone();
                }
            }
            (
                1,
                Tile::new(out_arr.to_shared(), in_data.bytes_per_elem, write_back_mu),
            )
        }
        None => (
            1,
            Tile::new_blank(
                vec![row_size, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
            ),
        ),
    }
}

pub fn select_to_scalar<SEL: SelectAdapter>(
    in_data: &SEL,
    _comp_bw: u64,
    write_back_mu: bool,
) -> (u64, Tile<u64>) {
    let sel_vec = in_data.to_sel_vec();
    let val = if sel_vec.is_empty() {
        0
    } else {
        sel_vec[0] as u64
    };
    (
        1,
        Tile::new(
            Array2::from_shape_vec((1, 1), vec![val])
                .unwrap()
                .to_shared(),
            8,
            write_back_mu,
        ),
    )
}

pub fn multihot_to_u64(
    in_data: &MultiHotN,
    _comp_bw: u64,
    write_back_mu: bool,
) -> (u64, Tile<u64>) {
    // Blank (timing-only) input carries no selection data, so we can't know how many
    // indices it would encode. Propagate a blank tile whose column count is the upper
    // bound (one per candidate) for buffer sizing.
    if in_data.is_blank() {
        return (
            1,
            Tile::new_blank(vec![1, in_data.len()], 8, write_back_mu),
        );
    }
    let sel_vec = in_data.to_sel_vec();
    let n = sel_vec.len();
    if n == 0 {
        return (
            1,
            Tile::new(
                Array2::from_shape_vec((1, 1), vec![0]).unwrap().to_shared(),
                8,
                write_back_mu,
            ),
        );
    }
    let vals: Vec<u64> = sel_vec.into_iter().map(|x| x as u64).collect();
    (
        1,
        Tile::new(
            Array2::from_shape_vec((1, n), vals).unwrap().to_shared(),
            8,
            write_back_mu,
        ),
    )
}

pub fn u64_to_multihot(
    in_data: &Tile<u64>,
    width: usize,
    _comp_bw: u64,
    write_back_mu: bool,
) -> (u64, MultiHotN) {
    match &in_data.underlying {
        Some(arr) => {
            let mut hot = vec![false; width];
            for &val in arr.iter() {
                let idx = val as usize;
                if idx < width {
                    hot[idx] = true;
                }
            }
            (1, MultiHotN::new(hot, write_back_mu))
        }
        None => (1, MultiHotN::new_blank(width, write_back_mu)),
    }
}

// index_to_multihot(x): encode the index values held in a single-row tile into a
// multihot vector of `num_classes` booleans (position i set iff i appears in x).
// Generic over the element type so it works for both integer index tiles (i64/u64,
// e.g. topk indices) and f32 index tiles; each value is cast to a usize index.
// Costs 1 cycle per input tile regardless of size.
pub fn index_to_multihot<T: DAMType + num_traits::AsPrimitive<usize>>(
    in_data: &Tile<T>,
    num_classes: usize,
    _comp_bw: u64,
    write_back_mu: bool,
) -> (u64, MultiHotN) {
    match &in_data.underlying {
        Some(arr) => {
            let mut hot = vec![false; num_classes];
            for &val in arr.iter() {
                let idx: usize = val.as_();
                if idx < num_classes {
                    hot[idx] = true;
                }
            }
            (1, MultiHotN::new(hot, write_back_mu))
        }
        None => (1, MultiHotN::new_blank(num_classes, write_back_mu)),
    }
}

pub fn to_const_int<T: DAMType>(_: &T, constant: u64, write_back_mu: bool) -> (u64, Tile<u64>) {
    (
        1,
        Tile::new(
            Array2::from_shape_vec((1, 1), vec![constant])
                .unwrap()
                .to_shared(),
            8,
            write_back_mu,
        ),
    )
}

// ============================================================================
// Elementwise functions added for ATen op lowering
// (see plan/0630_0708/elementwise). Each is dispatched by UnaryMap in
// `proto_driver` based on the (dtype_a, dtype_b) pair.
// ============================================================================

// sigmoid(x) = 1 / (1 + e^-x)
// Counted as 7 FLOPs/elem: negate (1) + exp (~4) + add (1) + reciprocal (1).
// (Consistent with silu = x * sigmoid(x) counted as 8.)
pub fn sigmoid<T: Debug + num_traits::Float + Copy>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 7) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| T::one() / (T::one() + (-x).exp())).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 7) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// clamp(x, min, max): either bound may be `None` (aten.clamp allows one-sided
// clamping). Counted as 2 FLOPs/elem (up to two comparisons).
pub fn clamp<T: Debug + Copy + PartialOrd + Default>(
    in_data: &Tile<T>,
    min: Option<T>,
    max: Option<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 2) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| {
                    let mut v = x;
                    if let Some(lo) = min {
                        if v < lo {
                            v = lo;
                        }
                    }
                    if let Some(hi) = max {
                        if v > hi {
                            v = hi;
                        }
                    }
                    v
                })
                .to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 2) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// ge_scalar(x, s) = (x >= s) -> bool tile. Counted as 1 FLOP/elem.
pub fn ge_scalar<T: Debug + Copy + PartialOrd>(
    in_data: &Tile<T>,
    scalar: T,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<bool>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        // bool tiles use 1 byte per element
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| x >= scalar).to_shared(),
                1,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![shape_0, shape_1], 1, write_back_mu, offset),
        ),
    }
}

// bitwise_not(x) = !x for boolean tiles (logical NOT). 1 FLOP/elem.
pub fn bitwise_not(
    in_data: &Tile<bool>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<bool>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| !x).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// floor_divide_scalar(x, d) = floor(x / d) for integer tiles.
// Integer division truncates toward zero; for the non-negative index tensors this
// matches PyTorch's floor semantics. Counted as 2 FLOPs/elem (div + floor).
pub fn floor_divide_scalar<T: Debug + Copy + num_traits::PrimInt>(
    in_data: &Tile<T>,
    divisor: T,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1 * 2) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.mapv(|x| x / divisor).to_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        None => (
            div_ceil((shape_0 * shape_1 * 2) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

// empty_like(x): uninitialized tensor with the same shape/dtype as the input. The
// contents are unspecified in PyTorch; we emit zeros so functional simulation stays
// deterministic. 1 FLOP/elem.
pub fn empty_like<T: Debug + Clone + num_traits::Zero>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    (
        div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
        Tile::new_zero_padded(
            [shape_0, shape_1],
            in_data.bytes_per_elem,
            write_back_mu,
            offset,
        ),
    )
}

// zeros_like(x): zero-filled tensor with the same shape/dtype as the input. 1 FLOP/elem.
pub fn zeros_like<T: Debug + Clone + num_traits::Zero>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    (
        div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
        Tile::new_zero_padded(
            [shape_0, shape_1],
            in_data.bytes_per_elem,
            write_back_mu,
            offset,
        ),
    )
}

// _to_copy cast f32 -> bf16. In the simulator bf16 is modelled as Tile<f32> with
// bytes_per_elem = 2; the data is unchanged, only the byte width is updated.
// 1 FLOP/elem.
pub fn f32_bf16<T: Debug + Clone>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(arr.clone(), 2, write_back_mu, offset),
        ),
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![shape_0, shape_1], 2, write_back_mu, offset),
        ),
    }
}

// _to_copy cast bf16 -> f32. Modelled as Tile<f32> with bytes_per_elem = 4.
// 1 FLOP/elem.
pub fn bf16_f32<T: Debug + Clone>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(arr.clone(), 4, write_back_mu, offset),
        ),
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![shape_0, shape_1], 4, write_back_mu, offset),
        ),
    }
}

// _to_copy cast f32 -> bool. Nonzero elements become true (matches torch's bool cast).
// Output is a bool tile (1 byte per element). 1 FLOP/elem.
pub fn f32_bool(
    in_data: &Tile<f32>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<bool>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(arr.mapv(|x| x != 0.0).to_shared(), 1, write_back_mu, offset),
        ),
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![shape_0, shape_1], 1, write_back_mu, offset),
        ),
    }
}

// _to_copy cast i64 -> f32 (value-preserving numeric cast). Output is an f32 tile
// (4 bytes per element). 1 FLOP/elem.
pub fn i64_f32(in_data: &Tile<i64>, flop_per_cycle: u64, write_back_mu: bool) -> (u64, Tile<f32>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(arr.mapv(|x| x as f32).to_shared(), 4, write_back_mu, offset),
        ),
        None => (
            div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(vec![shape_0, shape_1], 4, write_back_mu, offset),
        ),
    }
}

// transpose(x): tile-wise 2D transpose ([m, n] -> [n, m]). A pure data-movement op
// used by UnaryMap to realize aten.transpose when the parallel-stream permute also
// requires transposing each tile's data (see `_lower_transpose` /
// `need_elementwise_transpose`). Generic over the element type so it works for
// f32/bf16 (bf16 is modelled as Tile<f32>) and any other tile element. Counted as
// 1 FLOP/elem, consistent with the other data-movement/creation ops.
pub fn transpose<T: Debug + Clone>(
    in_data: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let offset = in_data.offset;

    match &in_data.underlying {
        Some(arr) => (
            1, // div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_padded(
                arr.t().to_owned().into_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
        // Timing-only (blank) tile: no data, just swap the shape.
        None => (
            1, // div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
            Tile::new_blank_padded(
                vec![shape_1, shape_0],
                in_data.bytes_per_elem,
                write_back_mu,
                offset,
            ),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::elem::Bufferizable;

    #[test]
    fn test_row_wise_sum() {
        let arr = ndarray::Array2::from_shape_fn((3, 4), |(i, j)| i as f32 + j as f32);
        println!("input arr: {:?}", arr);
        let in_data = Tile::new_padded(arr.to_shared(), 4, false, 3);
        let (flop_count, out_data) = row_wise_sum(&in_data, 6, false);

        println!("output arr: {:?}", out_data.underlying.unwrap());
        assert_eq!(flop_count, 2);
    }

    #[test]
    fn test_sub() {
        let arr1 = ndarray::Array2::from_shape_vec((2, 2), vec![5.0f32, 1.0, -3.0, 0.0]).unwrap();
        let arr2 = ndarray::Array2::from_shape_vec((2, 2), vec![2.0f32, 4.0, -8.0, 0.5]).unwrap();
        let in1 = Tile::new_padded(arr1.to_shared(), 4, false, 2);
        let in2 = Tile::new_padded(arr2.to_shared(), 4, false, 2);
        let (flop_count, out_data) = sub(&in1, &in2, 2, false);

        let out = out_data.underlying.unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        // Order matters: lhs - rhs, not the other way round.
        assert_eq!(out.as_slice().unwrap(), &[3.0f32, -3.0, 5.0, -0.5]);
        assert_eq!(flop_count, div_ceil(2 * 2, 2));
    }

    #[test]
    fn test_sub_broadcasts_and_blank() {
        // [2,3] - [2,1]: broadcasting the row-wise max out of an online softmax.
        let arr1 =
            ndarray::Array2::from_shape_vec((2, 3), vec![1.0f32, 5.0, 2.0, 9.0, 0.0, 4.0]).unwrap();
        let arr2 = ndarray::Array2::from_shape_vec((2, 1), vec![5.0f32, 9.0]).unwrap();
        let in1 = Tile::new_padded(arr1.to_shared(), 4, false, 2);
        let in2 = Tile::new_padded(arr2.to_shared(), 4, false, 2);
        let (_, out_data) = sub(&in1, &in2, 8, false);
        let out = out_data.underlying.unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.as_slice().unwrap(),
            &[-4.0f32, 0.0, -3.0, 0.0, -9.0, -5.0]
        );

        // Timing-only path: one blank input means no data, but shape/cycles hold.
        let blank: Tile<f32> = Tile::new_blank_padded(vec![2, 3], 2, false, 1);
        let in2 = Tile::new_padded(arr2.to_shared(), 2, false, 1);
        let (flop_count, out_data) = sub(&blank, &in2, 8, true);
        assert!(out_data.underlying.is_none());
        assert_eq!(out_data.shape, vec![2, 3]);
        assert_eq!(flop_count, div_ceil(2 * 3, 8));
    }

    #[test]
    fn test_row_wise_max() {
        // Row i holds [i, i+1, i+2, i+3], so the max of row i is i + 3.
        let arr = ndarray::Array2::from_shape_fn((3, 4), |(i, j)| i as f32 + j as f32);
        let in_data = Tile::new_padded(arr.to_shared(), 4, false, 3);
        let (flop_count, out_data) = row_wise_max(&in_data, 6, false);

        let out = out_data.underlying.unwrap();
        assert_eq!(out.shape(), &[3, 1]);
        assert_eq!(out.as_slice().unwrap(), &[3.0f32, 4.0, 5.0]);
        // div_ceil(3 * 4, 6) == 2, same roofline as row_wise_sum
        assert_eq!(flop_count, 2);
        assert_eq!(out_data.offset, 3);
        assert_eq!(out_data.bytes_per_elem, 4);
    }

    #[test]
    fn test_row_wise_max_negative_and_blank() {
        // All-negative rows: the reduction must not seed from zero.
        let arr =
            ndarray::Array2::from_shape_vec((2, 3), vec![-5.0f32, -1.0, -3.0, -9.0, -7.0, -8.0])
                .unwrap();
        let in_data = Tile::new_padded(arr.to_shared(), 2, true, 0);
        let (_, out_data) = row_wise_max(&in_data, 4, true);
        assert_eq!(
            out_data.underlying.unwrap().as_slice().unwrap(),
            &[-1.0f32, -7.0]
        );

        // Timing-only ("blank") tile: no data, but shape/cycles still collapse to [R, 1].
        let blank: Tile<f32> = Tile::new_blank_padded(vec![2, 3], 2, false, 1);
        let (flop_count, out_data) = row_wise_max(&blank, 4, true);
        assert!(out_data.underlying.is_none());
        assert_eq!(out_data.shape, vec![2, 1]);
        assert_eq!(out_data.offset, 1);
        assert_eq!(flop_count, div_ceil(2 * 3, 4));
    }

    #[test]
    fn test_row_wise_append() {
        let arr =
            ndarray::Array2::from_shape_fn(
                (6, 4),
                |(i, j)| if i < 3 { i as f32 + j as f32 } else { 0.0 },
            );
        println!("input arr: {:?}", arr);
        let in_data = Tile::new_padded(arr.to_shared(), 4, false, 3);
        let arr_to_append = ndarray::Array2::from_shape_fn((1, 4), |(i, j)| 3 as f32 + j as f32);
        let data_to_append = Tile::new_padded(arr_to_append.to_shared(), 4, false, 1);
        println!(
            "data_to_append: {:?}",
            data_to_append.underlying.as_ref().unwrap()
        );

        let (flop_count, out_data) = row_wise_append(&in_data, &data_to_append, false);
        println!("output arr: {:?}", out_data.underlying.as_ref().unwrap());
        assert_eq!(out_data.offset, 4);
        assert_eq!(flop_count, 1);
    }

    #[test]
    fn test_mask_row() {
        // Test case 1: Create a mask for row index 2 in a 5x3 matrix
        let idx_arr = Array2::from_shape_vec((1, 1), vec![2u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);

        let (cycles, out_data) = mask_row::<u64, f32>(&in_data, false, 5, 3, 8);

        println!("output arr:\n{:?}", out_data.underlying.as_ref().unwrap());

        // Check the cycle count
        assert_eq!(cycles, 1);

        // Check the output shape
        assert_eq!(out_data.shape, vec![5, 3]);

        // Verify the output array
        let result = out_data.underlying.as_ref().unwrap();
        for i in 0..5 {
            for j in 0..3 {
                if i == 2 {
                    assert_eq!(result[[i, j]], 1.0, "Row {} col {} should be 1.0", i, j);
                } else {
                    assert_eq!(result[[i, j]], 0.0, "Row {} col {} should be 0.0", i, j);
                }
            }
        }
    }

    #[test]
    fn test_mask_row_first_row() {
        // Test case 2: Mask the first row (index 0)
        let idx_arr = Array2::from_shape_vec((1, 1), vec![0u32]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 4, false);

        let (cycles, out_data) = mask_row::<u32, f64>(&in_data, false, 3, 4, 8);

        println!(
            "output arr (first row):\n{:?}",
            out_data.underlying.as_ref().unwrap()
        );

        assert_eq!(cycles, 1);
        assert_eq!(out_data.shape, vec![3, 4]);

        let result = out_data.underlying.as_ref().unwrap();
        // First row should be all 1.0
        for j in 0..4 {
            assert_eq!(result[[0, j]], 1.0);
        }
        // Other rows should be all 0.0
        for i in 1..3 {
            for j in 0..4 {
                assert_eq!(result[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn test_multihot_to_u64_single() {
        // MultiHot with a single true at index 2
        let mh = MultiHotN::new(vec![false, false, true, false], false);
        let (cycles, tile) = multihot_to_u64(&mh, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(tile.shape, vec![1, 1]);
        let arr = tile.underlying.as_ref().unwrap();
        assert_eq!(arr[[0, 0]], 2);
    }

    #[test]
    fn test_multihot_to_u64_multiple() {
        // MultiHot with true at indices 0, 2, 3
        let mh = MultiHotN::new(vec![true, false, true, true, false], false);
        let (cycles, tile) = multihot_to_u64(&mh, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(tile.shape, vec![1, 3]);
        let arr = tile.underlying.as_ref().unwrap();
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[0, 1]], 2);
        assert_eq!(arr[[0, 2]], 3);
    }

    #[test]
    fn test_multihot_to_u64_none_selected() {
        // MultiHot with no true values
        let mh = MultiHotN::new(vec![false, false, false], false);
        let (cycles, tile) = multihot_to_u64(&mh, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(tile.shape, vec![1, 1]);
        let arr = tile.underlying.as_ref().unwrap();
        assert_eq!(arr[[0, 0]], 0);
    }

    #[test]
    fn test_multihot_to_u64_all_selected() {
        // MultiHot with all true
        let mh = MultiHotN::new(vec![true, true, true], false);
        let (cycles, tile) = multihot_to_u64(&mh, 1, true);
        assert_eq!(cycles, 1);
        assert_eq!(tile.shape, vec![1, 3]);
        assert!(tile.read_from_mu);
        let arr = tile.underlying.as_ref().unwrap();
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[0, 1]], 1);
        assert_eq!(arr[[0, 2]], 2);
    }

    #[test]
    fn test_multihot_to_u64_blank() {
        // Blank (timing-only) input has no data: output is a blank tile whose column
        // count is the candidate-count upper bound, with no underlying data.
        let mh = MultiHotN::new_blank(4, false);
        let (cycles, tile) = multihot_to_u64(&mh, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(tile.shape, vec![1, 4]);
        assert!(tile.underlying.is_none());
    }

    #[test]
    fn test_u64_to_multihot_single() {
        let arr = Array2::from_shape_vec((1, 1), vec![2u64]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = u64_to_multihot(&tile, 4, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 4);
        assert_eq!(mh.to_sel_vec(), vec![2]);
    }

    #[test]
    fn test_u64_to_multihot_multiple() {
        let arr = Array2::from_shape_vec((1, 3), vec![0u64, 2, 3]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = u64_to_multihot(&tile, 5, 1, true);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 5);
        assert_eq!(mh.to_sel_vec(), vec![0, 2, 3]);
        assert!(mh.read_from_mu());
    }

    #[test]
    fn test_u64_to_multihot_empty() {
        // All indices out of range: concrete (not blank) but nothing selected.
        let arr = Array2::from_shape_vec((1, 1), vec![10u64]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = u64_to_multihot(&tile, 3, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.to_sel_vec(), Vec::<usize>::new());
        assert!(!mh.is_blank());
    }

    #[test]
    fn test_u64_to_multihot_roundtrip() {
        // multihot -> u64 -> multihot should reproduce the original
        let original = MultiHotN::new(vec![true, false, true, false, true], false);
        let (_, tile) = multihot_to_u64(&original, 1, false);
        let (_, reconstructed) = u64_to_multihot(&tile, 5, 1, false);
        assert_eq!(*original, *reconstructed);
    }

    #[test]
    fn test_index_to_multihot_single() {
        let arr = Array2::from_shape_vec((1, 1), vec![2.0f32]).unwrap();
        let tile = Tile::new(arr.to_shared(), 4, false);
        let (cycles, mh) = index_to_multihot(&tile, 4, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 4);
        assert_eq!(mh.to_sel_vec(), vec![2]);
    }

    #[test]
    fn test_index_to_multihot_multiple() {
        let arr = Array2::from_shape_vec((1, 3), vec![0.0f32, 2.0, 3.0]).unwrap();
        let tile = Tile::new(arr.to_shared(), 4, false);
        let (cycles, mh) = index_to_multihot(&tile, 5, 1, true);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 5);
        assert_eq!(mh.to_sel_vec(), vec![0, 2, 3]);
        assert!(mh.read_from_mu());
    }

    #[test]
    fn test_index_to_multihot_out_of_range() {
        // Index >= num_classes is ignored: concrete (not blank) but nothing selected.
        let arr = Array2::from_shape_vec((1, 1), vec![10.0f32]).unwrap();
        let tile = Tile::new(arr.to_shared(), 4, false);
        let (cycles, mh) = index_to_multihot(&tile, 3, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.to_sel_vec(), Vec::<usize>::new());
        assert!(!mh.is_blank());
    }

    #[test]
    fn test_index_to_multihot_blank() {
        // Timing-only tile (no underlying data) yields a blank multihot (no data),
        // distinct from a concrete all-false selection.
        let tile: Tile<f32> = Tile::new_blank(vec![1, 1], 4, false);
        let (cycles, mh) = index_to_multihot(&tile, 3, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 3);
        assert!(mh.is_blank());
        assert_eq!(mh.to_sel_vec(), Vec::<usize>::new());
    }

    #[test]
    fn test_index_to_multihot_i64() {
        // Integer index tile (e.g. topk indices, dtype int64).
        let arr = Array2::from_shape_vec((1, 3), vec![0i64, 2, 3]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = index_to_multihot(&tile, 5, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 5);
        assert_eq!(mh.to_sel_vec(), vec![0, 2, 3]);
    }

    #[test]
    fn test_index_to_multihot_u64() {
        let arr = Array2::from_shape_vec((1, 2), vec![1u64, 4]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = index_to_multihot(&tile, 5, 1, true);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 5);
        assert_eq!(mh.to_sel_vec(), vec![1, 4]);
        assert!(mh.read_from_mu());
    }

    #[test]
    fn test_mask_row_last_row() {
        // Test case 3: Mask the last row
        let idx_arr = Array2::from_shape_vec((1, 1), vec![4u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);

        let (cycles, out_data) = mask_row::<u64, f32>(&in_data, false, 5, 2, 8);

        println!(
            "output arr (last row):\n{:?}",
            out_data.underlying.as_ref().unwrap()
        );

        assert_eq!(cycles, 1);
        assert_eq!(out_data.shape, vec![5, 2]);

        let result = out_data.underlying.as_ref().unwrap();
        // Last row should be all 1.0
        for j in 0..2 {
            assert_eq!(result[[4, j]], 1.0);
        }
        // Other rows should be all 0.0
        for i in 0..4 {
            for j in 0..2 {
                assert_eq!(result[[i, j]], 0.0);
            }
        }
    }

    // ---- Tests for the elementwise ops added for ATen lowering ----

    #[test]
    fn test_sigmoid() {
        let arr = Array2::from_shape_vec((1, 3), vec![0.0f32, 100.0, -100.0]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 4, false);
        let (cycles, out) = sigmoid(&in_data, 1, false);
        let o = out.underlying.unwrap();
        assert!((o[[0, 0]] - 0.5).abs() < 1e-6);
        assert!(o[[0, 1]] > 0.99);
        assert!(o[[0, 2]] < 0.01);
        // numel(3) * 7 FLOPs / 1 FLOP-per-cycle
        assert_eq!(cycles, 21);
    }

    #[test]
    fn test_clamp_one_sided() {
        // clamp(None, 127): only the upper bound is applied.
        let arr = Array2::from_shape_vec((1, 3), vec![-5i64, 50, 200]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 8, false);
        let (_cycles, out) = clamp(&in_data, None, Some(127i64), 1, false);
        let o = out.underlying.unwrap();
        assert_eq!(o[[0, 0]], -5);
        assert_eq!(o[[0, 1]], 50);
        assert_eq!(o[[0, 2]], 127);
    }

    #[test]
    fn test_clamp_two_sided() {
        let arr = Array2::from_shape_vec((1, 3), vec![-5i64, 50, 200]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 8, false);
        let (_cycles, out) = clamp(&in_data, Some(0i64), Some(127i64), 1, false);
        let o = out.underlying.unwrap();
        assert_eq!(o[[0, 0]], 0);
        assert_eq!(o[[0, 1]], 50);
        assert_eq!(o[[0, 2]], 127);
    }

    #[test]
    fn test_ge_scalar() {
        let arr = Array2::from_shape_vec((1, 3), vec![10i64, 128, 200]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 8, false);
        let (_cycles, out) = ge_scalar(&in_data, 128i64, 1, false);
        let o = out.underlying.as_ref().unwrap();
        assert_eq!(o[[0, 0]], false);
        assert_eq!(o[[0, 1]], true);
        assert_eq!(o[[0, 2]], true);
        // bool tiles are 1 byte per element
        assert_eq!(out.bytes_per_elem, 1);
    }

    #[test]
    fn test_bitwise_not() {
        let arr = Array2::from_shape_vec((1, 2), vec![true, false]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 1, false);
        let (_cycles, out) = bitwise_not(&in_data, 1, false);
        let o = out.underlying.unwrap();
        assert_eq!(o[[0, 0]], false);
        assert_eq!(o[[0, 1]], true);
    }

    #[test]
    fn test_floor_divide_scalar() {
        let arr = Array2::from_shape_vec((1, 4), vec![0i64, 5, 9, 12]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 8, false);
        let (_cycles, out) = floor_divide_scalar(&in_data, 4i64, 1, false);
        let o = out.underlying.unwrap();
        assert_eq!(o[[0, 0]], 0);
        assert_eq!(o[[0, 1]], 1);
        assert_eq!(o[[0, 2]], 2);
        assert_eq!(o[[0, 3]], 3);
    }

    #[test]
    fn test_zeros_like() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 4, false);
        let (_cycles, out) = zeros_like(&in_data, 1, false);
        assert_eq!(out.shape, vec![2, 2]);
        assert!(out.underlying.unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_empty_like_shape() {
        let arr = Array2::from_shape_vec((1, 3), vec![7i64, 8, 9]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 8, false);
        let (_cycles, out) = empty_like(&in_data, 1, false);
        assert_eq!(out.shape, vec![1, 3]);
        assert_eq!(out.bytes_per_elem, 8);
    }

    #[test]
    fn test_cast_changes_bytes_per_elem() {
        // f32 -> bf16: data unchanged, byte width becomes 2.
        let arr = Array2::from_shape_vec((1, 2), vec![1.5f32, 2.5]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 4, false);
        let (_c, bf) = f32_bf16(&in_data, 1, false);
        assert_eq!(bf.bytes_per_elem, 2);
        assert_eq!(bf.underlying.as_ref().unwrap()[[0, 0]], 1.5);

        // bf16 -> f32: byte width becomes 4.
        let arr2 = Array2::from_shape_vec((1, 2), vec![1.5f32, 2.5]).unwrap();
        let in_bf = Tile::new(arr2.to_shared(), 2, false);
        let (_c2, f) = bf16_f32(&in_bf, 1, false);
        assert_eq!(f.bytes_per_elem, 4);
        assert_eq!(f.underlying.as_ref().unwrap()[[0, 1]], 2.5);
    }

    #[test]
    fn test_f32_to_bool() {
        let arr = Array2::from_shape_vec((1, 3), vec![0.0f32, 1.0, -2.5]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 4, false);
        let (_c, out) = f32_bool(&in_data, 1, false);
        let o = out.underlying.unwrap();
        assert_eq!(o[[0, 0]], false); // 0.0 -> false
        assert_eq!(o[[0, 1]], true); // nonzero -> true
        assert_eq!(o[[0, 2]], true);
        assert_eq!(out.bytes_per_elem, 1);
    }

    #[test]
    fn test_i64_to_f32() {
        let arr = Array2::from_shape_vec((1, 3), vec![0i64, 7, -3]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 8, false);
        let (_c, out) = i64_f32(&in_data, 1, false);
        let o = out.underlying.unwrap();
        assert_eq!(o[[0, 0]], 0.0f32);
        assert_eq!(o[[0, 1]], 7.0f32);
        assert_eq!(o[[0, 2]], -3.0f32);
        assert_eq!(out.bytes_per_elem, 4);
    }

    #[test]
    fn test_transpose() {
        // [[1,2,3],[4,5,6]] (2x3) -> [[1,4],[2,5],[3,6]] (3x2)
        let arr =
            Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let in_data = Tile::new(arr.to_shared(), 4, false);
        let (cycles, out) = transpose(&in_data, 1, false);

        assert_eq!(out.shape, vec![3, 2]);
        let o = out.underlying.as_ref().unwrap();
        assert_eq!(o[[0, 0]], 1.0);
        assert_eq!(o[[0, 1]], 4.0);
        assert_eq!(o[[1, 0]], 2.0);
        assert_eq!(o[[1, 1]], 5.0);
        assert_eq!(o[[2, 0]], 3.0);
        assert_eq!(o[[2, 1]], 6.0);
        // 6 elems * 1 FLOP / 1 FLOP-per-cycle
        assert_eq!(cycles, 6);
        // dtype byte width is preserved
        assert_eq!(out.bytes_per_elem, 4);
    }

    #[test]
    fn test_transpose_blank_swaps_shape() {
        // Timing-only tile: no data, shape [2,5] -> [5,2].
        let in_data: Tile<f32> = Tile::new_blank_padded(vec![2, 5], 2, false, 0);
        let (cycles, out) = transpose(&in_data, 1, false);
        assert_eq!(out.shape, vec![5, 2]);
        assert!(out.underlying.is_none());
        // bf16-width (2 bytes) is preserved through the transpose
        assert_eq!(out.bytes_per_elem, 2);
        assert_eq!(cycles, 10);
    }
}
