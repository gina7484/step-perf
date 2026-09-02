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

pub fn max<T: Debug + ndarray::LinalgScalar + Default + PartialOrd>(
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
                    let val1 = *arr1.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = *arr2.get((i1, j1)).unwrap();
                    let out_val = if val1 > val2 { val1 } else { val2 };
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
            // Row-wise max: reduce each row's shape_1 columns down to a
            // single running max, mirroring row_wise_sum's shape but with
            // no generic "max" method on this file's Num-only bound, so
            // fold by hand instead of ndarray's sum_axis.
            let mut out_arr = ndarray::Array2::from_elem((shape_0, 1), arr[[0, 0]]);
            for i in 0..shape_0 {
                let mut row_max = arr[[i, 0]];
                for j in 1..shape_1 {
                    let v = arr[[i, j]];
                    if v > row_max {
                        row_max = v;
                    }
                }
                out_arr[[i, 0]] = row_max;
            }
            (
                div_ceil((shape_0 * shape_1) as u64, flop_per_cycle),
                Tile::new_padded(
                    out_arr.to_shared(),
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

/// Multiply every element of an F32 tile by a scalar carried in a U64 tile.
///
/// Needed by the analytic ragged-padding correction (route 3):
/// `l_true = l_polluted - n_invalid * exp(-m_final)`, where `n_invalid` is a
/// per-request count that necessarily arrives as U64 -- step-perf's
/// `MetadataGen` reads u64 ONLY (memory/metadata_gen.rs:29 panics on a float
/// .npy), while `exp(-m)` is F32. The proto_driver already has an
/// (F32, U64, F32) BinaryMap arm, but it implemented only `SetOffset`.
///
/// Scalar semantics match `set_offset`: element [0,0] of the U64 tile.
/// Subtract a scalar carried in a U64 tile from every element of an F32 tile.
/// Route 3's C>1 form: in the shift-free merge the exp weights cancel, so the
/// whole correction is `l_glob - n_invalid_total`.
pub fn sub_u64_scalar(
    in_data: &Tile<f32>,
    scalar: &Tile<u64>,
    write_back_mu: bool,
) -> (u64, Tile<f32>) {
    assert_eq!(in_data.shape.len(), 2);
    let (s0, s1) = (in_data.shape[0], in_data.shape[1]);
    let k = scalar.underlying.as_ref().unwrap()[[0, 0]] as f32;
    match &in_data.underlying {
        Some(arr) => (1, Tile::new(arr.mapv(|v| v - k).into_shared(),
                                   in_data.bytes_per_elem, write_back_mu)),
        None => (1, Tile::new_blank(vec![s0, s1], in_data.bytes_per_elem, write_back_mu)),
    }
}

pub fn mul_by_u64_scalar(
    in_data: &Tile<f32>,
    scalar: &Tile<u64>,
    write_back_mu: bool,
) -> (u64, Tile<f32>) {
    assert_eq!(in_data.shape.len(), 2);
    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];

    let k = scalar.underlying.as_ref().unwrap()[[0, 0]] as f32;

    match &in_data.underlying {
        Some(arr) => (
            1,
            Tile::new(
                arr.mapv(|v| v * k).into_shared(),
                in_data.bytes_per_elem,
                write_back_mu,
            ),
        ),
        // A blank tile stays blank: 0 * k == 0.
        None => (
            1,
            Tile::new_blank(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
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
    mock_bf16: bool,
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
                Tile::new(
                    out_arr.to_shared(),
                    if mock_bf16 {
                        2
                    } else {
                        std::mem::size_of::<D>()
                    },
                    write_back_mu,
                ),
            )
        }
        None => (
            1,
            Tile::new_blank(
                vec![row, col],
                if mock_bf16 {
                    2
                } else {
                    std::mem::size_of::<D>()
                },
                write_back_mu,
            ),
        ),
    }
}

/// Column-oriented sibling of `mask_row` -- and NOT a transpose of it.
///
/// `mask_row` is a ONE-HOT ROW selector (`out[i][*] = 1` for the single index
/// `i`). This is a PREFIX COLUMN mask: `1.0` in columns `[0, n)` and `0.0`
/// from `n` on, for EVERY row, with `n` clamped to `[0, col]`.
///
/// The clamp is load-bearing: it lets ONE count cover all three
/// ragged-padding cases (GH#3 [T5]) with no "is this the last tile?" select --
///   * `n >= col`  -> all ones   (a fully-valid KV tile)
///   * `0 < n < col` -> partial  (the ragged tail tile)
///   * `n <= 0`    -> all zeros  (a whole pad tile, which C>1 padding creates)
/// The caller feeds `total_valid - kv_tile_index * col` and all three fall out.
///
/// NEGATIVE COUNTS: the count arrives unsigned (`u64` on the proto path), and
/// `total_valid - n*col` is genuinely negative on pad tiles, so it has already
/// WRAPPED by the time it arrives. Left alone that reads as an enormous count,
/// i.e. `>= col`, i.e. ALL ONES -- a whole pad tile would come out entirely
/// unmasked, silently, which is the exact failure the inert all-ones mask
/// already has. So the value is reinterpreted as signed and floored at zero.
/// Real counts are bounded by the cache extent (maxN = 4096), nowhere near
/// the sign bit, so the reinterpretation cannot misfire on a legitimate count.
pub fn mask_col<
    T: Debug + Default + Clone + TryInto<usize>,
    D: num_traits::Float + Debug + Default + Clone,
>(
    in_data: &Tile<T>,
    write_back_mu: bool,
    row: usize,
    col: usize,
    mock_bf16: bool,
) -> (u64, Tile<D>) {
    assert_eq!(in_data.shape, vec![1, 1]);

    match &in_data.underlying {
        Some(arr) => {
            let val = arr[[0, 0]].clone();
            let raw: usize = val
                .try_into()
                .unwrap_or_else(|_| panic!("Failed to convert value to usize"));
            // See the doc comment: an upstream subtraction that went negative
            // arrives here wrapped, and must floor to zero rather than saturate
            // to "all valid".
            let n = if (raw as i64) < 0 { 0 } else { raw.min(col) };

            let mut out_arr = Array2::<D>::default((row, col));
            for i in 0..row {
                for j in 0..n {
                    out_arr[[i, j]] = D::one();
                }
            }

            (
                1,
                Tile::new(
                    out_arr.to_shared(),
                    if mock_bf16 {
                        2
                    } else {
                        std::mem::size_of::<D>()
                    },
                    write_back_mu,
                ),
            )
        }
        None => (
            1,
            Tile::new_blank(
                vec![row, col],
                if mock_bf16 {
                    2
                } else {
                    std::mem::size_of::<D>()
                },
                write_back_mu,
            ),
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

pub fn col_wise_append<T: Debug + Default + Clone>(
    in_data: &Tile<T>,
    data_to_append: &Tile<T>,
    write_back_mu: bool,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(data_to_append.shape.len(), 2);
    // data_to_append arrives in the same row-major (K, D) convention as
    // RowWiseAppend's rhs (K new items, D features each, D matching
    // in_data's row count) and is transposed here into (D, K) before being
    // spliced into in_data as K new columns. This lets callers append a
    // natural per-token feature row without a separate STeP-graph-level
    // transpose.
    assert_eq!(in_data.shape[0], data_to_append.shape[1]);

    let shape_0 = in_data.shape[0];
    let shape_1 = in_data.shape[1];
    let n_new = data_to_append.shape[0];

    let offset = in_data.offset;
    assert!(
        offset + n_new <= shape_1,
        "should have enough space to append new columns"
    );

    match (&in_data.underlying, &data_to_append.underlying) {
        (Some(arr), Some(arr_to_append)) => {
            // Create a mutable copy of the original array
            let mut result = arr.to_owned();
            let col_to_append = arr_to_append.t();

            // Copy the transposed data into the slice starting at offset
            let mut slice = result.slice_mut(ndarray::s![.., offset..offset + n_new]);
            slice.assign(&col_to_append);

            (
                1,
                Tile::new_padded(
                    result.into_shared(),
                    in_data.bytes_per_elem,
                    write_back_mu,
                    (offset + n_new) as usize,
                ),
            )
        }
        _ => (
            1,
            Tile::new_blank_padded(
                vec![shape_0, shape_1],
                in_data.bytes_per_elem,
                write_back_mu,
                (offset + n_new) as usize,
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
        None => (1, MultiHotN::new(vec![false; width], write_back_mu)),
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
    fn test_col_wise_append() {
        // in_data: (D=4 rows, N=6 cols), columns 0..3 already filled,
        // columns 3..6 blank (0.0), offset=3 (append at column 3 next).
        let arr = ndarray::Array2::from_shape_fn(
            (4, 6),
            |(i, j)| if j < 3 { i as f32 + j as f32 } else { 0.0 },
        );
        println!("input arr: {:?}", arr);
        let in_data = Tile::new_padded(arr.to_shared(), 4, false, 3);

        // data_to_append arrives in the natural (1, D) row-major shape
        // (same convention as RowWiseAppend's rhs), D=4 matching in_data's
        // row count - it should be transposed internally into a (4,1)
        // column before being spliced in.
        let arr_to_append = ndarray::Array2::from_shape_fn((1, 4), |(_, j)| 10.0 * (j as f32 + 1.0));
        let data_to_append = Tile::new_padded(arr_to_append.to_shared(), 4, false, 1);
        println!(
            "data_to_append: {:?}",
            data_to_append.underlying.as_ref().unwrap()
        );

        let (flop_count, out_data) = col_wise_append(&in_data, &data_to_append, false);
        let out_arr = out_data.underlying.as_ref().unwrap();
        println!("output arr: {:?}", out_arr);

        assert_eq!(out_data.offset, 4);
        assert_eq!(flop_count, 1);
        // Column 3 should now hold the transposed data_to_append row:
        // [10, 20, 30, 40] down rows 0..4, not written across row 0.
        for i in 0..4 {
            assert_eq!(out_arr[[i, 3]], 10.0 * (i as f32 + 1.0));
        }
        // Untouched columns must be unchanged.
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(out_arr[[i, j]], i as f32 + j as f32);
            }
        }
        for j in 4..6 {
            for i in 0..4 {
                assert_eq!(out_arr[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn test_mask_col_partial_is_the_intra_tile_case() {
        // The ragged tail: 3 valid columns of 8, every row identical.
        let idx_arr = Array2::from_shape_vec((1, 1), vec![3u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);
        let (cycles, out) = mask_col::<u64, f32>(&in_data, false, 2, 8, false);
        assert_eq!(cycles, 1);
        assert_eq!(out.shape, vec![2, 8]);
        let r = out.underlying.as_ref().unwrap();
        for i in 0..2 {
            for j in 0..8 {
                let want = if j < 3 { 1.0 } else { 0.0 };
                assert_eq!(r[[i, j]], want, "row {i} col {j}");
            }
        }
    }

    #[test]
    fn test_mask_col_clamps_to_all_ones() {
        // A fully-valid tile: the caller feeds a count larger than the width
        // rather than testing "is this the last tile?".
        for count in [8u64, 9, 4096] {
            let idx_arr = Array2::from_shape_vec((1, 1), vec![count]).unwrap();
            let in_data = Tile::new(idx_arr.to_shared(), 8, false);
            let (_c, out) = mask_col::<u64, f32>(&in_data, false, 1, 8, false);
            let r = out.underlying.as_ref().unwrap();
            for j in 0..8 {
                assert_eq!(r[[0, j]], 1.0, "count {count} col {j} should be valid");
            }
        }
    }

    #[test]
    fn test_mask_col_zero_is_all_zeros() {
        let idx_arr = Array2::from_shape_vec((1, 1), vec![0u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);
        let (_c, out) = mask_col::<u64, f32>(&in_data, false, 1, 8, false);
        let r = out.underlying.as_ref().unwrap();
        for j in 0..8 {
            assert_eq!(r[[0, j]], 0.0, "col {j}");
        }
    }

    /// THE critical case. `total_valid - kv_index*col` is negative on a whole
    /// pad tile and arrives here already wrapped. Untreated it reads as an
    /// enormous count -> all ones -> the pad tile is entirely unmasked, with
    /// no error anywhere. At C=16 a 91-token request pads to 512 slots, so
    /// getting this wrong silently reinstates most of the bug.
    #[test]
    fn test_mask_col_wrapped_negative_count_masks_everything() {
        for count in [u64::MAX, u64::MAX - 31, 1u64 << 63] {
            let idx_arr = Array2::from_shape_vec((1, 1), vec![count]).unwrap();
            let in_data = Tile::new(idx_arr.to_shared(), 8, false);
            let (_c, out) = mask_col::<u64, f32>(&in_data, false, 1, 8, false);
            let r = out.underlying.as_ref().unwrap();
            for j in 0..8 {
                assert_eq!(
                    r[[0, j]], 0.0,
                    "wrapped count {count} must mask column {j}, not saturate to valid"
                );
            }
        }
    }

    /// `mask_col` is not a transposed `mask_row`: same input, different shape
    /// of answer. Pins the distinction so nobody "simplifies" one into the
    /// other -- hwsim already reuses MaskRow as a dataflow stand-in, which is
    /// fine for latency and wrong for values.
    #[test]
    fn test_mask_col_is_not_a_transposed_mask_row() {
        let idx_arr = Array2::from_shape_vec((1, 1), vec![2u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);
        let (_c, col_out) = mask_col::<u64, f32>(&in_data, false, 4, 4, false);
        let (_c2, row_out) = mask_row::<u64, f32>(&in_data, false, 4, 4, false);
        let c = col_out.underlying.as_ref().unwrap();
        let rw = row_out.underlying.as_ref().unwrap();
        // prefix-of-columns in every row ...
        assert_eq!((c[[0, 0]], c[[0, 1]], c[[0, 2]]), (1.0, 1.0, 0.0));
        assert_eq!((c[[3, 0]], c[[3, 1]], c[[3, 2]]), (1.0, 1.0, 0.0));
        // ... versus one whole row set.
        assert_eq!((rw[[2, 0]], rw[[2, 3]]), (1.0, 1.0));
        assert_eq!((rw[[0, 0]], rw[[3, 0]]), (0.0, 0.0));
    }

    #[test]
    fn test_mask_row() {
        // Test case 1: Create a mask for row index 2 in a 5x3 matrix
        let idx_arr = Array2::from_shape_vec((1, 1), vec![2u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);

        let (cycles, out_data) = mask_row::<u64, f32>(&in_data, false, 5, 3, false);

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

        let (cycles, out_data) = mask_row::<u32, f64>(&in_data, false, 3, 4, false);

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
    fn test_u64_to_multihot_single() {
        let arr = Array2::from_shape_vec((1, 1), vec![2u64]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = u64_to_multihot(&tile, 4, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 4);
        assert_eq!(*mh, vec![false, false, true, false]);
    }

    #[test]
    fn test_u64_to_multihot_multiple() {
        let arr = Array2::from_shape_vec((1, 3), vec![0u64, 2, 3]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = u64_to_multihot(&tile, 5, 1, true);
        assert_eq!(cycles, 1);
        assert_eq!(mh.len(), 5);
        assert_eq!(*mh, vec![true, false, true, true, false]);
        assert!(mh.read_from_mu());
    }

    #[test]
    fn test_u64_to_multihot_empty() {
        // All indices out of range
        let arr = Array2::from_shape_vec((1, 1), vec![10u64]).unwrap();
        let tile = Tile::new(arr.to_shared(), 8, false);
        let (cycles, mh) = u64_to_multihot(&tile, 3, 1, false);
        assert_eq!(cycles, 1);
        assert_eq!(*mh, vec![false, false, false]);
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
    fn test_mask_row_last_row() {
        // Test case 3: Mask the last row
        let idx_arr = Array2::from_shape_vec((1, 1), vec![4u64]).unwrap();
        let in_data = Tile::new(idx_arr.to_shared(), 8, false);

        let (cycles, out_data) = mask_row::<u64, f32>(&in_data, false, 5, 2, false);

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
}
