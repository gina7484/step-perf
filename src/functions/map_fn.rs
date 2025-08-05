use ndarray::Array2;

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
#[cfg(test)]
mod tests {
    use super::*;

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
}
