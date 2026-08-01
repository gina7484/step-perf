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

pub fn mul<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    id: u32,
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

    let offset = accum_offset(in1, in2);

    match (&in1.underlying, &in2.underlying) {
        (Some(in1_arr), Some(in2_arr)) => {
            let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
            for i in 0..out_shape_0 {
                for j in 0..out_shape_1 {
                    let i0 = i.min(in1_shape_0 - 1);
                    let j0 = j.min(in1_shape_1 - 1);
                    let val1 = in1_arr.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = in2_arr.get((i1, j1)).unwrap();
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
        _ => (
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

/// Valid-row count for one accumulation step.
///
/// `in1` is the incoming data tile and `in2` the running accumulator, so the
/// padding is described by `in1`: the accumulator covers the same rows and, on
/// the first step, is a neutral `init` tile whose `offset` is meaningless (an
/// `InitFn::Zero` tile reports the *full* height). Taking `max` of the two would
/// therefore let that neutral tile unmask rows the input marked as padding.
/// Only when `in1` is a broadcast row against a taller accumulator does `in2`
/// carry the row count. Mirrors `map_accum_fn::matmul`, which also keys the
/// output offset off `in1`.
fn accum_offset<T: Debug + Clone>(in1: &Tile<T>, in2: &Tile<T>) -> usize {
    if in1.shape[0] == 1 && in2.shape[0] != 1 {
        in2.offset
    } else {
        in1.offset
    }
}

pub fn add<T: Debug + ndarray::LinalgScalar + Default>(
    in1: &Tile<T>,
    in2: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    id: u32,
) -> (u64, Tile<T>) {
    assert_eq!(in1.shape.len(), 2);
    assert_eq!(in2.shape.len(), 2);
    let in1_shape_0 = in1.shape[0];
    let in1_shape_1 = in1.shape[1];
    let in2_shape_0 = in2.shape[0];
    let in2_shape_1 = in2.shape[1];

    // On the first accumulation step the accumulator is an empty (0-row) tile:
    // `init=zero` with a dynamic row count serializes as tile_row=0, so it starts
    // as [0, N] (or [0, 0]). It is the additive identity, so adopt the input
    // tile's shape instead of broadcasting against 0 rows (which would fail the
    // row assertion below and underflow `in2_shape_0 - 1` in the add loop).
    if in2_shape_0 == 0 {
        let cycles = div_ceil((in1_shape_0 * in1_shape_1) as u64, flop_per_cycle);
        return match &in1.underlying {
            Some(in1_arr) => (
                cycles,
                Tile::new_padded(
                    in1_arr.clone(),
                    in1.bytes_per_elem,
                    write_back_mu,
                    in1.offset,
                ),
            ),
            None => (
                cycles,
                Tile::new_blank_padded(
                    vec![in1_shape_0, in1_shape_1],
                    in1.bytes_per_elem,
                    write_back_mu,
                    in1.offset,
                ),
            ),
        };
    }

    assert!((in1_shape_0 == in2_shape_0) || (in1_shape_0 == 1) || (in2_shape_0 == 1), "Accum_{}", id);
    assert!((in1_shape_1 == in2_shape_1) || (in1_shape_1 == 1) || (in2_shape_1 == 1), "Accum_{}", id);

    let out_shape_0 = in1_shape_0.max(in2_shape_0);
    let out_shape_1 = in1_shape_1.max(in2_shape_1);

    let offset = accum_offset(in1, in2);

    match (&in1.underlying, &in2.underlying) {
        (Some(in1_arr), Some(in2_arr)) => {
            let mut out_arr = ndarray::Array2::default((out_shape_0, out_shape_1));
            for i in 0..out_shape_0 {
                for j in 0..out_shape_1 {
                    let i0 = i.min(in1_shape_0 - 1);
                    let j0 = j.min(in1_shape_1 - 1);
                    let val1 = in1_arr.get((i0, j0)).unwrap();
                    let i1 = i.min(in2_shape_0 - 1);
                    let j1 = j.min(in2_shape_1 - 1);
                    let val2 = in2_arr.get((i1, j1)).unwrap();
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
        _ => (
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

pub fn retile_col<T: Debug + Clone>(
    in_data: &Tile<T>,
    accumulator: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    id: u32,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(accumulator.shape.len(), 2);

    let accum_offset = accumulator.offset;
    let in_offset = in_data.offset;
    // This won't be used as the offset field was for a syntactic sugar.
    // We will move on to deprecating the offset field in the future.

    // Functional simulation only happens when both sides carry data. A stream
    // can mix data-carrying and blank tiles (e.g. Reshape pads a blank stream
    // with an `InitFn::Zero` tile, which materializes an array), so falling
    // back to the timing-only path here is not optional.
    match (&in_data.underlying, &accumulator.underlying) {
        (Some(in_arr), Some(accum_arr)) => {
            let cur_arr = if accum_arr.shape() == [0, 0] {
                // Initial accumulation
                Array2::from_shape_vec((in_arr.shape()[0], 0), vec![])
                    .unwrap()
                    .to_shared()
            } else {
                accum_arr.clone()
            };

            (
                0, // TODO: Add cycles it took for grouping smaller tiles into larger tiles
                ndarray::concatenate(ndarray::Axis(1), &[cur_arr.view(), in_arr.view()])
                    .map(|arr| {
                        Tile::new_padded(
                            arr.to_shared(),
                            in_data.bytes_per_elem,
                            in_data.read_from_mu,
                            in_offset,
                        )
                    })
                    .unwrap_or_else(|_| {
                        panic!(
                            "Failed to concatenate input data and accumulator data (Accum_{})",
                            id
                        )
                    }),
            )
        }
        _ => {
            if accumulator.shape != vec![0, 0] {
                // In the initial accumulation, the accumulator's shape is [0,0]. Therefore we use in_data's shape[0].
                // However, afterwards, we need to make sure the number of rows match.
                assert_eq!(in_data.shape[0], accumulator.shape[0]);
            }

            (
                0,
                Tile::new_blank_padded(
                    vec![in_data.shape[0], in_data.shape[1] + accumulator.shape[1]],
                    in_data.bytes_per_elem,
                    in_data.read_from_mu,
                    in_offset,
                ),
            )
        }
    }
}

pub fn retile_row<T: Debug + Clone>(
    in_data: &Tile<T>,
    accumulator: &Tile<T>,
    flop_per_cycle: u64,
    write_back_mu: bool,
    id: u32,
) -> (u64, Tile<T>) {
    assert_eq!(in_data.shape.len(), 2);
    assert_eq!(accumulator.shape.len(), 2);

    let accum_offset = accumulator.offset;
    let in_offset = in_data.offset;

    // Functional simulation only happens when both sides carry data. A stream
    // can mix data-carrying and blank tiles (e.g. Reshape pads a blank stream
    // with an `InitFn::Zero` tile, which materializes an array), so falling
    // back to the timing-only path here is not optional.
    match (&in_data.underlying, &accumulator.underlying) {
        (Some(in_arr), Some(accum_arr)) => {
            let cur_arr = if accum_arr.shape() == [0, 0] {
                // Initial accumulation
                Array2::from_shape_vec((0, in_arr.shape()[1]), vec![])
                    .unwrap()
                    .to_shared()
            } else {
                accum_arr.clone()
            };

            (
                0, // TODO: Add cycles it took for grouping smaller tiles into larger tiles
                ndarray::concatenate(ndarray::Axis(0), &[cur_arr.view(), in_arr.view()])
                    .map(|arr| {
                        Tile::new_padded(
                            arr.to_shared(),
                            in_data.bytes_per_elem,
                            in_data.read_from_mu,
                            accum_offset + in_offset,
                        )
                    })
                    .unwrap_or_else(|_| {
                        panic!(
                            "Failed to concatenate input data and accumulator data (Accum_{})",
                            id
                        )
                    }),
            )
        }
        _ => {
            assert_eq!(in_data.shape[1], accumulator.shape[1], "Accum_{}", id);
            let new_rows = if (in_data.shape[0] == in_offset) || (in_offset == 0) {
                in_offset
            } else {
                panic!("Invalid offset for input data (Accum_{})", id);
            };

            (
                0,
                Tile::new_blank_padded(
                    vec![
                        in_data.shape[0] + accumulator.shape[0],
                        accumulator.shape[1],
                    ],
                    in_data.bytes_per_elem,
                    in_data.read_from_mu,
                    accum_offset + new_rows,
                ),
            )
        }
    }
}

pub fn signal_req_all_read<T: Debug>(
    in_data: &Tile<T>,
    _: &Tile<u64>,
    write_back_mu: bool,
    id: u32,
) -> (u64, Tile<u64>) {
    match &in_data.underlying {
        Some(_) => (
            1,
            Tile::new(
                Array2::from_shape_vec((1, 1), vec![1]).unwrap().to_shared(),
                8,
                write_back_mu,
            ),
        ),
        None => (1, Tile::new_blank(vec![1, 1], 8, write_back_mu)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArcArray2;

    // Regression: an add-accumulation step must not resurrect the rows that the
    // input tile marks as padding. In the MoE flow `Reshape` pads a short expert
    // chunk and `Accum(RetileRow)` records the real row count in `offset`; if the
    // downstream `Accum(fn: Add)` rebuilds its output with a full-height offset,
    // every consumer (notably `RetileStreamify(filter_mask=true)`) sees a fully
    // valid tile and the padding is silently unmasked.
    #[test]
    fn add_preserves_input_offset() {
        const TILE_ROW: usize = 64;
        const TILE_COL: usize = 512;
        const VALID_ROWS: usize = 16;

        // Timing-only path: both tiles are blank.
        let in1: Tile<f32> = Tile::new_blank_padded(vec![TILE_ROW, TILE_COL], 2, false, VALID_ROWS);
        let acc: Tile<f32> = Tile::new_zero([TILE_ROW, TILE_COL], 2, false);
        let (_cycles, out) = add(&in1, &acc, 6400, false, 128);
        assert_eq!(out.offset, VALID_ROWS, "blank path dropped the offset");

        // Functional path: both tiles carry data.
        let in1: Tile<f32> = Tile::new_padded(
            ArcArray2::from_elem((TILE_ROW, TILE_COL), 1.0f32),
            2,
            false,
            VALID_ROWS,
        );
        let acc: Tile<f32> = Tile::new_zero([TILE_ROW, TILE_COL], 2, false);
        let (_cycles, out) = add(&in1, &acc, 6400, false, 128);
        assert_eq!(out.offset, VALID_ROWS, "functional path dropped the offset");
    }

    #[test]
    fn mul_preserves_input_offset() {
        const TILE_ROW: usize = 64;
        const TILE_COL: usize = 512;
        const VALID_ROWS: usize = 16;

        let in1: Tile<f32> = Tile::new_blank_padded(vec![TILE_ROW, TILE_COL], 2, false, VALID_ROWS);
        let acc: Tile<f32> = Tile::new_zero([TILE_ROW, TILE_COL], 2, false);
        let (_cycles, out) = mul(&in1, &acc, 6400, false, 128);
        assert_eq!(out.offset, VALID_ROWS, "blank path dropped the offset");

        let in1: Tile<f32> = Tile::new_padded(
            ArcArray2::from_elem((TILE_ROW, TILE_COL), 1.0f32),
            2,
            false,
            VALID_ROWS,
        );
        let acc: Tile<f32> = Tile::new_zero([TILE_ROW, TILE_COL], 2, false);
        let (_cycles, out) = mul(&in1, &acc, 6400, false, 128);
        assert_eq!(out.offset, VALID_ROWS, "functional path dropped the offset");
    }

    // Regression: in the MoE dynamic-M add-accumulation path the accumulator is
    // initialised via `init=zero` with a dynamic row count, which serializes as
    // tile_row=0 -> a [0, N] tile. On the first step `add` must treat it as the
    // additive identity and adopt the input tile's shape, rather than panicking
    // on the row-broadcast assertion (or underflowing `in2_shape_0 - 1`).
    #[test]
    fn add_empty_accumulator_adopts_input_shape() {
        let in1: Tile<f32> = Tile::new(ArcArray2::from_elem((13, 64), 1.0f32), 4, false);
        let acc: Tile<f32> = Tile::new_zero([0, 64], 4, false);
        let (_cycles, out) = add(&in1, &acc, 6400, false, 122);
        assert_eq!(out.shape, vec![13, 64]);
        let out_arr = out.underlying.expect("output should carry data");
        assert_eq!(out_arr.shape(), &[13, 64]);
        assert!(out_arr.iter().all(|&v| v == 1.0f32));
    }

    // A subsequent step (matching shapes) still accumulates elementwise.
    #[test]
    fn add_matching_shapes_accumulates() {
        let in1: Tile<f32> = Tile::new(ArcArray2::from_elem((13, 64), 1.0f32), 4, false);
        let acc: Tile<f32> = Tile::new(ArcArray2::from_elem((13, 64), 2.0f32), 4, false);
        let (_cycles, out) = add(&in1, &acc, 6400, false, 122);
        let out_arr = out.underlying.expect("output should carry data");
        assert_eq!(out_arr.shape(), &[13, 64]);
        assert!(out_arr.iter().all(|&v| v == 3.0f32));
    }
}
