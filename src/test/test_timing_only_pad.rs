//! Regression test for pure-timing simulation of `graph.pb`'s early stage.
//!
//! `LinearOffChipLoad_0` has no `.npy` file, so it emits tiles with
//! `underlying: None` (timing-only). `Reshape_6` however pads short chunks with
//! a tile built from `InitFn::Zero`, which materializes an `underlying` array.
//! The resulting stream therefore mixes blank tiles with data-carrying pad
//! tiles, and `Accum_15` (`retile_row`) has to cope with both.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{CheckerContext, GeneratorContext};

    use crate::functions::accum_fn;
    use crate::operator::accum::{Accum, AccumConfig};
    use crate::operator::flatten::Flatten;
    use crate::operator::reshape::Reshape;
    use crate::primitives::{elem::Elem, tile::Tile};
    use crate::utils::events::SimpleEvent;

    /// Mirrors LinearOffChipLoad_0 -> Reshape_6 -> Flatten_14 -> Accum_15.
    ///
    /// 9 blank 1x512 tiles into a chunk_size-16 Reshape means 7 pad tiles are
    /// appended, so the Accum sees blank tiles first and pad tiles afterwards.
    #[test]
    fn timing_only_reshape_pad_into_retile_row() {
        const BYTES_PER_ELEM: usize = 2; // bf16
        const TILE_ROW: usize = 1;
        const TILE_COL: usize = 512;
        const CHUNK_SIZE: usize = 16;
        const NUM_TILES: usize = 9;

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (reshape_snd, reshape_rcv) = ctx.unbounded();
        let (flatten_snd, flatten_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // LinearOffChipLoad_0 with a missing .npy: blank tiles, rank-0 stream.
        ctx.add_child(GeneratorContext::new(
            || {
                (0..NUM_TILES)
                    .map(|_| {
                        Elem::Val(Tile::<f32>::new_blank(
                            vec![TILE_ROW, TILE_COL],
                            BYTES_PER_ELEM,
                            false,
                        ))
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_snd,
        ));

        // Reshape_6: pad_func = Zero, exactly as proto_driver builds it.
        ctx.add_child(Reshape::new(
            in_rcv,
            reshape_snd,
            0, // split_dim
            CHUNK_SIZE,
            Some(Tile::<f32>::new_zero_padded(
                [TILE_ROW, TILE_COL],
                BYTES_PER_ELEM,
                false,
                0,
            )),
            0,    // input_stream_rank
            true, // add_outer_dim
            6,
        ));

        // Flatten_14 (1 D, 2 D)
        ctx.add_child(Flatten::new(reshape_rcv, flatten_snd, 1, 2));

        // Accum_15: RetileRow, init Empty, tile_row 0, tile_col 512, rank 1.
        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            flatten_rcv,
            out_snd,
            Arc::new(|tile1, tile2, comp_bw, write_back_mu| {
                accum_fn::retile_row(tile1, tile2, comp_bw, write_back_mu, 15)
            }),
            Arc::new(|| Tile::<f32>::new_empty([0, TILE_COL], BYTES_PER_ELEM, false)),
            1, // rank
            AccumConfig {
                compute_bw: 4096,
                write_back_mu: false,
            },
            15,
        ));

        // The retiled tile stays blank, is CHUNK_SIZE rows tall, and its offset
        // counts only the NUM_TILES real rows -- the pad tiles carry offset 0.
        ctx.add_child(CheckerContext::new(
            || {
                vec![Elem::Val(Tile::<f32>::new_blank_padded(
                    vec![CHUNK_SIZE, TILE_COL],
                    BYTES_PER_ELEM,
                    false,
                    NUM_TILES,
                ))]
                .into_iter()
            },
            out_rcv,
        ));

        let summary = ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());

        // dam swallows Context panics, so assert on the summary rather than
        // relying on the test harness reporting a failure.
        assert!(
            summary.passed(),
            "simulation did not complete cleanly: a context panicked"
        );
    }
}
