use super::counter::count_range_elems;
use crate::primitives::elem::{Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::request_profile::{ProfiledReceiver, ProfiledSender};
use dam::context_tools::*;
use dam::types::DAMType;

#[context_macro]
pub struct FlatmapRetileStreamify<T: Clone> {
    in_stream: ProfiledReceiver<Elem<Tile<T>>>,
    out_stream: ProfiledSender<Elem<Tile<T>>>,
    axis: usize,
    chunk_size: usize,
    id: u32,
}

impl<T: Clone> FlatmapRetileStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        axis: usize,
        chunk_size: usize,
        id: u32,
    ) -> Self {
        assert!(axis < 2, "Retile direction must be row or column");
        assert!(chunk_size > 0, "Retile chunk size must be positive");
        let ctx = Self {
            in_stream: in_stream.into(),
            out_stream: out_stream.into(),
            axis,
            chunk_size,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    fn streamify(&mut self, data: &Tile<T>, stop: Option<StopType>) {
        assert_eq!(data.shape.len(), 2);
        let extent = data.shape[self.axis];
        assert!(
            extent > 0,
            "FlatmapRetileStreamify_{}: split tile dimension must be positive",
            self.id
        );
        assert_eq!(
            extent % self.chunk_size,
            0,
            "FlatmapRetileStreamify_{}: chunk size must divide the tile dimension",
            self.id
        );
        for start in (0..extent).step_by(self.chunk_size) {
            let end = start + self.chunk_size;
            let chunk = match &data.underlying {
                Some(arr) => Tile::new(
                    arr.slice_axis(ndarray::Axis(self.axis), ndarray::Slice::from(start..end))
                        .to_shared(),
                    data.bytes_per_elem,
                    data.read_from_mu,
                ),
                None => {
                    let mut shape = data.shape.clone();
                    shape[self.axis] = self.chunk_size;
                    Tile::new_blank(shape, data.bytes_per_elem, data.read_from_mu)
                }
            };
            let elem = match stop {
                Some(level) if end == extent => Elem::ValStop(chunk, level),
                _ => Elem::Val(chunk),
            };
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem,
                    },
                )
                .unwrap();
        }
    }
}

impl<T: Clone> Context for FlatmapRetileStreamify<T>
where
    Tile<T>: DAMType,
{
    fn run(&mut self) {
        while let Ok(element) = self.in_stream.dequeue(&self.time) {
            match element.data {
                Elem::Val(data) => self.streamify(&data, None),
                Elem::ValStop(data, stop) => self.streamify(&data, Some(stop)),
            }
        }
    }
}

#[context_macro]
pub struct FlatmapFilterRowStreamify<T: Clone> {
    in_stream: ProfiledReceiver<Elem<Tile<T>>>, // tile shape: [R,C]
    mask_stream: Option<ProfiledReceiver<Elem<Tile<bool>>>>, // tile shape: [R,1]
    out_stream: ProfiledSender<Elem<Tile<T>>>,  // tile shape: [1,C]
    id: u32,
}

impl<T: Clone> FlatmapFilterRowStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        mask_stream: Option<Receiver<Elem<Tile<bool>>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream: in_stream.into(),
            mask_stream: mask_stream.map(Into::into),
            out_stream: out_stream.into(),
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        if let Some(mask_stream) = &ctx.mask_stream {
            mask_stream.attach_receiver(&ctx);
        }
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    fn filter_streamify(
        &mut self,
        data: &Tile<T>,
        mask: Option<&Tile<bool>>,
        stop_level: Option<StopType>,
    ) {
        let rows = data.shape[0];
        let mask_arr = mask.map(|mask| {
            assert_eq!(mask.shape, vec![rows, 1], "Row mask must have shape [R,1]");
            mask.underlying
                .as_ref()
                .expect("Row filtering requires concrete mask values")
        });
        let count = mask_arr.map_or(rows, |arr| arr.iter().filter(|&&value| value).count());
        let mut emitted = 0;
        for row in 0..rows {
            if mask_arr.is_some_and(|arr| !arr[[row, 0]]) {
                continue;
            }
            let out_data = match &data.underlying {
                Some(arr) => Tile::new(
                    arr.slice(ndarray::s![row..row + 1, ..]).to_shared(),
                    data.bytes_per_elem,
                    data.read_from_mu,
                ),
                None => Tile::new_blank(
                    vec![1, data.shape[1]],
                    data.bytes_per_elem,
                    data.read_from_mu,
                ),
            };
            emitted += 1;
            let elem = match stop_level {
                Some(stop) if emitted == count => Elem::ValStop(out_data, stop),
                _ => Elem::Val(out_data),
            };
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem,
                    },
                )
                .unwrap();
        }
    }
}

impl<T: Clone> Context for FlatmapFilterRowStreamify<T>
where
    Tile<T>: DAMType,
{
    fn run(&mut self) {
        while let Ok(element) = self.in_stream.dequeue(&self.time) {
            let (data, stop) = match element.data {
                Elem::Val(data) => (data, None),
                Elem::ValStop(data, stop) => (data, Some(stop)),
            };
            let mask = self.mask_stream.as_ref().map(|mask_stream| {
                let element = mask_stream
                    .dequeue(&self.time)
                    .expect("Mask stream terminated before data stream");
                let (mask, mask_stop) = match element.data {
                    Elem::Val(mask) => (mask, None),
                    Elem::ValStop(mask, stop) => (mask, Some(stop)),
                };
                assert_eq!(mask_stop, stop, "Stop token position mismatch");
                mask
            });
            self.filter_streamify(&data, mask.as_ref(), stop);
        }
    }
}

#[context_macro]
pub struct FlatmapCounter<T: Clone> {
    in_stream: ProfiledReceiver<Elem<Tile<T>>>, // tile shape: [1,1]
    out_stream: ProfiledSender<Elem<Tile<T>>>,  // tile shape: [1,1]
    id: u32,
}

impl<T: Clone + TryInto<usize> + TryFrom<usize>> FlatmapCounter<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream: in_stream.into(),
            out_stream: out_stream.into(),
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    pub fn gen_stream(&self, scalar_tile: Tile<T>, stop_level: Option<u32>) {
        let tile_value = &scalar_tile.underlying.as_ref().unwrap()[[0, 0]];
        let data_usize: usize = tile_value.clone().try_into().unwrap_or_else(|_| {
            panic!(
                "[Counter {}] Failed to convert tile value to usize",
                self.id
            );
        });
        for elem in count_range_elems(
            data_usize,
            scalar_tile.bytes_per_elem,
            scalar_tile.read_from_mu,
            stop_level,
            self.id,
        ) {
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem,
                    },
                )
                .unwrap();
        }
    }
}

impl<T: Clone + TryInto<usize> + TryFrom<usize>> Context for FlatmapCounter<T>
where
    Tile<T>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => self.gen_stream(data, None),
                    Elem::ValStop(data, s) => self.gen_stream(data, Some(s)),
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };
    use ndarray::{ArcArray, Array2};

    use super::{FlatmapCounter, FlatmapFilterRowStreamify, FlatmapRetileStreamify};
    use crate::primitives::{buffer::Buffer, elem::Elem, tile::Tile};
    use crate::{functions::accum_fn::retile_row, operator::accum::AccumConfig};

    use crate::{
        operator::{accum::Accum, reshape::ReshapePadStream},
        utils::events::SimpleEvent,
    };

    fn check_retile_chunks(axis: usize, blank: bool) {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();
        let data = Tile::new(
            ndarray::arr2(&[
                [0u32, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ])
            .into_shared(),
            4,
            true,
        );
        let data = if blank { data.drop_underlying() } else { data };
        let expected = if axis == 0 {
            vec![
                ndarray::arr2(&[[0u32, 1, 2, 3], [4, 5, 6, 7]]),
                ndarray::arr2(&[[8, 9, 10, 11], [12, 13, 14, 15]]),
            ]
        } else {
            vec![
                ndarray::arr2(&[[0u32, 1], [4, 5], [8, 9], [12, 13]]),
                ndarray::arr2(&[[2, 3], [6, 7], [10, 11], [14, 15]]),
            ]
        };
        let chunks: Vec<_> = expected
            .into_iter()
            .map(|arr| {
                let tile = Tile::new(arr.into_shared(), 4, true);
                if blank {
                    tile.drop_underlying()
                } else {
                    tile
                }
            })
            .collect();
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::Val(data.clone()),
                    Elem::ValStop(data.clone(), 1),
                    Elem::ValStop(data.clone(), 3),
                ]
                .into_iter()
            },
            in_snd,
        ));
        ctx.add_child(FlatmapRetileStreamify::new(in_rcv, out_snd, axis, 2, 0));
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(chunks[0].clone()),
                    Elem::Val(chunks[1].clone()),
                    Elem::Val(chunks[0].clone()),
                    Elem::ValStop(chunks[1].clone(), 1),
                    Elem::Val(chunks[0].clone()),
                    Elem::ValStop(chunks[1].clone(), 3),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatmap_retile_preserves_data_and_stops() {
        check_retile_chunks(0, false);
        check_retile_chunks(1, false);
    }

    #[test]
    fn flatmap_retile_preserves_blank_shapes_and_stops() {
        check_retile_chunks(0, true);
        check_retile_chunks(1, true);
    }

    #[test]
    fn flatmap_retile_accepts_runtime_row_sizes_and_empty_streams() {
        for empty in [false, true] {
            let mut ctx = ProgramBuilder::default();
            let (in_snd, in_rcv) = ctx.unbounded();
            let (out_snd, out_rcv) = ctx.unbounded();
            ctx.add_child(GeneratorContext::new(
                move || {
                    if empty {
                        vec![]
                    } else {
                        vec![
                            Elem::ValStop(Tile::<u32>::new_blank(vec![3, 4], 4, false), 1),
                            Elem::ValStop(Tile::<u32>::new_blank(vec![1, 4], 4, false), 2),
                        ]
                    }
                    .into_iter()
                },
                in_snd,
            ));
            ctx.add_child(FlatmapRetileStreamify::new(in_rcv, out_snd, 1, 2, 0));
            ctx.add_child(ApproxCheckerContext::new(
                move || {
                    if empty {
                        vec![]
                    } else {
                        vec![
                            Elem::Val(Tile::<u32>::new_blank(vec![3, 2], 4, false)),
                            Elem::ValStop(Tile::<u32>::new_blank(vec![3, 2], 4, false), 1),
                            Elem::Val(Tile::<u32>::new_blank(vec![1, 2], 4, false)),
                            Elem::ValStop(Tile::<u32>::new_blank(vec![1, 2], 4, false), 2),
                        ]
                    }
                    .into_iter()
                },
                out_rcv,
                |x, y| x == y,
            ));
            ctx.initialize(Default::default())
                .unwrap()
                .run(Default::default());
        }
    }

    #[test]
    #[should_panic(expected = "chunk size must divide the tile dimension")]
    fn flatmap_retile_rejects_partial_runtime_chunks() {
        let mut ctx = ProgramBuilder::default();
        let (_in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, _out_rcv) = ctx.unbounded();
        let mut op = FlatmapRetileStreamify::new(in_rcv, out_snd, 1, 2, 0);
        op.streamify(&Tile::<u32>::new_blank(vec![3, 5], 4, false), Some(1));
    }

    #[test]
    fn roundtrip_test() {
        // Reshape: (1, 3, 9) => (1, 3, 3, 4), (1, 3, 3, 4)
        // * The last three vector tiles will be padded values

        // Accum (RetileRow): (1, 3, 3, 4) => (1, 3, 3)
        // * This will be done both for data and mask stream

        // FlatmapFilterRowStreamify: (1, 3, 3) => (1, 3, 9)

        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let tile_shape: Vec<usize> = vec![1, 4];

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (mask_snd, mask_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();
        let (accum_data_snd, accum_data_rcv) = ctx.unbounded();
        let (accum_mask_snd, accum_mask_rcv) = ctx.unbounded();
        let (final_snd, final_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 9
            ])
            .into_shape_with_order((3, 9))
            .unwrap(),
        );
        ctx.add_child(GeneratorContext::new(
            move || {
                Buffer::new((*in_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(ReshapePadStream::new(
            in_rcv,
            out_snd,
            mask_snd,
            0,
            4,
            Some(Tile::new_blank(
                tile_shape.clone(),
                BYTES_PER_ELEM,
                READ_FROM_MU,
            )),
            2,
            false,
            0,
        ));

        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            out_rcv,
            accum_data_snd,
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                retile_row(tile1, tile2, comp_bw, write_back_mu, 0)
            }),
            Arc::new(move || Tile::new_empty([0, 4], BYTES_PER_ELEM, false)),
            1,
            AccumConfig {
                compute_bw: 1024,
                write_back_mu: false,
            },
            0,
        ));

        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            mask_rcv,
            accum_mask_snd,
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                retile_row(tile1, tile2, comp_bw, write_back_mu, 0)
            }),
            Arc::new(move || Tile::new_empty([0, 1], BYTES_PER_ELEM, false)),
            1,
            AccumConfig {
                compute_bw: 1024,
                write_back_mu: false,
            },
            0,
        ));

        ctx.add_child(FlatmapFilterRowStreamify::new(
            accum_data_rcv,
            Some(accum_mask_rcv),
            final_snd,
            0,
        ));

        let mut output_tile_vec = vec![];
        for _ in 0..3 {
            output_tile_vec.extend(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                9
            ]);
            output_tile_vec.extend(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3
            ]);
        }

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 9
            ])
            .into_shape_with_order((3, 9))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            final_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatmap_counter_test() {
        // Test that FlatmapCounter generates a sequence from 0 to N-1
        // when given a scalar tile with value N
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;
        const READ_FROM_MU: bool = false;

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // Test with count = 5, should generate 0, 1, 2, 3, 4
        let count: u32 = 5;
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![Elem::ValStop(
                    Tile::<VT>::new(
                        ndarray::arr2(&[[count]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    ),
                    0,
                )]
                .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(FlatmapCounter::new(in_rcv, out_snd, 0));

        // Expected output: 0, 1, 2, 3, 4 (last one with ValStop(1))
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                (0..count)
                    .map(|i| {
                        let tile = Tile::<VT>::new(
                            ndarray::arr2(&[[i]]).into_shared(),
                            BYTES_PER_ELEM,
                            READ_FROM_MU,
                        );
                        if i == count - 1 {
                            Elem::ValStop(tile, 1)
                        } else {
                            Elem::Val(tile)
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatmap_counter_multiple_inputs_test() {
        // Test that FlatmapCounter correctly handles multiple input tiles
        // and propagates stop levels correctly
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;
        const READ_FROM_MU: bool = false;

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // Send multiple count values: 3, 2, 4
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[3u32]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[2u32]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::ValStop(
                        Tile::<VT>::new(
                            ndarray::arr2(&[[4u32]]).into_shared(),
                            BYTES_PER_ELEM,
                            READ_FROM_MU,
                        ),
                        2,
                    ),
                ]
                .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(FlatmapCounter::new(in_rcv, out_snd, 0));

        // Expected output:
        // From first input (3): 0, 1, 2 (last with stop_level=1)
        // From second input (2): 0, 1 (last with stop_level=1)
        // From third input (4): 0, 1, 2, 3 (last with stop_level=3)
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                let mut expected = vec![];

                // First input (count=3): generates 0, 1, 2
                for i in 0..3 {
                    let tile = Tile::<VT>::new(
                        ndarray::arr2(&[[i]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    );
                    if i == 2 {
                        expected.push(Elem::ValStop(tile, 1));
                    } else {
                        expected.push(Elem::Val(tile));
                    }
                }

                // Second input (count=2): generates 0, 1
                for i in 0..2 {
                    let tile = Tile::<VT>::new(
                        ndarray::arr2(&[[i]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    );
                    if i == 1 {
                        expected.push(Elem::ValStop(tile, 1));
                    } else {
                        expected.push(Elem::Val(tile));
                    }
                }

                // Third input closes two outer dimensions. FlatmapCounter
                // appends its range dimension, so the final stop is 3.
                for i in 0..4 {
                    let tile = Tile::<VT>::new(
                        ndarray::arr2(&[[i]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    );
                    if i == 3 {
                        expected.push(Elem::ValStop(tile, 3));
                    } else {
                        expected.push(Elem::Val(tile));
                    }
                }

                expected.into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    #[should_panic(expected = "count must be positive")]
    fn flatmap_counter_rejects_zero_count() {
        let mut ctx = ProgramBuilder::default();
        let (_in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, _out_rcv) = ctx.unbounded();
        let counter = FlatmapCounter::<u64>::new(in_rcv, out_snd, 23);
        counter.gen_stream(
            Tile::new(ndarray::arr2(&[[0_u64]]).into_shared(), 8, false),
            None,
        );
    }

    #[test]
    fn flatmap_row_streamify_with_data_test() {
        // Test that Unmasked flatmap outputs all rows from a [3,4] tile
        // as three [1,4] tiles
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;
        const READ_FROM_MU: bool = false;

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // Input: a single [3,4] tile with actual data, wrapped in ValStop
        let input_arr = ndarray::arr2(&[[1u32, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
        let input_tile = Tile::<VT>::new(input_arr.into_shared(), BYTES_PER_ELEM, READ_FROM_MU);

        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(input_tile.clone(), 0)].into_iter(),
            in_snd,
        ));

        ctx.add_child(FlatmapFilterRowStreamify::new(in_rcv, None, out_snd, 0));

        // Expected: three [1,4] tiles; last one carries the stop level
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[1u32, 2, 3, 4]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[5u32, 6, 7, 8]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::ValStop(
                        Tile::<VT>::new(
                            ndarray::arr2(&[[9u32, 10, 11, 12]]).into_shared(),
                            BYTES_PER_ELEM,
                            READ_FROM_MU,
                        ),
                        0,
                    ),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatmap_row_streamify_blank_test() {
        // Test that Unmasked flatmap handles blank tiles (no underlying data)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;
        const READ_FROM_MU: bool = false;

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // Input: a blank [2,5] tile
        let input_tile = Tile::<VT>::new_blank(vec![2, 5], BYTES_PER_ELEM, READ_FROM_MU);

        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(input_tile.clone(), 0)].into_iter(),
            in_snd,
        ));

        ctx.add_child(FlatmapFilterRowStreamify::new(in_rcv, None, out_snd, 0));

        // Expected: two blank [1,5] tiles; last carries stop level
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new_blank(
                        vec![1, 5],
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::ValStop(
                        Tile::<VT>::new_blank(vec![1, 5], BYTES_PER_ELEM, READ_FROM_MU),
                        0,
                    ),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatmap_row_streamify_multiple_inputs_test() {
        // Test with multiple input tiles (Val then ValStop)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;
        const READ_FROM_MU: bool = false;

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // Two input tiles: a [2,3] Val, then a [2,3] ValStop
        let arr1 = ndarray::arr2(&[[1u32, 2, 3], [4, 5, 6]]);
        let tile1 = Tile::<VT>::new(arr1.into_shared(), BYTES_PER_ELEM, READ_FROM_MU);

        let arr2 = ndarray::arr2(&[[7u32, 8, 9], [10, 11, 12]]);
        let tile2 = Tile::<VT>::new(arr2.into_shared(), BYTES_PER_ELEM, READ_FROM_MU);

        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(tile1.clone()), Elem::ValStop(tile2.clone(), 0)].into_iter(),
            in_snd,
        ));

        ctx.add_child(FlatmapFilterRowStreamify::new(in_rcv, None, out_snd, 0));

        // Expected: 4 row tiles total
        // From tile1 (Val, no stop): [1,2,3] Val, [4,5,6] Val
        // From tile2 (ValStop(0)): [7,8,9] Val, [10,11,12] ValStop(0)
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[1u32, 2, 3]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[4u32, 5, 6]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::Val(Tile::<VT>::new(
                        ndarray::arr2(&[[7u32, 8, 9]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    )),
                    Elem::ValStop(
                        Tile::<VT>::new(
                            ndarray::arr2(&[[10u32, 11, 12]]).into_shared(),
                            BYTES_PER_ELEM,
                            READ_FROM_MU,
                        ),
                        0,
                    ),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
    #[test]
    fn filtered_rows_keep_values_and_stop_after_last_selected_row() {
        for blank in [false, true] {
            let mut ctx = ProgramBuilder::default();
            let (data_snd, data_rcv) = ctx.unbounded();
            let (mask_snd, mask_rcv) = ctx.unbounded();
            let (out_snd, out_rcv) = ctx.unbounded();
            let data = if blank {
                Tile::new_blank(vec![5, 2], 4, true)
            } else {
                Tile::new(
                    ndarray::arr2(&[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]).into_shared(),
                    4,
                    true,
                )
            };
            let mask = Tile::new(
                ndarray::arr2(&[[false], [true], [false], [true], [false]]).into_shared(),
                1,
                false,
            );
            ctx.add_child(GeneratorContext::new(
                move || vec![Elem::ValStop(data, 2)].into_iter(),
                data_snd,
            ));
            ctx.add_child(GeneratorContext::new(
                move || vec![Elem::ValStop(mask, 2)].into_iter(),
                mask_snd,
            ));
            ctx.add_child(FlatmapFilterRowStreamify::new(
                data_rcv,
                Some(mask_rcv),
                out_snd,
                0,
            ));
            let selected = [[2, 3], [6, 7]].map(|row| {
                if blank {
                    Tile::new_blank(vec![1, 2], 4, true)
                } else {
                    Tile::new(ndarray::arr2(&[row]).into_shared(), 4, true)
                }
            });
            let [first, last] = selected;
            ctx.add_child(ApproxCheckerContext::new(
                move || vec![Elem::Val(first), Elem::ValStop(last, 2)].into_iter(),
                out_rcv,
                |actual, expected| actual == expected,
            ));
            ctx.initialize(Default::default())
                .unwrap()
                .run(Default::default());
        }
    }

    #[test]
    fn filter_rows_with_explicit_mask() {
        // [1,3] => [1,7]
        // [3,4] tile => [1,4] tile (last tile is padded with 2 vectors)
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 6 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(0),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new(concatenated_array.to_shared(), 4, read_from_mu),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays_input: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let arrays_output: Vec<Array2<i32>> = (0..7)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays_input, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays_output, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        let (mask_snd, mask_rcv) = ctx.unbounded();
        let masks = [
            vec![true, true, true],
            vec![true, true, true],
            vec![true, false, false],
        ]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            let tile = Tile::new(
                Array2::from_shape_vec((3, 1), values).unwrap().to_shared(),
                1,
                false,
            );
            if i == 2 {
                Elem::ValStop(tile, 1)
            } else {
                Elem::Val(tile)
            }
        })
        .collect::<Vec<_>>();
        ctx.add_child(GeneratorContext::new(|| masks.into_iter(), mask_snd));
        ctx.add_child(
            crate::operator::flatmap_decomp::FlatmapFilterRowStreamify::new(
                in_data_rcv,
                Some(mask_rcv),
                out_data_snd,
                0,
            ),
        );
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            |actual, expected| actual == expected,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
