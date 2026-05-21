use crate::primitives::elem::{Elem, StopType};
use crate::primitives::tile::Tile;
use dam::context_tools::*;
use crate::trace::TracingSender as Sender;
use dam::types::DAMType;

#[context_macro]
pub struct FlatmapFilterRowStreamify<T: Clone> {
    in_stream: Receiver<Elem<Tile<T>>>,      // tile shape: [R,C]
    mask_stream: Receiver<Elem<Tile<bool>>>, // tile shape: [R,1]
    out_stream: Sender<Elem<Tile<T>>>,       // tile shape: [1,C]
    id: u32,
}

impl<T: Clone> FlatmapFilterRowStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        mask_stream: Receiver<Elem<Tile<bool>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            mask_stream,
            out_stream,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.mask_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
    fn filter_streamify(
        &mut self,
        data: &Tile<T>,
        mask: &Tile<bool>,
        stop_level: Option<StopType>,
    ) {
        let mask_iter: ndarray::iter::LanesIter<'_, bool, ndarray::Dim<[usize; 1]>> =
            mask.underlying.as_ref().unwrap().rows().into_iter(); // [1] (bool) x R
        let mut count = 0;
        for mask_i in mask_iter {
            if mask_i[0] {
                count += 1;
            }
        }

        match &data.underlying {
            Some(arr) => {
                // split_row: [R,C] => [C] x R
                // Later if we want to do column-wise, we can do that here ([R,C] => [R] x C)
                // let vec_iter = if self.split_row {
                //     arr.rows().into_iter()
                // } else {
                //     arr.columns().into_iter()
                // };
                let vec_iter = arr.rows().into_iter(); // [C] x R
                let mask_iter = mask.underlying.as_ref().unwrap().rows().into_iter(); // [1] (bool) x R

                let mut curr_cnt = 0;
                for (row, mask_i) in vec_iter.zip(mask_iter) {
                    // Only process rows where the mask is true
                    if mask_i[0] {
                        let out_data = Tile::<T>::new_padded(
                            row.to_shared().insert_axis(ndarray::Axis(0)), // [N] => [1,N]
                            data.bytes_per_elem,
                            data.read_from_mu,
                            1,
                        );
                        curr_cnt += 1;

                        // check whether this is the last value and set the stop level if needed
                        let elem = if stop_level.is_some() && (curr_cnt == count) {
                            Elem::ValStop(out_data, stop_level.unwrap())
                        } else {
                            Elem::Val(out_data)
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
            None => {
                // let vec_size = if self.split_row {
                //     data.shape[1]
                // } else {
                //     data.shape[0]
                // };
                let vec_size = data.shape[1];

                for idx in 0..count {
                    let out_data = Tile::<T>::new_blank(
                        vec![1, vec_size],
                        data.bytes_per_elem,
                        data.read_from_mu,
                    );

                    // check whether this is the last value and set the stop level if needed
                    let elem = if stop_level.is_some() && (idx + 1 == count) {
                        Elem::ValStop(out_data, stop_level.unwrap())
                    } else {
                        Elem::Val(out_data)
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
    }
}

impl<T: Clone> Context for FlatmapFilterRowStreamify<T>
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
                    Elem::Val(data) => match self.mask_stream.dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: mask_enum,
                        }) => match mask_enum {
                            Elem::Val(mask) => self.filter_streamify(&data, &mask, None),
                            Elem::ValStop(_, s) => {
                                panic!(
                                    "Stop token position mismatch: data(val), mask(valstop({}))",
                                    s
                                );
                            }
                        },
                        Err(_) => {
                            panic!("Mask stream terminated before data stream");
                        }
                    },
                    Elem::ValStop(data, s) => match self.mask_stream.dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: mask_enum,
                        }) => match mask_enum {
                            Elem::Val(_) => {
                                panic!(
                                    "Stop token position mismatch: data(valstop({})), mask(val)",
                                    s
                                );
                            }
                            Elem::ValStop(mask, mask_s) => {
                                assert_eq!(mask_s, s);
                                self.filter_streamify(&data, &mask, Some(s))
                            }
                        },
                        Err(_) => {
                            panic!("Mask stream terminated before data stream");
                        }
                    },
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[context_macro]
pub struct FlatmapRowStreamify<T: Clone> {
    in_stream: Receiver<Elem<Tile<T>>>, // tile shape: [R,C]
    out_stream: Sender<Elem<Tile<T>>>,  // tile shape: [1,C]
    id: u32,
}

impl<T: Clone> FlatmapRowStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    fn streamify(&mut self, data: &Tile<T>, stop_level: Option<StopType>) {
        let num_rows = data.shape[0];

        match &data.underlying {
            Some(arr) => {
                let vec_iter = arr.rows().into_iter(); // [C] x R

                for (idx, row) in vec_iter.enumerate() {
                    let out_data = Tile::<T>::new_padded(
                        row.to_shared().insert_axis(ndarray::Axis(0)), // [N] => [1,N]
                        data.bytes_per_elem,
                        data.read_from_mu,
                        1,
                    );

                    let elem = if stop_level.is_some() && (idx + 1 == num_rows) {
                        Elem::ValStop(out_data, stop_level.unwrap())
                    } else {
                        Elem::Val(out_data)
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
            None => {
                let vec_size = data.shape[1];

                for idx in 0..num_rows {
                    let out_data = Tile::<T>::new_blank(
                        vec![1, vec_size],
                        data.bytes_per_elem,
                        data.read_from_mu,
                    );

                    let elem = if stop_level.is_some() && (idx + 1 == num_rows) {
                        Elem::ValStop(out_data, stop_level.unwrap())
                    } else {
                        Elem::Val(out_data)
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
    }
}

impl<T: Clone> Context for FlatmapRowStreamify<T>
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
                    Elem::Val(data) => self.streamify(&data, None),
                    Elem::ValStop(data, s) => self.streamify(&data, Some(s)),
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[context_macro]
pub struct FlatmapCounter<T: Clone> {
    in_stream: Receiver<Elem<Tile<T>>>, // tile shape: [1,1]
    out_stream: Sender<Elem<Tile<T>>>,  // tile shape: [1,1]
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
            in_stream,
            out_stream,
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
        for i in 0..data_usize - 1 {
            let t_value = <T>::try_from(i).unwrap_or_else(|_| {
                panic!("[Counter {}] Failed to convert index to T", self.id);
            });
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: Elem::Val(Tile::<T>::new(
                            ndarray::arr2(&[[t_value]]).into_shared(),
                            scalar_tile.bytes_per_elem,
                            scalar_tile.read_from_mu,
                        )),
                    },
                )
                .unwrap();
        }

        let t_value = <T>::try_from(data_usize - 1).unwrap_or_else(|_| {
            panic!("[Counter {}] Failed to convert index to T", self.id);
        });
        let stop_level = if stop_level.is_none() {
            1
        } else {
            stop_level.unwrap() + 1
        };
        self.out_stream
            .enqueue(
                &self.time,
                ChannelElement {
                    time: self.time.tick(),
                    data: Elem::ValStop(
                        Tile::<T>::new(
                            ndarray::arr2(&[[t_value]]).into_shared(),
                            scalar_tile.bytes_per_elem,
                            scalar_tile.read_from_mu,
                        ),
                        stop_level,
                    ),
                },
            )
            .unwrap();
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
    use ndarray::ArcArray;

    use super::{FlatmapCounter, FlatmapFilterRowStreamify, FlatmapRowStreamify};
    use crate::primitives::{buffer::Buffer, elem::Elem, tile::Tile};
    use crate::{functions::accum_fn::retile_row, operator::accum::AccumConfig};
    use crate::{
        operator::{accum::Accum, reshape::ReshapePadStream},
        utils::events::SimpleEvent,
    };

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
            Some(Tile::new_blank_padded(
                tile_shape.clone(),
                BYTES_PER_ELEM,
                READ_FROM_MU,
                0,
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
            accum_mask_rcv,
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
                Tile::<VT>::new_blank_padded(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU,
                    0
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
                        0,
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
        // From third input (4): 0, 1, 2, 3 (last with stop_level=1)
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

                // Third input (count=4, with stop_level=0): generates 0, 1, 2, 3
                for i in 0..4 {
                    let tile = Tile::<VT>::new(
                        ndarray::arr2(&[[i]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                    );
                    if i == 3 {
                        expected.push(Elem::ValStop(tile, 1));
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
    fn flatmap_row_streamify_with_data_test() {
        // Test that FlatmapRowStreamify outputs all rows from a [3,4] tile
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

        ctx.add_child(FlatmapRowStreamify::new(in_rcv, out_snd, 0));

        // Expected: three [1,4] tiles; last one carries the stop level
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new_padded(
                        ndarray::arr2(&[[1u32, 2, 3, 4]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                        1,
                    )),
                    Elem::Val(Tile::<VT>::new_padded(
                        ndarray::arr2(&[[5u32, 6, 7, 8]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                        1,
                    )),
                    Elem::ValStop(
                        Tile::<VT>::new_padded(
                            ndarray::arr2(&[[9u32, 10, 11, 12]]).into_shared(),
                            BYTES_PER_ELEM,
                            READ_FROM_MU,
                            1,
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
        // Test that FlatmapRowStreamify handles blank tiles (no underlying data)
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

        ctx.add_child(FlatmapRowStreamify::new(in_rcv, out_snd, 0));

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

        ctx.add_child(FlatmapRowStreamify::new(in_rcv, out_snd, 0));

        // Expected: 4 row tiles total
        // From tile1 (Val, no stop): [1,2,3] Val, [4,5,6] Val
        // From tile2 (ValStop(0)): [7,8,9] Val, [10,11,12] ValStop(0)
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new_padded(
                        ndarray::arr2(&[[1u32, 2, 3]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                        1,
                    )),
                    Elem::Val(Tile::<VT>::new_padded(
                        ndarray::arr2(&[[4u32, 5, 6]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                        1,
                    )),
                    Elem::Val(Tile::<VT>::new_padded(
                        ndarray::arr2(&[[7u32, 8, 9]]).into_shared(),
                        BYTES_PER_ELEM,
                        READ_FROM_MU,
                        1,
                    )),
                    Elem::ValStop(
                        Tile::<VT>::new_padded(
                            ndarray::arr2(&[[10u32, 11, 12]]).into_shared(),
                            BYTES_PER_ELEM,
                            READ_FROM_MU,
                            1,
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
}
