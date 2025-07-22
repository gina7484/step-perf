use crate::primitives::elem::{Elem, StopType};
use dam::context_tools::*;

#[context_macro]
pub struct Reshape<InputType: Clone> {
    in_stream: Receiver<Elem<InputType>>,
    out_stream: Sender<Elem<InputType>>,
    split_dim: usize,
    chunk_size: usize,
    pad_val: Option<InputType>,
    input_stream_rank: StopType,
    add_outer_dim: bool,
    id: u32,
}

impl<InputType: DAMType> Reshape<InputType>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<InputType>>,
        out_stream: Sender<Elem<InputType>>,
        split_dim: usize,
        chunk_size: usize,
        pad_val: Option<InputType>,
        input_stream_rank: StopType,
        add_outer_dim: bool,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            split_dim,
            chunk_size,
            pad_val,
            input_stream_rank,
            add_outer_dim,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<InputType: DAMType> Context for Reshape<InputType> {
    fn run(&mut self) {
        if self.split_dim == 0 {
            let mut counter = 0;
            loop {
                match self.in_stream.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => match data {
                        Elem::Val(x) => {
                            counter += 1;

                            let output_elem = if counter == self.chunk_size {
                                counter = 0;
                                if self.add_outer_dim {
                                    let stop_level = match self.in_stream.peek_next(&self.time) {
                                        Ok(ChannelElement { time: _, data: _ }) => 1,
                                        Err(_) => 2,
                                    };
                                    Elem::ValStop(x.clone(), stop_level)
                                } else {
                                    Elem::ValStop(x.clone(), 1)
                                }
                            } else {
                                Elem::Val(x.clone())
                            };
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: output_elem,
                                    },
                                )
                                .unwrap();
                        }
                        Elem::ValStop(x, s) => {
                            counter += 1;
                            if counter == self.chunk_size {
                                counter = 0;
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::ValStop(x.clone(), s + 1),
                                        },
                                    )
                                    .unwrap();
                            } else {
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::Val(x.clone()),
                                        },
                                    )
                                    .unwrap();

                                assert!(
                                    self.pad_val.is_some(),
                                    "When splitting the innermost dimension, \
                                    we pad if the dimension is not exactly divisible by the chunk size. \
                                    Therefore, the pad_val must be provided."
                                );

                                // pad so that the dimension is divisible by the chunk size
                                for i in 0..self.chunk_size - counter {
                                    let padded_val = if i == self.chunk_size - counter - 1 {
                                        Elem::ValStop(self.pad_val.clone().unwrap(), s + 1)
                                    } else {
                                        Elem::Val(self.pad_val.clone().unwrap())
                                    };

                                    self.out_stream
                                        .enqueue(
                                            &self.time,
                                            ChannelElement {
                                                time: self.time.tick(),
                                                data: padded_val,
                                            },
                                        )
                                        .unwrap();
                                }
                                counter = 0;
                            };
                        }
                    },
                    Err(_) => {
                        if 0 < counter && counter < self.chunk_size {
                            // use this as if we got a done token
                            assert!(
                                self.pad_val.is_some(),
                                "When splitting the innermost dimension, \
                                we pad if the dimension is not exactly divisible by the chunk size. \
                                Therefore, the pad_val must be provided."
                            );
                            assert!(
                                self.input_stream_rank == 0,
                                "input stream rank should be 0 to enter here"
                            );
                            // pad so that the dimension is divisible by the chunk size
                            for i in 0..self.chunk_size - counter {
                                let is_last = i == self.chunk_size - counter - 1;
                                let padded_val = if is_last {
                                    let stop_lev = if self.add_outer_dim { 2 } else { 1 };
                                    Elem::ValStop(self.pad_val.clone().unwrap(), stop_lev)
                                } else {
                                    Elem::Val(self.pad_val.clone().unwrap())
                                };

                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: padded_val,
                                        },
                                    )
                                    .unwrap();
                            }
                        }
                        return;
                    }
                }
            }
        } else {
            let mut counter = 0;
            loop {
                match self.in_stream.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => match data {
                        Elem::Val(x) => {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(x.clone()),
                                    },
                                )
                                .unwrap();
                        }
                        Elem::ValStop(x, s) => {
                            if s >= self.split_dim as StopType {
                                counter += 1;
                            }
                            if counter == self.chunk_size {
                                counter = 0;

                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::ValStop(x.clone(), s + 1),
                                        },
                                    )
                                    .unwrap();
                            } else {
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::ValStop(x.clone(), s),
                                        },
                                    )
                                    .unwrap();
                            };
                        }
                    },
                    Err(_) => return,
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
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, GeneratorContext, PrinterContext,
        },
    };
    use ndarray::ArcArray;

    use crate::primitives::{buffer::Buffer, elem::Elem, tile::Tile};

    use super::Reshape;

    #[test]
    fn reshape_0d_with_pad() {
        // (1, 3, 9) => (1, 3, 3, 4)
        // the last three vector tiles will be padded values
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let tile_shape: Vec<usize> = vec![1, 4];

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

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

        ctx.add_child(Reshape::new(
            in_rcv,
            out_snd,
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
            ArcArray::from_vec(output_tile_vec)
                .into_shape_with_order((3, 3, 4))
                .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn reshape_0d_with_pad_0d_stream() {
        // (9) => (3, 4)
        // the last three vector tiles will be padded values
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;

        let tile_shape: Vec<usize> = vec![1, 4];

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr =
            vec![Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU); 9]
                .into_iter()
                .collect::<Vec<_>>();
        ctx.add_child(GeneratorContext::new(
            move || in_arr.into_iter().map(|x| Elem::Val(x.clone())).into_iter(),
            in_snd,
        ));

        ctx.add_child(Reshape::new(
            in_rcv,
            out_snd,
            0,
            4,
            Some(Tile::new_blank_padded(
                tile_shape.clone(),
                BYTES_PER_ELEM,
                READ_FROM_MU,
                0,
            )),
            0,
            false,
            0,
        ));

        let val_tile = Elem::Val(Tile::<VT>::new_blank(
            tile_shape.clone(),
            BYTES_PER_ELEM,
            READ_FROM_MU,
        ));
        let val_stop_tile = Elem::ValStop(
            Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU),
            1,
        );
        let val_tile_pad = Elem::Val(Tile::<VT>::new_blank_padded(
            tile_shape.clone(),
            BYTES_PER_ELEM,
            READ_FROM_MU,
            0,
        ));
        let val_stop_tile_pad = Elem::ValStop(
            Tile::<VT>::new_blank_padded(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU, 0),
            1,
        );
        let output_tile_vec = vec![
            val_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile.clone(),
            val_tile.clone(),
            val_tile_pad.clone(),
            val_tile_pad.clone(),
            val_stop_tile_pad.clone(),
        ];
        ctx.add_child(ApproxCheckerContext::new(
            move || output_tile_vec.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn reshape_0d_with_pad_0d_stream_add_outer_dim_divisible() {
        // (9) => (1, 3, 3)
        // the last three vector tiles will be padded values
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;

        let tile_shape: Vec<usize> = vec![1, 4];

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr =
            vec![Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU); 9]
                .into_iter()
                .collect::<Vec<_>>();
        ctx.add_child(GeneratorContext::new(
            move || in_arr.into_iter().map(|x| Elem::Val(x.clone())).into_iter(),
            in_snd,
        ));

        ctx.add_child(Reshape::new(
            in_rcv,
            out_snd,
            0,
            3,
            Some(Tile::new_blank_padded(
                tile_shape.clone(),
                BYTES_PER_ELEM,
                READ_FROM_MU,
                0,
            )),
            0,
            true,
            0,
        ));

        let val_tile = Elem::Val(Tile::<VT>::new_blank(
            tile_shape.clone(),
            BYTES_PER_ELEM,
            READ_FROM_MU,
        ));
        let val_stop_tile = Elem::ValStop(
            Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU),
            1,
        );
        let val_stop_tile_last = Elem::ValStop(
            Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU),
            2,
        );
        let output_tile_vec = vec![
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile_last.clone(),
        ];
        ctx.add_child(ApproxCheckerContext::new(
            move || output_tile_vec.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn reshape_0d_with_pad_0d_stream_add_outer_dim() {
        // (9) => (1, 3, 4)
        // the last three vector tiles will be padded values
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;

        let tile_shape: Vec<usize> = vec![1, 4];

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr =
            vec![Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU); 9]
                .into_iter()
                .collect::<Vec<_>>();
        ctx.add_child(GeneratorContext::new(
            move || in_arr.into_iter().map(|x| Elem::Val(x.clone())).into_iter(),
            in_snd,
        ));

        ctx.add_child(Reshape::new(
            in_rcv,
            out_snd,
            0,
            4,
            Some(Tile::new_blank_padded(
                tile_shape.clone(),
                BYTES_PER_ELEM,
                READ_FROM_MU,
                0,
            )),
            0,
            true,
            0,
        ));

        let val_tile = Elem::Val(Tile::<VT>::new_blank(
            tile_shape.clone(),
            BYTES_PER_ELEM,
            READ_FROM_MU,
        ));
        let val_stop_tile = Elem::ValStop(
            Tile::<VT>::new_blank(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU),
            1,
        );
        let val_tile_pad = Elem::Val(Tile::<VT>::new_blank_padded(
            tile_shape.clone(),
            BYTES_PER_ELEM,
            READ_FROM_MU,
            0,
        ));
        let val_stop_tile_pad = Elem::ValStop(
            Tile::<VT>::new_blank_padded(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU, 0),
            1,
        );
        let val_stop_tile_pad_last = Elem::ValStop(
            Tile::<VT>::new_blank_padded(tile_shape.clone(), BYTES_PER_ELEM, READ_FROM_MU, 0),
            2,
        );
        let output_tile_vec = vec![
            val_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_tile.clone(),
            val_stop_tile.clone(),
            val_tile.clone(),
            val_tile_pad.clone(),
            val_tile_pad.clone(),
            val_stop_tile_pad_last.clone(),
        ];
        ctx.add_child(ApproxCheckerContext::new(
            move || output_tile_vec.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn reshape_as_promote() {
        // (1,3, 2 * 1) => (1,3, 2, 1)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 1
            ])
            .into_shape_with_order((3, 2 * 1))
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

        ctx.add_child(Reshape::new(in_rcv, out_snd, 0, 1, None, 2, false, 0));

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 1
            ])
            .into_shape_with_order((3, 2, 1))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
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
    fn reshape_1d() {
        // (5, 2 * 3, 4) => (5, 2, 3, 4)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                5 * 2 * 3 * 4
            ])
            .into_shape_with_order((5, 2 * 3, 4))
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

        ctx.add_child(Reshape::new(in_rcv, out_snd, 1, 3, None, 2, false, 0));

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                5 * 2 * 3 * 4
            ])
            .into_shape_with_order((5, 2, 3, 4))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
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
}
