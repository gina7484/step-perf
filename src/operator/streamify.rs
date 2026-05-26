use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;

use crate::primitives::buffer::Buffer;
use crate::primitives::elem::Bufferizable;
use crate::primitives::elem::{Elem, StopType};

use crate::utils::events::LoggableEventSimple;

/// * `repeat_factor``: The number of repeated linear reads to do for each buffer. The final output shape will be `repeat_factor * buffer.shape()`.
#[context_macro]
pub struct Streamify<E: LoggableEventSimple, T: Bufferizable + Clone> {
    pub repeat_factor: Vec<usize>, // The number of repeated linear reads to do for each buffer
    pub rank: StopType,
    pub in_stream: Receiver<Elem<Buffer<T>>>,
    pub out_stream: Sender<Elem<T>>,
    pub id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
    > Streamify<E, T>
where
    Buffer<T>: DAMType,
{
    pub fn new(
        repeat_factor: Vec<usize>, // The number of repeated linear reads to do for each buffer
        rank: StopType,
        in_stream: Receiver<Elem<Buffer<T>>>,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            repeat_factor,
            rank,
            in_stream,
            out_stream,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
    > Context for Streamify<E, T>
where
    Buffer<T>: DAMType,
{
    fn run(&mut self) {
        let mut tensor_shape_tiled: Vec<usize>;

        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: buff_elem,
                }) => {
                    let start_time = self.time.tick().time();
                    match buff_elem {
                        Elem::Val(buff) => {
                            if self.repeat_factor.is_empty() {
                                for elem in buff.to_elem_iter() {
                                    self.out_stream
                                        .enqueue(
                                            &self.time,
                                            ChannelElement {
                                                time: self.time.tick(),
                                                data: elem,
                                            },
                                        )
                                        .unwrap();
                                    self.time.incr_cycles(1);
                                }
                            } else {
                                for (i, repeat_factor) in
                                    self.repeat_factor.iter().rev().enumerate()
                                {
                                    // For each buffer, we will repeat the elements based on the repeat factor
                                    for repeat_i in 0..*repeat_factor {
                                        let buff_clone = buff.clone();
                                        for elem in buff_clone.to_elem_iter() {
                                            match elem {
                                                Elem::Val(tile) => {
                                                    self.out_stream
                                                        .enqueue(
                                                            &self.time,
                                                            ChannelElement {
                                                                time: self.time.tick(),
                                                                data: Elem::Val(tile),
                                                            },
                                                        )
                                                        .unwrap();

                                                    self.time.incr_cycles(1);
                                                }
                                                Elem::ValStop(tile, stop_lev) => {
                                                    let new_stop_level = if stop_lev == self.rank
                                                        && repeat_i == (*repeat_factor - 1)
                                                    {
                                                        stop_lev + 1 + i as StopType
                                                    } else {
                                                        stop_lev
                                                    };
                                                    self.out_stream
                                                        .enqueue(
                                                            &self.time,
                                                            ChannelElement {
                                                                time: self.time.tick(),
                                                                data: Elem::ValStop(
                                                                    tile,
                                                                    new_stop_level,
                                                                ),
                                                            },
                                                        )
                                                        .unwrap();

                                                    self.time.incr_cycles(1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Elem::ValStop(buff, outer_stop_lev) => {
                            if self.repeat_factor.is_empty() {
                                for elem in buff.to_elem_iter() {
                                    match elem {
                                        Elem::Val(tile) => {
                                            self.out_stream
                                                .enqueue(
                                                    &self.time,
                                                    ChannelElement {
                                                        time: self.time.tick(),
                                                        data: Elem::Val(tile),
                                                    },
                                                )
                                                .unwrap();

                                            self.time.incr_cycles(1);
                                        }
                                        Elem::ValStop(tile, stop_lev) => {
                                            let new_stop_level = if stop_lev == self.rank {
                                                stop_lev + outer_stop_lev
                                            } else {
                                                stop_lev
                                            };
                                            self.out_stream
                                                .enqueue(
                                                    &self.time,
                                                    ChannelElement {
                                                        time: self.time.tick(),
                                                        data: Elem::ValStop(tile, new_stop_level),
                                                    },
                                                )
                                                .unwrap();

                                            self.time.incr_cycles(1);
                                        }
                                    }
                                }
                            } else {
                                for (i, repeat_factor) in
                                    self.repeat_factor.iter().rev().enumerate()
                                {
                                    // For each buffer, we will repeat the elements based on the repeat factor
                                    for repeat_i in 0..*repeat_factor {
                                        let buff_clone = buff.clone();
                                        for elem in buff_clone.to_elem_iter() {
                                            match elem {
                                                Elem::Val(tile) => {
                                                    self.out_stream
                                                        .enqueue(
                                                            &self.time,
                                                            ChannelElement {
                                                                time: self.time.tick(),
                                                                data: Elem::Val(tile),
                                                            },
                                                        )
                                                        .unwrap();

                                                    self.time.incr_cycles(1);
                                                }
                                                Elem::ValStop(tile, stop_lev) => {
                                                    let new_stop_level = if stop_lev == self.rank
                                                        && repeat_i == (*repeat_factor - 1)
                                                    {
                                                        stop_lev
                                                            + outer_stop_lev
                                                            + 1
                                                            + i as StopType
                                                    } else {
                                                        stop_lev
                                                    };
                                                    self.out_stream
                                                        .enqueue(
                                                            &self.time,
                                                            ChannelElement {
                                                                time: self.time.tick(),
                                                                data: Elem::ValStop(
                                                                    tile,
                                                                    new_stop_level,
                                                                ),
                                                            },
                                                        )
                                                        .unwrap();

                                                    self.time.incr_cycles(1);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    self.in_stream.dequeue(&self.time).unwrap();

                    dam::logging::log_event(&E::new(
                        "Streamify".to_string(),
                        self.id,
                        start_time,
                        self.time.tick().time(),
                        false,
                    ))
                    .unwrap();
                }
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[context_macro]
pub struct StaticStreamify<E: LoggableEventSimple, T: Bufferizable + Clone> {
    pub stride: Vec<usize>,
    pub out_shape: Vec<usize>,
    pub in_stream: Receiver<Elem<Buffer<T>>>,
    pub out_stream: Sender<Elem<T>>,
    pub id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
    > StaticStreamify<E, T>
where
    Buffer<T>: DAMType,
{
    pub fn new(
        stride: Vec<usize>,
        out_shape: Vec<usize>,
        in_stream: Receiver<Elem<Buffer<T>>>,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            stride,
            out_shape,
            in_stream,
            out_stream,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    fn get_view(&self, buffer: Buffer<T>) -> impl Iterator<Item = Elem<T>> {
        let out_shape = self.out_shape.clone();
        let stride = self.stride.clone();
        assert_eq!(
            out_shape.len(),
            stride.len(),
            "stride and out_shape must have the same rank"
        );
        let ndim = out_shape.len();
        let total_elems: usize = out_shape.iter().product();

        // Flatten the underlying buffer in C-order so we can index by the linear
        // address computed from `stride * multi_index`.
        let flat: Vec<T> = buffer.iter().cloned().collect();

        (0..total_elems).map(move |flat_idx| {
            // Convert flat_idx into a multi-dim index in out_shape (row-major).
            let mut multi_index = vec![0usize; ndim];
            let mut remaining = flat_idx;
            for i in (0..ndim).rev() {
                multi_index[i] = remaining % out_shape[i];
                remaining /= out_shape[i];
            }

            // Map the output position into the buffer via stride.
            let buf_idx: usize = (0..ndim).map(|d| multi_index[d] * stride[d]).sum();
            let val = flat[buf_idx].clone();

            // Determine the highest-dimensional stop token needed, walking from
            // the innermost dim outward and only promoting the stop level while
            // all inner dims are at their last element.
            let mut highest_stop_token: Option<StopType> = None;
            let mut all_inner_dims_at_end = true;
            for dim in (0..ndim).rev() {
                if all_inner_dims_at_end {
                    let is_dim_size_one = out_shape[dim] == 1;
                    let is_last_elem = multi_index[dim] == out_shape[dim] - 1;
                    if is_last_elem || is_dim_size_one {
                        highest_stop_token = Some((ndim - dim) as StopType);
                    }
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            match highest_stop_token {
                Some(stop_type) => Elem::ValStop(val, stop_type),
                None => Elem::Val(val),
            }
        })
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
    > Context for StaticStreamify<E, T>
where
    Buffer<T>: DAMType,
{
    fn run(&mut self) {
        let mut tensor_shape_tiled: Vec<usize>;
        let rank = self.out_shape.len() as u32;

        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: buff_elem,
                }) => {
                    let start_time = self.time.tick().time();
                    match buff_elem {
                        Elem::Val(buff) => {
                            for elem in self.get_view(buff) {
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: elem,
                                        },
                                    )
                                    .unwrap();
                                self.time.incr_cycles(1);
                            }
                        }
                        Elem::ValStop(buff, outer_stop_lev) => {
                            for elem in self.get_view(buff) {
                                match elem {
                                    Elem::Val(tile) => {
                                        self.out_stream
                                            .enqueue(
                                                &self.time,
                                                ChannelElement {
                                                    time: self.time.tick(),
                                                    data: Elem::Val(tile),
                                                },
                                            )
                                            .unwrap();

                                        self.time.incr_cycles(1);
                                    }
                                    Elem::ValStop(tile, stop_lev) => {
                                        let new_stop_level = if stop_lev == rank {
                                            stop_lev + outer_stop_lev
                                        } else {
                                            stop_lev
                                        };
                                        self.out_stream
                                            .enqueue(
                                                &self.time,
                                                ChannelElement {
                                                    time: self.time.tick(),
                                                    data: Elem::ValStop(tile, new_stop_level),
                                                },
                                            )
                                            .unwrap();

                                        self.time.incr_cycles(1);
                                    }
                                }
                            }
                        }
                    }
                    self.in_stream.dequeue(&self.time).unwrap();

                    dam::logging::log_event(&E::new(
                        "Streamify".to_string(),
                        self.id,
                        start_time,
                        self.time.tick().time(),
                        false,
                    ))
                    .unwrap();
                }
                Err(_) => {
                    return;
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::utils::events::{LoggableEventSimple, DUMMY_ID};
    use crate::{
        define_simple_event,
        operator::bufferize::Bufferize,
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::dam_macros::event_type;
    use dam::{
        simulation::{
            LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
        },
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, FunctionContext, GeneratorContext,
        },
    };
    use serde::{Deserialize, Serialize};

    define_simple_event!(BufferizeEvent);
    define_simple_event!(StreamifyEvent);
    #[test]
    fn round_trip_test_3d() {
        // Tiled stream shape: [1,3, 2, 2] => [1,3,|2, 2] =>[1,3, 2, 2, 2] (2D repeat)
        //                                                        |_ repeated
        type VT = u32;

        let mut ctx = ProgramBuilder::default();
        let bufferize_rank = 2;

        // [1,3,2,2]
        let input_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
        ];

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || input_tiled_stream.into_iter(),
            snd,
        ));

        let (buff_snd, buff_rcv) = ctx.bounded(1);
        ctx.add_child(Bufferize::<BufferizeEvent, _>::new(
            rcv,
            buff_snd,
            bufferize_rank,
            DUMMY_ID,
        ));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(super::Streamify::<StreamifyEvent, _>::new(
            vec![2],
            bufferize_rank,
            buff_rcv,
            out_snd,
            DUMMY_ID,
        ));

        // [1,3,2,2,2]
        let output_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 4),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || output_tiled_stream.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        let logging = true;
        if logging {
            let initialized = ctx.initialize(Default::default()).unwrap();
            let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
                // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
                dam::logging::LogFilter::AllowAll,
            ));
            let run_options = run_options.logging(LoggingOptions::Mongo(
                MongoOptionsBuilder::default()
                    .db("test_streamify".to_string())
                    .uri("mongodb://127.0.0.1:27017".to_string())
                    .build()
                    .unwrap(),
            ));
            let summary = initialized.run(run_options.build().unwrap());
            // Check the summary
            println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
        } else {
            ctx.initialize(Default::default())
                .unwrap()
                .run(Default::default());
        }
    }

    #[test]
    fn static_streamify_test() {
        // Buffer shape [3, 2]:
        //   t0 t1
        //   t2 t3
        //   t4 t5
        // stride = [0, 2, 1], out_shape = [2, 3, 2]
        //
        // For output position (i, j, k), buf_idx = i*0 + j*2 + k*1 = j*2 + k.
        // Expected stream:
        //   t0, t1 S(1), t2, t3 S(1), t4, t5 S(2),
        //   t0, t1 S(1), t2, t3 S(1), t4, t5 S(3)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;

        let mut ctx = ProgramBuilder::default();

        // Six distinct tiles so we can verify the stride mapping, not just stop placement.
        let tile_vec: Vec<Tile<VT>> = (0u32..6)
            .map(|v| {
                Tile::<VT>::new(
                    ndarray::ArcArray2::from_elem((1, 1), v),
                    BYTES_PER_ELEM,
                    false,
                )
            })
            .collect();

        let buffer_arr = ndarray::ArcArray::from_vec(tile_vec.clone())
            .into_shape_with_order((3, 2))
            .unwrap()
            .into_dyn();
        let buffer = super::Buffer::new(buffer_arr, 0);

        let (in_snd, in_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(buffer)].into_iter(),
            in_snd,
        ));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(super::StaticStreamify::<SimpleEvent, _>::new(
            vec![0, 2, 1],
            vec![2, 3, 2],
            in_rcv,
            out_snd,
            DUMMY_ID,
        ));

        let expected = vec![
            Elem::Val(tile_vec[0].clone()),
            Elem::ValStop(tile_vec[1].clone(), 1),
            Elem::Val(tile_vec[2].clone()),
            Elem::ValStop(tile_vec[3].clone(), 1),
            Elem::Val(tile_vec[4].clone()),
            Elem::ValStop(tile_vec[5].clone(), 2),
            Elem::Val(tile_vec[0].clone()),
            Elem::ValStop(tile_vec[1].clone(), 1),
            Elem::Val(tile_vec[2].clone()),
            Elem::ValStop(tile_vec[3].clone(), 1),
            Elem::Val(tile_vec[4].clone()),
            Elem::ValStop(tile_vec[5].clone(), 3),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || expected.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn static_streamify_test2() {
        // Buffer shape [3, 2]:
        //   t0 t1
        //   t2 t3
        //   t4 t5
        // stride = [0, 2, 1], out_shape = [2, 3, 2]
        //
        // For output position (i, j, k), buf_idx = i*0 + j*2 + k*1 = j*2 + k.
        // Expected stream:
        //   t0, t1 S(1), t2, t3 S(1), t4, t5 S(2),
        //   t0, t1 S(1), t2, t3 S(1), t4, t5 S(3)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 4;

        let mut ctx = ProgramBuilder::default();

        // Six distinct tiles so we can verify the stride mapping, not just stop placement.
        let tile_vec: Vec<Tile<VT>> = (0u32..6)
            .map(|v| {
                Tile::<VT>::new(
                    ndarray::ArcArray2::from_elem((1, 1), v),
                    BYTES_PER_ELEM,
                    false,
                )
            })
            .collect();

        let buffer_arr = ndarray::ArcArray::from_vec(tile_vec.clone())
            .into_shape_with_order((3, 2))
            .unwrap()
            .into_dyn();
        let buffer = super::Buffer::new(buffer_arr, 0);

        let (in_snd, in_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(buffer, 1)].into_iter(),
            in_snd,
        ));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(super::StaticStreamify::<SimpleEvent, _>::new(
            vec![0, 2, 1],
            vec![2, 3, 2],
            in_rcv,
            out_snd,
            DUMMY_ID,
        ));

        let expected = vec![
            Elem::Val(tile_vec[0].clone()),
            Elem::ValStop(tile_vec[1].clone(), 1),
            Elem::Val(tile_vec[2].clone()),
            Elem::ValStop(tile_vec[3].clone(), 1),
            Elem::Val(tile_vec[4].clone()),
            Elem::ValStop(tile_vec[5].clone(), 2),
            Elem::Val(tile_vec[0].clone()),
            Elem::ValStop(tile_vec[1].clone(), 1),
            Elem::Val(tile_vec[2].clone()),
            Elem::ValStop(tile_vec[3].clone(), 1),
            Elem::Val(tile_vec[4].clone()),
            Elem::ValStop(tile_vec[5].clone(), 4),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || expected.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn round_trip_test_0d() {
        // Tiled stream shape: [2, 2, 2] => [2, |2, 2] => [2, 2, 2]
        type VT = u32;

        let mut ctx = ProgramBuilder::default();
        let bufferize_rank = 2;

        // [2,2,2]
        let input_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
        ];

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || input_tiled_stream.into_iter(),
            snd,
        ));

        let (buff_snd, buff_rcv) = ctx.bounded(1);
        ctx.add_child(Bufferize::<SimpleEvent, _>::new(
            rcv,
            buff_snd,
            bufferize_rank,
            DUMMY_ID,
        ));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(super::Streamify::<SimpleEvent, _>::new(
            vec![],
            bufferize_rank,
            buff_rcv,
            out_snd,
            DUMMY_ID,
        ));

        // [2, 2, 2]
        let output_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || output_tiled_stream.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
