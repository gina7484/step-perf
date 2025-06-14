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
