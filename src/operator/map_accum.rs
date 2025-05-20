use std::{marker::PhantomData, sync::Arc};

use crate::memory::{data::Tile, events::LoggableEventSimple, PMU_BW};
use crate::primitives::elem::{Elem, StopType};
use crate::utils::calculation::div_ceil;
use dam::dam_macros::event_type;
use dam::{context_tools::*, logging::LogEvent};
use serde::{Deserialize, Serialize};

/// This is necesssary for operation patterns like matmul where the
#[context_macro]
pub struct BinaryMapAccum<E> {
    in1_stream: Receiver<Elem<Tile>>,
    in2_stream: Receiver<Elem<Tile>>,
    out_stream: Sender<Elem<Tile>>,
    func: Arc<dyn Fn(&Tile, &Tile, &Tile, u64, bool) -> (u64, Tile) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    init_accum: Arc<dyn Fn() -> Tile + Sync + Send>,
    rank: StopType,
    compute_bw: u64,     // FLOPs / cycle
    write_back_mu: bool, // Whether the output is written to a memory unit
    _phantom: PhantomData<E>,
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> BinaryMapAccum<E> {
    pub fn new(
        in1_stream: Receiver<Elem<Tile>>,
        in2_stream: Receiver<Elem<Tile>>,
        out_stream: Sender<Elem<Tile>>,
        func: Arc<dyn Fn(&Tile, &Tile, &Tile, u64, bool) -> (u64, Tile) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
        init_accum: Arc<dyn Fn() -> Tile + Sync + Send>,
        rank: StopType,
        compute_bw: u64, // FLOPs / cycle
        write_back_mu: bool,
    ) -> Self {
        let ctx = Self {
            in1_stream,
            in2_stream,
            out_stream,
            func,
            init_accum,
            rank,
            compute_bw,
            write_back_mu,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in1_stream.attach_receiver(&ctx);
        ctx.in2_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Context
    for BinaryMapAccum<E>
{
    fn run(&mut self) {
        loop {
            let mut accumulator: Tile = (self.init_accum)();

            let in1 = self.in1_stream.peek_next(&self.time);
            let in2 = self.in2_stream.peek_next(&self.time);

            match (in1, in2) {
                (
                    Ok(ChannelElement {
                        time: _,
                        data: data1_enum,
                    }),
                    Ok(ChannelElement {
                        time: _,
                        data: data2_enum,
                    }),
                ) => match (data1_enum, data2_enum) {
                    (Elem::Val(data1), Elem::Val(data2)) => {
                        // compute the cycles based on a roofline model
                        let mut load_cycle: u64 = 0;
                        if data1.read_from_mu {
                            load_cycle += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
                        }
                        if data2.read_from_mu {
                            load_cycle += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
                        }
                        let (comp_cycles, out_tile) = (self.func)(
                            &data1,
                            &data2,
                            &accumulator,
                            self.compute_bw,
                            self.write_back_mu,
                        );

                        let roofline_cycles =
                            [load_cycle, comp_cycles].into_iter().max().unwrap_or(0);

                        // increment cycles and dequeue inputs
                        self.time.incr_cycles(roofline_cycles);

                        self.in1_stream.dequeue(&self.time).unwrap();
                        self.in2_stream.dequeue(&self.time).unwrap();

                        let curr_time = self.time.tick();

                        // update accumulator
                        accumulator = out_tile;

                        // log the time for accumulation
                        let time_block_start_ns = curr_time.time() - roofline_cycles;

                        dam::logging::log_event(&E::new(
                            time_block_start_ns,
                            curr_time.time(),
                            false,
                        ))
                        .unwrap();
                    }
                    (Elem::Stop(lev1), Elem::Stop(lev2)) => {
                        if lev1 != lev2 {
                            panic!("The two input streams' shape don't match!");
                        }

                        if lev1 == self.rank {
                            // If you see the accumulation rank:
                            // - enqueue the accumulator

                            let store_cycles = if self.write_back_mu {
                                div_ceil(accumulator.size_in_bytes() as u64, PMU_BW)
                            } else {
                                0_u64
                            };

                            if store_cycles > 0 {
                                self.time.incr_cycles(store_cycles);

                                let curr_time = self.time.tick();
                                // update accumulator
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: curr_time,
                                            data: Elem::Val(accumulator),
                                        },
                                    )
                                    .unwrap();

                                dam::logging::log_event(&E::new(
                                    curr_time.time() - store_cycles,
                                    curr_time.time(),
                                    false,
                                ))
                                .unwrap();
                            } else {
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::Val(accumulator),
                                        },
                                    )
                                    .unwrap();
                            }

                            // initialize the accumulator
                            accumulator = (self.init_accum)();
                        } else if lev1 > self.rank {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick() + 1,
                                        data: Elem::Stop(lev1 - self.rank),
                                    },
                                )
                                .unwrap();
                        }
                        // Record the cycle used to read the stop token
                        self.time.incr_cycles(1);
                        dam::logging::log_event(&E::new(
                            self.time.tick().time() - 1,
                            self.time.tick().time(),
                            true,
                        ))
                        .unwrap();

                        self.in1_stream.dequeue(&self.time).unwrap();
                        self.in2_stream.dequeue(&self.time).unwrap();
                    }
                    (_, _) => panic!("The two input streams' shape don't match!"),
                },
                (Ok(_), Err(_)) => panic!("One stream closed earlier"),
                (Err(_), Ok(_)) => panic!("One stream closed earlier"),
                (Err(_), Err(_)) => return,
            }
        }
    }
}
