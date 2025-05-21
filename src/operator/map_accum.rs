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
                        // Load
                        let mut load_cycle: u64 = 0;
                        if data1.read_from_mu {
                            load_cycle += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
                        }
                        if data2.read_from_mu {
                            load_cycle += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
                        }

                        // Compute
                        let (comp_cycles, out_tile) = (self.func)(
                            &data1,
                            &data2,
                            &accumulator,
                            self.compute_bw,
                            self.write_back_mu,
                        );
                        accumulator = out_tile; // update accumulator

                        let roofline_cycles =
                            [load_cycle, comp_cycles].into_iter().max().unwrap_or(0);

                        // increment cycles and dequeue inputs
                        self.time.incr_cycles(roofline_cycles);

                        self.in1_stream.dequeue(&self.time).unwrap();
                        self.in2_stream.dequeue(&self.time).unwrap();

                        // Logging
                        dam::logging::log_event(&E::new(
                            self.time.tick().time() - roofline_cycles,
                            self.time.tick().time(),
                            false,
                        ))
                        .unwrap();
                    }
                    (Elem::ValStop(data1, lev1), Elem::ValStop(data2, lev2)) => {
                        if lev1 != lev2 {
                            panic!("The two input streams' shape don't match!");
                        }

                        if lev1 < self.rank {
                            // Load
                            let mut load_cycle: u64 = 0;
                            if data1.read_from_mu {
                                load_cycle += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
                            }
                            if data2.read_from_mu {
                                load_cycle += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
                            }

                            // Compute
                            let (comp_cycles, out_tile) = (self.func)(
                                &data1,
                                &data2,
                                &accumulator,
                                self.compute_bw,
                                self.write_back_mu,
                            );
                            accumulator = out_tile; // update accumulator

                            let roofline_cycles =
                                [load_cycle, comp_cycles].into_iter().max().unwrap_or(0);

                            // increment cycles and dequeue inputs
                            self.time.incr_cycles(roofline_cycles);

                            self.in1_stream.dequeue(&self.time).unwrap();
                            self.in2_stream.dequeue(&self.time).unwrap();

                            // Logging
                            dam::logging::log_event(&E::new(
                                self.time.tick().time() - roofline_cycles,
                                self.time.tick().time(),
                                false,
                            ))
                            .unwrap();
                        } else if lev1 == self.rank {
                            // Load
                            let mut load_cycle: u64 = 0;
                            if data1.read_from_mu {
                                load_cycle += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
                            }
                            if data2.read_from_mu {
                                load_cycle += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
                            }

                            // Compute
                            let (comp_cycles, out_tile) = (self.func)(
                                &data1,
                                &data2,
                                &accumulator,
                                self.compute_bw,
                                self.write_back_mu,
                            );
                            accumulator = (self.init_accum)(); // Initialize accumulator

                            // Store
                            let store_cycles = if self.write_back_mu {
                                div_ceil(accumulator.size_in_bytes() as u64, PMU_BW)
                            } else {
                                0_u64
                            };

                            let roofline_cycles = [load_cycle, comp_cycles, store_cycles]
                                .into_iter()
                                .max()
                                .unwrap_or(0);

                            // increment cycles and dequeue inputs
                            self.time.incr_cycles(roofline_cycles);

                            self.in1_stream.dequeue(&self.time).unwrap();
                            self.in2_stream.dequeue(&self.time).unwrap();

                            // Logging
                            dam::logging::log_event(&E::new(
                                self.time.tick().time() - roofline_cycles,
                                self.time.tick().time(),
                                false,
                            ))
                            .unwrap();

                            // Enqueue
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(out_tile),
                                    },
                                )
                                .unwrap();
                        } else {
                            // lev1 > self.rank
                            // Load
                            let mut load_cycle: u64 = 0;
                            if data1.read_from_mu {
                                load_cycle += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
                            }
                            if data2.read_from_mu {
                                load_cycle += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
                            }

                            // Compute
                            let (comp_cycles, out_tile) = (self.func)(
                                &data1,
                                &data2,
                                &accumulator,
                                self.compute_bw,
                                self.write_back_mu,
                            );
                            accumulator = (self.init_accum)(); // Initialize accumulator

                            // Store
                            let store_cycles = if self.write_back_mu {
                                div_ceil(accumulator.size_in_bytes() as u64, PMU_BW)
                            } else {
                                0_u64
                            };

                            let roofline_cycles = [load_cycle, comp_cycles, store_cycles]
                                .into_iter()
                                .max()
                                .unwrap_or(0);

                            // increment cycles and dequeue inputs
                            self.time.incr_cycles(roofline_cycles);

                            self.in1_stream.dequeue(&self.time).unwrap();
                            self.in2_stream.dequeue(&self.time).unwrap();

                            // Logging
                            dam::logging::log_event(&E::new(
                                self.time.tick().time() - roofline_cycles,
                                self.time.tick().time(),
                                true,
                            ))
                            .unwrap();

                            // Enqueue
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(out_tile, lev1 - self.rank),
                                    },
                                )
                                .unwrap();
                        }
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
