use std::{marker::PhantomData, sync::Arc};

use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::dam_macros::event_type;
use dam::{context_tools::*, logging::LogEvent};
use serde::{Deserialize, Serialize};

/// This is necesssary for operation patterns like matmul where the
#[context_macro]
pub struct BinaryMapAccum<E, T: DAMType, OT: DAMType> {
    in1_stream: Receiver<Elem<Tile<T>>>,
    in2_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Tile<OT>>>,
    func: Arc<dyn Fn(&Tile<T>, &Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
    rank: StopType,
    compute_bw: u64,     // FLOPs / cycle
    write_back_mu: bool, // Whether the output is written to a memory unit
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > BinaryMapAccum<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    pub fn new(
        in1_stream: Receiver<Elem<Tile<T>>>,
        in2_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<OT>>>,
        func: Arc<
            dyn Fn(&Tile<T>, &Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync,
        >, // bytes, bytes, FLOPs per cycle -> cycles
        init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
        rank: StopType,
        compute_bw: u64, // FLOPs / cycle
        write_back_mu: bool,
        id: u32,
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
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in1_stream.attach_receiver(&ctx);
        ctx.in2_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    fn process_map_accum(&mut self, data1: Tile<T>, data2: Tile<T>, accumulator: &mut Tile<OT>) {
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
        *accumulator = out_tile; // update accumulator
        let roofline_cycles = [load_cycle, comp_cycles].into_iter().max().unwrap_or(0);

        // increment cycles and dequeue inputs
        self.time.incr_cycles(roofline_cycles);

        self.in1_stream.dequeue(&self.time).unwrap();
        self.in2_stream.dequeue(&self.time).unwrap();

        // Logging
        dam::logging::log_event(&E::new(
            "BinaryMapAccum".to_string(),
            self.id,
            self.time.tick().time() - roofline_cycles,
            self.time.tick().time(),
            false,
        ))
        .unwrap();
    }

    fn process_map_accum_init(
        &mut self,
        data1: Tile<T>,
        data2: Tile<T>,
        accumulator: &mut Tile<OT>,
        is_reduction_rank: bool,
    ) -> Tile<OT> {
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
        *accumulator = (self.init_accum)(); // Initialize accumulator

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
            "BinaryMapAccum".to_string(),
            self.id,
            self.time.tick().time() - roofline_cycles,
            self.time.tick().time(),
            !is_reduction_rank,
        ))
        .unwrap();

        out_tile
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Context for BinaryMapAccum<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    fn run(&mut self) {
        let mut accumulator = (self.init_accum)();
        loop {
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
                        self.process_map_accum(data1, data2, &mut accumulator);
                    }
                    (Elem::ValStop(data1, lev1), Elem::ValStop(data2, lev2)) => {
                        if lev1 != lev2 {
                            panic!("The two input streams' shape don't match!");
                        }

                        if lev1 < self.rank {
                            self.process_map_accum(data1, data2, &mut accumulator);
                        } else if lev1 == self.rank {
                            let out_tile = self.process_map_accum_init(
                                data1,
                                data2,
                                &mut accumulator,
                                lev1 == self.rank,
                            );

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
                            let out_tile = self.process_map_accum_init(
                                data1,
                                data2,
                                &mut accumulator,
                                lev1 == self.rank,
                            );

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
