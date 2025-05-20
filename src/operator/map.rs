use std::{marker::PhantomData, sync::Arc};

use crate::memory::{data::Tile, events::LoggableEventSimple, PMU_BW};
use crate::primitives::elem::Elem;
use crate::utils::calculation::div_ceil;
use dam::dam_macros::event_type;
use dam::{context_tools::*, logging::LogEvent};
use serde::{Deserialize, Serialize};

/// The function will be a binary function that returns the latency in cycles
/// based on the size of the operands and allocated bandwidth.
///
/// Assumptions used during the roofline analysis:
/// - Each operand is stored in separate PMUs
/// - No further on-chip tiling.
/// - When reading from / writing to a PMU, we use the full bandwidth. This gives an optimistic (upper) bound.
///   We can add flags to use a statically divided bandwidth to consider contention between read and write.
///   However, as this uses a statically divided bandwidth, there are limits in terms of how accurate we can model contention.
///   To accurately model on-chip memory accesses, one has to create a similar context as ramulator context for PMUs.
#[context_macro]
pub struct BinaryMap<E> {
    in1_stream: Receiver<Elem<Tile>>,
    in2_stream: Receiver<Elem<Tile>>,
    out_stream: Sender<Elem<Tile>>,
    func: Arc<dyn Fn(&Tile, &Tile, u64, bool) -> (u64, Tile) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    compute_bw: u64,                                                         // FLOPs / cycle
    write_back_mu: bool, // Whether the output is written to a memory unit
    _phantom: PhantomData<E>,
}
/* BinaryMap<E, A, B>
        in1_stream: Receiver<Elem<Tile<A>>>,
        in2_stream: Receiver<Elem<Tile<A>>>,
        out_stream: Sender<Elem<Tile<B>>>,
        func: Arc<dyn Fn(&Tile<A>, &Tile<A>, u64, bool) -> (u64, Tile<B>) + Send + Sync>,
*/

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> BinaryMap<E> {
    pub fn new(
        in1_stream: Receiver<Elem<Tile>>,
        in2_stream: Receiver<Elem<Tile>>,
        out_stream: Sender<Elem<Tile>>,
        func: Arc<dyn Fn(&Tile, &Tile, u64, bool) -> (u64, Tile) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
        compute_bw: u64,                                                         // FLOPs / cycle
        write_back_mu: bool,
    ) -> Self {
        let ctx = Self {
            in1_stream,
            in2_stream,
            out_stream,
            func,
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
    for BinaryMap<E>
{
    fn run(&mut self) {
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
                        let mut load_cycle: u64 = 0;
                        if data1.read_from_mu {
                            load_cycle += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
                        }
                        if data2.read_from_mu {
                            load_cycle += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
                        }
                        let (comp_cycles, out_tile) =
                            (self.func)(&data1, &data2, self.compute_bw, self.write_back_mu);
                        let store_cycles = if self.write_back_mu {
                            div_ceil(out_tile.size_in_bytes() as u64, PMU_BW)
                        } else {
                            0_u64
                        };
                        let roofline_cycles = [load_cycle, comp_cycles, store_cycles]
                            .into_iter()
                            .max()
                            .unwrap_or(0);

                        self.time.incr_cycles(roofline_cycles);

                        self.in1_stream.dequeue(&self.time).unwrap();
                        self.in2_stream.dequeue(&self.time).unwrap();

                        let curr_time = self.time.tick();
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: curr_time,
                                    data: Elem::Val(out_tile),
                                },
                            )
                            .unwrap();

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

                        let curr_time = self.time.tick();
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: curr_time + 1,
                                    data: Elem::Stop(lev1),
                                },
                            )
                            .unwrap();

                        // Also log the cycle spent on stop tokens to quantify its overhead
                        dam::logging::log_event(&E::new(
                            curr_time.time(),
                            curr_time.time() + 1,
                            true,
                        ))
                        .unwrap();
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
