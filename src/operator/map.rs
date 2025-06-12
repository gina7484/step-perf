use std::{marker::PhantomData, sync::Arc};

use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

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
pub struct BinaryMap<E, A: DAMType, B: DAMType> {
    in1_stream: Receiver<Elem<Tile<A>>>,
    in2_stream: Receiver<Elem<Tile<A>>>,
    out_stream: Sender<Elem<Tile<B>>>,
    func: Arc<dyn Fn(&Tile<A>, &Tile<A>, u64, bool) -> (u64, Tile<B>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    compute_bw: u64,     // FLOPs / cycle
    write_back_mu: bool, // Whether the output is written to a memory unit
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType,
        B: DAMType,
    > BinaryMap<E, A, B>
where
    Elem<Tile<A>>: DAMType,
    Elem<Tile<B>>: DAMType,
{
    pub fn new(
        in1_stream: Receiver<Elem<Tile<A>>>,
        in2_stream: Receiver<Elem<Tile<A>>>,
        out_stream: Sender<Elem<Tile<B>>>,
        func: Arc<dyn Fn(&Tile<A>, &Tile<A>, u64, bool) -> (u64, Tile<B>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
        compute_bw: u64, // FLOPs / cycle
        write_back_mu: bool,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in1_stream,
            in2_stream,
            out_stream,
            func,
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
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType,
        B: DAMType,
    > Context for BinaryMap<E, A, B>
where
    Elem<Tile<A>>: DAMType,
    Elem<Tile<B>>: DAMType,
{
    fn run(&mut self) {
        loop {
            let in1 = self.in1_stream.peek_next(&self.time);
            let in2 = self.in2_stream.peek_next(&self.time);

            let (tile1, tile2, stop_lev) = match (in1, in2) {
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
                    (Elem::Val(data1), Elem::Val(data2)) => (data1, data2, None),
                    (Elem::ValStop(data1, lev1), Elem::ValStop(data2, lev2)) => {
                        if lev1 != lev2 {
                            panic!("The two input streams' shape don't match!");
                        }
                        (data1, data2, Some(lev1))
                    }
                    (_, _) => panic!("The two input streams' shape don't match!"),
                },
                (Ok(_), Err(_)) => panic!("One stream closed earlier"),
                (Err(_), Ok(_)) => panic!("One stream closed earlier"),
                (Err(_), Err(_)) => {
                    return;
                }
            };

            let start_time = self.time.tick().time();

            let mut load_cycle: u64 = 0;
            if tile1.read_from_mu {
                load_cycle += div_ceil(tile1.size_in_bytes() as u64, PMU_BW);
            }
            if tile2.read_from_mu {
                load_cycle += div_ceil(tile2.size_in_bytes() as u64, PMU_BW);
            }
            let (comp_cycles, out_tile) =
                (self.func)(&tile1, &tile2, self.compute_bw, self.write_back_mu);
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

            let data = match stop_lev {
                Some(level) => Elem::ValStop(out_tile, level),
                None => Elem::Val(out_tile),
            };
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: data,
                    },
                )
                .unwrap();

            dam::logging::log_event(&E::new(
                self.id,
                start_time,
                self.time.tick().time(),
                stop_lev != None,
            ))
            .unwrap();

            self.in1_stream.dequeue(&self.time).unwrap();
            self.in2_stream.dequeue(&self.time).unwrap();
        }
    }
}

pub struct UnaryMapConfig {
    pub compute_bw: u64,     // FLOPs / cycle
    pub write_back_mu: bool, // Whether the output is written to a memory unit
}

#[context_macro]
pub struct UnaryMap<E, T: DAMType, OT: DAMType> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Tile<OT>>>,
    func: Arc<dyn Fn(&Tile<T>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, FLOPs per cycle -> cycles
    config: UnaryMapConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > UnaryMap<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<OT>>>,
        func: Arc<dyn Fn(&Tile<T>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, FLOPs per cycle -> cycles
        config: UnaryMapConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            func,
            config,
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
        T: DAMType,
        OT: DAMType,
    > Context for UnaryMap<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    fn run(&mut self) {
        loop {
            let in_elem = self.in_stream.peek_next(&self.time);
            let (in_tile, stop_lev) = match in_elem {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => (data, None),
                    Elem::ValStop(data, lev) => (data, Some(lev)),
                },
                Err(_) => {
                    return; // Stream closed
                }
            };

            let start_time = self.time.tick().time();
            let load_cycles = if in_tile.read_from_mu {
                div_ceil(in_tile.size_in_bytes() as u64, PMU_BW)
            } else {
                0
            };

            let (comp_cycles, out_tile) = (self.func)(&in_tile, self.config.compute_bw, self.config.write_back_mu);
            let store_cycles = if self.config.write_back_mu {
                div_ceil(out_tile.size_in_bytes() as u64, PMU_BW)
            } else {
                0
            };

            let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
                .into_iter()
                .max()
                .unwrap_or(0);
            self.time.incr_cycles(roofline_cycles);
            let data = match stop_lev {
                Some(level) => Elem::ValStop(out_tile, level),
                None => Elem::Val(out_tile),
            };
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: data,
                    },
                )
                .unwrap();
            dam::logging::log_event(&E::new(
                self.id,
                start_time,
                self.time.tick().time(),
                stop_lev != None,
            )).unwrap();
            self.in_stream.dequeue(&self.time).unwrap();
        }
    }
}