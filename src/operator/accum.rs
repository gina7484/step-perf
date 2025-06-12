
use std::{marker::PhantomData, sync::Arc};

use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

#[context_macro]
pub struct Accum<E, T: DAMType, OT: DAMType> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Tile<OT>>>,
    func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
    rank: StopType,
    compute_bw: u64,     // FLOPs / cycle
    write_back_mu: bool, // Whether the output is written to a memory unit
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Accum<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<OT>>>,
        func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
        init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
        rank: StopType,
        compute_bw: u64, // FLOPs / cycle
        write_back_mu: bool,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            func,
            init_accum,
            rank,
            compute_bw,
            write_back_mu,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    fn process_accum(&mut self, data: Tile<T>, accumulator: &mut Tile<OT>) {
        let load_cycles = if data.read_from_mu {
            div_ceil(data.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let (comp_cycles, out_tile) = (self.func)(
            &data,
            &accumulator,
            self.compute_bw,
            self.write_back_mu,
        );
        *accumulator = out_tile;

        let roofline_cycles = [load_cycles, comp_cycles].into_iter().max().unwrap_or(0);

        // increment cycles and dequeue inputs
        self.time.incr_cycles(roofline_cycles);

        self.in_stream.dequeue(&self.time).unwrap();

    }

    fn process_accum_init(&mut self, data: Tile<T>, accumulator: &mut Tile<OT>) -> Tile<OT> {
        let load_cycles = if data.read_from_mu {
            div_ceil(data.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let (comp_cycles, out_tile) = (self.func)(
            &data,
            &accumulator,
            self.compute_bw,
            self.write_back_mu,
        );
        *accumulator = (self.init_accum)();

        let store_cycles = if self.write_back_mu {
            div_ceil(accumulator.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
            .into_iter()
            .max()
            .unwrap_or(0);

        self.time.incr_cycles(roofline_cycles);
        self.in_stream.dequeue(&self.time).unwrap();

        out_tile
    }
    
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Context for Accum<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    fn run(&mut self) {
        let mut accumulator = (self.init_accum)();
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {time: _, data}) => match data {
                    Elem::Val(x) => {
                        self.process_accum(x, &mut accumulator);
                    }
                    Elem::ValStop(x, level) => {
                        if level < self.rank {
                            self.process_accum(x, &mut accumulator);
                        } else if level == self.rank {
                            let out_tile = self.process_accum_init(x, &mut accumulator);
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
                            let out_tile = self.process_accum_init(x, &mut accumulator);
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(out_tile, level - self.rank),
                                    },
                                )
                                .unwrap();
                        }
                    }
                }
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        operator::accum::Accum,
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext, PrinterContext};
    use ndarray::Array2;

    fn tolerance_fn(a: &Elem<Tile<i32>>, b: &Elem<Tile<i32>>) -> bool {
        match (a, b) {
            (Elem::Val(a_tile), Elem::Val(b_tile)) => a_tile == b_tile,
            (Elem::ValStop(a_tile, a_level), Elem::ValStop(b_tile, b_level)) => {
                a_tile == b_tile && a_level == b_level
            }
            _ => false,
        }
    }

}