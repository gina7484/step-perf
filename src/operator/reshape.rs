use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::{select::SelectAdapter, tile::Tile};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;

#[context_macro]
pub struct Reshape<E, A: DAMType> {
    in_stream: Receiver<Elem<A>>,
    out_stream: Sender<Elem<A>>,
    chunk_size: usize,
    reshape_rank: usize,
    compute_bw: u64,
    write_back_mu: bool,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType
    > Reshape<E, A>
where 
    Elem<A>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<A>>,
        out_stream: Sender<Elem<A>>,
        chunk_size: usize,
        reshape_rank: usize,
        compute_bw: u64,
        write_back_mu: bool,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            chunk_size,
            reshape_rank,
            compute_bw,
            write_back_mu,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    fn handle_load_cycles<T: Bufferizable>(&mut self, data: &T) {
        if data.read_from_mu() {
            let load_cycle = div_ceil(data.size_in_bytes() as u64, PMU_BW);
            self.time.incr_cycles(load_cycle);
        }
    }

    fn handle_write_cycles<T: Bufferizable>(&mut self, data: &T) {
        if self.write_back_mu {
            let write_cycle = div_ceil(data.size_in_bytes() as u64, PMU_BW);
            self.time.incr_cycles(write_cycle);
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType
    > Context for Reshape<E, A>
where 
    Elem<A>: DAMType
{
    fn run(&mut self) {
        let mut passed_chunks = 0;
        loop {
            let mut in_elem = self.in_stream.peek_next(&self.time);
            match in_elem {
                Ok(ChannelElement{time: _, data: in_data}) => {
                    Elem::Val(in_data) => {
                        
                    }
                }
            }
        }
    }
}
