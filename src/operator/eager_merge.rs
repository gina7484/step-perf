use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::{select::SelectAdapter, tile::Tile};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use core::panic;
use dam::channel::PeekResult;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;

#[context_macro]
pub struct EagerMerge<A: DAMType, SELT: DAMType> {
    in_streams: Vec<Receiver<Elem<A>>>,
    sel_stream: Sender<Elem<SELT>>,
    out_stream: Sender<Elem<A>>,
    input_rank: StopType,
    id: u32,
}

impl<A: DAMType, SELT: DAMType + SelectAdapter + Bufferizable> EagerMerge<A, SELT>
where
    Elem<A>: DAMType,
    Elem<SELT>: DAMType,
{
    pub fn new(
        in_streams: Vec<Receiver<Elem<A>>>,
        sel_stream: Sender<Elem<SELT>>,
        out_stream: Sender<Elem<A>>,
        input_rank: StopType,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_streams,
            sel_stream,
            out_stream,
            input_rank,
            id,
            context_info: Default::default(),
        };

        ctx.in_streams.iter().for_each(|s| s.attach_receiver(&ctx));
        ctx.sel_stream.attach_sender(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    fn get_earliest_input_idx(&mut self) -> Option<usize> {
        let mut earliest_time = u64::MAX;
        let mut earliest_input_idx = None;
        let mut peeked_something = vec![false; self.in_streams.len()];
        let mut closed_streams = vec![false; self.in_streams.len()];

        loop {
            for (i, stream) in self.in_streams.iter().enumerate() {
                if peeked_something[i] {
                    continue;
                }

                match stream.peek() {
                    PeekResult::Something(elem) => {
                        peeked_something[i] = true;
                        if earliest_input_idx.is_none() {
                            earliest_input_idx = Some(i);
                            earliest_time = elem.time.time();
                        } else if elem.time.time() < earliest_time {
                            earliest_input_idx = Some(i);
                            earliest_time = elem.time.time();
                        }
                    }
                    PeekResult::Nothing(_) => continue,
                    PeekResult::Closed => {
                        peeked_something[i] = true;
                        closed_streams[i] = true;
                        continue;
                    }
                }
            }

            match earliest_input_idx {
                Some(idx) => {
                    if peeked_something.contains(&false) && self.time.tick().time() < earliest_time
                    {
                        // As we peeked all the inputs, we have an guarantee that there
                        // will be no more elements that arrive before the current time.
                        // However, this doesn't mean that there will be no data arriving
                        // between the current time and the peeked element's time in the
                        // input streams that returned nothing in the current cycle.

                        // Therefore, we increment the time and repeat peeking the input
                        // streams that we didn't find a peek result yet.
                        self.time.incr_cycles(1);
                        continue;
                    }

                    // As we peeked all the inputs, we have an guarantee that there
                    // will be no more elements that arrive before the current time.

                    // If all the input streams returned something when peeked, then we
                    // can guarantee that the current earliest_input_idx has the earliest input.

                    // Even if not all the input streams returned something,
                    // if the earliest_time is not a future timestamp,
                    // this means that the input streams that returned nothing don't have
                    // any data arriving before the current time.
                    // Therefore, we can conclude that this is the earliest time the input
                    // is available
                    return Some(idx);
                }
                None => {
                    if !closed_streams.contains(&false) {
                        return None;
                    }
                    self.time.incr_cycles(1);
                    continue;
                }
            }
        }
    }
}

impl<A: DAMType, SELT: DAMType + SelectAdapter + Bufferizable> Context for EagerMerge<A, SELT>
where
    Elem<A>: DAMType,
    Elem<SELT>: DAMType,
{
    fn run(&mut self) {
        loop {
            let earliest_input_idx = self.get_earliest_input_idx();

            if earliest_input_idx.is_none() {
                return;
            }

            let earliest_input_idx = earliest_input_idx.unwrap();

            self.sel_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: Elem::Val(SELT::from_sel_vec(
                            vec![earliest_input_idx],
                            self.in_streams.len(),
                            false,
                        )),
                    },
                )
                .unwrap();

            loop {
                match self.in_streams[earliest_input_idx].dequeue(&self.time) {
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
                            if self.input_rank == 0 {
                                break;
                            }
                        }
                        Elem::ValStop(x, s) => {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(x.clone(), s),
                                    },
                                )
                                .unwrap();
                            if s == self.input_rank {
                                break;
                            }
                            if s > self.input_rank {
                                panic!("Found a stop token in the input stream that has a higher rank than the given input rank");
                            }
                        }
                    },
                    Err(_) => return,
                }
                self.time.incr_cycles(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::{
        channel::{ChannelElement, Sender},
        context::Context,
        dam_macros::context_macro,
        simulation::ProgramBuilder,
        types::DAMType,
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, ConsumerContext, FunctionContext,
            GeneratorContext, PrinterContext,
        },
    };

    use crate::{
        operator::partition::{FlatPartition, FlatPartitionConfig},
        primitives::{elem::Elem, select::MultiHotN, tile::Tile},
        utils::events::SimpleEvent,
    };

    use super::EagerMerge;

    #[context_macro]
    pub struct SenderContext<A: DAMType> {
        pub out_stream: Sender<Elem<Tile<A>>>,
        pub num_matrices: usize,
        pub offset: u64,
        pub bytes_per_element: usize, // separate the two contexts
    }

    impl<A: DAMType> SenderContext<A>
    where
        Elem<Tile<A>>: DAMType,
    {
        pub fn new(
            out_stream: Sender<Elem<Tile<A>>>,
            num_matrices: usize,
            offset: u64,
            bytes_per_element: usize,
        ) -> Self {
            let ctx = Self {
                out_stream,
                num_matrices,
                offset,
                bytes_per_element,
                context_info: Default::default(),
            };

            ctx.out_stream.attach_sender(&ctx);

            ctx
        }
    }

    impl<A: DAMType> Context for SenderContext<A>
    where
        Elem<Tile<A>>: DAMType,
    {
        fn run(&mut self) {
            self.time.incr_cycles(self.offset);
            for _i in 0..self.num_matrices {
                let tile = Tile::<A>::new_blank(vec![2, 2], self.bytes_per_element, false);
                for j in 0..4 {
                    let elem = if j % 4 == 3 {
                        Elem::ValStop(tile.clone(), 2)
                    } else if j % 2 == 1 {
                        Elem::ValStop(tile.clone(), 1)
                    } else {
                        Elem::Val(tile.clone())
                    };
                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick() + j as u64,
                                data: elem,
                            },
                        )
                        .unwrap();
                }

                self.time.incr_cycles(self.offset);
            }
        }
    }

    #[test]
    fn eager_merge_simple_2d() {
        // cargo test --package step_perf --lib -- operator::eager_merge::tests::eager_merge_round_trip_2d --exact --show-output
        type VT = u32;
        const DUMMY_ID: u32 = 0;
        let mut ctx = ProgramBuilder::default();

        let (snd0_snd, snd0_rcv) = ctx.unbounded();

        ctx.add_child(SenderContext::<VT>::new(snd0_snd, 3, 12, 2));

        let (snd1_snd, snd1_rcv) = ctx.unbounded();

        ctx.add_child(SenderContext::<VT>::new(snd1_snd, 1, 14, 4));

        let (merger_snd, merger_rcv) = ctx.unbounded();
        let (sel_snd, sel_rcv) = ctx.unbounded();

        // should be merged in the 0 1 0 0 order
        ctx.add_child(EagerMerge::<Tile<VT>, MultiHotN>::new(
            vec![snd0_rcv, snd1_rcv],
            sel_snd,
            merger_snd,
            2,
            DUMMY_ID,
        ));

        ctx.add_child(PrinterContext::new(merger_rcv));
        ctx.add_child(ConsumerContext::new(sel_rcv));

        // Initialize and run the context
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn eager_merge_round_trip_2d() {
        // cargo test --package step_perf --lib -- operator::eager_merge::tests::eager_merge_round_trip_2d --exact --show-output
        type VT = u32;
        const DUMMY_ID: u32 = 0;
        let mut ctx = ProgramBuilder::default();

        let (snd0_snd, snd0_rcv) = ctx.unbounded();

        ctx.add_child(SenderContext::<VT>::new(snd0_snd, 3, 12, 2));

        let (snd1_snd, snd1_rcv) = ctx.unbounded();

        ctx.add_child(SenderContext::<VT>::new(snd1_snd, 1, 14, 4));
        // should be merged in the 0 1 0 0 order

        let (merger_snd, merger_rcv) = ctx.unbounded();
        let (sel_snd, sel_rcv) = ctx.unbounded();

        ctx.add_child(EagerMerge::<Tile<VT>, MultiHotN>::new(
            vec![snd0_rcv, snd1_rcv],
            sel_snd,
            merger_snd,
            2,
            DUMMY_ID,
        ));

        let (out1_snd, out1_rcv) = ctx.unbounded();
        let (out2_snd, out2_rcv) = ctx.unbounded();

        ctx.add_child(FlatPartition::<SimpleEvent, _, _>::new(
            merger_rcv,
            sel_rcv,
            vec![out1_snd, out2_snd],
            2,
            FlatPartitionConfig {
                switch_cycles: vec![1, 1],
                write_back_mu: false,
            },
            DUMMY_ID,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            || {
                let tile0 = Tile::<VT>::new_blank(vec![2, 2], 2, false);

                let from_sender0 = vec![
                    Elem::Val(tile0.clone()),
                    Elem::ValStop(tile0.clone(), 1),
                    Elem::Val(tile0.clone()),
                    Elem::ValStop(tile0.clone(), 2),
                ];

                [
                    from_sender0.as_slice(),
                    from_sender0.as_slice(),
                    from_sender0.as_slice(),
                ]
                .concat()
                .into_iter()
            },
            out1_rcv,
            |x, y| x == y,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            || {
                let tile1 = Tile::<VT>::new_blank(vec![2, 2], 4, false);

                vec![
                    Elem::Val(tile1.clone()),
                    Elem::ValStop(tile1.clone(), 1),
                    Elem::Val(tile1.clone()),
                    Elem::ValStop(tile1.clone(), 2),
                ]
                .into_iter()
            },
            out2_rcv,
            |x, y| x == y,
        ));
        // Initialize and run the context
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
