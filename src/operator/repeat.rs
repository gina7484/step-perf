use crate::primitives::elem::{Elem, StopType};
use dam::context_tools::*;

#[context_macro]
pub struct RepeatStatic<T: Clone> {
    in_stream: Receiver<Elem<T>>,
    repeat_factor: usize,
    out_stream: Sender<Elem<T>>,
}

impl<T: DAMType> RepeatStatic<T>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        repeat_factor: usize,
        out_stream: Sender<Elem<T>>,
    ) -> Self {
        let ctx = Self {
            in_stream,
            repeat_factor,
            out_stream,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType> Context for RepeatStatic<T> {
    fn run(&mut self) {
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        for i in 0..(self.repeat_factor - 1) {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick() + i as u64,
                                        data: Elem::Val(x.clone()),
                                    },
                                )
                                .unwrap();
                        }

                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick() + (self.repeat_factor - 1) as u64,
                                    data: Elem::ValStop(x.clone(), 1),
                                },
                            )
                            .unwrap();

                        self.time.incr_cycles(self.repeat_factor as u64);

                        self.in_stream.dequeue(&self.time).unwrap();
                    }
                    Elem::ValStop(x, s) => {
                        for i in 0..(self.repeat_factor - 1) {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick() + i as u64,
                                        data: Elem::Val(x.clone()),
                                    },
                                )
                                .unwrap();
                        }

                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick() + (self.repeat_factor - 1) as u64,
                                    data: Elem::ValStop(x.clone(), s + 1),
                                },
                            )
                            .unwrap();

                        self.time.incr_cycles(self.repeat_factor as u64);

                        self.in_stream.dequeue(&self.time).unwrap();
                    }
                },
                Err(_) => return,
            };
        }
    }
}

#[context_macro]
pub struct RepeatRef<T: Clone, R: Clone> {
    in_stream: Receiver<Elem<T>>,
    ref_stream: Receiver<Elem<R>>,
    out_stream: Sender<Elem<T>>,
    rank: StopType,
    id: u32,
}

impl<T: DAMType, R: DAMType> RepeatRef<T, R>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        ref_stream: Receiver<Elem<R>>,
        out_stream: Sender<Elem<T>>,
        rank: StopType,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            ref_stream,
            out_stream,
            rank,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.ref_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType, R: DAMType> Context for RepeatRef<T, R> {
    fn run(&mut self) {
        loop {
            // 1. Dequeue from in_stream
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: in_data,
                }) => {
                    // 2. Inner loop: dequeue from ref_stream
                    loop {
                        match self.ref_stream.dequeue(&self.time) {
                            Ok(ChannelElement {
                                time: _,
                                data: ref_data,
                            }) => {
                                let ref_stop_level: StopType = match &ref_data {
                                    Elem::Val(_) => 0,
                                    Elem::ValStop(_, s) => *s,
                                };

                                if ref_stop_level < self.rank {
                                    // Sub-rank boundary: consume silently
                                    continue;
                                } else if ref_stop_level == self.rank {
                                    // At-rank: output current in_data value (stripped of stop)
                                    self.out_stream
                                        .enqueue(
                                            &self.time,
                                            ChannelElement {
                                                time: self.time.tick(),
                                                data: match &in_data {
                                                    Elem::Val(x) => Elem::Val(x.clone()),
                                                    Elem::ValStop(x, _) => Elem::Val(x.clone()),
                                                },
                                            },
                                        )
                                        .unwrap();
                                } else {
                                    // ref_stop_level > self.rank: terminal stop
                                    self.out_stream
                                        .enqueue(
                                            &self.time,
                                            ChannelElement {
                                                time: self.time.tick(),
                                                data: match &in_data {
                                                    Elem::Val(x) => {
                                                        Elem::ValStop(
                                                            x.clone(),
                                                            ref_stop_level - self.rank,
                                                        )
                                                    }
                                                    Elem::ValStop(x, in_s) => {
                                                        assert_eq!(
                                                            in_s + 1,
                                                            ref_stop_level - self.rank,
                                                            "RepeatRef {}: mismatch between input stop count {} and reference stop count {} (rank={})",
                                                            self.id, in_s + 1, ref_stop_level - self.rank, self.rank
                                                        );
                                                        Elem::ValStop(x.clone(), in_s + 1)
                                                    }
                                                },
                                            },
                                        )
                                        .unwrap();
                                    break; // Move to next element in in_stream
                                }
                            }
                            Err(_) => {
                                panic!(
                                    "RepeatRef {}: reference stream ended before input stream",
                                    self.id
                                );
                            }
                        }
                    }
                }
                Err(_) => return, // End of in_stream
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::primitives::elem::Elem;

    use super::RepeatRef;

    #[test]
    fn repeat_ref_simple() {
        // cargo test --package step_perf --lib -- operator::repeat::tests::repeat_ref_simple --exact --show-output
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // Input: [1, 2, 3]
        ctx.add_child(GeneratorContext::new(
            || vec![Elem::Val(1), Elem::Val(2), Elem::Val(3)].into_iter(),
            in_snd,
        ));

        // Reference: repeat 1 three times, repeat 2 two times, repeat 3 four times
        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::ValStop(0, 1), // 3 elements
                    Elem::Val(0),
                    Elem::ValStop(0, 1), // 2 elements
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::ValStop(0, 1), // 4 elements
                ]
                .into_iter()
            },
            ref_snd,
        ));

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 0, 0));

        // Expected output: [1, 1, 1(stop), 2, 2(stop), 3, 3, 3, 3(stop)]
        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![
                    Elem::Val(1),
                    Elem::Val(1),
                    Elem::ValStop(1, 1),
                    Elem::Val(2),
                    Elem::ValStop(2, 1),
                    Elem::Val(3),
                    Elem::Val(3),
                    Elem::Val(3),
                    Elem::ValStop(3, 1),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn repeat_ref_2d() {
        // cargo test --package step_perf --lib -- operator::repeat::tests::repeat_ref_2d --exact --show-output
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(1),
                    Elem::ValStop(2, 1),
                    Elem::Val(3),
                    Elem::ValStop(4, 2),
                ]
                .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::ValStop(0, 1), // 3 elements
                    Elem::Val(0),
                    Elem::ValStop(0, 2), // 2 elements
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::ValStop(0, 1), // 4 elements
                    Elem::Val(0),
                    Elem::ValStop(0, 3), // 4 elements
                ]
                .into_iter()
            },
            ref_snd,
        ));

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 0, 0));

        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![
                    Elem::Val(1),
                    Elem::Val(1),
                    Elem::ValStop(1, 1),
                    Elem::Val(2),
                    Elem::ValStop(2, 2),
                    Elem::Val(3),
                    Elem::Val(3),
                    Elem::Val(3),
                    Elem::ValStop(3, 1),
                    Elem::Val(4),
                    Elem::ValStop(4, 3),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn repeat_ref_rank1() {
        // cargo test --package step_perf --lib -- operator::repeat::tests::repeat_ref_rank1 --exact --show-output
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        // in_stream: 1 2 3 S1
        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(1),
                    Elem::Val(2),
                    Elem::ValStop(3, 1),
                ]
                .into_iter()
            },
            in_snd,
        ));

        // ref_stream: 9 9 S1 9 9 S2 9 9 9 S1 9 S2 9 S1 9 S3
        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(9),
                    Elem::Val(9),
                    Elem::ValStop(9, 1),
                    Elem::Val(9),
                    Elem::Val(9),
                    Elem::ValStop(9, 2),
                    Elem::Val(9),
                    Elem::Val(9),
                    Elem::Val(9),
                    Elem::ValStop(9, 1),
                    Elem::Val(9),
                    Elem::ValStop(9, 2),
                    Elem::Val(9),
                    Elem::ValStop(9, 1),
                    Elem::Val(9),
                    Elem::ValStop(9, 3),
                ]
                .into_iter()
            },
            ref_snd,
        ));

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 1, 0));

        // output: 1 1S1 2 2S1 3 3S2
        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![
                    Elem::Val(1),
                    Elem::ValStop(1, 1),
                    Elem::Val(2),
                    Elem::ValStop(2, 1),
                    Elem::Val(3),
                    Elem::ValStop(3, 2),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
