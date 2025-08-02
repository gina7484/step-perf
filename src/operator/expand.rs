use crate::primitives::elem::{Elem, StopType};
use dam::context_tools::*;

#[context_macro]
pub struct ExpandRef<T: Clone, R: Clone> {
    in_stream: Receiver<Elem<T>>,
    ref_stream: Receiver<Elem<R>>,
    expand_rank: StopType,
    out_stream: Sender<Elem<T>>,
    id: u32,
}

impl<T: DAMType, R: DAMType> ExpandRef<T, R>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        ref_stream: Receiver<Elem<R>>,
        expand_rank: StopType,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            ref_stream,
            expand_rank,
            out_stream,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.ref_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType, R: DAMType> Context for ExpandRef<T, R> {
    fn run(&mut self) {
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        // This means the input stream is rank 0
                        assert_eq!(self.expand_rank, 1);
                        loop {
                            match self.ref_stream.dequeue(&self.time) {
                                Ok(ChannelElement { time: _, data }) => match data {
                                    Elem::Val(_) => {
                                        self.out_stream
                                            .enqueue(
                                                &self.time,
                                                ChannelElement {
                                                    time: self.time.tick(),
                                                    data: Elem::Val(x.clone()),
                                                },
                                            )
                                            .unwrap();
                                    }
                                    Elem::ValStop(r, s) => {
                                        panic!(
                                            "ExpandRef {}: input stream should not have any Elem::ValStop values as the input stream is rank 0",
                                            self.id
                                        );
                                    }
                                },
                                Err(_) => {
                                    self.in_stream.dequeue(&self.time).unwrap();
                                    return;
                                }
                            }
                        }
                    }
                    Elem::ValStop(x, s) => {
                        loop {
                            match self.ref_stream.dequeue(&self.time) {
                                Ok(ChannelElement { time: _, data }) => match data {
                                    Elem::Val(r) => {
                                        self.out_stream
                                            .enqueue(
                                                &self.time,
                                                ChannelElement {
                                                    time: self.time.tick(),
                                                    data: Elem::Val(x.clone()),
                                                },
                                            )
                                            .unwrap();
                                    }
                                    Elem::ValStop(r, s) => {
                                        self.out_stream
                                            .enqueue(
                                                &self.time,
                                                ChannelElement {
                                                    time: self.time.tick(),
                                                    data: Elem::ValStop(x.clone(), s),
                                                },
                                            )
                                            .unwrap();
                                        if s >= self.expand_rank {
                                            self.in_stream.dequeue(&self.time).unwrap();
                                            break; // move on to the next element in the input stream
                                        }
                                    }
                                },
                                Err(_) => {
                                    panic!(
                                        "ExpandRef {} should not reach here as it should have exited the loop on a stop token",
                                        self.id
                                    );
                                }
                            }
                        }
                    }
                },
                Err(_) => return,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, GeneratorContext, PrinterContext,
        },
    };

    use crate::primitives::elem::Elem;

    use super::ExpandRef;

    #[test]
    fn expand_0d() {
        // cargo test --package step_perf --lib -- operator::expand::tests::expand_0d --exact --show-output
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || vec![Elem::Val(1)].into_iter(),
            in_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || vec![Elem::Val(2), Elem::Val(2), Elem::Val(2)].into_iter(),
            ref_snd,
        ));

        ctx.add_child(ExpandRef::new(in_rcv, ref_rcv, 1, out_snd, 0));

        ctx.add_child(ApproxCheckerContext::new(
            || vec![Elem::Val(1), Elem::Val(1), Elem::Val(1)].into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn expand_3d() {
        // cargo test --package step_perf --lib -- operator::expand::tests::expand_3d --exact --show-output
        // [2,3,1,1] => [2,3,2,4]
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::ValStop(1, 2),
                    Elem::ValStop(2, 2),
                    Elem::ValStop(3, 3),
                    Elem::ValStop(4, 2),
                    Elem::ValStop(5, 2),
                    Elem::ValStop(6, 3),
                ]
                .into_iter()
            },
            in_snd,
        ));

        let vec1 = vec![
            Elem::Val(1),
            Elem::Val(1),
            Elem::Val(1),
            Elem::ValStop(1, 1),
            Elem::Val(1),
            Elem::Val(1),
            Elem::Val(1),
            Elem::ValStop(2, 2),
        ];
        let vec2 = vec![
            Elem::Val(1),
            Elem::Val(1),
            Elem::Val(1),
            Elem::ValStop(1, 1),
            Elem::Val(1),
            Elem::Val(1),
            Elem::Val(1),
            Elem::ValStop(2, 3),
        ];

        // // Chain multiple vectors (borrowing)
        let chained: Vec<Elem<i32>> = vec1
            .iter()
            .chain(vec1.iter())
            .chain(vec2.iter())
            .chain(vec1.iter())
            .chain(vec1.iter())
            .chain(vec2.iter())
            .cloned() // Convert &i32 to i32
            .collect();

        ctx.add_child(GeneratorContext::new(move || chained.into_iter(), ref_snd));

        ctx.add_child(ExpandRef::new(in_rcv, ref_rcv, 2, out_snd, 0));

        ctx.add_child(ApproxCheckerContext::new(
            || vec![Elem::Val(1), Elem::Val(1), Elem::Val(1)].into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
