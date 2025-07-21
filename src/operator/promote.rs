use crate::primitives::elem::{Elem, StopType};
use dam::context_tools::*;

#[context_macro]
pub struct PromoteOld<T: DAMType> {
    in_stream: Receiver<Elem<T>>,
    out_stream: Sender<Elem<T>>,
    promote_rank: StopType,
}

impl<T: DAMType> PromoteOld<T>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        out_stream: Sender<Elem<T>>,
        promote_rank: StopType,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            promote_rank,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType> Context for PromoteOld<T> {
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        if self.promote_rank == 0 {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(x.clone(), 1),
                                    },
                                )
                                .unwrap();
                        } else {
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
                    }
                    Elem::ValStop(x, s) => {
                        let new_stop_rank = if self.promote_rank > s { s } else { s + 1 };
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: Elem::ValStop(x.clone(), new_stop_rank),
                                },
                            )
                            .unwrap();
                    }
                },
                Err(_) => return,
            };
        }
    }
}

#[context_macro]
pub struct Promote<T: DAMType> {
    in_stream: Receiver<Elem<T>>,
    out_stream: Sender<Elem<T>>,
    in_stream_rank: StopType,
}

impl<T: DAMType> Promote<T>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        out_stream: Sender<Elem<T>>,
        in_stream_rank: StopType,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            in_stream_rank,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}
impl<T: DAMType> Context for Promote<T> {
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        let new_elem = if self.in_stream_rank == 0 {
                            match self.in_stream.peek_next(&self.time) {
                                Ok(ChannelElement { time: _, data: _ }) => Elem::Val(x),
                                Err(_closed) => Elem::ValStop(x, 1),
                            }
                        } else {
                            Elem::Val(x)
                        };
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: new_elem,
                                },
                            )
                            .unwrap();
                    }
                    Elem::ValStop(x, s) => {
                        let new_stop_rank = if self.in_stream_rank == s {
                            match self.in_stream.peek_next(&self.time) {
                                Ok(ChannelElement { time: _, data: _ }) => s,
                                Err(_closed) => s + 1,
                            }
                        } else {
                            s
                        };

                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: Elem::ValStop(x, new_stop_rank),
                                },
                            )
                            .unwrap();
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
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::primitives::elem::Elem;

    use super::{Promote, PromoteOld};

    #[test]
    fn promote_old_2d() {
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || {
                (0..4u32)
                    .map(|x| Elem::Val(x))
                    .chain(std::iter::once(Elem::ValStop(5, 1)))
                    .chain((0..4u32).map(|x| Elem::Val(x)))
                    .chain(std::iter::once(Elem::ValStop(5, 2)))
            },
            in_snd,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || {
                (0..4u32)
                    .map(|x| Elem::Val(x))
                    .chain(std::iter::once(Elem::ValStop(5, 1)))
                    .chain((0..4u32).map(|x| Elem::Val(x)))
                    .chain(std::iter::once(Elem::ValStop(5, 3)))
            },
            out_rcv,
            |x, y| x == y,
        ));
        ctx.add_child(PromoteOld::new(in_rcv, out_snd, 2));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn promote_old_0d() {
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || vec![Elem::Val(0), Elem::Val(1), Elem::ValStop(2, 1)].into_iter(),
            in_snd,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![
                    Elem::ValStop(0, 1),
                    Elem::ValStop(1, 1),
                    Elem::ValStop(2, 2),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));
        ctx.add_child(PromoteOld::new(in_rcv, out_snd, 0));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn promote_old_0d_on_0d() {
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || (0..2u32).map(|x| Elem::Val(x)),
            in_snd,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![
                    Elem::ValStop(0, 1),
                    Elem::ValStop(1, 1),
                    Elem::ValStop(2, 1),
                ]
                .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));
        ctx.add_child(PromoteOld::new(in_rcv, out_snd, 0));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn promote_2d() {
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || {
                (0..4u32)
                    .map(|x| Elem::Val(x))
                    .chain(std::iter::once(Elem::ValStop(5, 1)))
                    .chain((0..4u32).map(|x| Elem::Val(x)))
                    .chain(std::iter::once(Elem::ValStop(5, 1)))
            },
            in_snd,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || {
                (0..4u32)
                    .map(|x| Elem::Val(x))
                    .chain(std::iter::once(Elem::ValStop(5, 1)))
                    .chain((0..4u32).map(|x| Elem::Val(x)))
                    .chain(std::iter::once(Elem::ValStop(5, 2)))
            },
            out_rcv,
            |x, y| x == y,
        ));
        ctx.add_child(Promote::new(in_rcv, out_snd, 1));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn promote_0d() {
        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || vec![Elem::Val(0), Elem::Val(1), Elem::Val(2)].into_iter(),
            in_snd,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || vec![Elem::Val(0), Elem::Val(1), Elem::ValStop(2, 1)].into_iter(),
            out_rcv,
            |x, y| x == y,
        ));
        ctx.add_child(Promote::new(in_rcv, out_snd, 0));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
