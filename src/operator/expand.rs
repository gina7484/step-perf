use crate::primitives::elem::{Elem, StopType};
use crate::utils::onchip::{self, OnChipBytes, OnChipCounter};
use crate::utils::request_profile::{ProfiledReceiver, ProfiledSender};
use dam::context_tools::*;

#[context_macro]
pub struct ExpandStatic<T: Clone> {
    in_stream: ProfiledReceiver<Elem<T>>,
    // Products of replacement dimensions, from innermost to outermost.
    stop_periods: Vec<usize>,
    out_stream: ProfiledSender<Elem<T>>,
    onchip: OnChipCounter,
}

impl<T: DAMType> ExpandStatic<T>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        expanded_dims: Vec<usize>,
        out_stream: Sender<Elem<T>>,
    ) -> Self {
        assert!(
            !expanded_dims.is_empty(),
            "ExpandStatic needs at least one dimension"
        );
        let mut period = 1usize;
        let stop_periods = expanded_dims
            .iter()
            .rev()
            .map(|&dim| {
                assert!(dim > 0, "ExpandStatic dimensions must be positive");
                period = period
                    .checked_mul(dim)
                    .expect("ExpandStatic count overflow");
                period
            })
            .collect();
        let ctx = Self {
            in_stream: in_stream.into(),
            stop_periods,
            out_stream: out_stream.into(),
            onchip: onchip::counter(),
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType + OnChipBytes> Context for ExpandStatic<T> {
    fn run(&mut self) {
        let count = *self.stop_periods.last().unwrap();
        let rank = self.stop_periods.len() as StopType;
        while let Ok(ChannelElement { data, .. }) = self.in_stream.peek_next(&self.time) {
            let (value, input_stop) = match data {
                Elem::ValStop(value, stop) if stop >= rank => (value, stop),
                _ => panic!("ExpandStatic input must close all expanded unit dimensions"),
            };
            self.onchip.observe(value.onchip_bytes());
            for index in 1..=count {
                let stop = if index == count {
                    // These axes already exist, so keep the input's outer stops.
                    input_stop
                } else {
                    self.stop_periods
                        .iter()
                        .take_while(|&&period| index % period == 0)
                        .count() as StopType
                };
                let data = if stop == 0 {
                    Elem::Val(value.clone())
                } else {
                    Elem::ValStop(value.clone(), stop)
                };
                self.out_stream
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick(),
                            data,
                        },
                    )
                    .unwrap();
                self.time.incr_cycles(1);
            }
            self.in_stream.dequeue(&self.time).unwrap();
        }
    }
}

#[context_macro]
pub struct ExpandRef<T: Clone, R: Clone> {
    in_stream: ProfiledReceiver<Elem<T>>,
    ref_stream: ProfiledReceiver<Elem<R>>,
    expand_rank: StopType,
    out_stream: ProfiledSender<Elem<T>>,
    id: u32,
    onchip: OnChipCounter,
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
        assert!(expand_rank > 0, "ExpandRef expand_rank must be positive");
        let ctx = Self {
            in_stream: in_stream.into(),
            ref_stream: ref_stream.into(),
            expand_rank,
            out_stream: out_stream.into(),
            id,
            onchip: onchip::counter(),
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.ref_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType + OnChipBytes, R: DAMType> Context for ExpandRef<T, R> {
    fn run(&mut self) {
        let mut saw_reference = false;
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        self.onchip.observe(x.onchip_bytes());
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
                        self.onchip.observe(x.onchip_bytes());
                        loop {
                            match self.ref_stream.dequeue(&self.time) {
                                Ok(ChannelElement { time: _, data }) => match data {
                                    Elem::Val(r) => {
                                        saw_reference = true;
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
                                        saw_reference = true;
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
                                Err(_) if !saw_reference => {
                                    // An entirely empty reference means zero expansions.
                                    // Drain input so its producer can finish sending.
                                    while self.in_stream.dequeue(&self.time).is_ok() {}
                                    return;
                                }
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
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };

    use crate::primitives::elem::Elem;

    use super::{ExpandRef, ExpandStatic};

    #[test]
    fn expand_static_preserves_outer_stops_and_inner_unit_axes() {
        // [2,1,1] -> [2,3,1], including the case where every axis stays unit.
        for count in [1, 3] {
            let mut ctx = ProgramBuilder::default();
            let (input_sender, input) = ctx.bounded(1);
            let (output, output_receiver) = ctx.bounded(1);
            ctx.add_child(GeneratorContext::new(
                || vec![Elem::ValStop(7, 2), Elem::ValStop(9, 3)].into_iter(),
                input_sender,
            ));
            ctx.add_child(ExpandStatic::new(input, vec![count, 1], output));
            let expected: Vec<_> = [(7, 2), (9, 3)]
                .into_iter()
                .flat_map(|(value, stop)| {
                    (1..=count).map(move |index| {
                        Elem::ValStop(value, if index == count { stop } else { 1 })
                    })
                })
                .collect();
            ctx.add_child(ApproxCheckerContext::new(
                move || expected.into_iter(),
                output_receiver,
                |x, y| x == y,
            ));
            assert!(ctx
                .initialize(Default::default())
                .unwrap()
                .run(Default::default())
                .passed());
        }
    }

    #[test]
    fn expand_ref_follows_variable_reference_counts() {
        let mut ctx = ProgramBuilder::default();
        let (input_sender, input) = ctx.bounded(1);
        let (ref_sender, reference) = ctx.bounded(1);
        let (output, output_receiver) = ctx.bounded(1);
        ctx.add_child(GeneratorContext::new(
            || vec![Elem::ValStop(7, 1), Elem::ValStop(9, 2)].into_iter(),
            input_sender,
        ));
        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(0),
                    Elem::ValStop(0, 1),
                    Elem::Val(0),
                    Elem::Val(0),
                    Elem::ValStop(0, 2),
                ]
                .into_iter()
            },
            ref_sender,
        ));
        ctx.add_child(ExpandRef::new(input, reference, 1, output, 0));
        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![
                    Elem::Val(7),
                    Elem::ValStop(7, 1),
                    Elem::Val(9),
                    Elem::Val(9),
                    Elem::ValStop(9, 2),
                ]
                .into_iter()
            },
            output_receiver,
            |x, y| x == y,
        ));
        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
    }

    #[test]
    fn expand_ref_with_empty_reference_drains_input() {
        let mut ctx = ProgramBuilder::default();
        let (input_sender, input) = ctx.bounded(1);
        let (ref_sender, reference) = ctx.bounded(1);
        let (output, output_receiver) = ctx.bounded(1);
        ctx.add_child(GeneratorContext::new(
            || (0..4).map(|value| Elem::ValStop(value, if value == 3 { 2 } else { 1 })),
            input_sender,
        ));
        ctx.add_child(GeneratorContext::new(
            || Vec::<Elem<i32>>::new().into_iter(),
            ref_sender,
        ));
        ctx.add_child(ExpandRef::new(input, reference, 1, output, 0));
        ctx.add_child(ApproxCheckerContext::new(
            || Vec::<Elem<i32>>::new().into_iter(),
            output_receiver,
            |x, y| x == y,
        ));
        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
    }

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

        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
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

        let expected: Vec<_> = (1..=6)
            .flat_map(|value| {
                (0..8).map(move |index| match index {
                    3 => Elem::ValStop(value, 1),
                    7 => Elem::ValStop(value, if value % 3 == 0 { 3 } else { 2 }),
                    _ => Elem::Val(value),
                })
            })
            .collect();
        ctx.add_child(ApproxCheckerContext::new(
            move || expected.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
    }
}
