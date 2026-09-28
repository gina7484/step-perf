use crate::primitives::elem::{Elem, StopType};
use crate::utils::request_profile::{ProfiledReceiver, ProfiledSender};
use dam::context_tools::*;

#[context_macro]
pub struct RepeatStatic<T: Clone> {
    in_stream: ProfiledReceiver<Elem<T>>,
    // Products of appended dimensions, from innermost to outermost.
    stop_periods: Vec<usize>,
    out_stream: ProfiledSender<Elem<T>>,
}

impl<T: DAMType> RepeatStatic<T>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        repeat_factor: Vec<usize>,
        out_stream: Sender<Elem<T>>,
    ) -> Self {
        let mut period = 1usize;
        let stop_periods = repeat_factor
            .iter()
            .rev()
            .map(|&factor| {
                assert!(factor > 0, "RepeatStatic repeat factors must be positive");
                period = period
                    .checked_mul(factor)
                    .expect("RepeatStatic repeat count overflow");
                period
            })
            .collect();
        let ctx = Self {
            in_stream: in_stream.into(),
            stop_periods,
            out_stream: out_stream.into(),
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType> Context for RepeatStatic<T> {
    fn run(&mut self) {
        // An empty factor list forwards each input unchanged.
        let repeat_count = self.stop_periods.last().copied().unwrap_or(1);
        while let Ok(ChannelElement { data, .. }) = self.in_stream.peek_next(&self.time) {
            let (value, input_stop) = match data {
                Elem::Val(value) => (value, 0),
                Elem::ValStop(value, stop) => (value, stop),
            };
            for index in 1..=repeat_count {
                // Close every trailing dimension whose last value was emitted.
                let mut stop = self
                    .stop_periods
                    .iter()
                    .take_while(|&&period| index % period == 0)
                    .count() as StopType;
                if index == repeat_count {
                    stop += input_stop;
                }
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
pub struct RepeatRef<T: Clone, R: Clone> {
    in_stream: ProfiledReceiver<Elem<T>>,
    ref_stream: ProfiledReceiver<Elem<R>>,
    out_stream: ProfiledSender<Elem<T>>,
    expanded_rank_cnt: StopType,
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
        expanded_rank_cnt: StopType,
        rank: StopType,
        id: u32,
    ) -> Self {
        assert!(
            expanded_rank_cnt > 0,
            "RepeatRef expanded_rank_cnt must be positive"
        );
        let ctx = Self {
            in_stream: in_stream.into(),
            ref_stream: ref_stream.into(),
            out_stream: out_stream.into(),
            expanded_rank_cnt,
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
        while let Ok(ChannelElement { data, .. }) = self.in_stream.dequeue(&self.time) {
            let (value, input_stop) = match data {
                Elem::Val(value) => (value, 0),
                Elem::ValStop(value, stop) => (value, stop),
            };
            loop {
                let ref_data = self
                    .ref_stream
                    .dequeue(&self.time)
                    .unwrap_or_else(|_| {
                        panic!(
                            "RepeatRef {}: reference stream ended before input stream",
                            self.id
                        )
                    })
                    .data;
                let ref_stop = match ref_data {
                    Elem::Val(_) => 0,
                    Elem::ValStop(_, stop) => stop,
                };
                if ref_stop < self.rank {
                    continue;
                }

                let stop = ref_stop - self.rank;
                let terminal = stop >= self.expanded_rank_cnt;
                if terminal {
                    assert_eq!(
                        input_stop + self.expanded_rank_cnt, stop,
                        "RepeatRef {}: mismatch between input stop count {} and reference stop count {} (rank={}, expanded_rank_cnt={})",
                        self.id, input_stop, ref_stop, self.rank, self.expanded_rank_cnt,
                    );
                }
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
                // Inner repeat boundaries retain this value. Only the outermost
                // appended dimension advances the input stream.
                if terminal {
                    break;
                }
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

    use crate::primitives::elem::{Elem, StopType};
    use dam::context_tools::*;

    use super::{RepeatRef, RepeatStatic};

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

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 1, 0, 0));

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

        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
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

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 1, 0, 0));

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

        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
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
            || vec![Elem::Val(1), Elem::Val(2), Elem::ValStop(3, 1)].into_iter(),
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

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 1, 1, 0));

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

        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
    }

    #[context_macro]
    struct ExactOutput {
        input: Receiver<Elem<u64>>,
        expected: Vec<Elem<u64>>,
    }

    impl ExactOutput {
        fn new(input: Receiver<Elem<u64>>, expected: Vec<Elem<u64>>) -> Self {
            let ctx = Self {
                input,
                expected,
                context_info: Default::default(),
            };
            ctx.input.attach_receiver(&ctx);
            ctx
        }
    }

    impl Context for ExactOutput {
        fn run(&mut self) {
            for expected in &self.expected {
                assert_eq!(&self.input.dequeue(&self.time).unwrap().data, expected);
            }
            assert!(
                self.input.dequeue(&self.time).is_err(),
                "Unexpected extra repeat output"
            );
        }
    }

    fn check_static(input: Vec<Elem<u64>>, factors: Vec<usize>, expected: Vec<Elem<u64>>) {
        let mut ctx = ProgramBuilder::default();
        // A depth of one also exercises backpressure while repeating a value.
        let (in_snd, in_rcv) = ctx.bounded(1);
        let (out_snd, out_rcv) = ctx.bounded(1);
        ctx.add_child(GeneratorContext::new(move || input.into_iter(), in_snd));
        ctx.add_child(RepeatStatic::new(in_rcv, factors, out_snd));
        ctx.add_child(ExactOutput::new(out_rcv, expected));
        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
    }

    fn check_ref(
        input: Vec<Elem<u64>>,
        reference: Vec<Elem<u64>>,
        expanded_rank_cnt: StopType,
        trigger_rank: StopType,
        expected: Vec<Elem<u64>>,
    ) {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(1);
        let (ref_snd, ref_rcv) = ctx.bounded(1);
        let (out_snd, out_rcv) = ctx.bounded(1);
        ctx.add_child(GeneratorContext::new(move || input.into_iter(), in_snd));
        ctx.add_child(GeneratorContext::new(
            move || reference.into_iter(),
            ref_snd,
        ));
        ctx.add_child(RepeatRef::new(
            in_rcv,
            ref_rcv,
            out_snd,
            expanded_rank_cnt,
            trigger_rank,
            17,
        ));
        ctx.add_child(ExactOutput::new(out_rcv, expected));
        assert!(ctx
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
    }

    #[test]
    fn repeat_static_single_dimension() {
        check_static(
            vec![Elem::Val(1), Elem::ValStop(2, 1)],
            vec![3],
            vec![
                Elem::Val(1),
                Elem::Val(1),
                Elem::ValStop(1, 1),
                Elem::Val(2),
                Elem::Val(2),
                Elem::ValStop(2, 2),
            ],
        );
    }

    #[test]
    fn repeat_static_multiple_dimensions() {
        check_static(
            vec![Elem::Val(1), Elem::ValStop(2, 1), Elem::ValStop(3, 2)],
            vec![2, 3],
            vec![
                Elem::Val(1),
                Elem::Val(1),
                Elem::ValStop(1, 1),
                Elem::Val(1),
                Elem::Val(1),
                Elem::ValStop(1, 2),
                Elem::Val(2),
                Elem::Val(2),
                Elem::ValStop(2, 1),
                Elem::Val(2),
                Elem::Val(2),
                Elem::ValStop(2, 3),
                Elem::Val(3),
                Elem::Val(3),
                Elem::ValStop(3, 1),
                Elem::Val(3),
                Elem::Val(3),
                Elem::ValStop(3, 4),
            ],
        );
    }

    #[test]
    fn repeat_static_three_dimensions_with_singletons() {
        check_static(
            vec![Elem::ValStop(7, 1)],
            vec![2, 1, 2],
            vec![
                Elem::Val(7),
                Elem::ValStop(7, 2),
                Elem::Val(7),
                Elem::ValStop(7, 4),
            ],
        );
        check_static(
            vec![Elem::ValStop(8, 2)],
            vec![1, 2, 1],
            vec![Elem::ValStop(8, 1), Elem::ValStop(8, 5)],
        );
        check_static(
            vec![Elem::ValStop(9, 1)],
            vec![1, 1, 1],
            vec![Elem::ValStop(9, 4)],
        );
    }

    #[test]
    fn repeat_static_empty_factors_are_identity() {
        let input = vec![Elem::Val(1), Elem::ValStop(2, 1), Elem::ValStop(3, 2)];
        check_static(input.clone(), vec![], input);
    }

    #[test]
    fn repeat_empty_streams() {
        check_static(vec![], vec![2, 3], vec![]);
        check_ref(vec![], vec![], 2, 0, vec![]);
    }

    #[test]
    #[should_panic(expected = "repeat factors must be positive")]
    fn repeat_static_rejects_zero_factor() {
        check_static(vec![], vec![2, 0], vec![]);
    }

    #[test]
    #[should_panic(expected = "repeat count overflow")]
    fn repeat_static_rejects_count_overflow() {
        check_static(vec![], vec![usize::MAX, 2], vec![]);
    }

    #[test]
    fn repeat_ref_multiple_ragged_dimensions() {
        check_ref(
            vec![
                Elem::Val(1),
                Elem::ValStop(2, 1),
                Elem::Val(3),
                Elem::ValStop(4, 2),
            ],
            vec![
                Elem::Val(0),
                Elem::ValStop(0, 1),
                Elem::ValStop(0, 2),
                Elem::ValStop(0, 1),
                Elem::Val(0),
                Elem::ValStop(0, 3),
                Elem::ValStop(0, 2),
                Elem::Val(0),
                Elem::ValStop(0, 1),
                Elem::ValStop(0, 4),
            ],
            2,
            0,
            vec![
                Elem::Val(1),
                Elem::ValStop(1, 1),
                Elem::ValStop(1, 2),
                Elem::ValStop(2, 1),
                Elem::Val(2),
                Elem::ValStop(2, 3),
                Elem::ValStop(3, 2),
                Elem::Val(4),
                Elem::ValStop(4, 1),
                Elem::ValStop(4, 4),
            ],
        );
    }

    #[test]
    fn repeat_ref_multiple_dimensions_with_trigger_rank() {
        check_ref(
            vec![Elem::Val(5), Elem::ValStop(6, 1)],
            vec![
                Elem::Val(9),
                Elem::ValStop(9, 1),
                Elem::ValStop(9, 2),
                Elem::ValStop(9, 3),
                Elem::ValStop(9, 4),
                Elem::ValStop(9, 1),
                Elem::ValStop(9, 3),
                Elem::Val(9),
                Elem::ValStop(9, 2),
                Elem::ValStop(9, 5),
            ],
            2,
            2,
            vec![
                Elem::Val(5),
                Elem::ValStop(5, 1),
                Elem::ValStop(5, 2),
                Elem::ValStop(6, 1),
                Elem::Val(6),
                Elem::ValStop(6, 3),
            ],
        );
    }

    #[test]
    fn repeat_ref_three_dimensions_with_singletons() {
        check_ref(
            vec![Elem::Val(7), Elem::ValStop(8, 1)],
            vec![
                Elem::ValStop(0, 1),
                Elem::ValStop(0, 2),
                Elem::ValStop(0, 3),
                Elem::Val(0),
                Elem::ValStop(0, 1),
                Elem::ValStop(0, 3),
                Elem::ValStop(0, 4),
                Elem::ValStop(0, 5),
            ],
            3,
            1,
            vec![
                Elem::Val(7),
                Elem::ValStop(7, 1),
                Elem::ValStop(7, 2),
                Elem::Val(7),
                Elem::ValStop(7, 2),
                Elem::ValStop(7, 3),
                Elem::ValStop(8, 4),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "expanded_rank_cnt must be positive")]
    fn repeat_ref_rejects_zero_expanded_rank() {
        check_ref(vec![], vec![], 0, 0, vec![]);
    }
}
