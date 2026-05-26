use crate::primitives::elem::Elem;
use std::sync::OnceLock;
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
    id: u32,
}

fn traced_repeat_ref_ids() -> &'static Vec<u32> {
    static IDS: OnceLock<Vec<u32>> = OnceLock::new();
    IDS.get_or_init(|| {
        std::env::var("STEP_TRACE_REPEAT_REF_IDS")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .expect("STEP_TRACE_REPEAT_REF_IDS entries must be u32 op ids")
            })
            .collect()
    })
}

fn trace_repeat_ref(id: u32) -> bool {
    traced_repeat_ref_ids().contains(&id)
}

impl<T: DAMType, R: DAMType> RepeatRef<T, R>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        ref_stream: Receiver<Elem<R>>,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            ref_stream,
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

impl<T: DAMType, R: DAMType> Context for RepeatRef<T, R> {
    fn run(&mut self) {
        let trace = trace_repeat_ref(self.id);
        let mut input_count = 0usize;
        let mut ref_count = 0usize;
        let mut output_count = 0usize;
        loop {
            // 1. Dequeue from in_stream
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: in_data,
                }) => {
                    input_count += 1;
                    let mut window_outputs = 0usize;
                    // 2. Dequeue from ref_stream and enqueue until we see a stop token
                    loop {
                        match self.ref_stream.dequeue(&self.time) {
                            Ok(ChannelElement {
                                time: _,
                                data: ref_data,
                            }) => {
                                ref_count += 1;
                                match ref_data {
                                    Elem::Val(_) => {
                                        // Not a stop token, enqueue the input element
                                        window_outputs += 1;
                                        output_count += 1;
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
                                    }
                                    Elem::ValStop(_, s) => {
                                        // 3. Stop token: enqueue with stop token + 1
                                        window_outputs += 1;
                                        output_count += 1;
                                        if trace {
                                            eprintln!(
                                                "[repeat-ref-trace] id={} input={} window_outputs={} ref_count={} output_count={} ref_stop={}",
                                                self.id,
                                                input_count,
                                                window_outputs,
                                                ref_count,
                                                output_count,
                                                s
                                            );
                                        }
                                        self.out_stream
                                            .enqueue(
                                                &self.time,
                                                ChannelElement {
                                                    time: self.time.tick(),
                                                    data: match &in_data {
                                                        Elem::Val(x) => {
                                                            Elem::ValStop(x.clone(), 0 + 1)
                                                        }
                                                        Elem::ValStop(x, in_s) => {
                                                            assert_eq!(in_s + 1, s,
                                                                "RepeatRef {}: mismatch between input stop count {} and reference stop count {}",
                                                                self.id, in_s + 1, s);
                                                            Elem::ValStop(x.clone(), s)
                                                        }
                                                    },
                                                },
                                            )
                                            .unwrap();
                                        break; // Move to next element in in_stream
                                    }
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
                Err(_) => {
                    if trace {
                        eprintln!(
                            "[repeat-ref-trace] id={} input_closed input_count={} ref_count={} output_count={}",
                            self.id, input_count, ref_count, output_count
                        );
                    }
                    return;
                } // End of in_stream
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

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 0));

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

        ctx.add_child(RepeatRef::new(in_rcv, ref_rcv, out_snd, 0));

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
}
