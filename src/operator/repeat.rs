use crate::primitives::elem::Elem;
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
