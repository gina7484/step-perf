use dam::context_tools::*;

#[context_macro]
pub struct ReceiverBackpressureContext {
    in_stream: Receiver<u32>,
    out_stream: Sender<u32>,
}

impl ReceiverBackpressureContext {
    pub fn new(in_stream: Receiver<u32>, out_stream: Sender<u32>) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for ReceiverBackpressureContext {
    fn run(&mut self) {
        loop {
            let data = match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: t, data }) => {
                    println!("ChannelElement data: {}", data);
                    println!("ChannelElement time: {}", t.time());
                    println!("Received time: {}", self.time.tick().time());
                    data
                }
                Err(_) => return,
            };

            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick() + 1,
                        data,
                    },
                )
                .unwrap();
            self.time.incr_cycles(3);
        }
    }
}

#[cfg(test)]
mod test_backpressure {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{CheckerContext, GeneratorContext},
    };

    use super::ReceiverBackpressureContext;

    #[test]
    fn test_with_dequeue() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);
        let (out_snd, out_rcv) = ctx.bounded(2);

        ctx.add_child(GeneratorContext::new(|| 0..10u32, in_snd));
        ctx.add_child(ReceiverBackpressureContext::new(in_rcv, out_snd));
        ctx.add_child(CheckerContext::new(|| 0..10u32, out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
