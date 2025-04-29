use dam::context_tools::*;
use std::fs::File;
use std::path::Path;

use super::{parse_csv, HBMEntry};

#[context_macro]
/// This is a context that corresponds to the `Matmul` in step-perf-py.
/// Becasue step-perf-py only models how long loading each tile takes,
/// we will read in the csv generated from step-perf-py and factor in potential
/// stalls between the tile loads due to backpressure in this context.
pub struct MatmulLoadContext {
    file_path: String,
    out_stream: Sender<HBMEntry>,
}

impl MatmulLoadContext {
    pub fn new(file_path: String, out_stream: Sender<HBMEntry>) -> Self {
        let ctx = Self {
            file_path,
            out_stream,
            context_info: Default::default(),
        };
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for MatmulLoadContext {
    fn run(&mut self) {
        // Read in the data generated from step-perf-py
        let entries = parse_csv(&self.file_path);

        for hbm_entry in entries {
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick() + 1,
                        data: hbm_entry,
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

    use super::MatmulLoadContext;

    #[test]
    fn test_with_dequeue() {
        // let mut ctx = ProgramBuilder::default();
        // let (in_snd, in_rcv) = ctx.bounded(2);
        // let (out_snd, out_rcv) = ctx.bounded(2);

        // ctx.add_child(GeneratorContext::new(|| 0..10u32, in_snd));
        // ctx.add_child(ReceiverBackpressureContext::new(in_rcv, out_snd));
        // ctx.add_child(CheckerContext::new(|| 0..10u32, out_rcv));

        // ctx.initialize(Default::default())
        //     .unwrap()
        //     .run(Default::default());
    }
}
