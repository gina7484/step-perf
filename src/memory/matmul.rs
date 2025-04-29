use dam::context_tools::*;
use std::fs::File;
use std::path::Path;

use super::{hbm_to_pmu, parse_csv, HBMEntry, PMUEntry};

#[context_macro]
/// This is a context that corresponds to the `Matmul` in step-perf-py.
/// Becasue step-perf-py only models how long loading each tile takes,
/// we will read in the csv generated from step-perf-py and factor in potential
/// stalls between the tile loads due to backpressure in this context.
pub struct HBMLoadContext {
    file_path: String,
    out_stream: Sender<PMUEntry>,
}

impl HBMLoadContext {
    pub fn new(file_path: String, out_stream: Sender<PMUEntry>) -> Self {
        let ctx = Self {
            file_path,
            out_stream,
            context_info: Default::default(),
        };
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for HBMLoadContext {
    fn run(&mut self) {
        // Read in the data generated from step-perf-py
        let entries = parse_csv(&self.file_path);

        for hbm_entry in entries {
            let time_block_ms = hbm_entry.end_ms - hbm_entry.start_ms;
            let time_block_ns = (time_block_ms * 1e6) as u64;

            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick() + time_block_ns,
                        data: hbm_to_pmu(&hbm_entry),
                    },
                )
                .unwrap();
            self.time.incr_cycles(time_block_ns);
        }
    }
}

#[cfg(test)]
mod test_backpressure {
    use dam::{
        channel::Receiver,
        simulation::ProgramBuilder,
        utility_contexts::{CheckerContext, ConsumerContext, GeneratorContext},
    };

    use super::HBMLoadContext;

    #[test]
    fn test_with_dequeue() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::new("gen_qkv.csv".to_string(), in_snd));
        ctx.add_child(ConsumerContext::new(in_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
