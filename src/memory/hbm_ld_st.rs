use dam::context_tools::*;
use std::fs::File;
use std::path::Path;

use super::events::GenQKV;
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

            dam::logging::log_event(&GenQKV {
                start: self.time.tick().time() - time_block_ns,
                end: self.time.tick().time(),
            })
            .unwrap();
        }
    }
}

#[cfg(test)]
mod test_hbm_load {
    use dam::dam_macros::event_type;
    use dam::{
        simulation::{
            LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
        },
        utility_contexts::ConsumerContext,
    };
    use serde::{Deserialize, Serialize};

    use crate::memory::events::LoggableEvent;

    use super::HBMLoadContext;

    #[test]
    fn simple_test() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::new("gen_qkv.csv".to_string(), in_snd));
        ctx.add_child(ConsumerContext::new(in_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    /*
        // Logging values
        #[derive(Serialize, Deserialize, Debug)]
        #[event_type]
        pub struct GenQKV {
            pub start: u64,
            pub end: u64,
        }

        // Implement the trait for GenQKV
        impl LoggableEvent for GenQKV {
            fn new(start: u64, end: u64) -> Self {
                GenQKV { start, end }
            }
        }
    */
    #[test]
    fn test_with_logging() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::new("gen_qkv.csv".to_string(), in_snd));
        ctx.add_child(ConsumerContext::new(in_rcv));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
            // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
            dam::logging::LogFilter::AllowAll,
        ));
        let run_options = run_options.logging(LoggingOptions::Mongo(
            MongoOptionsBuilder::default()
                .db("init_hbm_log".to_string())
                .uri("mongodb://127.0.0.1:27017".to_string())
                .build()
                .unwrap(),
        ));
        let summary = initialized.run(run_options.build().unwrap());
        // Check the summary
        println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
    }
}
