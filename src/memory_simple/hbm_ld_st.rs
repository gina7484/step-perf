use dam::context_tools::*;
use dam::logging::LogEvent;
use std::marker::PhantomData;

use super::{events::LoggableEvent, hbm_to_pmu, parse_csv, HBMEntry, PMUEntry};

#[context_macro]
/// This is a context that corresponds to the `Matmul` in step-perf-py.
/// Becasue step-perf-py only models how long loading each tile takes,
/// we will read in the csv generated from step-perf-py and factor in potential
/// stalls between the tile loads due to backpressure in this context.
pub struct HBMLoadContext<E: LoggableEvent> {
    file_path: String,
    out_stream: Sender<PMUEntry>,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<E: LoggableEvent + LogEvent + std::marker::Sync + std::marker::Send> HBMLoadContext<E> {
    pub fn new(file_path: String, out_stream: Sender<PMUEntry>) -> Self {
        let ctx = Self {
            file_path,
            out_stream,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<E: LoggableEvent + LogEvent + std::marker::Sync + std::marker::Send> Context
    for HBMLoadContext<E>
{
    fn run(&mut self) {
        // Read in the data generated from step-perf-py
        let entries = parse_csv(&self.file_path);

        let start_ns = (entries[0].start_ms * 1e6) as u64;
        if start_ns > 0 {
            self.time.incr_cycles(start_ns);
        }

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

            dam::logging::log_event(&E::new(
                hbm_entry.outer,
                hbm_entry.m,
                hbm_entry.n,
                hbm_entry.k,
                self.time.tick().time() - time_block_ns,
                self.time.tick().time(),
                hbm_entry.output_tile_available,
                hbm_entry.num_elems,
            ))
            .unwrap();
        }
    }
}

#[context_macro]
/// This is a context that corresponds to the `Matmul` in step-perf-py.
/// Becasue step-perf-py only models how long loading each tile takes,
/// we will read in the csv generated from step-perf-py and factor in potential
/// stalls between the tile loads due to backpressure in this context.
pub struct HBMStoreContext<E: LoggableEvent> {
    file_path: String,
    in_stream: Receiver<PMUEntry>,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<E: LoggableEvent + LogEvent + std::marker::Sync + std::marker::Send> HBMStoreContext<E> {
    pub fn new(file_path: String, in_stream: Receiver<PMUEntry>) -> Self {
        let ctx = Self {
            file_path,
            in_stream,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);

        ctx
    }
}

impl<E: LoggableEvent + LogEvent + std::marker::Sync + std::marker::Send> Context
    for HBMStoreContext<E>
{
    fn run(&mut self) {
        // Read in the data generated from step-perf-py
        let entries = parse_csv(&self.file_path);

        for hbm_entry in entries {
            let time_block_ms = hbm_entry.end_ms - hbm_entry.start_ms;
            let time_block_ns = (time_block_ms * 1e6) as u64;

            match self.in_stream.peek_next(&self.time) {
                Ok(_) => {}
                Err(_) => return,
            }
            self.time.incr_cycles(time_block_ns);

            self.in_stream.dequeue(&self.time).unwrap();

            dam::logging::log_event(&E::new(
                hbm_entry.outer,
                hbm_entry.m,
                hbm_entry.n,
                hbm_entry.k,
                self.time.tick().time() - time_block_ns,
                self.time.tick().time(),
                hbm_entry.output_tile_available,
                hbm_entry.num_elems,
            ))
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

    use crate::memory_simple::events::LoggableEvent;

    use super::HBMLoadContext;

    // Logging values
    #[derive(Serialize, Deserialize, Debug)]
    #[event_type]
    struct GenQKV {
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    }

    // Implement the trait for GenQKV
    impl LoggableEvent for GenQKV {
        fn new(
            outer: u32,
            m: u32,
            n: u32,
            k: u32,
            start_ns: u64,
            end_ns: u64,
            output_tile_available: bool,
            num_elems: u32,
        ) -> Self {
            GenQKV {
                outer,
                m,
                n,
                k,
                start_ns: start_ns,
                end_ns: end_ns,
                output_tile_available,
                num_elems,
            }
        }
    }
    impl GenQKV {
        pub const NAME: &'static str = "GenQKV";
    }

    // Logging values
    #[derive(Serialize, Deserialize, Debug)]
    #[event_type]
    struct Output {
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    }

    // Implement the trait for GenQKV
    impl LoggableEvent for Output {
        fn new(
            outer: u32,
            m: u32,
            n: u32,
            k: u32,
            start_ns: u64,
            end_ns: u64,
            output_tile_available: bool,
            num_elems: u32,
        ) -> Self {
            Output {
                outer,
                m,
                n,
                k,
                start_ns,
                end_ns,
                output_tile_available,
                num_elems,
            }
        }
    }
    impl Output {
        pub const NAME: &'static str = "Output";
    }

    #[test]
    fn simple_test() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::<GenQKV>::new(
            "gen_qkv.csv".to_string(),
            in_snd,
        ));
        ctx.add_child(ConsumerContext::new(in_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_with_logging() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::<GenQKV>::new(
            "gen_qkv.csv".to_string(),
            in_snd,
        ));
        ctx.add_child(ConsumerContext::new(in_rcv));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
            // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
            dam::logging::LogFilter::AllowAll,
        ));
        let run_options = run_options.logging(LoggingOptions::Mongo(
            MongoOptionsBuilder::default()
                .db("init_hbm_log_generic".to_string())
                .uri("mongodb://127.0.0.1:27017".to_string())
                .build()
                .unwrap(),
        ));
        let summary = initialized.run(run_options.build().unwrap());
        // Check the summary
        println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
    }

    #[test]
    fn test_log_two_events() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);
        let (out_snd, out_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::<GenQKV>::new(
            "gen_qkv.csv".to_string(),
            in_snd,
        ));
        ctx.add_child(ConsumerContext::new(in_rcv));

        ctx.add_child(HBMLoadContext::<Output>::new(
            "output.csv".to_string(),
            out_snd,
        ));
        ctx.add_child(ConsumerContext::new(out_rcv));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
            // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
            dam::logging::LogFilter::AllowAll,
        ));
        let run_options = run_options.logging(LoggingOptions::Mongo(
            MongoOptionsBuilder::default()
                .db("init_hbm_log_two".to_string())
                .uri("mongodb://127.0.0.1:27017".to_string())
                .build()
                .unwrap(),
        ));
        let summary = initialized.run(run_options.build().unwrap());
        // Check the summary
        println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
    }
}
