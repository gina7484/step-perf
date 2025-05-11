use super::ActEntry;

use crate::memory_simple::events::LoggableEvent;
use crate::memory_simple::PMUEntry;

use dam::context_tools::*;
use dam::dam_macros::event_type;
use serde::{Deserialize, Serialize};

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct CompGenQKV {
    pub counter: u32,
    pub start: u64,
    pub end: u64,
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct LoadGenQKV {
    pub counter: u32,
    pub start: u64,
    pub end: u64,
}

#[context_macro]
pub struct GenQKV {
    in_stream: Receiver<PMUEntry>,
    out_stream: Sender<ActEntry>,
    flop: u64,
    compute_bw: f64, // FLOPs / ns
}

impl GenQKV {
    pub fn new(
        in_stream: Receiver<PMUEntry>,
        out_stream: Sender<ActEntry>,
        flop: u64,
        compute_bw: f64,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            context_info: Default::default(),
            flop,
            compute_bw,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for GenQKV {
    fn run(&mut self) {
        let mut counter = 0;
        loop {
            let compute_latency_ns;
            let load_latency_ns;
            let latency: u64 = match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    PMUEntry {
                        outer: _,
                        m: _,
                        n: _,
                        k: _,
                        output_tile_available: _,
                        num_elems,
                    } => {
                        let load_bw = data.bw();
                        load_latency_ns = (num_elems as f64 / load_bw).ceil() as u64;
                        compute_latency_ns = ((self.flop) as f64 / self.compute_bw).ceil() as u64;

                        let total_latency_ns = if load_latency_ns > compute_latency_ns {
                            load_latency_ns
                        } else {
                            compute_latency_ns
                        };
                        total_latency_ns
                    }
                },
                Err(_) => return,
            };
            self.time.incr_cycles(latency);
            self.in_stream.dequeue(&self.time).unwrap();

            let curr_time = self.time.tick();
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: curr_time,
                        data: ActEntry {},
                    },
                )
                .unwrap();

            let time_block_start_ns = curr_time.time() - latency;

            dam::logging::log_event(&CompGenQKV {
                counter: counter,
                start: time_block_start_ns,
                end: time_block_start_ns + compute_latency_ns,
            })
            .unwrap();

            dam::logging::log_event(&LoadGenQKV {
                counter: counter,
                start: time_block_start_ns,
                end: time_block_start_ns + load_latency_ns,
            })
            .unwrap();

            counter += 1;
        }
    }
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct CompProj {
    pub counter: u32,
    pub start: u64,
    pub end: u64,
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct LoadProj {
    pub counter: u32,
    pub start: u64,
    pub end: u64,
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct StoreProj {
    pub counter: u32,
    pub start: u64,
    pub end: u64,
}

#[context_macro]
pub struct Proj {
    in_stream: Receiver<ActEntry>,
    weight_stream: Receiver<PMUEntry>,
    out_stream: Sender<PMUEntry>,
    flop: u64,
    compute_bw: f64, // FLOPs / ns
}

impl Proj {
    pub fn new(
        in_stream: Receiver<ActEntry>,
        weight_stream: Receiver<PMUEntry>,
        out_stream: Sender<PMUEntry>,
        flop: u64,
        compute_bw: f64,
    ) -> Self {
        let ctx = Self {
            in_stream,
            weight_stream,
            out_stream,
            context_info: Default::default(),
            flop,
            compute_bw,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.weight_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for Proj {
    fn run(&mut self) {
        let mut counter = 0;

        // Load weight
        let out_size: u32;
        let load_latency_ns;
        let mut compute_latency_ns;
        let mut store_latency_ns;
        let mut output_entry;
        let _ = self.in_stream.peek_next(&self.time);
        let latency: u64 = match self.weight_stream.peek_next(&self.time) {
            Ok(ChannelElement { time: _, data }) => match data {
                PMUEntry {
                    outer: _,
                    m: _,
                    n,
                    k: _,
                    output_tile_available: _,
                    num_elems,
                } => {
                    out_size = n;
                    let load_bw = data.bw();
                    load_latency_ns = (num_elems as f64 / load_bw).ceil() as u64;
                    compute_latency_ns = ((self.flop) as f64 / self.compute_bw).ceil() as u64;

                    let output_pmu_entry = PMUEntry {
                        outer: 0,
                        m: 0,
                        n: 0,
                        k: 0,
                        output_tile_available: false,
                        num_elems: out_size,
                    };
                    let store_bw = output_pmu_entry.bw();
                    output_entry = Some(output_pmu_entry);
                    store_latency_ns = (out_size as f64 / store_bw).ceil() as u64;

                    let total_latency_ns =
                        compute_latency_ns.max(load_latency_ns.max(store_latency_ns));
                    total_latency_ns
                }
            },
            Err(_) => return,
        };

        self.time.incr_cycles(latency);

        let curr_time = self.time.tick();
        self.in_stream.dequeue(&self.time).unwrap();
        self.weight_stream.dequeue(&self.time).unwrap();
        self.out_stream
            .enqueue(
                &self.time,
                ChannelElement {
                    time: curr_time,
                    data: output_entry.unwrap(),
                },
            )
            .unwrap();

        let time_block_start_ns = curr_time.time() - latency;

        dam::logging::log_event(&CompProj {
            counter: counter,
            start: time_block_start_ns,
            end: time_block_start_ns + compute_latency_ns,
        })
        .unwrap();

        dam::logging::log_event(&LoadProj {
            counter: counter,
            start: time_block_start_ns,
            end: time_block_start_ns + load_latency_ns,
        })
        .unwrap();

        dam::logging::log_event(&StoreProj {
            counter: counter,
            start: time_block_start_ns,
            end: time_block_start_ns + store_latency_ns,
        })
        .unwrap();

        counter += 1;

        loop {
            let latency: u64 = match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data: _ }) => {
                    // There's no load latency as the loaded weight is being reused in the registers
                    // and the input is coming from the previous compute unit
                    // => No load from PMU
                    compute_latency_ns = ((self.flop) as f64 / self.compute_bw).ceil() as u64;

                    let output_pmu_entry = PMUEntry {
                        outer: 0,
                        m: 0,
                        n: 0,
                        k: 0,
                        output_tile_available: false,
                        num_elems: out_size,
                    };
                    let store_bw = output_pmu_entry.bw();
                    output_entry = Some(output_pmu_entry);
                    store_latency_ns = (out_size as f64 / store_bw).ceil() as u64;

                    compute_latency_ns.max(store_latency_ns)
                }
                Err(_) => return,
            };
            self.time.incr_cycles(latency);

            let curr_time = self.time.tick();
            self.in_stream.dequeue(&self.time).unwrap();
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: curr_time,
                        data: output_entry.unwrap(),
                    },
                )
                .unwrap();

            let time_block_start_ns = curr_time.time() - latency;

            dam::logging::log_event(&CompProj {
                counter: counter,
                start: time_block_start_ns,
                end: time_block_start_ns + compute_latency_ns,
            })
            .unwrap();

            dam::logging::log_event(&StoreProj {
                counter: counter,
                start: time_block_start_ns,
                end: time_block_start_ns + store_latency_ns,
            })
            .unwrap();
            counter += 1;
        }
    }
}
