use dam::{context_tools::*, dam_macros::event_type};
use serde::{Deserialize, Serialize};

use super::ActEntry;
use crate::memory::PMUEntry;

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct CompQKt {
    pub start: u64,
    pub end: u64,
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct LoadQKt {
    pub start: u64,
    pub end: u64,
}

#[context_macro]
pub struct QKt {
    in1_stream: Receiver<ActEntry>,
    in2_stream: Receiver<PMUEntry>,
    out_stream: Sender<ActEntry>,
    flop: Vec<u64>,
    compute_bw: f64, // FLOPs / ns
}

impl QKt {
    pub fn new(
        in1_stream: Receiver<ActEntry>,
        in2_stream: Receiver<PMUEntry>,
        out_stream: Sender<ActEntry>,
        flop: Vec<u64>,
        compute_bw: f64,
    ) -> Self {
        let ctx = Self {
            in1_stream,
            in2_stream,
            out_stream,
            flop,
            compute_bw,
            context_info: Default::default(),
        };
        ctx.in1_stream.attach_receiver(&ctx);
        ctx.in2_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for QKt {
    fn run(&mut self) {
        for flop in &self.flop {
            let load_latency_ns;
            let compute_latency_ns;
            let _ = self.in1_stream.peek_next(&self.time);
            let latency: u64 = match self.in2_stream.peek_next(&self.time) {
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
                        compute_latency_ns = (*flop as f64 / self.compute_bw).ceil() as u64;

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

            let curr_time = self.time.tick();
            self.in1_stream.dequeue(&self.time).unwrap();
            self.in2_stream.dequeue(&self.time).unwrap();
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

            dam::logging::log_event(&CompQKt {
                start: time_block_start_ns,
                end: time_block_start_ns + compute_latency_ns,
            })
            .unwrap();

            dam::logging::log_event(&LoadQKt {
                start: time_block_start_ns,
                end: time_block_start_ns + load_latency_ns,
            })
            .unwrap();
        }
    }
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct CompAttnV {
    pub start: u64,
    pub end: u64,
}

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
struct LoadAttnV {
    pub start: u64,
    pub end: u64,
}

#[context_macro]
pub struct AttnV {
    in1_stream: Receiver<ActEntry>,
    in2_stream: Receiver<PMUEntry>,
    out_stream: Sender<ActEntry>,
    flop: Vec<u64>,
    compute_bw: f64, // FLOPs / ns
}

impl AttnV {
    pub fn new(
        in1_stream: Receiver<ActEntry>,
        in2_stream: Receiver<PMUEntry>,
        out_stream: Sender<ActEntry>,
        flop: Vec<u64>,
        compute_bw: f64,
    ) -> Self {
        let ctx = Self {
            in1_stream,
            in2_stream,
            out_stream,
            flop,
            compute_bw,
            context_info: Default::default(),
        };
        ctx.in1_stream.attach_receiver(&ctx);
        ctx.in2_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for AttnV {
    fn run(&mut self) {
        for flop in &self.flop {
            let load_latency_ns;
            let compute_latency_ns;
            let _ = self.in1_stream.peek_next(&self.time);
            let latency: u64 = match self.in2_stream.peek_next(&self.time) {
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
                        compute_latency_ns = (*flop as f64 / self.compute_bw).ceil() as u64;

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

            let curr_time = self.time.tick();
            self.in1_stream.dequeue(&self.time).unwrap();
            self.in2_stream.dequeue(&self.time).unwrap();
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

            dam::logging::log_event(&CompAttnV {
                start: time_block_start_ns,
                end: time_block_start_ns + compute_latency_ns,
            })
            .unwrap();

            dam::logging::log_event(&LoadAttnV {
                start: time_block_start_ns,
                end: time_block_start_ns + load_latency_ns,
            })
            .unwrap();
        }
    }
}
