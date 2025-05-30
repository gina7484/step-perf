use dam::{
    channel::{ChannelElement, PeekResult, Receiver, Sender},
    context::Context,
    dam_macros::context_macro,
    simulation::ProgramBuilder,
    types::StaticallySized,
};
use derive_more::Constructor;
use serde_json::de::Read;

pub struct HBMConfig {
    pub channel_num: usize,
    pub per_channel_latency: u64,
    pub per_channel_init_interval: u64,
    pub per_channel_outstanding: usize,
}

#[derive(Constructor, Clone, Default, Debug)]
pub struct Request {
    is_write: bool,
    address: u64,
    id: usize, // Index of the reader or the writer
}

impl StaticallySized for Request {
    const SIZE: usize = 1 + 8 + 8;
}

#[derive(Constructor, Clone, Default, Debug)]
pub struct Response {
    is_write: bool,
    address: u64,
    id: usize, // Index of the reader or the writer
}

impl StaticallySized for Response {
    const SIZE: usize = 1 + 8 + 8;
}

#[context_macro]
pub struct HBMChannelContext {
    in_request: Receiver<Request>,
    out_rsp: Sender<Response>,
    latency: u64,
    init_interval: u64,
    outstanding: usize,
}

impl HBMChannelContext {
    pub fn new(
        in_request: Receiver<Request>,
        out_rsp: Sender<Response>,
        per_channel_latency: u64,
        per_channel_init_interval: u64,
        per_channel_outstanding: usize,
    ) -> Self {
        let ctx = Self {
            in_request,
            out_rsp,
            latency: per_channel_latency,
            init_interval: per_channel_init_interval,
            outstanding: per_channel_outstanding,
            context_info: Default::default(),
        };
        ctx.in_request.attach_receiver(&ctx);
        ctx.out_rsp.attach_sender(&ctx);

        ctx
    }
}

impl Context for HBMChannelContext {
    fn run(&mut self) {
        loop {
            // check if there's enough slot for in-flight requests (self.outstanding)
            // to incorporate this, we might have to move to peek
            match self.in_request.dequeue(&self.time) {
                Ok(ChannelElement {
                    time,
                    data:
                        Request {
                            is_write,
                            address,
                            id,
                        },
                }) => self
                    .out_rsp
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick() + self.latency,
                            data: Response::new(is_write, address, id),
                        },
                    )
                    .unwrap(),
                Err(_) => return,
            }
            self.time.incr_cycles(self.init_interval);
        }
    }
}

#[derive(Constructor)]
pub struct ChannelBundle {
    pub snd: Sender<Request>,
    pub rcv: Receiver<Response>,
}

#[derive(Constructor)]
pub struct ReadBundle {
    pub addr: Receiver<u64>,
    pub resp: Sender<()>,
}

#[derive(Constructor)]
pub struct WriteBundle {
    pub addr: Receiver<u64>,
    pub resp: Sender<()>,
}

#[context_macro]
pub struct HBMContext {
    channels: Vec<ChannelBundle>,
    readers: Vec<ReadBundle>,
    writers: Vec<WriteBundle>,
}

impl Context for HBMContext {
    fn run(&mut self) {
        let channel_num = self.channels.len();
        let mut off_set: usize = 0;

        while self.continue_running() {
            // Collect finished requests in the current cycle and send the response back
            let responses = self.dequeue_responses_at_current_cycle();
            for resp in responses.iter() {
                if resp.is_write {
                    self.writers[resp.id]
                        .resp
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: (),
                            },
                        )
                        .unwrap();
                } else {
                    self.readers[resp.id]
                        .resp
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: (),
                            },
                        )
                        .unwrap();
                }
            }

            // Collect incoming requests in the current cycle and send it to channels
            let requests = self.dequeue_requests_at_current_cycle();

            for (i, request) in requests.into_iter().enumerate() {
                off_set = (off_set + i) % channel_num;

                self.channels[off_set]
                    .snd
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick(),
                            data: request,
                        },
                    )
                    .unwrap();

                // update the offset for the next set of requests
                off_set = (off_set + 1) % channel_num;
            }

            self.time.incr_cycles(1);
        }
    }
}

impl HBMContext {
    pub fn new<'a>(builder: &mut ProgramBuilder<'a>, config: HBMConfig) {
        let mut channels = vec![];
        // Create Channels and attach to the ProgramBuilder
        for _ in 0..config.channel_num {
            let (req_snd, req_rcv) = builder.unbounded();
            let (rsp_snd, rsp_rcv) = builder.unbounded();

            builder.add_child(HBMChannelContext::new(
                req_rcv,
                rsp_snd,
                config.per_channel_latency,
                config.per_channel_init_interval,
                config.per_channel_outstanding,
            ));

            channels.push(ChannelBundle::new(req_snd, rsp_rcv));
        }

        let ctx = Self {
            channels,
            readers: vec![],
            writers: vec![],
            context_info: Default::default(),
        };

        for bundle in ctx.channels.iter() {
            bundle.rcv.attach_receiver(&ctx);
            bundle.snd.attach_sender(&ctx);
        }

        builder.add_child(ctx);
    }

    pub fn add_reader(&mut self, ReadBundle { addr, resp }: ReadBundle) {
        addr.attach_receiver(self);
        resp.attach_sender(self);
        self.readers.push(ReadBundle { addr, resp });
    }

    pub fn add_writerr(&mut self, WriteBundle { addr, resp }: WriteBundle) {
        addr.attach_receiver(self);
        resp.attach_sender(self);
        self.writers.push(WriteBundle { addr, resp });
    }

    fn dequeue_requests_at_current_cycle(&mut self) -> Vec<Request> {
        let mut requests = vec![];

        for (i, reader) in self.readers.iter().enumerate() {
            match reader.addr.peek() {
                PeekResult::Something(ChannelElement {
                    time: _,
                    data: addr,
                }) => {
                    requests.push(Request::new(false, addr, i));
                    reader.addr.dequeue(&self.time).unwrap();
                }
                PeekResult::Nothing(_time) => continue,
                PeekResult::Closed => continue,
            }
        }

        for (i, writer) in self.writers.iter().enumerate() {
            match writer.addr.peek() {
                PeekResult::Something(ChannelElement {
                    time: _,
                    data: addr,
                }) => {
                    requests.push(Request::new(true, addr, i));
                    writer.addr.dequeue(&self.time).unwrap();
                }
                PeekResult::Nothing(_time) => continue,
                PeekResult::Closed => continue,
            }
        }
        requests
    }

    fn dequeue_responses_at_current_cycle(&mut self) -> Vec<Response> {
        let mut responses = vec![];

        for channel in self.channels.iter() {
            match channel.rcv.peek() {
                PeekResult::Something(ChannelElement {
                    time: _,
                    data: addr,
                }) => {
                    responses.push(addr);
                    channel.rcv.dequeue(&self.time).unwrap();
                }
                PeekResult::Nothing(_time) => continue,
                PeekResult::Closed => continue,
            }
        }
        responses
    }

    fn continue_running(&mut self) -> bool {
        // check all of the writers
        let mut writers_done =
            self.writers
                .iter()
                .all(|WriteBundle { addr, resp }| match addr.peek() {
                    PeekResult::Closed => true,
                    _ => false,
                });

        if self.writers.is_empty() {
            writers_done = true;
        }

        if !writers_done {
            return true;
        }

        let readers_done =
            self.readers
                .iter()
                .all(|ReadBundle { addr, resp: _ }| match addr.peek() {
                    PeekResult::Closed => true,
                    _ => false,
                });

        if !readers_done {
            return true;
        }

        false
    }
}
