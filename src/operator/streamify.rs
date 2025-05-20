use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;

use crate::{
    memory::{data::Tile, events::LoggableEventSimple},
    primitives::{
        buffer::Buffer,
        elem::{Elem, StopType},
    },
    ramulator::access::MemoryData,
};

pub enum HbmAddrEnum {
    ADDR(u64),
    STOP(StopType),
}

#[context_macro]
pub struct Streamify<E: LoggableEventSimple> {
    pub in_stream: Receiver<Elem<Buffer>>,
    pub out_stream: Sender<Elem<Tile>>,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Streamify<E> {
    pub fn new(in_stream: Receiver<Elem<Buffer>>, out_stream: Sender<Elem<Tile>>) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Context
    for Streamify<E>
{
    fn run(&mut self) {
        for addr_enum in self.generate_addr() {
            match addr_enum {
                HbmAddrEnum::ADDR(addr) => {
                    // Send read request to HBM
                    let send_request_time = self.time.tick();
                    self.addr_snd
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: send_request_time,
                                data: addr,
                            },
                        )
                        .unwrap();

                    // Wait until you get back the response
                    self.resp_addr_rcv.dequeue(&self.time).unwrap();
                    let read_finish_time = self.time.tick();

                    // Send the data to on-chip
                    // To properly the backpressure under the double buffering setting,
                    // this channel should have a depth of 1
                    self.on_chip_snd
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: read_finish_time,
                                data: Elem::Val(Tile {
                                    shape: vec![self.tile_row, self.tile_col],
                                    bytes_per_elem: self.n_byte,
                                    read_from_mu: true,
                                }),
                            },
                        )
                        .unwrap();

                    dam::logging::log_event(&E::new(
                        send_request_time.time(),
                        read_finish_time.time(),
                        false,
                    ))
                    .unwrap();
                }
                HbmAddrEnum::STOP(level) => {
                    self.on_chip_snd
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick() + (level as u64),
                                data: Elem::Stop(level),
                            },
                        )
                        .unwrap();
                }
            }
        }
    }
}
