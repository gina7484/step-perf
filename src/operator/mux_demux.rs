use dam::context_tools::*;
use dam::dam_macros::event_type;
use serde::{Deserialize, Serialize};

use crate::{memory::data::Tile, primitives::elem::Elem};

#[context_macro]
pub struct TileDemux {
    in_stream: Receiver<Elem<Tile>>,
    out_streams: Vec<Sender<Elem<Tile>>>,
}

impl TileDemux {
    pub fn new(in_stream: Receiver<Elem<Tile>>, out_streams: Vec<Sender<Elem<Tile>>>) -> Self {
        let ctx = Self {
            in_stream,
            out_streams,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        for out_stream in ctx.out_streams.iter() {
            out_stream.attach_sender(&ctx);
        }

        ctx
    }
}

impl Context for TileDemux {
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => {
                        assert_eq!(data.shape.len(), 2);
                        assert_eq!(self.out_streams.len(), data.shape[0]);

                        for out_stream in self.out_streams.iter() {
                            out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(), // No latency added. Treated like a swtich box op.
                                        data: Elem::Val(Tile {
                                            shape: vec![data.shape[1]],
                                            bytes_per_elem: data.bytes_per_elem,
                                            read_from_mu: data.read_from_mu,
                                        }),
                                    },
                                )
                                .unwrap();
                        }
                    }
                    Elem::Stop(lev) => {
                        for out_stream in self.out_streams.iter() {
                            out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Stop(lev),
                                    },
                                )
                                .unwrap();
                        }
                    }
                },
                Err(_) => todo!(),
            }
        }
    }
}
