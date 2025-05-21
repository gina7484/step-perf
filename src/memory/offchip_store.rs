use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;
use half::f16;

use crate::{
    primitives::elem::{Elem, StopType},
    ramulator::access::MemoryData,
};

use super::{data::Tile, events::LoggableEventSimple};

#[context_macro]
pub struct OffChipStore<E: LoggableEventSimple> {
    pub tensor_shape_tiled: Vec<usize>,
    pub tile_row: usize,
    pub tile_col: usize,
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub on_chip_rcv: Receiver<Elem<Tile>>,
    pub addr_snd: Sender<u64>,
    pub wdata_snd: Sender<MemoryData>,
    pub ack_rcv: Receiver<bool>,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> OffChipStore<E> {
    pub fn new(
        tensor_shape_tiled: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
        base_addr_byte: u64,
        addr_offset: u64,
        on_chip_rcv: Receiver<Elem<Tile>>,
        addr_snd: Sender<u64>,
        wdata_snd: Sender<MemoryData>,
        ack_rcv: Receiver<bool>,
    ) -> Self {
        let ctx = Self {
            tensor_shape_tiled,
            tile_row,
            tile_col,
            base_addr_byte,
            addr_offset,
            on_chip_rcv,
            addr_snd,
            wdata_snd,
            ack_rcv,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.on_chip_rcv.attach_receiver(&ctx);
        ctx.addr_snd.attach_sender(&ctx);
        ctx.wdata_snd.attach_sender(&ctx);
        ctx.ack_rcv.attach_receiver(&ctx);

        ctx
    }
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Context
    for OffChipStore<E>
{
    fn run(&mut self) {
        let mut tile_idx = 0;
        let mut n_bytes = None;
        loop {
            let tile_data = match self.on_chip_rcv.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: tile,
                }) => match tile {
                    Elem::Val(tile_data) => tile_data,
                    Elem::ValStop(tile_data, _) => tile_data,
                },
                Err(_) => return,
            };

            // Calculate the write addresses for the given tile
            assert_eq!(tile_data.shape[0], self.tile_row);
            assert_eq!(tile_data.shape[1], self.tile_col);
            if n_bytes == None {
                n_bytes = Some(tile_data.bytes_per_elem);
            } else {
                assert_eq!(n_bytes.unwrap(), tile_data.bytes_per_elem);
            }

            let tile_offset = tile_data.size_in_bytes();
            let base_addr_i = self.base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset =
                self.tensor_shape_tiled.last().unwrap() * self.tile_col * n_bytes.unwrap();

            let mut tile_addrs = vec![];
            for r in 0..self.tile_row {
                for c in (0..(self.tile_col * n_bytes.unwrap())).step_by(self.addr_offset as usize)
                {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    tile_addrs.push(addr);
                }
            }

            tile_idx += 1;

            // Send write request to HBM
            let send_request_time = self.time.tick();
            for (idx, addr) in tile_addrs.iter().enumerate() {
                self.addr_snd
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: send_request_time + idx as u64,
                            data: *addr,
                        },
                    )
                    .unwrap();

                self.wdata_snd
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: send_request_time + idx as u64,
                            data: MemoryData::F16([f16::from_f32(0.0); 32]),
                        },
                    )
                    .unwrap();
            }

            // Wait until you get back the response
            for _i in tile_addrs {
                self.ack_rcv.dequeue(&self.time).unwrap();
            }

            let read_finish_time = self.time.tick();

            dam::logging::log_event(&E::new(
                send_request_time.time(),
                read_finish_time.time(),
                false,
            ))
            .unwrap();

            // dequeue
            self.on_chip_rcv.dequeue(&self.time).unwrap();
        }
    }
}
