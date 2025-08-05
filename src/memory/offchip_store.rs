use std::{fs::File, marker::PhantomData};

use dam::context_tools::*;
use dam::logging::LogEvent;
use half::f16;
use itertools::Itertools;
use ndarray::{concatenate, Array2, Axis};

use crate::{
    primitives::elem::{Bufferizable, Elem, StopType},
    ramulator::{access::MemoryData, hbm_context::ParAddrs},
};

use crate::utils::events::LoggableEventSimple;

use crate::primitives::tile::Tile;

#[context_macro]
pub struct OffChipStore<E: LoggableEventSimple, T: DAMType> {
    // Tiling configurations
    pub tensor_shape_tiled: Vec<usize>,
    pub tile_row: usize,
    pub tile_col: usize,
    // Data
    pub store_path: Option<String>,
    // HBM Configurations & Addresses
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub par_dispatch: usize,
    // Sender & Receiver (DAM details)
    pub on_chip_rcv: Receiver<Elem<Tile<T>>>,
    pub addr_snd: Sender<ParAddrs>,
    pub ack_rcv: Receiver<u64>,
    pub id: u32,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType + npyz::AutoSerialize,
    > OffChipStore<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        tensor_shape_tiled: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
        store_path: Option<String>,
        base_addr_byte: u64,
        addr_offset: u64,
        par_dispatch: usize,
        on_chip_rcv: Receiver<Elem<Tile<T>>>,
        addr_snd: Sender<ParAddrs>,
        ack_rcv: Receiver<u64>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            tensor_shape_tiled,
            tile_row,
            tile_col,
            store_path,
            base_addr_byte,
            addr_offset,
            on_chip_rcv,
            par_dispatch,
            addr_snd,
            ack_rcv,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.on_chip_rcv.attach_receiver(&ctx);
        ctx.addr_snd.attach_sender(&ctx);
        ctx.ack_rcv.attach_receiver(&ctx);

        ctx
    }

    pub fn on_chip_req_elems(&self) -> usize {
        self.tile_row * self.tile_col
    }

    pub fn stored_elems(&self) -> usize {
        let total_tiles: usize = self.tensor_shape_tiled.iter().product();
        total_tiles * self.tile_row * self.tile_col
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType + npyz::AutoSerialize,
    > Context for OffChipStore<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        let mut accum: Array2<T> = Array2::from_shape_vec(
            (0, self.tensor_shape_tiled.last().unwrap() * self.tile_col),
            vec![],
        )
        .unwrap();
        let mut horizontal_accum: Array2<T> =
            Array2::from_shape_vec((self.tile_row, 0), vec![]).unwrap();

        let mut tile_idx = 0;
        let mut n_bytes = None;
        loop {
            // Get the tile data and concatenate if you're simulating with actual values
            let tile_data = match self.on_chip_rcv.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: tile,
                }) => match tile {
                    Elem::Val(tile_data) => {
                        if self.store_path.is_some() {
                            assert!(tile_data.underlying.is_some());

                            let concatenated = concatenate(
                                Axis(1),
                                &[
                                    horizontal_accum.view(),
                                    tile_data.underlying.clone().unwrap().view(),
                                ],
                            )
                            .unwrap_or_else(|_| panic!("Error concatenating tiles horizontally"));
                            horizontal_accum = concatenated;
                        }
                        tile_data
                    }
                    Elem::ValStop(tile_data, s) => {
                        if self.store_path.is_some() {
                            assert!(tile_data.underlying.is_some());

                            let concatenated_horizontal = concatenate(
                                Axis(1),
                                &[
                                    horizontal_accum.view(),
                                    tile_data.underlying.clone().unwrap().view(),
                                ],
                            )
                            .unwrap_or_else(|_| panic!("Error concatenating tiles horizontally"));
                            horizontal_accum = concatenated_horizontal;

                            let concatenated =
                                concatenate(Axis(0), &[accum.view(), horizontal_accum.view()])
                                    .unwrap_or_else(|_| {
                                        panic!("Error concatenating tiles horizontally")
                                    });
                            accum = concatenated;

                            horizontal_accum =
                                Array2::from_shape_vec((self.tile_row, 0), vec![]).unwrap();
                        }
                        tile_data
                    }
                },
                Err(_) => {
                    if self.store_path.is_some() {
                        // Save the collected so far and return

                        // Check whether the collected data is same as expected
                        assert_eq!(
                            accum.len(),
                            self.tensor_shape_tiled.iter().product::<usize>()
                                * self.tile_row
                                * self.tile_col
                        );
                        let data: Vec<T> = accum.into_raw_vec_and_offset().0;

                        // Save data in .npy
                        let data_file_path = format!("{}.npy", self.store_path.clone().unwrap());
                        match npyz::to_file_1d(data_file_path, data) {
                            Ok(_) => {}
                            Err(_) => panic!(
                                "Error while writing data to {}",
                                format!("{}.npy", self.store_path.clone().unwrap())
                            ),
                        }

                        // save metadata as json file
                        let total_cols = self.tile_col * self.tensor_shape_tiled.last().unwrap();
                        let total_rows = self.tile_row
                            * self.tensor_shape_tiled[self.tensor_shape_tiled.len() - 2];
                        let mut shape =
                            self.tensor_shape_tiled[..self.tensor_shape_tiled.len() - 2].to_vec();
                        shape.append(&mut vec![total_rows, total_cols]);

                        let meta_file_path: String =
                            format!("{}.json", self.store_path.clone().unwrap());
                        let meta_file = File::create(meta_file_path.clone()).unwrap();
                        match serde_json::to_writer(meta_file, &shape) {
                            Ok(_) => {}
                            Err(_) => panic!("Error while writing metadata to {}", meta_file_path),
                        }

                        println!(
                            "Successfully wrote the output to {}",
                            self.store_path.clone().unwrap()
                        );
                    }
                    return;
                }
            };

            assert_eq!(tile_data.shape[0], self.tile_row);
            assert_eq!(tile_data.shape[1], self.tile_col);

            // Calculate the write addresses for the given tile
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
            for (idx, addr_chunk) in tile_addrs
                .iter()
                .chunks(self.par_dispatch)
                .into_iter()
                .enumerate()
            {
                let chunk_vec: Vec<u64> = addr_chunk.cloned().collect();
                self.addr_snd
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: send_request_time + idx as u64,
                            data: ParAddrs::new(chunk_vec),
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
                "OffChipStore".to_string(),
                self.id,
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

#[context_macro]
pub struct OffChipStoreRamulator<E: LoggableEventSimple, T: DAMType> {
    pub tensor_shape_tiled: Vec<usize>,
    pub tile_row: usize,
    pub tile_col: usize,
    pub store_path: Option<String>,
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub on_chip_rcv: Receiver<Elem<Tile<T>>>,
    pub addr_snd: Sender<u64>,
    pub wdata_snd: Sender<MemoryData>,
    pub ack_rcv: Receiver<bool>,
    pub id: u32,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType + npyz::AutoSerialize,
    > OffChipStoreRamulator<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        tensor_shape_tiled: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
        store_path: Option<String>,
        base_addr_byte: u64,
        addr_offset: u64,
        on_chip_rcv: Receiver<Elem<Tile<T>>>,
        addr_snd: Sender<u64>,
        wdata_snd: Sender<MemoryData>,
        ack_rcv: Receiver<bool>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            tensor_shape_tiled,
            tile_row,
            tile_col,
            store_path,
            base_addr_byte,
            addr_offset,
            on_chip_rcv,
            addr_snd,
            wdata_snd,
            ack_rcv,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.on_chip_rcv.attach_receiver(&ctx);
        ctx.addr_snd.attach_sender(&ctx);
        ctx.wdata_snd.attach_sender(&ctx);
        ctx.ack_rcv.attach_receiver(&ctx);

        ctx
    }

    pub fn on_chip_req_elems(&self) -> usize {
        self.tile_row * self.tile_col
    }

    pub fn stored_elems(&self) -> usize {
        let total_tiles: usize = self.tensor_shape_tiled.iter().product();
        total_tiles * self.tile_row * self.tile_col
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType + npyz::AutoSerialize,
    > Context for OffChipStoreRamulator<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        let mut accum: Array2<T> = Array2::from_shape_vec(
            (0, self.tensor_shape_tiled.last().unwrap() * self.tile_col),
            vec![],
        )
        .unwrap();
        let mut horizontal_accum: Array2<T> =
            Array2::from_shape_vec((self.tile_row, 0), vec![]).unwrap();

        let mut tile_idx = 0;
        let mut n_bytes = None;
        loop {
            // Get the tile data and concatenate if you're simulating with actual values
            let tile_data = match self.on_chip_rcv.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: tile,
                }) => match tile {
                    Elem::Val(tile_data) => {
                        if self.store_path.is_some() {
                            assert!(tile_data.underlying.is_some());

                            let concatenated = concatenate(
                                Axis(1),
                                &[
                                    horizontal_accum.view(),
                                    tile_data.underlying.clone().unwrap().view(),
                                ],
                            )
                            .unwrap_or_else(|_| panic!("Error concatenating tiles horizontally"));
                            horizontal_accum = concatenated;
                        }
                        tile_data
                    }
                    Elem::ValStop(tile_data, _) => {
                        if self.store_path.is_some() {
                            assert!(tile_data.underlying.is_some());

                            let concatenated_horizontal = concatenate(
                                Axis(1),
                                &[
                                    horizontal_accum.view(),
                                    tile_data.underlying.clone().unwrap().view(),
                                ],
                            )
                            .unwrap_or_else(|_| panic!("Error concatenating tiles horizontally"));
                            horizontal_accum = concatenated_horizontal;

                            let concatenated =
                                concatenate(Axis(0), &[accum.view(), horizontal_accum.view()])
                                    .unwrap_or_else(|_| {
                                        panic!("Error concatenating tiles horizontally")
                                    });
                            accum = concatenated;

                            horizontal_accum =
                                Array2::from_shape_vec((self.tile_row, 0), vec![]).unwrap();
                        }
                        tile_data
                    }
                },
                Err(_) => {
                    if self.store_path.is_some() {
                        // Save the collected so far and return

                        // Check whether the collected data is same as expected
                        assert_eq!(
                            accum.len(),
                            self.tensor_shape_tiled.iter().product::<usize>()
                                * self.tile_row
                                * self.tile_col
                        );
                        let data: Vec<T> = accum.into_raw_vec_and_offset().0;

                        // Save data in .npy
                        let data_file_path = format!("output.npy");
                        match npyz::to_file_1d(self.store_path.clone().unwrap(), data) {
                            Ok(_) => {}
                            Err(_) => panic!("Error while writing data to {}", data_file_path),
                        }

                        // save metadata as json file
                        let total_cols = self.tile_col * self.tensor_shape_tiled.last().unwrap();
                        let total_rows = self.tile_row
                            * self.tensor_shape_tiled[self.tensor_shape_tiled.len() - 2];
                        let mut shape =
                            self.tensor_shape_tiled[..self.tensor_shape_tiled.len() - 2].to_vec();
                        shape.append(&mut vec![total_rows, total_cols]);

                        let meta_file_path: String = format!("output.json");
                        let meta_file = File::create(meta_file_path.clone()).unwrap();
                        match serde_json::to_writer(meta_file, &shape) {
                            Ok(_) => {}
                            Err(_) => panic!("Error while writing metadata to {}", meta_file_path),
                        }

                        println!("Successfully wrote the output");
                    }
                    return;
                }
            };

            assert_eq!(tile_data.shape[0], self.tile_row);
            assert_eq!(tile_data.shape[1], self.tile_col);

            // Calculate the write addresses for the given tile
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
                "OffChipStore".to_string(),
                self.id,
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
