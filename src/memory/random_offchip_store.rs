use std::fs::File;
use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;
use itertools::Itertools;
use ndarray::{IntoDimension, IxDyn, IxDynImpl};

use crate::primitives::elem::{Bufferizable, Elem};
use crate::primitives::tile::Tile;
use crate::ramulator::hbm_context::ParAddrs;
use crate::utils::events::LoggableEventSimple;

#[context_macro]
pub struct RandomOffChipStore<E: LoggableEventSimple, T: DAMType> {
    // Tiling configurations
    pub tensor_shape_tiled: Vec<usize>, // In terms of tiles.
    pub npy_path: Option<String>,
    pub underlying: Option<ndarray::ArcArray<T, IxDyn>>,
    pub tile_row: usize,
    pub tile_col: usize,
    pub n_byte: usize, // size of the datatype
    // HBM Configurations & Addresses
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub par_dispatch: usize,
    // Channels facing HBM memory
    pub addr_snd: Sender<ParAddrs>,
    pub ack_rcv: Receiver<u64>,
    // Channel facing on-chip memory
    pub waddr: Receiver<Elem<Tile<u64>>>,
    pub wdata: Receiver<Elem<Tile<T>>>,
    pub wack: Sender<Elem<bool>>,
    pub ack_based_on_waddr: bool, // if true, the ack stream's shape will be based on the waddr,
    // otherwise it is based on the wdata.
    pub id: u32,
    // Phantom data for the event type
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType + npyz::Deserialize + npyz::AutoSerialize,
    > RandomOffChipStore<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        tensor_shape_tiled: Vec<usize>,
        npy_path: Option<String>,
        tile_row: usize,
        tile_col: usize,
        n_byte: usize,
        base_addr_byte: u64,
        addr_offset: u64,
        par_dispatch: usize,
        // HBM context facing the channels
        addr_snd: Sender<ParAddrs>,
        ack_rcv: Receiver<u64>,
        // On-chip memory facing the channels
        waddr: Receiver<Elem<Tile<u64>>>,
        wdata: Receiver<Elem<Tile<T>>>,
        wack: Sender<Elem<bool>>,
        id: u32,
        ack_based_on_waddr: bool,
    ) -> Self {
        let underlying = match npy_path.clone() {
            Some(file_path) => {
                // Open the file
                let mut file = std::fs::File::open(file_path).unwrap();

                // Read the data and shape of the `.npy` file
                let file_data = npyz::NpyFile::new(&mut file).unwrap();
                let shape_vec = file_data
                    .shape()
                    .iter()
                    .map(|x| *x as usize)
                    .collect::<Vec<usize>>();

                let total_cols = tile_col * tensor_shape_tiled.last().unwrap();
                let total_rows = tile_row * tensor_shape_tiled[tensor_shape_tiled.len() - 2];
                let mut untiled_shape = tensor_shape_tiled[..tensor_shape_tiled.len() - 2].to_vec();
                untiled_shape.append(&mut vec![total_rows, total_cols]);

                assert_eq!(untiled_shape, shape_vec);

                let shape: ndarray::Dim<IxDynImpl> = shape_vec.into_dimension();

                let vec_data: Vec<T> = file_data.into_vec().unwrap();
                Some(ndarray::ArcArray::from_shape_vec(shape, vec_data).unwrap())
            }
            None => None,
        };
        assert_eq!(
            tensor_shape_tiled.len(),
            2,
            "Only 2D tensors are supported for now in RandomOffChipStore"
        );

        let ctx = Self {
            tensor_shape_tiled,
            npy_path,
            underlying,
            tile_row,
            tile_col,
            n_byte,
            base_addr_byte,
            addr_offset,
            par_dispatch,
            addr_snd,
            ack_rcv,
            waddr,
            wdata,
            wack,
            id,
            ack_based_on_waddr,
            _phantom: PhantomData,
            context_info: Default::default(),
        };
        ctx.waddr.attach_receiver(&ctx);
        ctx.wdata.attach_receiver(&ctx);
        ctx.addr_snd.attach_sender(&ctx);
        ctx.ack_rcv.attach_receiver(&ctx);
        ctx.wack.attach_sender(&ctx);
        ctx
    }

    fn send_write_request(&mut self, waddr: u64, wdata: &Tile<T>) {
        // Calculate the write addresses for the given tile
        let n_bytes = wdata.bytes_per_elem;

        let tile_offset = wdata.size_in_bytes();
        let base_addr_i = self.base_addr_byte + (waddr * tile_offset as u64);
        let row_offset = self.tensor_shape_tiled.last().unwrap() * self.tile_col * n_bytes;

        let mut tile_addrs = vec![];
        for r in 0..self.tile_row {
            for c in (0..(self.tile_col * n_bytes)).step_by(self.addr_offset as usize) {
                let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                tile_addrs.push(addr);
            }
        }

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
            "RandomOffChipStore".to_string(),
            self.id,
            send_request_time.time(),
            read_finish_time.time(),
            false,
        ))
        .unwrap();
    }

    fn update_underlying(&mut self, waddr: u64, wdata: Tile<T>) {
        match &mut self.underlying {
            Some(tensor) => {
                assert!(wdata.underlying.is_some());

                // Calculate the tile index in the tensor
                let tile_idx = waddr as usize;

                // Calculate the total number of tiles in each dimension
                let total_tiles_col = self.tensor_shape_tiled.last().unwrap();
                let total_tiles_row = self.tensor_shape_tiled[self.tensor_shape_tiled.len() - 2];

                // Calculate the tile position in the 2D grid of tiles
                let tile_row_idx = tile_idx / total_tiles_col;
                let tile_col_idx = tile_idx % total_tiles_col;

                // Calculate the starting position in the underlying tensor
                let start_row = tile_row_idx * self.tile_row;
                let start_col = tile_col_idx * self.tile_col;

                // Get the tile data from wdata
                let tile_data = wdata.underlying.as_ref().unwrap();

                // Update the corresponding region in the underlying tensor
                let mut tile_slice = tensor.slice_mut(ndarray::s![
                    start_row..start_row + self.tile_row,
                    start_col..start_col + self.tile_col
                ]);

                // Copy the tile data to the tensor
                tile_slice.assign(tile_data);
            }
            None => return,
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType + npyz::Deserialize + npyz::AutoSerialize,
    > Context for RandomOffChipStore<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        loop {
            let peek_waddr = self.waddr.peek_next(&self.time);
            let peek_wdata = self.wdata.peek_next(&self.time);

            match (peek_waddr, peek_wdata) {
                (
                    Ok(ChannelElement {
                        time: _,
                        data: waddr_tile,
                    }),
                    Ok(ChannelElement {
                        time: _,
                        data: wdata,
                    }),
                ) => {
                    match (waddr_tile, wdata) {
                        (Elem::Val(waddr_tile), Elem::Val(wdata)) => {
                            let waddr = waddr_tile.underlying.as_ref().unwrap()[[0, 0]];
                            // Send write request to HBM
                            self.send_write_request(waddr, &wdata);

                            // Update the tensor if underlying is not None
                            self.update_underlying(waddr, wdata);

                            self.wack
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(true),
                                    },
                                )
                                .unwrap();
                        }
                        (
                            Elem::ValStop(waddr_tile, waddr_stop),
                            Elem::ValStop(wdata, wdata_stop),
                        ) => {
                            let waddr = waddr_tile.underlying.as_ref().unwrap()[[0, 0]];
                            // Send write request to HBM
                            self.send_write_request(waddr, &wdata);

                            // Update the tensor if underlying is not None
                            self.update_underlying(waddr, wdata);

                            let stop_level = if self.ack_based_on_waddr {
                                waddr_stop
                            } else {
                                wdata_stop
                            };

                            self.wack
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(true, stop_level),
                                    },
                                )
                                .unwrap();
                        }
                        (Elem::Val(waddr_tile), Elem::ValStop(wdata, wdata_stop)) => {
                            let waddr = waddr_tile.underlying.as_ref().unwrap()[[0, 0]];
                            // Send write request to HBM
                            self.send_write_request(waddr, &wdata);

                            // Update the tensor if underlying is not None
                            self.update_underlying(waddr, wdata);

                            let out_elem = if self.ack_based_on_waddr {
                                Elem::Val(true)
                            } else {
                                Elem::ValStop(true, wdata_stop)
                            };

                            self.wack
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: out_elem,
                                    },
                                )
                                .unwrap();
                        }
                        (Elem::ValStop(waddr_tile, waddr_stop), Elem::Val(wdata)) => {
                            let waddr = waddr_tile.underlying.as_ref().unwrap()[[0, 0]];
                            // Send write request to HBM
                            self.send_write_request(waddr, &wdata);

                            // Update the tensor if underlying is not None
                            self.update_underlying(waddr, wdata);

                            let out_elem = if self.ack_based_on_waddr {
                                Elem::ValStop(true, waddr_stop)
                            } else {
                                Elem::Val(true)
                            };

                            self.wack
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: out_elem,
                                    },
                                )
                                .unwrap();
                        }
                    }
                }
                (Err(_), Err(_)) => {
                    if self.npy_path.is_some() {
                        // Save data in .npy
                        let data_file_path = format!("{}.npy", self.npy_path.clone().unwrap());
                        match npyz::to_file_1d(
                            data_file_path,
                            self.underlying
                                .as_ref()
                                .unwrap()
                                .as_slice()
                                .unwrap()
                                .to_vec(),
                        ) {
                            Ok(_) => {}
                            Err(_) => panic!(
                                "Error while writing data to {}",
                                format!("{}.npy", self.npy_path.clone().unwrap())
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
                            format!("{}.json", self.npy_path.clone().unwrap());
                        let meta_file = File::create(meta_file_path.clone()).unwrap();
                        match serde_json::to_writer(meta_file, &shape) {
                            Ok(_) => {}
                            Err(_) => panic!("Error while writing metadata to {}", meta_file_path),
                        }

                        println!(
                            "Successfully wrote the output to {}",
                            self.npy_path.clone().unwrap()
                        );
                    }
                    return;
                }
                _ => {
                    panic!("Invalid write address or data");
                }
            }

            self.waddr.dequeue(&self.time).unwrap();
            self.wdata.dequeue(&self.time).unwrap();
        }
    }
}
