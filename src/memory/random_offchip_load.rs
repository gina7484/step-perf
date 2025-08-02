use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;
use itertools::Itertools;
use ndarray::{IntoDimension, IxDyn, IxDynImpl};

use crate::primitives::elem::Elem;
use crate::primitives::tile::Tile;
use crate::ramulator::hbm_context::ParAddrs;
use crate::utils::events::LoggableEventSimple;

#[context_macro]
pub struct RandomOffChipLoad<E: LoggableEventSimple, T: DAMType> {
    // Tiling configurations
    pub tensor_shape_tiled: Vec<usize>, // In terms of tiles.
    pub underlying: Option<ndarray::ArcArray<T, IxDyn>>,
    pub tile_row: usize,
    pub tile_col: usize,
    pub n_byte: usize, // size of the datatype
    // Pre-computed tiles for efficient access
    pub pre_computed_tiles: Option<Vec<Tile<T>>>,
    // HBM Configurations & Addresses
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub par_dispatch: usize,
    // Channels facing HBM memory
    pub addr_snd: Sender<ParAddrs>,
    pub resp_addr_rcv: Receiver<u64>,
    // Channel facing on-chip memory
    pub raddr: Receiver<Elem<Tile<u64>>>,
    pub rdata: Sender<Elem<Tile<T>>>,
    pub id: u32,
    // Phantom data for the event type
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: npyz::Deserialize + DAMType,
    > RandomOffChipLoad<E, T>
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
        addr_snd: Sender<ParAddrs>,
        resp_addr_rcv: Receiver<u64>,
        raddr: Receiver<Elem<Tile<u64>>>,
        rdata: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        let underlying = match npy_path {
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

        // Pre-compute all tiles for efficient access
        let pre_computed_tiles = match &underlying {
            Some(arr) => {
                let ndim = arr.ndim();

                // Create window size and stride vectors, starting with all 1s
                let mut window_size = vec![1; ndim];
                let mut stride = vec![1; ndim];

                // Set the first two dimensions for tiling
                window_size[ndim - 2] = tile_row;
                stride[ndim - 2] = tile_row;

                window_size[ndim - 1] = tile_col;
                stride[ndim - 1] = tile_col;

                // Collect all tiles
                let tiles: Vec<_> = arr
                    .windows_with_stride(IxDyn(&window_size), IxDyn(&stride))
                    .into_iter()
                    .map(|tile_data| {
                        Tile::new(
                            tile_data
                                .to_shared()
                                .into_shape_with_order((tile_row, tile_col))
                                .unwrap(),
                            n_byte,
                            true,
                        )
                    })
                    .collect();

                Some(tiles)
            }
            None => None,
        };

        let ctx = Self {
            tensor_shape_tiled,
            underlying,
            tile_row,
            tile_col,
            n_byte,
            pre_computed_tiles,
            base_addr_byte,
            addr_offset,
            par_dispatch,
            addr_snd,
            resp_addr_rcv,
            raddr,
            rdata,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.addr_snd.attach_sender(&ctx);
        ctx.resp_addr_rcv.attach_receiver(&ctx);
        ctx.raddr.attach_receiver(&ctx);
        ctx.rdata.attach_sender(&ctx);

        ctx
    }

    /// Generate addresses for a specific tile index
    fn generate_tile_addresses(&self, tile_idx: u64) -> Vec<u64> {
        // Calculate the base address for this tile
        let tile_offset = self.tile_row * self.tile_col * self.n_byte;
        let base_addr_i = self.base_addr_byte + (tile_idx * tile_offset as u64);
        let row_offset = self.tensor_shape_tiled.last().unwrap() * self.tile_col * self.n_byte;

        // Generate all addresses for this tile
        let mut tile_addrs = vec![];
        for r in 0..self.tile_row {
            for c in (0..(self.tile_col * self.n_byte)).step_by(self.addr_offset as usize) {
                let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                tile_addrs.push(addr);
            }
        }

        tile_addrs
    }

    /// Create tile data for a specific tile index
    fn create_tile_data(&self, tile_idx: u64) -> Tile<T> {
        let tile_idx_usize = tile_idx as usize;

        match &self.pre_computed_tiles {
            Some(tiles) => {
                if tile_idx_usize < tiles.len() {
                    tiles[tile_idx_usize].clone()
                } else {
                    // Fallback to blank tile if index is out of bounds
                    Tile::new_blank(vec![self.tile_row, self.tile_col], self.n_byte, true)
                }
            }
            None => {
                // Create blank tile when no underlying data is available
                Tile::new_blank(vec![self.tile_row, self.tile_col], self.n_byte, true)
            }
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: npyz::Deserialize + DAMType,
    > Context for RandomOffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        // Process requests from raddr until we get a stop signal
        while let Ok(addr_elem) = self.raddr.dequeue(&self.time) {
            match addr_elem.data {
                Elem::Val(addr_tile) => {
                    // Generate addresses for the requested tile
                    let tile_arr = addr_tile.underlying.as_ref().unwrap();
                    let tile_idx = tile_arr[[0, 0]];
                    let tile_addrs = self.generate_tile_addresses(tile_idx);

                    // Send read request to HBM
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

                    // Wait for all responses
                    for _i in &tile_addrs {
                        self.resp_addr_rcv.dequeue(&self.time).unwrap();
                    }

                    let read_finish_time = self.time.tick();

                    // Log the event
                    dam::logging::log_event(&E::new(
                        "RandomOffChipLoad".to_string(),
                        self.id,
                        send_request_time.time(),
                        read_finish_time.time(),
                        false,
                    ))
                    .unwrap();

                    // Create the tile data
                    let tile_data = self.create_tile_data(tile_idx);

                    // Send the tile data
                    self.rdata
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: Elem::Val(tile_data),
                            },
                        )
                        .unwrap();
                }
                Elem::ValStop(addr_tile, stop_level) => {
                    // Generate addresses for the requested tile
                    let tile_arr = addr_tile.underlying.as_ref().unwrap();
                    let tile_idx: u64 = tile_arr[[0, 0]];
                    let tile_addrs = self.generate_tile_addresses(tile_idx);

                    // Send read request to HBM
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

                    // Wait for all responses
                    for _i in &tile_addrs {
                        self.resp_addr_rcv.dequeue(&self.time).unwrap();
                    }

                    let read_finish_time = self.time.tick();

                    // Log the event
                    dam::logging::log_event(&E::new(
                        "RandomOffChipLoad".to_string(),
                        self.id,
                        send_request_time.time(),
                        read_finish_time.time(),
                        true,
                    ))
                    .unwrap();

                    // Create the tile data
                    let tile_data = self.create_tile_data(tile_idx);

                    // Send the tile data with stop signal
                    self.rdata
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: Elem::ValStop(tile_data, stop_level),
                            },
                        )
                        .unwrap();
                }
            }
        }
    }
}
