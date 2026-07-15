use serde_json;
use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;
use itertools::Itertools;
use ndarray::{IntoDimension, IxDyn, IxDynImpl};

use crate::primitives::elem::Elem;
use crate::ramulator::hbm_context::ParAddrs;

use crate::memory::HbmAddrEnum;
use crate::primitives::tile::Tile;
use crate::utils::events::LoggableEventSimple;

#[context_macro]
pub struct DynLinearOffChipLoad<E: LoggableEventSimple, T: DAMType> {
    pub tensor_shape_tiled: Vec<usize>, // In terms of tiles.
    pub underlying: Option<ndarray::ArcArray<T, IxDyn>>,
    // Tiling configurations
    pub tile_row: usize,
    pub tile_col: usize,
    pub n_byte: usize, // size of the datatype
    // HBM Configurations & Addresses
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub par_dispatch: usize,
    // Sender & Receiver (DAM details)
    pub addr_snd: Sender<ParAddrs>,
    pub resp_addr_rcv: Receiver<u64>,
    pub on_chip_snd: Sender<Elem<Tile<T>>>,
    pub id: u32,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: npyz::Deserialize + DAMType,
    > DynLinearOffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        shape_path: String, // path to the json file with the untiled_shape
        npy_path: String,
        tile_row: usize,
        tile_col: usize,
        n_byte: usize,
        base_addr_byte: u64,
        addr_offset: u64,
        par_dispatch: usize,
        addr_snd: Sender<ParAddrs>,
        resp_addr_rcv: Receiver<u64>,
        on_chip_snd: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        // Read the shape from the JSON file
        let shape_file = std::fs::File::open(&shape_path)
            .unwrap_or_else(|_| panic!("Failed to open shape file: {}", shape_path));
        let untiled_shape: Vec<usize> = serde_json::from_reader(shape_file)
            .unwrap_or_else(|_| panic!("Failed to parse shape JSON from: {}", shape_path));

        let underlying = match std::fs::File::open(npy_path) {
            Ok(mut file) => {
                let file_data = npyz::NpyFile::new(&mut file).unwrap();

                let shape: ndarray::Dim<IxDynImpl> = untiled_shape.clone().into_dimension();

                let vec_data: Vec<T> = file_data.into_vec().unwrap();
                Some(ndarray::ArcArray::from_shape_vec(shape, vec_data).unwrap())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None, // Just timing simulation
            Err(error) => {
                // The file may exist, but opening it failed for another reason,
                // such as insufficient permissions.
                panic!("Failed to open file: {error}");
            }
        };

        let mut tensor_shape_tiled: Vec<usize> = untiled_shape.clone();
        if tensor_shape_tiled.len() >= 2 {
            let last_idx = tensor_shape_tiled.len() - 1;
            let second_last_idx = tensor_shape_tiled.len() - 2;
            tensor_shape_tiled[second_last_idx] /= tile_row;
            tensor_shape_tiled[last_idx] /= tile_col;
        } else {
            panic!("Tensor shape tiled must have at least 2 dimensions");
        }

        let ctx = Self {
            tensor_shape_tiled,
            underlying,
            tile_row,
            tile_col,
            n_byte,
            base_addr_byte,
            addr_offset,
            par_dispatch,
            addr_snd,
            resp_addr_rcv,
            on_chip_snd,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.addr_snd.attach_sender(&ctx);
        ctx.resp_addr_rcv.attach_receiver(&ctx);
        ctx.on_chip_snd.attach_sender(&ctx);

        ctx
    }

    fn generate_addr(&self) -> impl Iterator<Item = HbmAddrEnum<T>> {
        let mut tile_data = vec![];
        // Tile the actual data
        match &self.underlying {
            Some(arr) => {
                let ndim = arr.ndim();

                // Create window size and stride vectors, starting with all 1s
                let mut window_size = vec![1; ndim];
                let mut stride = vec![1; ndim];

                // Set the first two dimensions for tiling
                window_size[ndim - 2] = self.tile_row;
                stride[ndim - 2] = self.tile_row;

                window_size[ndim - 1] = self.tile_col;
                stride[ndim - 1] = self.tile_col;

                // Remaining dimensions keep size/stride of 1 (as you suggested)

                for tile_i in arr.windows_with_stride(IxDyn(&window_size), IxDyn(&stride)) {
                    tile_data.push(Tile::new(
                        tile_i
                            .to_shared()
                            .into_shape_with_order((self.tile_row, self.tile_col))
                            .unwrap(),
                        self.n_byte,
                        true,
                    ))
                }
            }
            None => {}
        };

        // Calculate total elements in the output tensor
        let total_tiles: usize = self.tensor_shape_tiled.iter().product();

        // Create a vector to hold all the addresses
        let mut addrs: Vec<HbmAddrEnum<T>> = vec![];

        for flat_idx in 0..total_tiles {
            // Convert flat index to multi-dimensional indices
            let mut remaining = flat_idx;
            let mut multi_index = vec![0; self.tensor_shape_tiled.len()];

            // Calculate multi-dimensional indices
            for i in (0..self.tensor_shape_tiled.len()).rev() {
                multi_index[i] = remaining % self.tensor_shape_tiled[i];
                remaining /= self.tensor_shape_tiled[i];
            }

            // For identity view, just use the flat index directly
            let mut tile_idx = flat_idx;

            // Ensure we don't go out of bounds of the original tensor
            let original_size: usize = self.tensor_shape_tiled.iter().product();
            if original_size > 0 {
                tile_idx = tile_idx % original_size;
            } else {
                return Vec::new().into_iter();
            }
            // println!("tile_idx: {}", tile_idx);

            // Generate addresses to fetch the given tile
            let tile_offset = self.tile_row * self.tile_col * self.n_byte;
            let base_addr_i = self.base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset = self.tensor_shape_tiled.last().unwrap() * self.tile_col * self.n_byte;

            // Generate all addresses for this tile
            let mut tile_addrs = vec![];
            for r in 0..self.tile_row {
                for c in (0..(self.tile_col * self.n_byte)).step_by(self.addr_offset as usize) {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    tile_addrs.push(addr);
                }
            }

            // Determine the highest-dimensional stop token needed
            let mut highest_stop_token: Option<u32> = None;
            let mut all_inner_dims_at_end = true;

            // Check from innermost to outermost
            for dim in (0..self.tensor_shape_tiled.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = self.tensor_shape_tiled[dim] == 1;
                    let is_last_elem = multi_index[dim] == self.tensor_shape_tiled[dim] - 1;

                    // If at end or dim size is 1, update the highest stop token
                    if is_last_elem || is_dim_size_one {
                        highest_stop_token = Some((self.tensor_shape_tiled.len() - dim) as u32);
                    }

                    // Update tracking for outer dimensions
                    // Only continue checking outer dimensions if this one is at its last element
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            match self.underlying {
                Some(_) => {
                    // Add the addresses to the result list
                    if !tile_addrs.is_empty() {
                        if let Some(stop_type) = highest_stop_token {
                            // If there's a stop token, add all addresses except the last one
                            addrs.push(HbmAddrEnum::ADDRSTOP(
                                tile_addrs,
                                tile_data[tile_idx].clone(),
                                stop_type,
                            ));
                        } else {
                            // No stop token, add all addresses normally
                            addrs.push(HbmAddrEnum::ADDR(tile_addrs, tile_data[tile_idx].clone()));
                        }
                    }
                }
                None => {
                    if !tile_addrs.is_empty() {
                        if let Some(stop_type) = highest_stop_token {
                            // If there's a stop token, add all addresses except the last one
                            addrs.push(HbmAddrEnum::ADDRSTOP(
                                tile_addrs,
                                Tile::new_blank(
                                    vec![self.tile_row, self.tile_col],
                                    self.n_byte,
                                    true,
                                ),
                                stop_type,
                            ));
                        } else {
                            // No stop token, add all addresses normally
                            addrs.push(HbmAddrEnum::ADDR(
                                tile_addrs,
                                Tile::new_blank(
                                    vec![self.tile_row, self.tile_col],
                                    self.n_byte,
                                    true,
                                ),
                            ));
                        }
                    }
                }
            }
        }

        addrs.into_iter()
    }

    pub fn on_chip_req_elems(&self) -> usize {
        self.tile_row * self.tile_col
    }

    pub fn loaded_elems(&self) -> usize {
        let total_tiles: usize = self.tensor_shape_tiled.iter().product();
        total_tiles * self.tile_row * self.tile_col
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: npyz::Deserialize + DAMType,
    > Context for DynLinearOffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        for addr_enum in self.generate_addr() {
            let (tile_addrs, elem_tile, is_stop) = match addr_enum {
                HbmAddrEnum::ADDR(addrs, tile) => (addrs, Elem::Val(tile), false),
                HbmAddrEnum::ADDRSTOP(addrs, tile, level) => {
                    (addrs, Elem::ValStop(tile, level), true)
                }
            };

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

            for _i in tile_addrs {
                // Wait until you get back the response
                self.resp_addr_rcv.dequeue(&self.time).unwrap();
            }

            let read_finish_time = self.time.tick();

            dam::logging::log_event(&E::new(
                "DynLinearOffChipLoad".to_string(),
                self.id,
                send_request_time.time(),
                read_finish_time.time(),
                is_stop,
            ))
            .unwrap();

            // Send the data to on-chip
            // To properly the backpressure under the double buffering setting,
            // this channel should have a depth of 1

            self.on_chip_snd
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem_tile,
                    },
                )
                .unwrap();
        }
    }
}

// Mongodb: logging for the time used to load each tile

#[cfg(test)]
mod test {

    use std::sync::Arc;

    use dam::{simulation::ProgramBuilder, utility_contexts::ApproxCheckerContext};

    use ndarray::ArcArray;

    use crate::{
        memory::dyn_linear_offchip_load::DynLinearOffChipLoad,
        primitives::{buffer::Buffer, tile::Tile},
        ramulator::hbm_context::{HBMConfig, HBMContext, ReadBundle},
        utils::events::SimpleEvent,
    };

    #[test]
    fn round_trip_test_4d() {
        // matrix: [2,2]
        // output = [1,2,2]
        type VT = u32;

        const BYTES_PER_ELEM: usize = 4;
        const TILE_ROW: usize = 16;
        const TILE_COL: usize = 16;

        const ADDR_OFFSET: u64 = 64; // The number of bytes to read per request

        let mut ctx = ProgramBuilder::default();
        let (addr_snd, addr_rcv) = ctx.unbounded();
        let (resp_snd, resp_rcv) = ctx.unbounded();
        let (snd, rcv) = ctx.unbounded();

        let mut mem_context = HBMContext::new(
            &mut ctx,
            HBMConfig {
                addr_offset: ADDR_OFFSET,
                channel_num: 8,
                per_channel_init_interval: 2,
                per_channel_latency: 2,
                per_channel_outstanding: 1,
                per_channel_start_up_time: 14,
            },
        );
        mem_context.add_reader(ReadBundle {
            addr: addr_rcv,
            resp: resp_snd,
        });

        ctx.add_child(mem_context);

        // Create a temporary JSON file for the shape
        let temp_dir = std::env::temp_dir();
        let shape_file_path = temp_dir.join("test_shape.json");
        std::fs::write(&shape_file_path, "[32, 32]").unwrap();

        ctx.add_child(DynLinearOffChipLoad::<SimpleEvent, VT>::new(
            shape_file_path.to_string_lossy().to_string(),
            "dummy_path.npy".to_string(), // No NPY file for this test
            TILE_ROW,
            TILE_COL,
            BYTES_PER_ELEM,
            0,
            ADDR_OFFSET,
            4,
            addr_snd,
            resp_rcv,
            snd,
            0,
        ));

        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;
        let tile_vec =
            vec![
                Tile::<VT>::new_blank(vec![TILE_ROW, TILE_COL], BYTES_PER_ELEM, READ_FROM_MU);
                2 * 2
            ];

        // =============== Expected Output [2,2] ================
        // Create 2x2 Buffers (each are a buffer of 2x2 tiles)
        let arr = Arc::new(
            ArcArray::from_vec(tile_vec)
                .into_shape_with_order((2, 2))
                .unwrap(),
        );
        let buff = Buffer::new((*arr).clone().into_dyn(), DUMMY_CREATION_TIME);

        // =============== Output Stream [2,2] ================
        ctx.add_child(ApproxCheckerContext::new(
            move || buff.to_elem_iter().collect::<Vec<_>>().into_iter(),
            rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());

        // Clean up temporary file
        std::fs::remove_file(shape_file_path).unwrap();
    }
}
