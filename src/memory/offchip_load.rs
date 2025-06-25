use std::marker::PhantomData;

use dam::logging::LogEvent;
use dam::{context_tools::*, types::StaticallySized};
use itertools::Itertools;
use ndarray::{IntoDimension, Ix2, IxDyn, IxDynImpl};

use crate::ramulator::hbm_context::ParAddrs;
use crate::{
    primitives::elem::{Elem, StopType},
    ramulator::access::MemoryData,
};

use crate::memory::HbmAddrEnum;
use crate::primitives::tile::Tile;
use crate::utils::events::LoggableEventSimple;

#[context_macro]
pub struct OffChipLoad<E: LoggableEventSimple, T: DAMType> {
    // Tiling configurations
    pub tensor_shape_tiled: Vec<usize>, // In terms of tiles.
    pub stride: Vec<usize>,             // Express the view information with strides
    pub out_shape_tiled: Vec<usize>,    // stride and out_shape are both in terms of tiles
    pub underlying: Option<ndarray::ArcArray<T, IxDyn>>,
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
    > OffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        tensor_shape_tiled: Vec<usize>,
        stride: Vec<usize>,
        out_shape_tiled: Vec<usize>,
        npy_path: Option<String>,
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

        let ctx = Self {
            tensor_shape_tiled,
            stride,
            out_shape_tiled,
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
        let total_tiles: usize = self.out_shape_tiled.iter().product();

        // Create a vector to hold all the addresses
        let mut addrs: Vec<HbmAddrEnum<T>> = vec![];

        for flat_idx in 0..total_tiles {
            // Convert flat index to multi-dimensional indices
            let mut remaining = flat_idx;
            let mut multi_index = vec![0; self.out_shape_tiled.len()];

            // Calculate multi-dimensional indices
            for i in (0..self.out_shape_tiled.len()).rev() {
                multi_index[i] = remaining % self.out_shape_tiled[i];
                remaining /= self.out_shape_tiled[i];
            }

            // Calculate the index in the original flat tensor using strides
            let mut tile_idx = 0;
            for (dim, &idx_in_dim) in multi_index.iter().enumerate() {
                tile_idx += idx_in_dim * self.stride[dim];
            }

            // Ensure we don't go out of bounds of the original tensor
            let original_size: usize = self.tensor_shape_tiled.iter().product();
            if original_size > 0 {
                tile_idx = tile_idx % original_size;
            } else {
                tile_idx = 0; // Handle empty tensor case
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
            for dim in (0..self.out_shape_tiled.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = self.out_shape_tiled[dim] == 1;
                    let is_last_elem = multi_index[dim] == self.out_shape_tiled[dim] - 1;

                    // If at end or dim size is 1, update the highest stop token
                    if is_last_elem || is_dim_size_one {
                        highest_stop_token = Some((self.out_shape_tiled.len() - dim) as u32);
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
        let total_tiles: usize = self.out_shape_tiled.iter().product();
        total_tiles * self.tile_row * self.tile_col
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: npyz::Deserialize + DAMType,
    > Context for OffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        // Ensure stride and out_shape have the same length
        assert_eq!(
            self.stride.len(),
            self.out_shape_tiled.len(),
            "Stride and output shape must have the same number of dimensions"
        );
        // assert!(((self.tile_col * self.n_byte) as u64) % self.addr_offset == 0);

        // println!("Started run of OFFHCIP LOAD");

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
                "OffChipLoad".to_string(),
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
    use super::HbmAddrEnum;
    use crate::primitives::tile::Tile;

    #[test]
    fn test_generate_addr() {
        /*
        ADDR_OFFSET = (Channel Width) x (Burst Length) = 64 bytes
        - Channel Width: 16 bytes/channel
            - HBM2 standard (JEDEC HBM2 specification) defines each pseudo-channel width explicitly as 16 bytes/channel
        - Burst Length: 4
            - HBM2 standard (JEDEC HBM2 specification) specifies a burst length of 4 beats per DRAM access.
         */
        const ADDR_OFFSET: u64 = 64;

        // Identity view (size 1 dim)
        // let tensor_shape_tiled = [2, 1];
        // let stride = vec![1, 1];
        // let out_shape_tiled = vec![2, 1];

        // Identity view
        // let tensor_shape_tiled = [2, 3];
        // let stride = vec![3, 1];
        // let out_shape_tiled = vec![2, 3];

        // 2D repeat view (size-1 dim)
        // let tensor_shape_tiled = [2, 1];
        // let stride = vec![0, 1, 1];
        // let out_shape_tiled = vec![2, 2, 1];

        // 2D repeat view (size-1 dim)
        const B: usize = 32;
        const H: usize = 64;

        let n_byte = 2;

        let par_b = 16;
        let tile_m_gen_q = par_b;
        let tile_k_gen_q = H; // Same as the dimension's size as we don't tile this dim.
        let tile_n_gen_q = 32;

        let tensor_shape_tiled = [H / tile_n_gen_q, H / tile_k_gen_q]; // As we don't tile K, the second element is 1
        let stride = vec![0, H / tile_k_gen_q, 1];
        let out_shape_tiled = vec![B / tile_m_gen_q, H / tile_n_gen_q, H / tile_k_gen_q];
        // 2D repeat view
        // let tensor_shape_tiled = [3, 2];
        // let stride = vec![0, 2, 1];
        // let out_shape_tiled = vec![2, 3, 2];

        // 1D repeat view (size-1 dim)
        // let tensor_shape_tiled = [2, 1];
        // let stride = vec![1, 0, 1];
        // let out_shape_tiled = vec![2, 2, 1];

        // 1D repeat view
        // let tensor_shape_tiled = [2, 3];
        // let stride = vec![3, 0, 1];
        // let out_shape_tiled = vec![2, 2, 3];

        let tile_row = 16;
        let tile_col = 32;
        let n_byte = 2;
        let base_addr_byte = 0;

        let total_tiles: usize = out_shape_tiled.iter().product();

        // Create a vector to hold all the addresses
        let mut addrs: Vec<HbmAddrEnum<f32>> = vec![];

        for flat_idx in 0..total_tiles {
            // Convert flat index to multi-dimensional indices
            let mut remaining = flat_idx;
            let mut multi_index = vec![0; out_shape_tiled.len()];

            // Calculate multi-dimensional indices
            for i in (0..out_shape_tiled.len()).rev() {
                multi_index[i] = remaining % out_shape_tiled[i];
                remaining /= out_shape_tiled[i];
            }

            // Calculate the index in the original flat tensor using strides
            let mut tile_idx = 0;
            for (dim, &idx_in_dim) in multi_index.iter().enumerate() {
                tile_idx += idx_in_dim * stride[dim];
            }

            // Ensure we don't go out of bounds of the original tensor
            let original_size: usize = tensor_shape_tiled.iter().product();
            if original_size > 0 {
                tile_idx = tile_idx % original_size;
            } else {
                tile_idx = 0; // Handle empty tensor case
            }
            println!("tile_idx: {}", tile_idx);

            // Generate addresses to fetch the given tile
            let tile_offset = tile_row * tile_col * n_byte;
            let base_addr_i = base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset = tensor_shape_tiled[1] * tile_col * n_byte;

            // Generate all addresses for this tile
            let mut tile_addrs = vec![];
            for r in 0..tile_row {
                for c in (0..(tile_col * n_byte)).step_by(ADDR_OFFSET as usize) {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    tile_addrs.push(addr);
                }
            }

            // Determine the highest-dimensional stop token needed
            let mut highest_stop_token: Option<u32> = None;
            let mut all_inner_dims_at_end = true;

            // Check from innermost to outermost
            for dim in (0..out_shape_tiled.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = out_shape_tiled[dim] == 1;
                    let is_last_elem = multi_index[dim] == out_shape_tiled[dim] - 1;

                    // If at end or dim size is 1, update the highest stop token
                    if is_last_elem || is_dim_size_one {
                        highest_stop_token = Some((out_shape_tiled.len() - dim) as u32);
                    }

                    // Update tracking for outer dimensions
                    // Only continue checking outer dimensions if this one is at its last element
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            // Add the addresses to the result list
            if !tile_addrs.is_empty() {
                if let Some(stop_type) = highest_stop_token {
                    addrs.push(HbmAddrEnum::ADDRSTOP(
                        tile_addrs,
                        Tile::new_blank(vec![tile_row, tile_col], n_byte, true),
                        stop_type,
                    ));
                } else {
                    // No stop token, add all addresses normally
                    addrs.push(HbmAddrEnum::ADDR(
                        tile_addrs,
                        Tile::new_blank(vec![tile_row, tile_col], n_byte, true),
                    ));
                }
            }
        }

        for i in addrs.iter() {
            println!("Addr: {:?}", i);
        }
    }
}
