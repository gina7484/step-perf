use std::{fs::File, marker::PhantomData};

use dam::context_tools::*;
use dam::logging::LogEvent;
use itertools::Itertools;
use ndarray::{concatenate, Array2, Axis};
use serde_json;

use crate::primitives::elem::Bufferizable;
use crate::{primitives::elem::Elem, ramulator::hbm_context::ParAddrs};

use crate::utils::events::LoggableEventSimple;

use crate::memory::aw_trace::trace_aw_write;
use crate::primitives::tile::Tile;

#[context_macro]
pub struct DynOffChipStore<E: LoggableEventSimple, T: DAMType> {
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
    > DynOffChipStore<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        shape_path: String, // path to the json file with the untiled_shape
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
        // Read the shape from the JSON file
        let shape_file = std::fs::File::open(&shape_path)
            .unwrap_or_else(|_| panic!("Failed to open shape file: {}", shape_path));
        let untiled_shape: Vec<usize> = serde_json::from_reader(shape_file)
            .unwrap_or_else(|_| panic!("Failed to parse shape JSON from: {}", shape_path));

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
    > Context for DynOffChipStore<E, T>
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
            let mut cur_st: u32 = 0;
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
                        cur_st = s;
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

            trace_aw_write(
                self.id,
                read_finish_time.time(),
                &self.tensor_shape_tiled,
                cur_st,
                false,
            );

            // dequeue
            self.on_chip_rcv.dequeue(&self.time).unwrap();
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use dam::{simulation::ProgramBuilder, utility_contexts::GeneratorContext};
    use ndarray::ArcArray;

    use crate::{
        memory::dyn_offchip_store::DynOffChipStore,
        primitives::{buffer::Buffer, tile::Tile},
        ramulator::hbm_context::{HBMConfig, HBMContext, WriteBundle},
        utils::events::SimpleEvent,
    };

    #[test]
    fn round_trip_test_store() {
        // Test storing a 2x2 tensor (4 tiles total)
        type VT = u32;

        const BYTES_PER_ELEM: usize = 4;
        const TILE_ROW: usize = 16;
        const TILE_COL: usize = 16;

        const ADDR_OFFSET: u64 = 64; // The number of bytes to write per request

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
        mem_context.add_writer(WriteBundle {
            addr: addr_rcv,
            resp: resp_snd,
        });

        ctx.add_child(mem_context);

        // Create a temporary JSON file for the shape
        let temp_dir = std::env::temp_dir();
        let shape_file_path = temp_dir.join("test_store_shape.json");
        std::fs::write(&shape_file_path, "[32, 32]").unwrap();

        // Create a temporary store path
        let store_path = temp_dir.join("test_output");

        ctx.add_child(DynOffChipStore::<SimpleEvent, VT>::new(
            shape_file_path.to_string_lossy().to_string(),
            TILE_ROW,
            TILE_COL,
            Some(store_path.to_string_lossy().to_string()),
            0,
            ADDR_OFFSET,
            4,
            rcv,
            addr_snd,
            resp_rcv,
            0,
        ));

        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        // Create tiles with actual data for comparison
        // We create 4 tiles (2x2 grid) with sequential data for easy verification:
        // - Tile 0 (top-left):   values 0-255     (16x16 = 256 values)
        // - Tile 1 (top-right):  values 256-511   (16x16 = 256 values)
        // - Tile 2 (bottom-left): values 512-767   (16x16 = 256 values)
        // - Tile 3 (bottom-right): values 768-1023  (16x16 = 256 values)
        // This creates a 32x32 tensor with values 0-1023 in row-major order
        let mut tile_vec = Vec::new();
        for i in 0..(2 * 2) {
            // Create a tile with sequential data for easy verification
            let tile_data = ArcArray::from_shape_vec(
                (TILE_ROW, TILE_COL),
                (0..(TILE_ROW * TILE_COL))
                    .map(|j| (i * TILE_ROW * TILE_COL + j) as VT)
                    .collect(),
            )
            .unwrap();
            tile_vec.push(Tile::new(tile_data, BYTES_PER_ELEM, READ_FROM_MU));
        }

        // =============== Input Tiles [2,2] ================
        // Create 2x2 Buffers (each are a buffer of 16x16 tiles)
        let arr = Arc::new(
            ArcArray::from_vec(tile_vec)
                .into_shape_with_order((2, 2))
                .unwrap(),
        );
        let buff = Buffer::new((*arr).clone().into_dyn(), DUMMY_CREATION_TIME);

        // =============== Input Stream [2,2] ================
        ctx.add_child(GeneratorContext::new(
            move || buff.to_elem_iter().collect::<Vec<_>>().into_iter(),
            snd,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());

        // Verify the stored tensor matches the expected tensor
        let stored_npy_path = store_path.with_extension("npy");
        let stored_json_path = store_path.with_extension("json");

        if stored_npy_path.exists() && stored_json_path.exists() {
            // Read the stored shape from JSON
            let stored_shape_file = std::fs::File::open(&stored_json_path).unwrap_or_else(|_| {
                panic!("Failed to open stored shape file: {:?}", stored_json_path)
            });
            let stored_shape: Vec<usize> = serde_json::from_reader(stored_shape_file)
                .unwrap_or_else(|_| panic!("Failed to parse stored shape JSON"));

            // Verify the shape is correct
            assert_eq!(
                stored_shape,
                vec![32, 32],
                "Stored tensor shape should be [32, 32]"
            );

            // Read the stored data from NPY
            let mut stored_file = std::fs::File::open(&stored_npy_path).unwrap_or_else(|_| {
                panic!("Failed to open stored NPY file: {:?}", stored_npy_path)
            });
            let stored_data = npyz::NpyFile::new(&mut stored_file).unwrap();
            let stored_values: Vec<VT> = stored_data.into_vec().unwrap();

            // Verify the data size is correct
            let expected_size = 32 * 32; // 32x32 tensor
            assert_eq!(
                stored_values.len(),
                expected_size,
                "Stored tensor should have {} elements",
                expected_size
            );

            // Verify the stored data matches the expected pattern
            // DynOffChipStore concatenates tiles horizontally first, then vertically when it encounters stop tokens
            // Tile order: Val, ValStop(1), Val, ValStop(2)
            // This creates a 32x32 tensor with the following pattern:
            // - First 16 rows: Tile 0 (0-255) + Tile 1 (256-511) horizontally
            // - Last 16 rows: Tile 2 (512-767) + Tile 3 (768-1023) horizontally

            for row in 0..32 {
                for col in 0..32 {
                    let index = row * 32 + col;
                    let stored_value = stored_values[index];

                    let expected_value = if row < 16 {
                        // First 16 rows: Tile 0 (0-255) + Tile 1 (256-511)
                        if col < 16 {
                            // Tile 0: values 0-255
                            row * 16 + col
                        } else {
                            // Tile 1: values 256-511
                            256 + row * 16 + (col - 16)
                        }
                    } else {
                        // Last 16 rows: Tile 2 (512-767) + Tile 3 (768-1023)
                        if col < 16 {
                            // Tile 2: values 512-767
                            512 + (row - 16) * 16 + col
                        } else {
                            // Tile 3: values 768-1023
                            768 + (row - 16) * 16 + (col - 16)
                        }
                    } as VT;

                    assert_eq!(
                        stored_value, expected_value,
                        "Stored value at position [{}, {}] (index {}) should be {}, but got {}",
                        row, col, index, expected_value, stored_value
                    );
                }
            }

            println!("Successfully verified stored tensor matches expected [32, 32] tensor with correct tile concatenation pattern");
        } else {
            panic!(
                "Expected stored files not found: {:?} and {:?}",
                stored_npy_path, stored_json_path
            );
        }

        // Clean up temporary files
        std::fs::remove_file(shape_file_path).unwrap();
        if stored_npy_path.exists() {
            std::fs::remove_file(stored_npy_path).unwrap();
        }
        if stored_json_path.exists() {
            std::fs::remove_file(stored_json_path).unwrap();
        }
    }
}
