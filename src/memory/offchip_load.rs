use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;

use crate::{
    primitives::elem::{Elem, StopType},
    ramulator::access::MemoryData,
};

use super::{data::Tile, events::LoggableEventSimple};

#[derive(Debug)]
pub enum HbmAddrEnum {
    ADDR(u64),
    STOP(StopType),
}

#[context_macro]
pub struct OffChipLoad2D<E: LoggableEventSimple> {
    pub tensor_shape_tiled: [usize; 2], // In terms of tiles.
    pub stride: Vec<usize>,             // Express the view information with strides
    pub out_shape_tiled: Vec<usize>,    // stride and out_shape are both in terms of tiles
    pub tile_row: usize,
    pub tile_col: usize,
    pub n_byte: usize,       // size of the datatype
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_offset: u64,    // The data received per request
    pub addr_snd: Sender<u64>,
    pub resp_addr_rcv: Receiver<u64>,
    pub rdata_rcv: Receiver<MemoryData>,
    pub on_chip_snd: Sender<Elem<Tile>>,
    _phantom: PhantomData<E>, // Needed to use the generic parameter E
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> OffChipLoad2D<E> {
    pub fn new(
        tensor_shape_tiled: [usize; 2],
        stride: Vec<usize>,
        out_shape_tiled: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
        n_byte: usize,
        base_addr_byte: u64,
        addr_offset: u64,
        addr_snd: Sender<u64>,
        resp_addr_rcv: Receiver<u64>,
        rdata_rcv: Receiver<MemoryData>,
        on_chip_snd: Sender<Elem<Tile>>,
    ) -> Self {
        let ctx = Self {
            tensor_shape_tiled,
            stride,
            out_shape_tiled,
            tile_row,
            tile_col,
            n_byte,
            base_addr_byte,
            addr_offset,
            addr_snd,
            resp_addr_rcv,
            rdata_rcv,
            on_chip_snd,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.addr_snd.attach_sender(&ctx);
        ctx.resp_addr_rcv.attach_receiver(&ctx);
        ctx.rdata_rcv.attach_receiver(&ctx);
        ctx.on_chip_snd.attach_sender(&ctx);

        ctx
    }

    fn generate_addr(&self) -> impl Iterator<Item = HbmAddrEnum> {
        // Calculate total elements in the output tensor
        let total_tiles: usize = self.out_shape_tiled.iter().product();

        // Create an iterator that generates indices
        let mut addrs: Vec<HbmAddrEnum> = vec![];
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
            println!("tile_idx: {}", tile_idx);

            // append addresses to fetch the given tile
            let tile_offset = self.tile_row * self.tile_col * self.n_byte;
            let base_addr_i = self.base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset = self.tensor_shape_tiled[1] * self.tile_col * self.n_byte;
            let mut addr_for_curr_tile = vec![];
            for r in 0..self.tile_row {
                for c in (0..(self.tile_col * self.n_byte)).step_by(self.addr_offset as usize) {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    addr_for_curr_tile.push(HbmAddrEnum::ADDR(addr));
                }
            }
            addrs.append(&mut addr_for_curr_tile);

            // Check which dimensions need stop tokens
            let mut stop_tokens = vec![];

            // We'll track if all inner dimensions are at their final positions
            let mut all_inner_dims_at_end = true;

            // Check from innermost to outermost
            for dim in (0..self.out_shape_tiled.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = self.out_shape_tiled[dim] == 1;
                    let is_last_elem = multi_index[dim] == self.out_shape_tiled[dim] - 1;

                    // Add stop token if at end or dim size is 1
                    if is_last_elem || is_dim_size_one {
                        stop_tokens
                            .push(HbmAddrEnum::STOP((self.out_shape_tiled.len() - dim) as u32));
                    }

                    // Update tracking for outer dimensions
                    // Only continue checking outer dimensions if this one is at its last element
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            // Add the stop tokens
            addrs.append(&mut stop_tokens);
        }

        addrs.into_iter()
    }
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Context
    for OffChipLoad2D<E>
{
    fn run(&mut self) {
        // Ensure stride and out_shape have the same length
        assert_eq!(
            self.stride.len(),
            self.out_shape_tiled.len(),
            "Stride and output shape must have the same number of dimensions"
        );
        assert!(((self.tile_col * self.n_byte) as u64) % self.addr_offset == 0);

        println!("Started run of OFFHCIP LOAD");

        for addr_enum in self.generate_addr() {
            println!("Started to iterate {:?}", addr_enum);
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

                    dam::logging::log_event(&E::new(
                        send_request_time.time(),
                        read_finish_time.time(),
                        false,
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
                                data: Elem::Val(Tile {
                                    shape: vec![self.tile_row, self.tile_col],
                                    bytes_per_elem: self.n_byte,
                                    read_from_mu: true,
                                }),
                            },
                        )
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
// Mongodb: logging for the time used to load each tile

#[cfg(test)]
mod test {
    use std::default;

    use super::{HbmAddrEnum, OffChipLoad2D};
    use crate::ramulator::ramulator_context::{Memory, RamulatorContext, ReadBundle};

    use crate::define_simple_event;
    use crate::memory::events::LoggableEventSimple;
    use dam::dam_macros::event_type;
    use dam::simulation::RunOptions;
    use dam::utility_contexts::FunctionContext;
    use dam::{
        simulation::{
            LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
        },
        utility_contexts::ConsumerContext,
    };
    use serde::{Deserialize, Serialize};

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

        // 2D repeat view
        // let tensor_shape_tiled = [2, 3];
        // let stride = vec![0, 3, 1];
        // let out_shape_tiled = vec![2, 2, 3];

        // 1D repeat view (size-1 dim)
        // let tensor_shape_tiled = [2, 1];
        // let stride = vec![1, 0, 1];
        // let out_shape_tiled = vec![2, 2, 1];

        // 1D repeat view
        let tensor_shape_tiled = [2, 3];
        let stride = vec![3, 0, 1];
        let out_shape_tiled = vec![2, 2, 3];

        let tile_row = 16;
        let tile_col = 32;
        let n_byte = 2;
        let base_addr_byte = 0;

        let total_tiles: usize = out_shape_tiled.iter().product();

        // Create an iterator that generates indices
        let mut addrs: Vec<HbmAddrEnum> = vec![];
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

            // append addresses to fetch the given tile
            let tile_offset = tile_row * tile_col * n_byte;
            let base_addr_i = base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset = tensor_shape_tiled[1] * tile_col * n_byte;
            let mut addr_for_curr_tile = vec![];
            for r in 0..tile_row {
                for c in (0..(tile_col * n_byte)).step_by(ADDR_OFFSET as usize) {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    addr_for_curr_tile.push(HbmAddrEnum::ADDR(addr));
                }
            }
            addrs.append(&mut addr_for_curr_tile);

            // Check which dimensions need stop tokens
            let mut stop_tokens = vec![];

            // We'll track if all inner dimensions are at their final positions
            let mut all_inner_dims_at_end = true;

            // Check from innermost to outermost
            for dim in (0..out_shape_tiled.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = out_shape_tiled[dim] == 1;
                    let is_last_elem = multi_index[dim] == out_shape_tiled[dim] - 1;

                    // Add stop token if at end or dim size is 1
                    if is_last_elem || is_dim_size_one {
                        stop_tokens.push(HbmAddrEnum::STOP((out_shape_tiled.len() - dim) as u32));
                    }

                    // Update tracking for outer dimensions
                    // Only continue checking outer dimensions if this one is at its last element
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            // Add the stop tokens
            println!("{:?}", stop_tokens);
            addrs.append(&mut stop_tokens);
        }

        // for i in addrs.iter() {
        //     println!("Addr: {:?}", i);
        // }
    }
    define_simple_event!(InputLoad);
    // define_simple_event!(WeightQLoad);
    #[test]
    fn test_with_ramulator() {
        /*
        ADDR_OFFSET = (Channel Width) x (Burst Length) = 64 bytes
        - Channel Width: 16 bytes/channel
            - HBM2 standard (JEDEC HBM2 specification) defines each pseudo-channel width explicitly as 16 bytes/channel
        - Burst Length: 4
            - HBM2 standard (JEDEC HBM2 specification) specifies a burst length of 4 beats per DRAM access.
         */
        const ADDR_OFFSET: u64 = 64;

        /*
        Dataflow: ijk
        [32, 128] x [128, 64] = [32, 64]

        Stream: [ 2,   1] x [  1,  4] = [ 2,  4]
        Tile:   [16, 128] x [128, 16] = [16, 16]
         */

        let mut ctx: ProgramBuilder<'_> = ProgramBuilder::default();

        // ====================== Two matrix loaders ======================
        let n_byte = 2;
        let mat1_base = 0;

        let (addr_snd1, addr_rcv1) = ctx.unbounded();
        let (resp_addr_snd1, resp_addr_rcv1) = ctx.unbounded();
        let (rdata_snd1, rdata_rcv1) = ctx.unbounded();
        let (on_chip_snd1, on_chip_rcv1) = ctx.unbounded();

        let mat1 = OffChipLoad2D::<InputLoad>::new(
            [2, 1], // As we don't tile K, the second element is 1
            vec![1, 0, 1],
            vec![2, 4, 1],
            16,
            128,
            n_byte,
            mat1_base,
            ADDR_OFFSET,
            addr_snd1,
            resp_addr_rcv1,
            rdata_rcv1,
            on_chip_snd1,
        );

        ctx.add_child(mat1);

        // ====================== Ramulator Context ======================

        let config_file = "/home/ginasohn/step-perf/external/ramulator2_wrapper/configs/hbm2.yaml";
        let mut mem_context = RamulatorContext::new(config_file, (1u32, 1u32), None);
        mem_context.add_reader(ReadBundle {
            addr: Box::new(addr_rcv1),
            resp: Box::new(rdata_snd1),
            resp_addr: Box::new(resp_addr_snd1),
        });

        ctx.add_child(mem_context);

        // ====================== Consumer ======================
        ctx.add_child(ConsumerContext::new(on_chip_rcv1));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let summary = initialized.run(RunOptions::default());
        // Check the summary
        println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
    }

    #[test]
    fn test_with_ramulator_logging() {
        /*
        ADDR_OFFSET = (Channel Width) x (Burst Length) = 64 bytes
        - Channel Width: 16 bytes/channel
            - HBM2 standard (JEDEC HBM2 specification) defines each pseudo-channel width explicitly as 16 bytes/channel
        - Burst Length: 4
            - HBM2 standard (JEDEC HBM2 specification) specifies a burst length of 4 beats per DRAM access.
         */
        const ADDR_OFFSET: u64 = 64;

        /*
        Dataflow: ijk
        [32, 128] x [128, 64] = [32, 64]

        Stream: [ 2,   1] x [  1,  4] = [ 2,  4]
        Tile:   [16, 128] x [128, 16] = [16, 16]
         */

        let mut ctx: ProgramBuilder<'_> = ProgramBuilder::default();

        // ====================== Two matrix loaders ======================
        let n_byte = 2;
        let mat1_base = 0;
        let mat2_base = mat1_base + 32 * 128 * n_byte;

        let (addr_snd1, addr_rcv1) = ctx.unbounded();
        let (resp_addr_snd1, resp_addr_rcv1) = ctx.unbounded();
        let (rdata_snd1, rdata_rcv1) = ctx.unbounded();
        let (on_chip_snd1, on_chip_rcv1) = ctx.unbounded();

        let mat1 = OffChipLoad2D::<InputLoad>::new(
            [2, 1], // As we don't tile K, the second element is 1
            vec![1, 0, 1],
            vec![2, 4, 1],
            16,
            128,
            2,
            mat1_base,
            ADDR_OFFSET,
            addr_snd1,
            resp_addr_rcv1,
            rdata_rcv1,
            on_chip_snd1,
        );

        ctx.add_child(mat1);

        // let (addr_snd2, addr_rcv2) = ctx.unbounded();
        // let (resp_addr_snd2, resp_addr_rcv2) = ctx.unbounded();
        // let (rdata_snd2, rdata_rcv2) = ctx.unbounded();
        // let (on_chip_snd2, on_chip_rcv2) = ctx.unbounded();

        // let mat2 = OffChipLoad2D::<WeightQLoad>::new(
        //     [1, 4], // As we don't tile K, the second element is 1
        //     vec![0, 4, 1],
        //     vec![2, 1, 4],
        //     128,
        //     16,
        //     2,
        //     mat2_base,
        //     addr_snd2,
        //     resp_addr_rcv2,
        //     rdata_rcv2,
        //     on_chip_snd2,
        // );

        // ====================== Ramulator Context ======================

        let config_file = "/home/ginasohn/step-perf/external/ramulator2_wrapper/configs/hbm2.yaml";
        let mut mem_context = RamulatorContext::new(config_file, (1u32, 1u32), None);
        mem_context.add_reader(ReadBundle {
            addr: Box::new(addr_rcv1),
            resp: Box::new(rdata_snd1),
            resp_addr: Box::new(resp_addr_snd1),
        });

        ctx.add_child(mem_context);

        // ====================== Consumer ======================
        ctx.add_child(ConsumerContext::new(on_chip_rcv1));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
            // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
            dam::logging::LogFilter::AllowAll,
        ));
        let run_options = run_options.logging(LoggingOptions::Mongo(
            MongoOptionsBuilder::default()
                .db("off_chip_loader".to_string())
                .uri("mongodb://127.0.0.1:27017".to_string())
                .build()
                .unwrap(),
        ));
        let summary = initialized.run(run_options.build().unwrap());
        // Check the summary
        println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
    }
}
