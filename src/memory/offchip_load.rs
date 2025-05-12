use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;

use crate::ramulator::{access::MemoryData, ramulator_context::ADDR_OFFSET};

use super::{data::DataSizeInfo, events::LoggableEventSimple};

#[context_macro]
pub struct OffChipLoad2D<E: LoggableEventSimple> {
    pub tensor_shape_tiled: [usize; 2], // In terms of tiles.
    pub stride: Vec<usize>,             // Express the view information with strides
    pub out_shape_tiled: Vec<usize>,    // stride and out_shape are both in terms of tiles
    pub tile_row: usize,
    pub tile_col: usize,
    pub n_byte: usize,       // size of the datatype
    pub base_addr_byte: u64, // The base address for the given tensor
    pub addr_snd: Sender<u64>,
    pub resp_addr_rcv: Receiver<u64>,
    pub rdata_rcv: Receiver<MemoryData>,
    pub on_chip_snd: Sender<DataSizeInfo>,
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
        addr_snd: Sender<u64>,
        resp_addr_rcv: Receiver<u64>,
        rdata_rcv: Receiver<MemoryData>,
        on_chip_snd: Sender<DataSizeInfo>,
    ) -> Self {
        let ctx = Self {
            tensor_shape_tiled,
            stride,
            out_shape_tiled,
            tile_row,
            tile_col,
            n_byte,
            base_addr_byte,
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

    fn generate_addr(&self) -> impl Iterator<Item = u64> {
        // Calculate total elements in the output tensor
        let total_tiles: usize = self.out_shape_tiled.iter().product();

        // Create an iterator that generates indices
        let mut addrs: Vec<u64> = vec![];
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
            // Get original tensor size (total number of elements)
            let original_size: usize = self.tensor_shape_tiled.iter().product();
            if original_size > 0 {
                tile_idx = tile_idx % original_size
            } else {
                tile_idx = 0 // Handle empty tensor case
            }

            // append addresses to fetch the given tile
            let tile_offset = self.tile_row * self.tile_col * self.n_byte;
            let base_addr_i = self.base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset = self.tensor_shape_tiled[1] * self.tile_col * self.n_byte;

            let mut addr_for_curr_tile = vec![];
            for r in 0..self.tile_row {
                for c in (0..(self.tile_col * self.n_byte)).step_by(ADDR_OFFSET as usize) {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    addr_for_curr_tile.push(addr);
                }
            }
            addrs.append(&mut addr_for_curr_tile);
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
        assert!(((self.tile_col * self.n_byte) as u64) % ADDR_OFFSET == 0);

        for addr in self.generate_addr() {
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
                        data: DataSizeInfo {
                            bytes: self.tile_row * self.tile_col * self.n_byte,
                        },
                    },
                )
                .unwrap();

            dam::logging::log_event(&E::new(send_request_time.time(), read_finish_time.time()))
                .unwrap();
        }
    }
}
// Mongodb: logging for the time used to load each tile

#[cfg(test)]
mod test {
    use super::OffChipLoad2D;
    use crate::ramulator::ramulator_context::{Memory, RamulatorContext, ReadBundle};
    use crate::ramulator::{access::MemoryData, ramulator_context::ADDR_OFFSET};

    use crate::define_simple_event;
    use crate::memory::events::LoggableEventSimple;
    use dam::dam_macros::event_type;
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
        let tensor_shape_tiled = [1, 4];
        let stride = vec![0, 4, 1];
        let out_shape_tiled = vec![2, 1, 4];
        let tile_row = 128;
        let tile_col = 16;
        let n_byte = 2;
        let base_addr_byte = 0;

        // let tensor_shape_tiled = [2, 1];
        // let stride = vec![1, 0, 1];
        // let out_shape_tiled = vec![2, 4, 1];
        // let tile_row = 16;
        // let tile_col = 128;
        // let n_byte = 2;
        // let base_addr_byte = 0;

        // Calculate total elements in the output tensor
        let total_tiles: usize = out_shape_tiled.iter().product();

        // Create an iterator that generates indices
        let mut addrs: Vec<u64> = vec![];
        for flat_idx in 0..total_tiles {
            // Convert flat index to multi-dimensional indices
            let mut remaining = flat_idx;
            let mut multi_index = vec![0; out_shape_tiled.len()];

            // Calculate multi-dimensional indices
            for i in (0..out_shape_tiled.len()).rev() {
                multi_index[i] = remaining % out_shape_tiled[i];
                remaining /= out_shape_tiled[i];
            }
            println!("multi_index: {:?}", multi_index);

            // Calculate the index in the original flat tensor using strides
            let mut tile_idx = 0;
            for (dim, &idx_in_dim) in multi_index.iter().enumerate() {
                tile_idx += idx_in_dim * stride[dim];
            }

            // Ensure we don't go out of bounds of the original tensor
            // Get original tensor size (total number of elements)
            let original_size: usize = tensor_shape_tiled.iter().product();
            if original_size > 0 {
                tile_idx = tile_idx % original_size
            } else {
                tile_idx = 0 // Handle empty tensor case
            }

            println!("tile_idx: {}", tile_idx);

            // append addresses to fetch the given tile
            let tile_offset = tile_row * tile_col * n_byte;
            let base_addr_i = base_addr_byte + (tile_idx * tile_offset) as u64;
            let row_offset = tensor_shape_tiled[1] * tile_col * n_byte;

            println!("base_addr_i: {}", base_addr_i);

            let mut addr_for_curr_tile = vec![];
            for r in 0..tile_row {
                for c in (0..(tile_col * n_byte)).step_by(ADDR_OFFSET as usize) {
                    let addr: u64 = base_addr_i + (r * row_offset + c) as u64;
                    addr_for_curr_tile.push(addr);
                }
            }
            println!("addr_for_curr_tile: {:?}", addr_for_curr_tile);
            addrs.append(&mut addr_for_curr_tile);
        }

        for i in addrs.iter() {
            println!("Addr: {}", i);
        }
    }

    define_simple_event!(InputLoad);
    // define_simple_event!(WeightQLoad);
    #[test]
    fn test_with_ramulator() {
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
