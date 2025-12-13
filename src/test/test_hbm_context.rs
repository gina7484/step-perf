#[cfg(test)]
mod test {
    use std::default;
    use std::sync::Arc;

    use crate::memory::offchip_load::OffChipLoad;
    use crate::memory::offchip_store::OffChipStore;

    use crate::functions::{map_accum_fn, map_fn};
    use crate::operator::map::BinaryMap;

    use crate::operator::repeat::RepeatStatic;
    use crate::ramulator::hbm_context::{HBMConfig, HBMContext, ReadBundle, WriteBundle};

    use crate::utils::events::{SimpleEvent, DUMMY_ID};
    use dam::simulation::{DotConvertible, RunOptions};
    use dam::{
        simulation::{
            LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
        },
        utility_contexts::ConsumerContext,
    };
    use serde::{Deserialize, Serialize};

    use std::collections::HashMap;

    /*
    Dataflow: ijk
    [32, 128] x [128, 64] = [32, 64]

    Stream: [ 2,   1] x [  1,  4] = [ 2,  4]
    Tile:   [16, 128] x [128, 16] = [16, 16]
     */

    #[test]
    fn test_gen_q_with_data() {
        /*
        ADDR_OFFSET = (Channel Width) x (Burst Length) = 64 bytes
        - Channel Width: 16 bytes/channel
            - HBM2 standard (JEDEC HBM2 specification) defines each pseudo-channel width explicitly as 16 bytes/channel
        - Burst Length: 4
            - HBM2 standard (JEDEC HBM2 specification) specifies a burst length of 4 beats per DRAM access.
         */
        const ADDR_OFFSET: u64 = 64; // 32 elements in this test case
        const PAR_DISPATCH: usize = 8;

        const B: usize = 32;
        const H: usize = 64;

        let n_byte = 4;

        let par_b = 16;

        let mut ctx: ProgramBuilder<'_> = ProgramBuilder::default();

        // ====================== Base for all matrices ======================
        let tensor_sizes = HashMap::from([("Input", B * H), ("W_Q", H * H), ("Output", B * H)]);

        let mut tensor_addrs: HashMap<&'static str, usize> = HashMap::new();
        let mut offset: usize = 0;
        for (i, j) in tensor_sizes {
            tensor_addrs.insert(i, offset);
            offset += j * n_byte;
        }

        // ====================== Generate Q ======================
        // Tiling Scheme
        let tile_m_gen_q = par_b;
        let tile_k_gen_q = H; // Same as the dimension's size as we don't tile this dim.
        let tile_n_gen_q = 32;

        // Operand 1 (Input) [B,H]
        // Stream shape: [B/16, H]
        // Tile shape: [16, H]
        // Repeat1D by number of N tiles

        let (addr_snd1, addr_rcv1) = ctx.unbounded();
        let (resp_addr_snd1, resp_addr_rcv1) = ctx.unbounded();
        let (repeat_snd1, repeat_rcv1) = ctx.bounded(1);

        let mat1 = OffChipLoad::<SimpleEvent, f32>::new(
            vec![B / tile_m_gen_q, H / tile_k_gen_q], // As we don't tile K, the second element is 1
            vec![H / tile_k_gen_q, 1],
            vec![B / tile_m_gen_q, H / tile_k_gen_q],
            None, //Some("./step-perf/input.npy".to_string()),
            tile_m_gen_q,
            tile_k_gen_q,
            n_byte as usize,
            tensor_addrs.get("Input").unwrap().clone() as u64,
            ADDR_OFFSET,
            PAR_DISPATCH,
            addr_snd1,
            resp_addr_rcv1,
            repeat_snd1,
            0,
        );

        let (on_chip_snd1, on_chip_rcv1) = ctx.bounded(1);

        let repeat_mat1 = RepeatStatic::new(repeat_rcv1, H / tile_n_gen_q, on_chip_snd1);

        // Operand 2 (W_Q): [H,H]
        // Stream shape: [H, H/tileN]
        // Tile shape: [H, tileN]
        // Repeat2D by number of M tiles
        let (addr_snd2, addr_rcv2) = ctx.unbounded();
        let (resp_addr_snd2, resp_addr_rcv2) = ctx.unbounded();
        let (on_chip_snd2, on_chip_rcv2) = ctx.bounded(1);

        // For the weights, we will assume it's saved in a transposed order
        let mat2_stride = if H / tile_k_gen_q == 1 || H / tile_n_gen_q == 1 {
            vec![0, H / tile_n_gen_q, 1]
        } else {
            vec![0, 1, H / tile_n_gen_q]
        };
        let mat2 = OffChipLoad::<SimpleEvent, f32>::new(
            vec![H / tile_k_gen_q, H / tile_n_gen_q], // As we don't tile K, the second element is 1
            mat2_stride,
            vec![B / tile_m_gen_q, H / tile_k_gen_q, H / tile_n_gen_q],
            None, //Some("./step-perf/w_q.npy".to_string()),
            tile_k_gen_q,
            tile_n_gen_q,
            n_byte as usize,
            tensor_addrs.get("W_Q").unwrap().clone() as u64,
            ADDR_OFFSET,
            PAR_DISPATCH,
            addr_snd2,
            resp_addr_rcv2,
            on_chip_snd2,
            1,
        );

        // ====================== Matmul Context ======================
        // Size of per-tile computation: [16, H] * [H, tileN] = [16 ,tileN]
        /* As the computation has a long reduction dimension, use a output stationary systolic array of shape [16, tileN] = [16, 16]
           This roughly maps to 3 PCUs (16*6 * 3), which has an approximate of 638 * 1e12 (FLOPs/s) / 1040 * 3 * 1/(1.8*1e9) (s/cycle) = 1022 (FLOPs/cycle)
        */
        let (mm_snd, mm_rcv) = ctx.bounded(1);
        // let gen_q = BinaryMapAccum::<GenQ>::new(
        //     on_chip_rcv1,
        //     on_chip_rcv2,
        //     mm_snd,
        //     Arc::new(|tile1, tile2, accumulator, comp_bw, write_back_mu| {
        //         map_accum_fn::matmul(tile1, tile2, accumulator, comp_bw, write_back_mu, false)
        //     }),
        //     Arc::new(move || Tile {
        //         shape: vec![16, tile_n_gen_q],
        //         bytes_per_elem: n_byte,
        //         read_from_mu: true,
        //     }),
        //     1,
        //     1022,
        //     true,
        // );
        let gen_q = BinaryMap::<SimpleEvent, f32, f32, f32>::new(
            on_chip_rcv1,
            on_chip_rcv2,
            mm_snd,
            Arc::new(|tile1, tile2, comp_bw, write_back_mu| {
                map_fn::matmul(tile1, tile2, comp_bw, write_back_mu, false)
            }),
            1022,
            true,
            3,
        );

        // ====================== Store Context ======================
        let (waddr_snd, waddr_rcv) = ctx.unbounded();
        let (ack_snd, ack_rcv) = ctx.unbounded();
        let store_ctx = OffChipStore::<SimpleEvent, f32>::new(
            vec![B / tile_m_gen_q, H / tile_n_gen_q],
            tile_m_gen_q,
            tile_n_gen_q,
            None, //Some("./step-perf/output.npy".to_string()),
            tensor_addrs.get("Output").unwrap().clone() as u64,
            ADDR_OFFSET,
            PAR_DISPATCH,
            mm_rcv,
            waddr_snd,
            ack_rcv,
            4,
        );

        // ====================== HBM Context ======================

        let mut mem_context = HBMContext::new(
            &mut ctx,
            HBMConfig {
                addr_offset: ADDR_OFFSET, // 32 elements in this test case
                channel_num: 8,
                per_channel_latency: 2,
                per_channel_init_interval: 2,
                per_channel_outstanding: 1, // For now, this does not have any effect
                per_channel_start_up_time: 14, // Time to wait before the first request can be processed
            },
        );
        mem_context.add_reader(ReadBundle {
            addr: addr_rcv1,
            resp: resp_addr_snd1,
        });

        mem_context.add_reader(ReadBundle {
            addr: addr_rcv2,
            resp: resp_addr_snd2,
        });
        mem_context.add_writer(WriteBundle {
            addr: waddr_rcv,
            resp: ack_snd,
        });

        // ====================== Simple Metrics ======================
        let on_chip_req_bytes =
            (mat1.on_chip_req_elems() + mat2.on_chip_req_elems() + store_ctx.on_chip_req_elems())
                * n_byte;
        let data_movement_bytes =
            (mat1.loaded_elems() + mat2.loaded_elems() + store_ctx.stored_elems()) * n_byte;
        println!("on_chip_req_bytes: {}", on_chip_req_bytes);
        println!("data_movement_bytes: {}", data_movement_bytes);

        // ====================== Add contexts ======================
        ctx.add_child(mat1);
        ctx.add_child(repeat_mat1);
        ctx.add_child(mat2);
        ctx.add_child(gen_q);
        ctx.add_child(store_ctx);
        ctx.add_child(mem_context);

        // ====================== Run Simulation ======================

        let initialized = ctx.initialize(Default::default()).unwrap();
        let run_with_mongo = false;

        if run_with_mongo {
            let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
                // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
                dam::logging::LogFilter::AllowAll,
            ));
            let run_options = run_options.logging(LoggingOptions::Mongo(
                MongoOptionsBuilder::default()
                    .db("test_mm".to_string())
                    .uri("mongodb://127.0.0.1:27017".to_string())
                    .build()
                    .unwrap(),
            ));
            let summary = initialized.run(run_options.build().unwrap());
            // Check the summary
            println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());

            #[cfg(feature = "dot")]
            {
                println!("{}", summary.to_dot_string());
            }
        } else {
            let summary = initialized.run(Default::default());
            // Check the summary
            println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());

            #[cfg(feature = "dot")]
            {
                println!("{}", summary.to_dot_string());
            }
        }
    }
}
