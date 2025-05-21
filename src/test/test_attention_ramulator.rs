#[cfg(test)]
mod test {
    use std::default;
    use std::sync::Arc;

    use crate::memory::data::Tile;
    use crate::memory::offchip_load::OffChipLoad2D;

    use crate::functions::map_accum_fn::matmul;
    use crate::operator::map_accum::BinaryMapAccum;
    use crate::operator::repeat::RepeatStatic;
    use crate::ramulator::ramulator_context::{Memory, RamulatorContext, ReadBundle};

    use crate::define_simple_event;
    use crate::memory::events::LoggableEventSimple;
    use dam::dam_macros::event_type;
    use dam::simulation::{DotConvertible, RunOptions};
    use dam::types::tensor;
    use dam::utility_contexts::{FunctionContext, PrinterContext};
    use dam::{
        simulation::{
            LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
        },
        utility_contexts::ConsumerContext,
    };
    use frunk::labelled::chars::P;
    use serde::{Deserialize, Serialize};

    use std::collections::HashMap;

    define_simple_event!(InputLoad);
    define_simple_event!(WeightQLoad);
    define_simple_event!(GenQ);

    /*
    Dataflow: ijk
    [32, 128] x [128, 64] = [32, 64]

    Stream: [ 2,   1] x [  1,  4] = [ 2,  4]
    Tile:   [16, 128] x [128, 16] = [16, 16]
     */

    /*
        #[test]
        fn test_static_parallel() {
            /*
            ADDR_OFFSET = (Channel Width) x (Burst Length) = 64 bytes
            - Channel Width: 16 bytes/channel
                - HBM2 standard (JEDEC HBM2 specification) defines each pseudo-channel width explicitly as 16 bytes/channel
            - Burst Length: 4
                - HBM2 standard (JEDEC HBM2 specification) specifies a burst length of 4 beats per DRAM access.
             */
            const ADDR_OFFSET: u64 = 64;

            let SEQ_LENGTHS = [
                374, 396, 879, 91, 91, 381, 1313, 388, 242, 209, 394, 394, 1315, 2221, 389, 415, 120,
                369, 206, 1353, 197, 181, 388, 4085, 2584, 203, 126, 389, 2548, 91, 4081, 181, 191, 27,
                203, 398, 126, 209, 209, 28, 437, 181, 203, 200, 4073, 91, 1087, 382, 412, 194, 203,
                200, 64, 458, 1352, 874, 378, 91, 4074, 389, 212, 1085, 407, 396, 1029, 962, 203, 898,
                181, 320, 1119, 124, 1313, 1096, 400, 1057, 203, 1314, 2, 1065, 372, 4094, 1104, 1075,
                4088, 1225, 1118, 380, 1133, 42, 2590, 1140, 126, 431, 1115, 206, 400, 1123, 862, 859,
                890, 1192, 899, 163, 181, 416, 1163, 243, 1041, 1350, 1104, 1127, 1114, 1083, 1313,
                888, 13, 874, 1065, 1157, 1077, 2903, 4079, 1149, 996, 990, 197, 4107, 197, 1378, 1214,
                1016, 1186, 4076, 1100, 218, 1232, 975, 209, 181, 1029, 888, 1113, 1035, 1009, 869,
                1183, 972, 1097, 206, 181, 1316, 908, 1094, 996, 420, 1112, 1133, 890,
            ];
            let B = 32;
            let H = 64;
            let N_LAYERS = 80;
            let N_HEADS = 64;
            let N_KV_HEADS = 8; // Number of key-value heads for GQA
            let HEAD_DIM = H / N_HEADS;
            let MLP_HID = 28672; // Corrected from config
            let VOCAB_SIZE = 128256; // Corrected from config
            let n_byte = 2;

            let par_b = 16;

            let mut ctx: ProgramBuilder<'_> = ProgramBuilder::default();

            // ====================== Base for all matrices ======================
            let tensor_sizes = HashMap::from([
                ("Input", B * H),
                ("W_Q", H * H),
                ("W_K", H * (H / N_HEADS * N_KV_HEADS)),
                ("W_V", H * (H / N_HEADS * N_KV_HEADS)),
                ("Output", B * H),
            ]);

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
            let tile_n_gen_q = 16;

            // Operand 1 (Input) [B,H]
            // Stream shape: [B/16, H]
            // Tile shape: [16, H]
            // Repeat1D by number of N tiles

            let (addr_snd1, addr_rcv1) = ctx.unbounded();
            let (resp_addr_snd1, resp_addr_rcv1) = ctx.unbounded();
            let (rdata_snd1, rdata_rcv1) = ctx.unbounded();
            let (repeat_snd1, repeat_rcv1) = ctx.bounded(1);

            let mat1 = OffChipLoad2D::<InputLoad>::new(
                [B / tile_m_gen_q, H / tile_k_gen_q], // As we don't tile K, the second element is 1
                vec![H / tile_k_gen_q, 1],
                vec![B / tile_m_gen_q, H / tile_k_gen_q],
                tile_m_gen_q,
                tile_k_gen_q,
                n_byte as usize,
                tensor_addrs.get("Input").unwrap().clone() as u64,
                ADDR_OFFSET,
                addr_snd1,
                resp_addr_rcv1,
                rdata_rcv1,
                repeat_snd1,
            );

            let (on_chip_snd1, on_chip_rcv1) = ctx.bounded(1);
            let repeat_mat1 = RepeatStatic::new(repeat_rcv1, H / tile_n_gen_q, on_chip_snd1);

            ctx.add_child(mat1);
            ctx.add_child(repeat_mat1);

            // Operand 2 (W_Q): [H,H]
            // Stream shape: [H, H/tileN]
            // Tile shape: [H, tileN]
            // Repeat2D by number of M tiles
            let (addr_snd2, addr_rcv2) = ctx.unbounded();
            let (resp_addr_snd2, resp_addr_rcv2) = ctx.unbounded();
            let (rdata_snd2, rdata_rcv2) = ctx.unbounded();
            let (on_chip_snd2, on_chip_rcv2) = ctx.bounded(1);

            let mat2 = OffChipLoad2D::<WeightQLoad>::new(
                [H / tile_k_gen_q, H / tile_n_gen_q], // As we don't tile K, the second element is 1
                vec![0, H / tile_n_gen_q, 1],
                vec![B / tile_m_gen_q, H / tile_k_gen_q, H / tile_n_gen_q],
                tile_k_gen_q,
                tile_n_gen_q,
                n_byte as usize,
                tensor_addrs.get("W_Q").unwrap().clone() as u64,
                ADDR_OFFSET,
                addr_snd2,
                resp_addr_rcv2,
                rdata_rcv2,
                on_chip_snd2,
            );

            ctx.add_child(mat2);

            // ====================== Matmul Context ======================
            // Size of per-tile computation: [16, H] * [H, tileN] = [16 ,tileN]
            /* As the computation has a long reduction dimension, use a output stationary systolic array of shape [16, tileN] = [16, 16]
               This roughly maps to 3 PCUs (16*6 * 3), which has an approximate of 638 * 1e12 (FLOPs/s) / 1040 * 3 * 1/(1.8*1e9) (s/cycle) = 1022 (FLOPs/cycle)
            */
            let (mm_snd, mm_rcv) = ctx.bounded(1);
            let gen_q = BinaryMap::<GenQ>::new(
                on_chip_rcv1,
                on_chip_rcv2,
                mm_snd,
                Arc::new(|tile1, tile2, comp_bw, read_from_mu| {
                    matmul(tile1, tile2, comp_bw, read_from_mu)
                }),
                1022,
                true,
            );

            ctx.add_child(gen_q);

            // ====================== Store Context ======================

            // ====================== Ramulator Context ======================

            let config_file = "/home/ginasohn/step-perf/external/ramulator2_wrapper/configs/hbm2.yaml";
            let mut mem_context = RamulatorContext::new(config_file, (5u32, 5u32), None);
            mem_context.add_reader(ReadBundle {
                addr: Box::new(addr_rcv1),
                resp: Box::new(rdata_snd1),
                resp_addr: Box::new(resp_addr_snd1),
            });

            mem_context.add_reader(ReadBundle {
                addr: Box::new(addr_rcv2),
                resp: Box::new(rdata_snd2),
                resp_addr: Box::new(resp_addr_snd2),
            });

            ctx.add_child(mem_context);

            // ====================== Consumer ======================
            ctx.add_child(ConsumerContext::new(mm_rcv));

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
    */

    #[test]
    fn test_gen_q() {
        /*
        ADDR_OFFSET = (Channel Width) x (Burst Length) = 64 bytes
        - Channel Width: 16 bytes/channel
            - HBM2 standard (JEDEC HBM2 specification) defines each pseudo-channel width explicitly as 16 bytes/channel
        - Burst Length: 4
            - HBM2 standard (JEDEC HBM2 specification) specifies a burst length of 4 beats per DRAM access.
         */
        const ADDR_OFFSET: u64 = 64; // 32 elements in this test case

        const B: usize = 32;
        const H: usize = 64;

        let n_byte = 2;

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
        let (rdata_snd1, rdata_rcv1) = ctx.unbounded();
        let (repeat_snd1, repeat_rcv1) = ctx.bounded(1);

        let mat1 = OffChipLoad2D::<InputLoad>::new(
            [B / tile_m_gen_q, H / tile_k_gen_q], // As we don't tile K, the second element is 1
            vec![H / tile_k_gen_q, 1],
            vec![B / tile_m_gen_q, H / tile_k_gen_q],
            tile_m_gen_q,
            tile_k_gen_q,
            n_byte as usize,
            tensor_addrs.get("Input").unwrap().clone() as u64,
            ADDR_OFFSET,
            addr_snd1,
            resp_addr_rcv1,
            rdata_rcv1,
            repeat_snd1,
        );

        let (on_chip_snd1, on_chip_rcv1) = ctx.bounded(1);

        let repeat_mat1 = RepeatStatic::new(repeat_rcv1, H / tile_n_gen_q, on_chip_snd1);
        ctx.add_child(mat1);
        ctx.add_child(repeat_mat1);

        // Operand 2 (W_Q): [H,H]
        // Stream shape: [H, H/tileN]
        // Tile shape: [H, tileN]
        // Repeat2D by number of M tiles
        let (addr_snd2, addr_rcv2) = ctx.unbounded();
        let (resp_addr_snd2, resp_addr_rcv2) = ctx.unbounded();
        let (rdata_snd2, rdata_rcv2) = ctx.unbounded();
        let (on_chip_snd2, on_chip_rcv2) = ctx.bounded(1);

        // For the weights, we will assume it's saved in a transposed order
        let mat2 = OffChipLoad2D::<WeightQLoad>::new(
            [H / tile_k_gen_q, H / tile_n_gen_q], // As we don't tile K, the second element is 1
            vec![0, 1, H / tile_n_gen_q],
            vec![B / tile_m_gen_q, H / tile_k_gen_q, H / tile_n_gen_q],
            tile_k_gen_q,
            tile_n_gen_q,
            n_byte as usize,
            tensor_addrs.get("W_Q").unwrap().clone() as u64,
            ADDR_OFFSET,
            addr_snd2,
            resp_addr_rcv2,
            rdata_rcv2,
            on_chip_snd2,
        );

        ctx.add_child(mat2);

        // ====================== Matmul Context ======================
        // Size of per-tile computation: [16, H] * [H, tileN] = [16 ,tileN]
        /* As the computation has a long reduction dimension, use a output stationary systolic array of shape [16, tileN] = [16, 16]
           This roughly maps to 3 PCUs (16*6 * 3), which has an approximate of 638 * 1e12 (FLOPs/s) / 1040 * 3 * 1/(1.8*1e9) (s/cycle) = 1022 (FLOPs/cycle)
        */
        let (mm_snd, mm_rcv) = ctx.bounded(1);
        let gen_q = BinaryMapAccum::<GenQ>::new(
            on_chip_rcv1,
            on_chip_rcv2,
            mm_snd,
            Arc::new(|tile1, tile2, accumulator, comp_bw, write_back_mu| {
                matmul(tile1, tile2, accumulator, comp_bw, write_back_mu, false)
            }),
            Arc::new(move || Tile {
                shape: vec![16, tile_n_gen_q],
                bytes_per_elem: n_byte,
                read_from_mu: true,
            }),
            1,
            1022,
            true,
        );

        ctx.add_child(gen_q);

        // ====================== Store Context ======================

        // ====================== Ramulator Context ======================

        let config_file = "/home/ginasohn/step-perf/external/ramulator2_wrapper/configs/hbm2.yaml";
        let mut mem_context = RamulatorContext::new(config_file, (5u32, 5u32), None);
        mem_context.add_reader(ReadBundle {
            addr: Box::new(addr_rcv1),
            resp: Box::new(rdata_snd1),
            resp_addr: Box::new(resp_addr_snd1),
        });

        mem_context.add_reader(ReadBundle {
            addr: Box::new(addr_rcv2),
            resp: Box::new(rdata_snd2),
            resp_addr: Box::new(resp_addr_snd2),
        });

        ctx.add_child(mem_context);

        // ====================== Consumer ======================
        ctx.add_child(ConsumerContext::new(mm_rcv));

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
