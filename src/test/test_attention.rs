use dam::dam_macros::event_type;
use dam::{
    simulation::{
        LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
    },
    utility_contexts::ConsumerContext,
};
use serde::{Deserialize, Serialize};

use crate::memory::events::LoggableEvent;

#[derive(Serialize, Debug)]
#[event_type]
struct HBMGenQKV {
    outer: u32,
    m: u32,
    n: u32,
    k: u32,
    start_ns: u64,
    end_ns: u64,
    output_tile_available: bool,
    num_elems: u32,
}

// Implement the trait for HBMGenQKV
impl LoggableEvent for HBMGenQKV {
    fn new(
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    ) -> Self {
        HBMGenQKV {
            outer,
            m,
            n,
            k,
            start_ns,
            end_ns,
            output_tile_available,
            num_elems,
        }
    }
}
impl HBMGenQKV {
    pub const NAME: &'static str = "HBMGenQKV";
}

#[derive(Serialize, Debug)]
#[event_type]
struct HBMQKt {
    outer: u32,
    m: u32,
    n: u32,
    k: u32,
    start_ns: u64,
    end_ns: u64,
    output_tile_available: bool,
    num_elems: u32,
}

// Implement the trait for HBMGenQKV
impl LoggableEvent for HBMQKt {
    fn new(
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    ) -> Self {
        HBMQKt {
            outer,
            m,
            n,
            k,
            start_ns,
            end_ns,
            output_tile_available,
            num_elems,
        }
    }
}
impl HBMQKt {
    pub const NAME: &'static str = "HBMQKt";
}

#[derive(Serialize, Debug)]
#[event_type]
struct HBMAttnV {
    outer: u32,
    m: u32,
    n: u32,
    k: u32,
    start_ns: u64,
    end_ns: u64,
    output_tile_available: bool,
    num_elems: u32,
}

// Implement the trait for HBMGenQKV
impl LoggableEvent for HBMAttnV {
    fn new(
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    ) -> Self {
        HBMAttnV {
            outer,
            m,
            n,
            k,
            start_ns,
            end_ns,
            output_tile_available,
            num_elems,
        }
    }
}
impl HBMAttnV {
    pub const NAME: &'static str = "HBMAttnV";
}

#[derive(Serialize, Debug)]
#[event_type]
struct HBMProj {
    outer: u32,
    m: u32,
    n: u32,
    k: u32,
    start_ns: u64,
    end_ns: u64,
    output_tile_available: bool,
    num_elems: u32,
}

// Implement the trait for HBMProj
impl LoggableEvent for HBMProj {
    fn new(
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    ) -> Self {
        HBMProj {
            outer,
            m,
            n,
            k,
            start_ns,
            end_ns,
            output_tile_available,
            num_elems,
        }
    }
}
impl HBMProj {
    pub const NAME: &'static str = "HBMProj";
}

#[derive(Serialize, Debug)]
#[event_type]
struct HBMOutput {
    outer: u32,
    m: u32,
    n: u32,
    k: u32,
    start_ns: u64,
    end_ns: u64,
    output_tile_available: bool,
    num_elems: u32,
}

// Implement the trait for HBMProj
impl LoggableEvent for HBMOutput {
    fn new(
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    ) -> Self {
        HBMOutput {
            outer,
            m,
            n,
            k,
            start_ns,
            end_ns,
            output_tile_available,
            num_elems,
        }
    }
}
impl HBMOutput {
    pub const NAME: &'static str = "HBMOutput";
}

#[cfg(test)]
mod test_attention {
    use dam::{
        simulation::{
            DotConvertible, LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder,
            RunOptionsBuilder,
        },
        utility_contexts::CheckerContext,
    };

    use super::{HBMAttnV, HBMGenQKV, HBMOutput, HBMProj, HBMQKt};
    use crate::memory::hbm_ld_st::{HBMLoadContext, HBMStoreContext};
    use crate::operator::batchedmatvec::{AttnV, QKt};
    use crate::operator::matmul::{GenQKV, Proj};

    #[test]
    fn test_attention() {
        let mut ctx = ProgramBuilder::default();

        // HBM Loaders
        let (load_gen_qkv_snd, load_gen_qkv_rcv) = ctx.bounded(2);
        let (load_q_kt_snd, load_q_kt_rcv) = ctx.bounded(2);
        let (load_attn_v_snd, load_attn_v_rcv) = ctx.bounded(2);
        let (load_proj_snd, load_proj_rcv) = ctx.bounded(2);

        ctx.add_child(HBMLoadContext::<HBMGenQKV>::new(
            "gen_qkv.csv".to_string(),
            load_gen_qkv_snd,
        ));

        ctx.add_child(HBMLoadContext::<HBMQKt>::new(
            "q_kt.csv".to_string(),
            load_q_kt_snd,
        ));

        ctx.add_child(HBMLoadContext::<HBMAttnV>::new(
            "attn_v.csv".to_string(),
            load_attn_v_snd,
        ));

        ctx.add_child(HBMLoadContext::<HBMProj>::new(
            "proj.csv".to_string(),
            load_proj_snd,
        ));

        // Compute Allocation

        let rda_compute_bw = 613.0; // FLOPs/ns

        let gen_qkv_compute = rda_compute_bw * 0.31;
        let q_kt_compute = rda_compute_bw * 0.0091;
        let attn_v_compute = rda_compute_bw * 0.0087;
        let proj_compute = rda_compute_bw * 0.1;

        // Flop count
        let H = 8192; // hidden size;
        let gen_qkv_flop = 1 * H * 3 * H;
        let q_kt_flop = vec![
            3087744, 3269376, 7257024, 751296, 751296, 3145536, 10840128, 3203328, 1997952,
            1725504, 3252864, 3252864, 10856640, 18336576, 3211584, 3426240, 990720, 3046464,
            1700736, 11170368, 1626432, 1494336, 3203328, 33725760,
        ];
        let attn_v_flop = vec![
            3063808, 3244032, 7200768, 745472, 745472, 3121152, 10756096, 3178496, 1982464,
            1712128, 3227648, 3227648, 10772480, 18194432, 3186688, 3399680, 983040, 3022848,
            1687552, 11083776, 1613824, 1482752, 3178496, 33464320,
        ];

        let q_kt_flop_repeated: Vec<_> = q_kt_flop
            .iter()
            .flat_map(|&x| std::iter::repeat(x).take(64))
            .collect();
        let attn_v_flop_repeated: Vec<_> = attn_v_flop
            .iter()
            .flat_map(|&x| std::iter::repeat(x).take(64))
            .collect();

        let proj_flop = 1 * H * H;

        // Operations
        let (gen_qkv_snd, gen_qkv_rcv) = ctx.bounded(2);
        let (q_kt_snd, q_kt_rcv) = ctx.bounded(2);
        let (attn_v_snd, attn_v_rcv) = ctx.bounded(2);
        let (proj_snd, proj_rcv) = ctx.bounded(2);

        ctx.add_child(GenQKV::new(
            load_gen_qkv_rcv,
            gen_qkv_snd,
            gen_qkv_flop,
            gen_qkv_compute,
        ));
        ctx.add_child(QKt::new(
            gen_qkv_rcv,
            load_q_kt_rcv,
            q_kt_snd,
            q_kt_flop_repeated,
            q_kt_compute,
        ));
        ctx.add_child(AttnV::new(
            q_kt_rcv,
            load_attn_v_rcv,
            attn_v_snd,
            attn_v_flop_repeated,
            attn_v_compute,
        ));
        ctx.add_child(Proj::new(
            attn_v_rcv,
            load_proj_rcv,
            proj_snd,
            proj_flop,
            proj_compute,
        ));

        // HBM Store Context
        ctx.add_child(HBMStoreContext::<HBMOutput>::new(
            "output.csv".to_string(),
            proj_rcv,
        ));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
            // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
            dam::logging::LogFilter::AllowAll,
        ));
        let run_options = run_options.logging(LoggingOptions::Mongo(
            MongoOptionsBuilder::default()
                .db("attn_log".to_string())
                .uri("mongodb://127.0.0.1:27017".to_string())
                .build()
                .unwrap(),
        ));
        let summary = initialized.run(run_options.build().unwrap());
        // Check the summary
        println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
    }
}
