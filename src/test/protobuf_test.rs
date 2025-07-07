#[cfg(test)]
mod test {
    use std::fs;
    use std::io::repeat;
    use std::sync::Arc;

    use crate::functions;
    use crate::proto_driver::configs::SimConfig;
    use crate::proto_driver::parse_proto;
    use dam::dam_macros::event_type;
    use dam::simulation::{
        DotConvertible, LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder,
        RunOptionsBuilder,
    };
    use frunk::labelled::chars::V;
    use prost::Message;
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use serde::{Deserialize, Serialize};

    use crate::build_sim::channel::ChannelMapCollection;
    use crate::memory::offchip_load::OffChipLoad;
    use crate::memory::offchip_store::OffChipStore;
    use crate::operator::{map::BinaryMap, repeat::RepeatStatic};
    use crate::primitives::tile::Tile;
    use crate::proto_driver::proto_headers::graph_proto::{
        data_type::Type, elemto_elem_func, operation::OpType, ProgramGraph,
    };
    use crate::ramulator::hbm_context::{HBMConfig, HBMContext, ReadBundle, WriteBundle};
    use crate::utils::{cast::to_usize_vec, events::SimpleEvent};

    #[test]
    fn run_graph() {
        let proto = "graph.pb";
        let logging: bool = false;
        let db_name = None;
        let step_graph: ProgramGraph = {
            let file_contents = fs::read(proto).unwrap();
            ProgramGraph::decode(file_contents.as_slice()).unwrap()
        };

        println!("Successfully read proto file");

        let (passed, cycles) = parse_proto(
            step_graph,
            logging,
            HBMConfig {
                addr_offset: 64, // 32 elements in this test case
                channel_num: 8,
                per_channel_latency: 2,
                per_channel_init_interval: 2,
                per_channel_outstanding: 1,
                per_channel_start_up_time: 14,
            },
            SimConfig {
                channel_depth: Some(1024),
            },
            db_name,
        );

        println!("Passed: {}, Elapsed Cycles: {}", passed, cycles);
    }
}
