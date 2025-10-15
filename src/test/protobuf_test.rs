use crate::proto_driver::proto_headers::graph_proto::ProgramGraph;

pub fn print_proto(step_graph: ProgramGraph) {
    for operation in step_graph.operators {
        println!("processing {:?}\n", operation);
    }
}

#[cfg(test)]
mod test {
    use crate::proto_driver::configs::SimConfig;
    use crate::proto_driver::parse_proto;
    use prost::Message;
    use std::collections::HashMap;
    use std::fs;

    use super::print_proto;

    use crate::proto_driver::proto_headers::graph_proto::ProgramGraph;
    use crate::ramulator::hbm_context::HBMConfig;

    #[test]
    fn test_print_proto() {
        let proto = "graph.pb";
        let step_graph: ProgramGraph = {
            let file_contents = fs::read(proto).unwrap();
            ProgramGraph::decode(file_contents.as_slice()).unwrap()
        };
        print_proto(step_graph);
    }

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

        let (passed, cycles, duration) = parse_proto(
            step_graph,
            logging,
            HBMConfig {
                addr_offset: 64, // 32 elements in this test case
                channel_num: 32,
                per_channel_latency: 2,
                per_channel_init_interval: 2,
                per_channel_outstanding: 1,
                per_channel_start_up_time: 14,
            },
            SimConfig {
                channel_depth: Some(16),
                functional_sim: false,
                mock_bf16: false,
                config_dict: HashMap::new(),
            },
            db_name,
        );

        println!(
            "Passed: {}, Elapsed Cycles: {}, Duration: {:?}",
            passed, cycles, duration
        );
    }
}
