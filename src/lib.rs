pub mod build_sim;
pub mod functions;
pub mod memory;
pub mod operator;
pub mod primitives;
pub mod proto_driver;
pub mod ramulator;
pub mod test;
pub mod utils;

use std::collections::HashMap;
use std::fs;
use std::io::repeat;
use std::sync::Arc;

use prost::Message;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::proto_driver::configs::SimConfig;
use crate::proto_driver::parse_proto;
use crate::proto_driver::proto_headers::graph_proto::ProgramGraph;
use crate::ramulator::hbm_context::HBMConfig;

#[pyfunction]
#[pyo3(signature = (proto, logging, hbm_config, sim_config, db_name=None, dump_prefix=None))]
/// Return passed, cycles, duration_ms, duration_s, and a traffic dictionary.
/// Traffic contains total_bytes, read_bytes, and write_bytes measured by HBM.
fn run_graph(
    py: Python,
    proto: String,
    logging: bool,
    hbm_config: HBMConfig,
    sim_config: SimConfig,
    db_name: Option<String>,
    // When set, `build_from_proto` dumps `<dump_prefix>.proto.txt` and
    // `<dump_prefix>.nodes.txt` describing the graph it built. Useful for
    // debugging `builder.initialize` failures.
    dump_prefix: Option<String>,
) -> (bool, u64, u128, u64, HashMap<String, u64>) {
    let step_graph: ProgramGraph = {
        let file_contents = fs::read(proto).unwrap();
        ProgramGraph::decode(file_contents.as_slice()).unwrap()
    };

    println!("Successfully read proto file");

    let (passed, cycles, duration, traffic_stats) =
        parse_proto(step_graph, logging, hbm_config, sim_config, db_name.clone(), dump_prefix);

    if logging {
        println!(
            "Log saved to {}",
            db_name.unwrap_or("sim_default_name".to_string())
        );
    }

    let duration_milliseconds = duration.as_millis();
    let duration_seconds = duration.as_secs();
    let traffic = HashMap::from([
        ("total_bytes".to_string(), traffic_stats.total_bytes()),
        ("read_bytes".to_string(), traffic_stats.read_bytes()),
        ("write_bytes".to_string(), traffic_stats.write_bytes()),
    ]);
    (passed, cycles, duration_milliseconds, duration_seconds, traffic)
}

#[pymodule]
fn step_perf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_graph, m)?)?;
    //    m.add_function(wrap_pyfunction!(run_graph_f64, m)?)?;
    Ok(())
}
