pub mod build_sim;
pub mod functions;
pub mod memory;
pub mod operator;
pub mod primitives;
pub mod proto_driver;
pub mod ramulator;
pub mod test;
pub mod utils;

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
fn run_graph(
    py: Python,
    proto: String,
    logging: bool,
    hbm_config: HBMConfig,
    sim_config: SimConfig,
    db_name: Option<String>,
) -> (bool, u64, u128, u64) {
    let step_graph: ProgramGraph = {
        let file_contents = fs::read(proto).unwrap();
        ProgramGraph::decode(file_contents.as_slice()).unwrap()
    };

    println!("Successfully read proto file");

    let (passed, cycles, duration) =
        parse_proto(step_graph, logging, hbm_config, sim_config, db_name.clone());

    println!(
        "Passed: {}, Elapsed Cycles: {}, Duration: {:?}",
        passed, cycles, duration
    );

    if logging {
        println!(
            "Log saved to {}",
            db_name.unwrap_or("sim_default_name".to_string())
        );
    }

    // Convert duration to milliseconds as f64 for Python (better precision for short durations)
    let duration_milliseconds = duration.as_millis();
    let duration_seconds = duration.as_secs();
    return (passed, cycles, duration_milliseconds, duration_seconds);
}

#[pymodule]
fn step_perf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_graph, m)?)?;
    //    m.add_function(wrap_pyfunction!(run_graph_f64, m)?)?;
    Ok(())
}
