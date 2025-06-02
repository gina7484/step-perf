pub mod functions;
pub mod memory;
pub mod operator;
pub mod primitives;
pub mod proto_driver;
pub mod ramulator;
pub mod test;
pub mod utils;

use std::fs;

use prost::Message;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::proto_driver::proto_headers::graph_proto::ProgramGraph;

#[pyfunction]
fn run_graph(py: Python, proto: String) -> (bool, u64) {
    let step_graph: ProgramGraph = {
        let file_contents = fs::read(proto).unwrap();
        ProgramGraph::decode(file_contents.as_slice()).unwrap()
    };

    // let (passed, cycles) = parse_proto(step_graph);

    // println!("Passed: {}, Elapsed Cycles: {}", passed, cycles);

    // return (passed, cycles);
    (true, 0)
}

#[pymodule]
fn step_perf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_graph, m)?)?;
    //    m.add_function(wrap_pyfunction!(run_graph_f64, m)?)?;
    Ok(())
}
