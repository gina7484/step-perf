pub mod functions;
pub mod memory;
pub mod operator;
pub mod primitives;
pub mod ramulator;
pub mod test;
pub mod utils;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[pyfunction]
fn run_graph(py: Python) -> (bool) {
    println!("From run_graph in Rust");

    return true;
}

#[pymodule]
fn step_perf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_graph, m)?)?;
    //    m.add_function(wrap_pyfunction!(run_graph_f64, m)?)?;
    Ok(())
}
