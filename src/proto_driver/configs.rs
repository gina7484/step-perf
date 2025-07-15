use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub channel_depth: Option<usize>,
    pub functional_sim: bool,
}

impl<'py> FromPyObject<'py> for SimConfig {
    fn extract_bound(obj: &pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        // Retrieve the channel_depth attribute from the object
        let channel_depth_obj = obj.getattr("channel_depth").map_err(|_| {
            PyTypeError::new_err("Expected 'channel_depth' attribute in SimConfig object")
        })?;

        // Extract the field into the appropriate type - handle None case
        let channel_depth: Option<usize> = if channel_depth_obj.is_none() {
            None
        } else {
            let value: usize = channel_depth_obj.extract().map_err(|_| {
                PyTypeError::new_err("Expected 'channel_depth' to be an integer or None")
            })?;
            Some(value)
        };

        // Retrieve the functional_sim attribute from the object
        let functional_sim_obj = obj.getattr("functional_sim").map_err(|_| {
            PyTypeError::new_err("Expected 'functional_sim' attribute in SimConfig object")
        })?;

        // Extract the functional_sim field
        let functional_sim: bool = functional_sim_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("Expected 'functional_sim' to be a boolean"))?;

        Ok(SimConfig {
            channel_depth,
            functional_sim,
        })
    }
}
