use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub channel_depth: Option<usize>,
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

        Ok(SimConfig { channel_depth })
    }
}
