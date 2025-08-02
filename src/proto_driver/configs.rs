use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub channel_depth: Option<usize>,
    pub functional_sim: bool,
    pub mock_bf16: bool,
    pub config_dict: HashMap<u32, usize>,
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

        // Retrieve the mock_bf16 attribute from the object
        let mock_bf16_obj = obj.getattr("mock_bf16").map_err(|_| {
            PyTypeError::new_err("Expected 'mock_bf16' attribute in SimConfig object")
        })?;

        // Extract the mock_bf16 field
        let mock_bf16: bool = mock_bf16_obj
            .extract()
            .map_err(|_| PyTypeError::new_err("Expected 'mock_bf16' to be a boolean"))?;

        // Retrieve the config_dict attribute from the object
        let config_dict_obj = obj.getattr("config_dict").map_err(|_| {
            PyTypeError::new_err("Expected 'config_dict' attribute in SimConfig object")
        })?;

        // Extract the config_dict field - handle None case
        let config_dict: HashMap<u32, usize> = if config_dict_obj.is_none() {
            HashMap::new()
        } else {
            config_dict_obj.extract().map_err(|_| {
                PyTypeError::new_err("Expected 'config_dict' to be a dictionary with integer keys and values, or None")
            })?
        };

        Ok(SimConfig {
            channel_depth,
            functional_sim,
            mock_bf16,
            config_dict,
        })
    }
}
