pub mod batchedmatvec;
pub mod matmul;

use dam::types::StaticallySized;
use serde::Deserialize;

/// This expresses the data being streamed directly between compute units
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct ActEntry {}

impl StaticallySized for ActEntry {
    const SIZE: usize = 1;
}
