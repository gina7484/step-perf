pub mod accum;
pub mod bufferize;
pub mod map;
pub mod map_accum;
// pub mod mux_demux;
pub mod broadcast;
pub mod dynstreamify;
pub mod eager_merge;
pub mod expand;
pub mod flatmap;
pub mod flatten;
pub mod parallelize;
pub mod partition;
pub mod promote;
pub mod reassemble;
pub mod repeat;
pub mod reshape;
pub mod streamify;

use dam::types::StaticallySized;
use serde::Deserialize;

/// This expresses the data being streamed directly between compute units
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct ActEntry {}

impl StaticallySized for ActEntry {
    const SIZE: usize = 1;
}
