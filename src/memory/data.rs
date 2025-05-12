use dam::types::StaticallySized;

use crate::ramulator::access::MemoryData;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DataSizeInfo {
    pub bytes: usize,
}

impl StaticallySized for DataSizeInfo {
    const SIZE: usize = 8;
}
