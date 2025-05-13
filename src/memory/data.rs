use dam::types::StaticallySized;

use crate::ramulator::access::MemoryData;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tile {
    pub shape: Vec<usize>,
    pub bytes_per_elem: usize,
    pub read_from_mu: bool,
}

impl StaticallySized for Tile {
    const SIZE: usize = 8;
}

impl Tile {
    pub fn size_in_bytes(&self) -> usize {
        let total_elems: usize = self.shape.iter().product();
        self.bytes_per_elem * total_elems
    }
}
