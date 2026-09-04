pub mod aw_trace;
pub mod dyn_linear_offchip_load;
pub mod dyn_offchip_store;
pub mod linear_offchip_load;
pub mod linear_offchip_load_ref;
pub mod metadata_gen;
pub mod offchip_store;
pub mod random_offchip_load;
pub mod random_offchip_store;

/// PMU bandwidth (bytes/cycle)
pub static PMU_BW: u64 = 512;

/// Byte offset of a tile in a row-major tensor, including tiled columns.
pub fn tile_base_addr(index: u64, columns: usize, rows: usize, cols: usize, bytes: usize) -> u64 {
    let columns = columns as u64;
    ((index / columns) * rows as u64 * columns + index % columns)
        * cols as u64 * bytes as u64
}

use crate::primitives::{elem::StopType, tile::Tile};
use dam::types::DAMType;

#[derive(Debug)]
pub enum HbmAddrEnum<T: DAMType> {
    ADDR(Vec<u64>, Tile<T>),
    ADDRSTOP(Vec<u64>, Tile<T>, StopType),
}
