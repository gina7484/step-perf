pub mod dyn_offchip_load;
pub mod metadata_gen;
pub mod offchip_load;
pub mod offchip_store;
pub mod random_offchip_load;
pub mod random_offchip_store;

/// PMU bandwidth (bytes/cycle)
pub static PMU_BW: u64 = 64;

use crate::primitives::{elem::StopType, tile::Tile};
use dam::types::DAMType;

#[derive(Debug)]
pub enum HbmAddrEnum<T: DAMType> {
    ADDR(Vec<u64>, Tile<T>),
    ADDRSTOP(Vec<u64>, Tile<T>, StopType),
}
