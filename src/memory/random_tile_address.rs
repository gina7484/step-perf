use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RandomTileAddressError {
    #[error("random tile addressing requires allocation rank >= 2, got {0}")]
    Rank(usize),
    #[error("random tile addressing requires nonzero tile geometry and element width")]
    ZeroGeometry,
    #[error("random tile addressing requires a nonzero HBM address increment")]
    ZeroAddressIncrement,
    #[error("random tile index {index} is outside allocation tile count {tile_count}")]
    OutOfBounds { index: u64, tile_count: u64 },
    #[error("random tile address calculation overflowed u64")]
    Overflow,
}

fn checked_mul(lhs: u64, rhs: u64) -> Result<u64, RandomTileAddressError> {
    lhs.checked_mul(rhs).ok_or(RandomTileAddressError::Overflow)
}

fn checked_add(lhs: u64, rhs: u64) -> Result<u64, RandomTileAddressError> {
    lhs.checked_add(rhs).ok_or(RandomTileAddressError::Overflow)
}

/// Translate one row-major tile index into the byte addresses issued to HBM.
///
/// Leading allocation dimensions form consecutive two-dimensional slabs. The
/// last two dimensions are the tile-row and tile-column grid. A tile with more
/// than one row therefore issues each row at the allocation's physical row
/// stride instead of treating the whole tile as one contiguous byte interval.
pub fn random_tile_byte_addresses(
    tensor_shape_tiled: &[usize],
    tile_idx: u64,
    tile_row: usize,
    tile_col: usize,
    n_byte: usize,
    base_addr_byte: u64,
    addr_offset: u64,
) -> Result<Vec<u64>, RandomTileAddressError> {
    if tensor_shape_tiled.len() < 2 {
        return Err(RandomTileAddressError::Rank(tensor_shape_tiled.len()));
    }
    if tile_row == 0 || tile_col == 0 || n_byte == 0 {
        return Err(RandomTileAddressError::ZeroGeometry);
    }
    if addr_offset == 0 {
        return Err(RandomTileAddressError::ZeroAddressIncrement);
    }

    let shape = tensor_shape_tiled
        .iter()
        .map(|&dim| u64::try_from(dim).map_err(|_| RandomTileAddressError::Overflow))
        .collect::<Result<Vec<_>, _>>()?;
    let tile_count = shape
        .iter()
        .try_fold(1_u64, |count, &dim| checked_mul(count, dim))?;
    if tile_idx >= tile_count {
        return Err(RandomTileAddressError::OutOfBounds {
            index: tile_idx,
            tile_count,
        });
    }

    let grid_rows = shape[shape.len() - 2];
    let grid_cols = shape[shape.len() - 1];
    let tiles_per_slab = checked_mul(grid_rows, grid_cols)?;
    let slab_idx = tile_idx / tiles_per_slab;
    let within_slab = tile_idx % tiles_per_slab;
    let tile_grid_row = within_slab / grid_cols;
    let tile_grid_col = within_slab % grid_cols;

    let tile_row = u64::try_from(tile_row).map_err(|_| RandomTileAddressError::Overflow)?;
    let tile_col = u64::try_from(tile_col).map_err(|_| RandomTileAddressError::Overflow)?;
    let n_byte = u64::try_from(n_byte).map_err(|_| RandomTileAddressError::Overflow)?;
    let tile_row_bytes = checked_mul(tile_col, n_byte)?;
    let physical_row_bytes = checked_mul(grid_cols, tile_row_bytes)?;
    let physical_rows_per_slab = checked_mul(grid_rows, tile_row)?;
    let first_physical_row = checked_add(
        checked_mul(slab_idx, physical_rows_per_slab)?,
        checked_mul(tile_grid_row, tile_row)?,
    )?;
    let first_col_byte = checked_mul(tile_grid_col, tile_row_bytes)?;

    let mut addresses = Vec::new();
    for row in 0..tile_row {
        let physical_row = checked_add(first_physical_row, row)?;
        let row_addr = checked_add(
            base_addr_byte,
            checked_add(
                checked_mul(physical_row, physical_row_bytes)?,
                first_col_byte,
            )?,
        )?;
        let mut col_byte = 0_u64;
        while col_byte < tile_row_bytes {
            addresses.push(checked_add(row_addr, col_byte)?);
            col_byte = checked_add(col_byte, addr_offset)?;
        }
    }
    Ok(addresses)
}

#[cfg(test)]
mod tests {
    use super::{random_tile_byte_addresses, RandomTileAddressError};

    #[test]
    fn rank_two_row_major_addresses() {
        let addresses = random_tile_byte_addresses(&[2, 3], 4, 1, 2, 4, 100, 4).unwrap();
        assert_eq!(addresses, vec![132, 136]);
    }

    #[test]
    fn rank_three_flattens_leading_dimensions() {
        let addresses = random_tile_byte_addresses(&[2, 2, 3], 7, 1, 2, 4, 100, 4).unwrap();
        assert_eq!(addresses, vec![156, 160]);
    }

    #[test]
    fn multirow_tile_uses_physical_row_stride() {
        let addresses = random_tile_byte_addresses(&[2, 3], 1, 2, 4, 1, 100, 2).unwrap();
        assert_eq!(addresses, vec![104, 106, 116, 118]);
    }

    #[test]
    fn rejects_out_of_bounds_tile_before_address_generation() {
        assert_eq!(
            random_tile_byte_addresses(&[2, 3, 4], 24, 1, 1, 4, 0, 4),
            Err(RandomTileAddressError::OutOfBounds {
                index: 24,
                tile_count: 24,
            })
        );
    }

    #[test]
    fn preserves_a_base_above_four_gibibytes() {
        let base = (1_u64 << 32) + 128;
        assert_eq!(
            random_tile_byte_addresses(&[1, 1], 0, 1, 1, 8, base, 8).unwrap(),
            vec![base]
        );
    }
}
