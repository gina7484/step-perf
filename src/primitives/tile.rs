use dam::types::StaticallySized;
use ndarray::Array2;

use super::elem::{Bufferizable, EmptyFiber};

/// Tile
/// - offset: If the tile has a padded value, this is the offset expressing
///     non-padded rows (same as the number of non-padded rows in the tile).
///     If none of the rows in the tile are padded, this is same as the number of rows in the tile.
///     If the tile is a padded value, this is 0.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tile<T> {
    pub shape: Vec<usize>,
    pub bytes_per_elem: usize,
    pub read_from_mu: bool,
    pub underlying: Option<ndarray::ArcArray2<T>>,
    pub offset: usize,
    empty_fiber: bool,
    // As tile is treated as 'value' instead of 'reference,
    // we will use Array instead of ArcArray
}
impl<T: StaticallySized> StaticallySized for Tile<T> {
    const SIZE: usize = T::SIZE + 1;
}
impl<T> Bufferizable for Tile<T> {
    fn size_in_bytes(&self) -> usize {
        if self.empty_fiber {
            return 0;
        }
        let total_elems: usize = self.shape.iter().product();
        self.bytes_per_elem * total_elems
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu && !self.empty_fiber
    }

    fn clone_with_updated_read_from_mu(&self, read_from_mu: bool) -> Self {
        Self {
            shape: self.shape.clone(),
            bytes_per_elem: self.bytes_per_elem,
            read_from_mu: read_from_mu && !self.empty_fiber,
            empty_fiber: self.empty_fiber,
            underlying: self.underlying.clone(),
            offset: self.offset,
        }
    }
}
impl<T> Tile<T> {
    pub fn new_empty_fiber(shape: Vec<usize>, bytes_per_elem: usize) -> Self {
        Self {
            shape,
            bytes_per_elem,
            read_from_mu: false,
            underlying: None,
            offset: 0,
            empty_fiber: true,
        }
    }
    pub fn is_empty_fiber(&self) -> bool {
        self.empty_fiber
    }
}

impl<T> EmptyFiber for Tile<T> {
    fn is_empty_fiber(&self) -> bool {
        self.empty_fiber
    }
    fn empty_fiber_like(&self) -> Self {
        Self::new_empty_fiber(self.shape.clone(), self.bytes_per_elem)
    }
}

impl<T: Clone> Tile<T> {
    /// This creates a tile with no underlying data
    pub fn new_blank(shape: Vec<usize>, bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: shape.clone(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: None,
            offset: shape[0],
        }
    }
    pub fn new(arr: ndarray::ArcArray2<T>, bytes_per_elem: usize, read_from_mu: bool) -> Self {
        let rows = arr.shape().to_vec()[0];
        Self {
            shape: arr.shape().to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: Some(arr),
            offset: rows,
        }
    }

    /// Returns a copy of this tile with the `underlying` data dropped (set to `None`).
    /// All other fields (shape, bytes_per_elem, read_from_mu, offset) are preserved.
    pub fn drop_underlying(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            bytes_per_elem: self.bytes_per_elem,
            read_from_mu: self.read_from_mu,
            empty_fiber: self.empty_fiber,
            underlying: None,
            offset: self.offset,
        }
    }

    /// This creates a tile with no underlying data and a padded value
    pub fn new_blank_padded(
        shape: Vec<usize>,
        bytes_per_elem: usize,
        read_from_mu: bool,
        offset: usize,
    ) -> Self {
        Self {
            shape: shape,
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: None,
            offset: offset,
        }
    }

    /// This creates a tile with underlying data and a padded value
    pub fn new_padded(
        arr: ndarray::ArcArray2<T>,
        bytes_per_elem: usize,
        read_from_mu: bool,
        offset: usize,
    ) -> Self {
        Self {
            shape: arr.shape().to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: Some(arr),
            offset: offset,
        }
    }

    /// This is used for the accumulator in the retile_col or retile_row function.
    /// It contains an 0-sized dimension.
    /// * Tile Shape: arr_shape (should contain 0-sized dimension)
    /// * Tile content: [] (empty array)
    /// * Offset: arr_shape[0]
    pub fn new_empty(arr_shape: [usize; 2], bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: Some(
                Array2::from_shape_vec((arr_shape[0], arr_shape[1]), vec![])
                    .unwrap()
                    .to_shared(),
            ),
            offset: arr_shape[0],
        }
    }
}

// Functions to initialize tiles
impl<T: Clone + num::Float> Tile<T> {
    /// Returns a tile filled with negative infinity (All rows are active. No padding.)
    ///
    /// This is the identity for a max reduction: seeding with zero would clamp an
    /// all-negative reduction group to 0. Bounded to `num::Float` because the
    /// integer tile types have no -inf representation.
    /// * Tile Shape: arr_shape
    /// * Tile content: all -inf
    /// * Offset: arr_shape[0]
    pub fn new_neg_inf(arr_shape: [usize; 2], bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: Some(ndarray::ArcArray2::from_elem(arr_shape, T::neg_infinity())),
            offset: arr_shape[0],
        }
    }
}

impl<T: Clone + num::Zero> Tile<T> {
    /// Returns a zero tile (All rows are active. No padding.)
    /// * Tile Shape: arr_shape
    /// * Tile content: all zeros
    /// * Offset: arr_shape[0]
    pub fn new_zero(arr_shape: [usize; 2], bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: Some(ndarray::ArcArray2::zeros(arr_shape)),
            offset: arr_shape[0],
        }
    }

    /// Returns a zero tile specifying that this is tile added due to padding
    /// * Tile Shape: arr_shape
    /// * Tile content: all zeros
    /// * Offset: offset
    pub fn new_zero_padded(
        arr_shape: [usize; 2],
        bytes_per_elem: usize,
        read_from_mu: bool,
        offset: usize,
    ) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            empty_fiber: false,
            underlying: Some(ndarray::ArcArray2::zeros(arr_shape)),
            offset: offset,
        }
    }
}

#[cfg(test)]
mod empty_fiber_tests {
    use super::*;
    use dam::types::DAMType;

    #[test]
    fn empty_fiber_keeps_metadata_but_has_no_data_memory() {
        let empty = Tile::<f32>::new_empty_fiber(vec![2, 4], 4);
        assert!(empty.is_empty_fiber());
        assert_eq!(empty.shape, vec![2, 4]);
        assert_eq!(empty.bytes_per_elem, 4);
        assert!(empty.underlying.is_none());
        assert_eq!(empty.size_in_bytes(), 0);
        assert!(!empty.read_from_mu());
        assert!(!empty.clone_with_updated_read_from_mu(true).read_from_mu);
        assert_eq!(empty.clone_with_updated_read_from_mu(true), empty);
        assert_eq!(empty.drop_underlying(), empty);
        assert_eq!(empty.clone(), empty);
        assert_eq!(empty.dam_size(), 33);
        assert_eq!(
            crate::primitives::elem::Elem::ValStop(empty, 2).dam_size(),
            65
        );
    }

    #[test]
    fn numerical_zero_blank_padding_and_retile_initializer_are_not_empty_fibers() {
        let tiles = [
            Tile::<f32>::new_zero([2, 4], 4, false),
            Tile::new_blank(vec![2, 4], 4, false),
            Tile::new_blank_padded(vec![2, 4], 4, false, 0),
            Tile::new_empty([0, 4], 4, false),
            Tile::default(),
        ];
        for tile in tiles {
            assert!(!tile.is_empty_fiber());
        }
        assert_ne!(
            Tile::<f32>::new_empty_fiber(vec![2, 4], 4),
            Tile::new_blank_padded(vec![2, 4], 4, false, 0)
        );
    }
}
