use dam::types::StaticallySized;
use ndarray::Array2;

use super::elem::Bufferizable;

/// A tile value. Padding validity is carried by a separate boolean stream.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tile<T> {
    pub shape: Vec<usize>,
    pub bytes_per_elem: usize,
    pub read_from_mu: bool,
    pub underlying: Option<ndarray::ArcArray2<T>>,
    // As tile is treated as 'value' instead of 'reference,
    // we will use Array instead of ArcArray
}
impl<T: StaticallySized> StaticallySized for Tile<T> {
    const SIZE: usize = T::SIZE;
}
impl<T> Bufferizable for Tile<T> {
    fn size_in_bytes(&self) -> usize {
        let total_elems: usize = self.shape.iter().product();
        self.bytes_per_elem * total_elems
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu
    }

    fn clone_with_updated_read_from_mu(&self, read_from_mu: bool) -> Self {
        Self {
            shape: self.shape.clone(),
            bytes_per_elem: self.bytes_per_elem,
            read_from_mu: read_from_mu,
            underlying: self.underlying.clone(),
        }
    }
}
impl<T: Clone> Tile<T> {
    /// This creates a tile with no underlying data
    pub fn new_blank(shape: Vec<usize>, bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: shape.clone(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            underlying: None,
        }
    }
    pub fn new(arr: ndarray::ArcArray2<T>, bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr.shape().to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            underlying: Some(arr),
        }
    }

    /// Returns a copy of this tile with the `underlying` data dropped (set to `None`).
    /// All other fields (shape, bytes_per_elem, read_from_mu) are preserved.
    pub fn drop_underlying(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            bytes_per_elem: self.bytes_per_elem,
            read_from_mu: self.read_from_mu,
            underlying: None,
        }
    }

    /// This is used for the accumulator in the retile_col or retile_row function.
    /// It contains an 0-sized dimension.
    /// * Tile Shape: arr_shape (should contain 0-sized dimension)
    /// * Tile content: [] (empty array)
    pub fn new_empty(arr_shape: [usize; 2], bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            underlying: Some(
                Array2::from_shape_vec((arr_shape[0], arr_shape[1]), vec![])
                    .unwrap()
                    .to_shared(),
            ),
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
    pub fn new_neg_inf(arr_shape: [usize; 2], bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            underlying: Some(ndarray::ArcArray2::from_elem(arr_shape, T::neg_infinity())),
        }
    }
}

impl<T: Clone + num::Zero> Tile<T> {
    /// Returns a zero tile (All rows are active. No padding.)
    /// * Tile Shape: arr_shape
    /// * Tile content: all zeros
    pub fn new_zero(arr_shape: [usize; 2], bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: bytes_per_elem,
            read_from_mu: read_from_mu,
            underlying: Some(ndarray::ArcArray2::zeros(arr_shape)),
        }
    }
}
