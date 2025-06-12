use dam::types::StaticallySized;
use ndarray::Array2;

use super::elem::Bufferizable;

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
}
impl<T> Tile<T> {
    pub fn new_blank(shape: Vec<usize>, bytes_per_elem: usize, read_from_mu: bool) -> Self {
        Self {
            shape: shape,
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
}

impl<T: Clone + num::Zero> Tile<T> {
    pub fn new_zero(arr_shape: [usize; 2], read_from_mu: bool) -> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: std::mem::size_of::<T>(),
            read_from_mu: read_from_mu,
            underlying: Some(ndarray::ArcArray2::zeros(arr_shape)),
        }
    }

    pub fn new_empty(arr_shape: [usize; 2],read_from_mu: bool)-> Self {
        Self {
            shape: arr_shape.to_vec(),
            bytes_per_elem: std::mem::size_of::<T>(),
            read_from_mu: read_from_mu,
            underlying: Some(Array2::from_shape_vec((arr_shape[0], arr_shape[1]), vec![],).unwrap().to_shared()),
        }
    }
}
