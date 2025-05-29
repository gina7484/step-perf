use dam::types::StaticallySized;

use super::elem::Bufferizable;

pub trait SelectAdapter {
    fn to_sel_vec(&self) -> Vec<usize>;
}

// Two options for the select type
// 1. Multi Hot [Bool; N]
// 2. Index [Option<usize>; K]
//      - K=number of experts to choose each time
//      - When we don't choose any of them, it's None
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MultiHotN<const N: usize> {
    underlying: Option<[bool; N]>,
    read_from_mu: bool,
}

impl<const N: usize> MultiHotN<N> {
    pub fn new(arr: [bool; N], read_from_mu: bool) -> Self {
        Self {
            underlying: Some(arr),
            read_from_mu,
        }
    }
}

impl<const N: usize> std::ops::Deref for MultiHotN<N> {
    type Target = [bool; N];

    fn deref(&self) -> &Self::Target {
        self.underlying
            .as_ref()
            .expect("Can't deref a null buffer!")
    }
}

impl<const N: usize> std::ops::DerefMut for MultiHotN<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.underlying
            .as_mut()
            .expect("Can't deref_mut a null buffer!")
    }
}

impl<const N: usize> StaticallySized for MultiHotN<N> {
    const SIZE: usize = bool::SIZE * N;
}

impl<const N: usize> SelectAdapter for MultiHotN<N> {
    fn to_sel_vec(&self) -> Vec<usize> {
        let vec: Vec<bool> = self.underlying.unwrap().to_vec();
        let mut res_vec: Vec<usize> = vec![];
        for (idx, data) in vec.iter().enumerate() {
            if *data {
                res_vec.push(idx);
            }
        }
        res_vec
    }
}

impl<const N: usize> Bufferizable for MultiHotN<N> {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<bool>() * N
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndexN<const N: usize> {
    underlying: Option<[Option<usize>; N]>,
    read_from_mu: bool,
}

impl<const N: usize> IndexN<N> {
    pub fn new(arr: [Option<usize>; N], read_from_mu: bool) -> Self {
        Self {
            underlying: Some(arr),
            read_from_mu,
        }
    }
}

impl<const N: usize> std::ops::Deref for IndexN<N> {
    type Target = [Option<usize>; N];

    fn deref(&self) -> &Self::Target {
        self.underlying
            .as_ref()
            .expect("Can't deref a null buffer!")
    }
}

impl<const N: usize> std::ops::DerefMut for IndexN<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.underlying
            .as_mut()
            .expect("Can't deref_mut a null buffer!")
    }
}

impl<const N: usize> StaticallySized for IndexN<N> {
    const SIZE: usize = N;
}

impl<const N: usize> SelectAdapter for IndexN<N> {
    fn to_sel_vec(&self) -> Vec<usize> {
        let vec: Vec<Option<usize>> = self.underlying.unwrap().to_vec();
        let mut res_vec: Vec<usize> = vec![];
        for data in vec.iter() {
            match data {
                Some(x) => {
                    res_vec.push(*x);
                }
                None => {}
            }
        }
        res_vec
    }
}

impl<const N: usize> Bufferizable for IndexN<N> {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<usize>() * N
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu
    }
}

#[cfg(test)]
mod tests {
    use super::SelectAdapter;
    use crate::primitives::select::{IndexN, MultiHotN};
    use dam::types::DAMType;

    #[test]
    fn test_one_hot() {
        let one_hot_a: MultiHotN<2> = MultiHotN::new([false, true], false);
        let one_hot_b: MultiHotN<3> = MultiHotN::new([false, true, true], false);
        let one_hot_c: MultiHotN<16> = MultiHotN::new([
            false, true, true, false, false, false, false, false, false, false, false, false,
            false, false, false, false,
        ], false);

        assert!(one_hot_a.to_sel_vec() == vec![1usize]);
        assert!(one_hot_b.to_sel_vec() == vec![1usize, 2usize]);
        assert!(one_hot_c.to_sel_vec() == vec![1usize, 2usize]);

        assert!(one_hot_a.dam_size() == 2);
        assert!(one_hot_b.dam_size() == 3);
        assert!(one_hot_c.dam_size() == 16);

        dbg!(one_hot_a);
        dbg!(one_hot_b);
        dbg!(one_hot_c);
    }

    #[test]
    fn test_index_list() {
        let index_a: IndexN<2> = IndexN::new([Some(1), Some(2)], false);
        let index_b: IndexN<2> = IndexN::new([Some(1), None],false);
        let index_c: IndexN<3> = IndexN::new([Some(1), Some(2), Some(3)],false);
        let index_d: IndexN<6> =
            IndexN::new([Some(0), Some(1), Some(2), Some(11), Some(20), Some(21)],false);

        assert!(index_a.to_sel_vec() == vec![1usize, 2usize]);
        assert!(index_b.to_sel_vec() == vec![1usize]);
        assert!(index_c.to_sel_vec() == vec![1, 2, 3]);
        assert!(index_d.to_sel_vec() == vec![0, 1, 2, 11, 20, 21]);

        assert!(index_a.dam_size() == 2);
        assert!(index_b.dam_size() == 2);
        assert!(index_c.dam_size() == 3);
        assert!(index_d.dam_size() == 6);

        dbg!(index_a);
        dbg!(index_b);
        dbg!(index_c);
        dbg!(index_d);
    }
}
