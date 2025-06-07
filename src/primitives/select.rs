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
pub struct MultiHotN {
    underlying: Vec<bool>,
    read_from_mu: bool,
}

impl MultiHotN {
    pub fn new(arr: Vec<bool>, read_from_mu: bool) -> Self {
        Self {
            underlying: arr,
            read_from_mu,
        }
    }

    pub fn len(&self) -> usize {
        self.underlying.len()
    }
}

impl std::ops::Deref for MultiHotN {
    type Target = Vec<bool>;

    fn deref(&self) -> &Self::Target {
        self.underlying.as_ref()
    }
}

impl StaticallySized for MultiHotN {
    const SIZE: usize = bool::SIZE * 64;
}

impl SelectAdapter for MultiHotN {
    fn to_sel_vec(&self) -> Vec<usize> {
        let mut res_vec: Vec<usize> = vec![];
        for (idx, data) in self.underlying.iter().enumerate() {
            if *data {
                res_vec.push(idx);
            }
        }
        res_vec
    }
}

impl Bufferizable for MultiHotN {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<bool>() * self.underlying.len()
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndexN {
    underlying: Vec<Option<usize>>,
    read_from_mu: bool,
}

impl IndexN {
    pub fn new(data: Vec<Option<usize>>, read_from_mu: bool) -> Self {
        Self {
            underlying: data,
            read_from_mu,
        }
    }
}

impl std::ops::Deref for IndexN {
    type Target = Vec<Option<usize>>;

    fn deref(&self) -> &Self::Target {
        self.underlying.as_ref()
    }
}

impl StaticallySized for IndexN {
    const SIZE: usize = 64;
}

impl SelectAdapter for IndexN {
    fn to_sel_vec(&self) -> Vec<usize> {
        let mut res_vec: Vec<usize> = vec![];
        for data in self.underlying.iter() {
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

impl Bufferizable for IndexN {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<usize>() * self.underlying.len()
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
        let one_hot_a: MultiHotN = MultiHotN::new(vec![false, true], false);
        let one_hot_b: MultiHotN = MultiHotN::new(vec![false, true, true], false);
        let one_hot_c: MultiHotN = MultiHotN::new(
            vec![
                false, true, true, false, false, false, false, false, false, false, false, false,
                false, false, false, false,
            ],
            false,
        );

        assert!(one_hot_a.to_sel_vec() == vec![1usize]);
        assert!(one_hot_b.to_sel_vec() == vec![1usize, 2usize]);
        assert!(one_hot_c.to_sel_vec() == vec![1usize, 2usize]);

        assert!(one_hot_a.dam_size() == 64);
        assert!(one_hot_b.dam_size() == 64);
        assert!(one_hot_c.dam_size() == 64);

        dbg!(one_hot_a);
        dbg!(one_hot_b);
        dbg!(one_hot_c);
    }

    #[test]
    fn test_index_list() {
        let index_a: IndexN = IndexN::new(vec![Some(1), Some(2)], false);
        let index_b: IndexN = IndexN::new(vec![Some(1), None], false);
        let index_c: IndexN = IndexN::new(vec![Some(1), Some(2), Some(3)], false);
        let index_d: IndexN = IndexN::new(
            vec![Some(0), Some(1), Some(2), Some(11), Some(20), Some(21)],
            false,
        );

        assert!(index_a.to_sel_vec() == vec![1usize, 2usize]);
        assert!(index_b.to_sel_vec() == vec![1usize]);
        assert!(index_c.to_sel_vec() == vec![1, 2, 3]);
        assert!(index_d.to_sel_vec() == vec![0, 1, 2, 11, 20, 21]);

        assert!(index_a.dam_size() == 64);
        assert!(index_b.dam_size() == 64);
        assert!(index_c.dam_size() == 64);
        assert!(index_d.dam_size() == 64);

        dbg!(index_a);
        dbg!(index_b);
        dbg!(index_c);
        dbg!(index_d);
    }
}
