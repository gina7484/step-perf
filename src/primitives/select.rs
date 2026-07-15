use dam::types::StaticallySized;

use super::elem::Bufferizable;

pub trait SelectAdapter {
    fn to_sel_vec(&self) -> Vec<usize>;
    fn from_sel_vec(sel_vec: Vec<usize>, size: usize, read_from_mu: bool) -> Self;
    /// True if this is a dummy carrying no data (every slot is `None`), as produced
    /// by `new_blank`. Distinct from a real selection with no chosen candidates.
    fn is_blank(&self) -> bool;
}

// Two options for the select type
// 1. Multi Hot [Bool; N]
// 2. Index [Option<usize>; K]
//      - K=number of experts to choose each time
//      - When we don't choose any of them, it's None
// Each slot is `Option<bool>`: `Some(_)` carries real selection data (`Some(true)`
// = selected, `Some(false)` = not selected), while `None` marks a dummy slot with
// no data. A `new_blank` MultiHotN has every slot `None`, which is distinguishable
// from an all-`false` (all `Some(false)`) real selection.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MultiHotN {
    underlying: Vec<Option<bool>>,
    read_from_mu: bool,
}

impl MultiHotN {
    pub fn new(arr: Vec<bool>, read_from_mu: bool) -> Self {
        Self {
            underlying: arr.into_iter().map(Some).collect(),
            read_from_mu,
        }
    }

    /// Creates a dummy MultiHotN carrying only length metadata (no data).
    /// Every candidate slot is `None`, so `to_sel_vec()` is empty and `is_blank()`
    /// is true, while `len()`/`size_in_bytes()` still reflect `len` candidates.
    /// `len` is the number of candidates (the width, same as `from_sel_vec`'s `size`).
    pub fn new_blank(len: usize, read_from_mu: bool) -> Self {
        Self {
            underlying: vec![None; len],
            read_from_mu,
        }
    }

    pub fn len(&self) -> usize {
        self.underlying.len()
    }
}

impl std::ops::Deref for MultiHotN {
    type Target = Vec<Option<bool>>;

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
            if *data == Some(true) {
                res_vec.push(idx);
            }
        }
        res_vec
    }

    fn from_sel_vec(sel_vec: Vec<usize>, size: usize, read_from_mu: bool) -> Self {
        let mut underlying = vec![Some(false); size];
        for idx in sel_vec {
            underlying[idx] = Some(true);
        }
        Self {
            underlying,
            read_from_mu,
        }
    }

    /// A blank (all `None`) MultiHotN is distinct from an all-`false` selection
    /// (all `Some(false)`), which carries real data with no chosen candidates.
    fn is_blank(&self) -> bool {
        self.underlying.iter().all(Option::is_none)
    }
}

impl Bufferizable for MultiHotN {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<bool>() * self.underlying.len()
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu
    }
    fn clone_with_updated_read_from_mu(&self, read_from_mu: bool) -> Self {
        Self {
            underlying: self.underlying.clone(),
            read_from_mu,
        }
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

    /// Creates a dummy IndexN carrying only length metadata (no selection).
    /// All index slots are set to `None`, so `to_sel_vec()` is empty while
    /// `len()`/`size_in_bytes()` still reflect `len` slots.
    /// `len` is the number of index slots (K).
    pub fn new_blank(len: usize, read_from_mu: bool) -> Self {
        Self {
            underlying: vec![None; len],
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

    fn from_sel_vec(sel_vec: Vec<usize>, size: usize, read_from_mu: bool) -> Self {
        let mut underlying = vec![None; sel_vec.len()];
        for (i, &idx) in sel_vec.iter().enumerate() {
            underlying[i] = Some(idx);
        }
        Self {
            underlying,
            read_from_mu,
        }
    }

    /// A blank IndexN has every slot `None`, as produced by `new_blank`.
    fn is_blank(&self) -> bool {
        self.underlying.iter().all(Option::is_none)
    }
}

impl Bufferizable for IndexN {
    fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<usize>() * self.underlying.len()
    }
    fn read_from_mu(&self) -> bool {
        self.read_from_mu
    }
    fn clone_with_updated_read_from_mu(&self, read_from_mu: bool) -> Self {
        Self {
            underlying: self.underlying.clone(),
            read_from_mu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SelectAdapter;
    use crate::primitives::elem::Bufferizable;
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

    #[test]
    fn test_new_blank() {
        // MultiHotN dummy: len candidates, no data (every slot None)
        let blank_mh = MultiHotN::new_blank(8, false);
        assert_eq!(blank_mh.to_sel_vec(), Vec::<usize>::new());
        assert_eq!(blank_mh.len(), 8);
        assert_eq!(blank_mh.size_in_bytes(), std::mem::size_of::<bool>() * 8);
        assert_eq!(blank_mh.read_from_mu, false);
        assert!(blank_mh.is_blank());
        // A blank (all None) is distinguishable from an all-false real selection.
        let all_false = MultiHotN::new(vec![false; 8], false);
        assert!(!all_false.is_blank());
        assert_eq!(all_false.to_sel_vec(), Vec::<usize>::new());
        assert_ne!(*blank_mh, *all_false);

        // IndexN dummy: len slots, all None
        let blank_idx = IndexN::new_blank(3, true);
        assert_eq!(blank_idx.to_sel_vec(), Vec::<usize>::new());
        assert_eq!(blank_idx.len(), 3);
        assert_eq!(blank_idx.size_in_bytes(), std::mem::size_of::<usize>() * 3);
        assert_eq!(blank_idx.read_from_mu, true);
    }

    #[test]
    fn test_from_sel_vec() {
        // Test MultiHotN from_sel_vec
        let sel_vec = vec![1, 3, 5];
        let multi_hot = MultiHotN::from_sel_vec(sel_vec.clone(), 8, false);
        assert_eq!(multi_hot.to_sel_vec(), sel_vec);
        assert_eq!(multi_hot.len(), 8);
        assert_eq!(multi_hot[1], Some(true));
        assert_eq!(multi_hot[3], Some(true));
        assert_eq!(multi_hot[5], Some(true));
        assert_eq!(multi_hot[0], Some(false));
        assert_eq!(multi_hot[2], Some(false));
        assert_eq!(multi_hot[4], Some(false));

        // Test IndexN from_sel_vec
        let index_n = IndexN::from_sel_vec(sel_vec.clone(), 8, false);
        assert_eq!(index_n.to_sel_vec(), sel_vec);
        assert_eq!(index_n.len(), 3);
        assert_eq!(index_n[0], Some(1));
        assert_eq!(index_n[1], Some(3));
        assert_eq!(index_n[2], Some(5));

        // Test empty selection
        let empty_sel = vec![];
        let empty_multi_hot = MultiHotN::from_sel_vec(empty_sel.clone(), 4, true);
        let empty_index = IndexN::from_sel_vec(empty_sel.clone(), 4, true);

        assert_eq!(empty_multi_hot.to_sel_vec(), empty_sel);
        assert_eq!(empty_index.to_sel_vec(), empty_sel);
        assert_eq!(empty_multi_hot.read_from_mu, true);
        assert_eq!(empty_index.read_from_mu, true);
    }
}
