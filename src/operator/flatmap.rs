use crate::utils::request_profile::{ProfiledReceiver, ProfiledSender};
// This is an operator that will be abstracted as a FlatMap operator

use crate::primitives::elem::{Elem, StopType};
use crate::primitives::select::{MultiHotN, SelectAdapter};
use crate::primitives::tile::Tile;
use dam::context_tools::*;
use dam::types::DAMType;
use ndarray::Array2;

#[context_macro]
pub struct ExpertAddrGen<SEL: Clone + SelectAdapter> {
    in_stream: ProfiledReceiver<Elem<SEL>>, // Index of the expert
    out_stream: ProfiledSender<Elem<Tile<u64>>>,
    num_tile_per_expert: u64,
    expert_addr_base: u64,
    id: u32,
}

impl<SEL: Clone + SelectAdapter> ExpertAddrGen<SEL>
where
    SEL: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<SEL>>,
        out_stream: Sender<Elem<Tile<u64>>>,
        num_tile_per_expert: u64,
        expert_addr_base: u64,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream: in_stream.into(),
            out_stream: out_stream.into(),
            num_tile_per_expert,
            expert_addr_base,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<SEL: Clone + SelectAdapter> Context for ExpertAddrGen<SEL>
where
    SEL: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => {
                        let expert_idx_list = data.to_sel_vec();
                        assert_eq!(expert_idx_list.len(), 1);

                        let expert_addr: u64 = self.expert_addr_base
                            + expert_idx_list[0] as u64 * self.num_tile_per_expert;

                        for i in 0..self.num_tile_per_expert {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(
                                            Tile::new(
                                                Array2::from_shape_vec(
                                                    (1, 1),
                                                    vec![expert_addr + i],
                                                )
                                                .unwrap()
                                                .to_shared(),
                                                8,
                                                false,
                                            ),
                                            if i < self.num_tile_per_expert - 1 {
                                                1
                                            } else {
                                                2
                                            },
                                        ),
                                    },
                                )
                                .unwrap();
                        }
                    }
                    Elem::ValStop(_data, _s) => {
                        panic!("This function is designed to only be used for 0d input streams");
                    }
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

/// The scalar an input element contributes to the base tile index.
///
/// `DynAddrGen` accepts either a Select stream (the `ExpertAddrGen` case: one
/// chosen index per element) or a rank-0 index tile.
pub trait DynAddrBase {
    fn to_base_idx(&self) -> u64;
}

impl DynAddrBase for MultiHotN {
    fn to_base_idx(&self) -> u64 {
        // A blank select carries no data (timing-only simulation). The number of
        // addresses this op emits is data-independent, so 0 keeps the timing
        // faithful without inventing an index.
        if self.is_blank() {
            return 0;
        }
        let sel_vec = self.to_sel_vec();
        assert_eq!(
            sel_vec.len(),
            1,
            "DynAddrGen expects exactly one selected index per input element"
        );
        sel_vec[0] as u64
    }
}

impl<T: DAMType + num_traits::AsPrimitive<u64>> DynAddrBase for Tile<T> {
    fn to_base_idx(&self) -> u64 {
        // `underlying == None` is a timing-only tile; see the MultiHotN impl.
        self.underlying.as_ref().map_or(0, |arr| arr[[0, 0]].as_())
    }
}

/// The general form of `ExpertAddrGen`: for every element of `in_stream`, emit
/// the tile indices that an `out_shape_tiled` view reads out of one slab of
/// `prod(tensor_shape_tiled)` tiles.
///
/// The input element picks the slab (its scalar times the slab size, plus
/// `addr_base`); `stride` and `out_shape_tiled` walk the view inside that slab
/// exactly as `LinearOffChipLoad` / `LinearOffChipLoadRef` do for a static
/// load. Every emitted tile is the `[1,1]` u64 tile index a
/// `RandomOffChipLoad` consumes.
///
/// `ExpertAddrGen` is the special case `tensor_shape_tiled = [n, 1]`,
/// `stride = [1, 1]`, `out_shape_tiled = [n, 1]`.
#[context_macro]
pub struct DynAddrGen<IN: Clone + DynAddrBase> {
    in_stream: ProfiledReceiver<Elem<IN>>,
    out_stream: ProfiledSender<Elem<Tile<u64>>>,
    /// Slab-relative tile index + the stop token that closes at it, for every
    /// position of the view. Precomputed once: the walk does not depend on the
    /// input element, only the base address does.
    view: Vec<(u64, Option<StopType>)>,
    /// `prod(tensor_shape_tiled)` -- how many tiles one input element steps past.
    slab_tiles: u64,
    addr_base: u64,
    /// `out_shape_tiled.len()`: the level of the stop token that closes the
    /// whole address grid, and so the one the input's own stop folds into.
    out_rank: StopType,
    id: u32,
}

impl<IN: Clone + DynAddrBase> DynAddrGen<IN>
where
    IN: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<IN>>,
        out_stream: Sender<Elem<Tile<u64>>>,
        tensor_shape_tiled: Vec<usize>,
        stride: Vec<usize>,
        out_shape_tiled: Vec<usize>,
        addr_base: u64,
        id: u32,
    ) -> Self {
        assert_eq!(
            stride.len(),
            out_shape_tiled.len(),
            "DynAddrGen {}: stride and out_shape_tiled must have the same number of dimensions",
            id
        );

        let ctx = Self {
            in_stream: in_stream.into(),
            out_stream: out_stream.into(),
            view: Self::generate_view(&tensor_shape_tiled, &stride, &out_shape_tiled),
            slab_tiles: tensor_shape_tiled.iter().product::<usize>() as u64,
            addr_base,
            out_rank: out_shape_tiled.len() as StopType,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    /// Walk `out_shape_tiled` in row-major order, mapping each position to the
    /// tile it reads through `stride` (wrapped into the tensor, as
    /// `LinearOffChipLoad` does) and to the highest stop token that closes
    /// there.
    fn generate_view(
        tensor_shape_tiled: &[usize],
        stride: &[usize],
        out_shape_tiled: &[usize],
    ) -> Vec<(u64, Option<StopType>)> {
        let total_tiles: usize = out_shape_tiled.iter().product();
        let tensor_tiles: usize = tensor_shape_tiled.iter().product();

        let mut view = Vec::with_capacity(total_tiles);

        for flat_idx in 0..total_tiles {
            // Convert flat index to multi-dimensional indices
            let mut remaining = flat_idx;
            let mut multi_index = vec![0; out_shape_tiled.len()];

            for i in (0..out_shape_tiled.len()).rev() {
                multi_index[i] = remaining % out_shape_tiled[i];
                remaining /= out_shape_tiled[i];
            }

            // Calculate the index in the original flat tensor using strides
            let mut tile_idx = 0;
            for (dim, &idx_in_dim) in multi_index.iter().enumerate() {
                tile_idx += idx_in_dim * stride[dim];
            }

            // Ensure we don't go out of bounds of the slab
            tile_idx = if tensor_tiles > 0 {
                tile_idx % tensor_tiles
            } else {
                0 // Handle empty tensor case
            };

            // Determine the highest-dimensional stop token needed
            let mut highest_stop_token: Option<StopType> = None;
            let mut all_inner_dims_at_end = true;

            // Check from innermost to outermost
            for dim in (0..out_shape_tiled.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = out_shape_tiled[dim] == 1;
                    let is_last_elem = multi_index[dim] == out_shape_tiled[dim] - 1;

                    // If at end or dim size is 1, update the highest stop token
                    if is_last_elem || is_dim_size_one {
                        highest_stop_token = Some((out_shape_tiled.len() - dim) as StopType);
                    }

                    // Only continue checking outer dimensions if this one is at
                    // its last element
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            view.push((tile_idx as u64, highest_stop_token));
        }

        view
    }
}

impl<IN: Clone + DynAddrBase> Context for DynAddrGen<IN>
where
    IN: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => {
                    // A stop on the input closes the ranks *above* the address
                    // grid, so it is folded into the token that closes the grid
                    // -- the same convention as `LinearOffChipLoadRef`.
                    let (data, in_stop) = match data_enum {
                        Elem::Val(data) => (data, None),
                        Elem::ValStop(data, s) => (data, Some(s)),
                    };

                    let base = self.addr_base + data.to_base_idx() * self.slab_tiles;

                    for &(tile_idx, stop_level) in self.view.iter() {
                        let addr_tile = Tile::new(
                            Array2::from_shape_vec((1, 1), vec![base + tile_idx])
                                .unwrap()
                                .to_shared(),
                            8,
                            false,
                        );

                        let elem = match stop_level {
                            None => Elem::Val(addr_tile),
                            Some(level) => {
                                let final_stop_lev = match in_stop {
                                    Some(in_stop) if level == self.out_rank => in_stop + level,
                                    _ => level,
                                };
                                Elem::ValStop(addr_tile, final_stop_lev)
                            }
                        };

                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: elem,
                                },
                            )
                            .unwrap();
                    }
                }
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[context_macro]
pub struct CacheReadAddrGen {
    idx_stream: ProfiledReceiver<Elem<Tile<u64>>>, // Index of the request
    seq_len_stream: ProfiledReceiver<Elem<Tile<u64>>>, // Sequence length
    offset_per_idx: u64,
    out_stream: ProfiledSender<Elem<Tile<u64>>>,
    id: u32,
}

impl CacheReadAddrGen {
    pub fn new(
        idx_stream: Receiver<Elem<Tile<u64>>>,
        seq_len_stream: Receiver<Elem<Tile<u64>>>,
        offset_per_idx: u64,
        out_stream: Sender<Elem<Tile<u64>>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            idx_stream: idx_stream.into(),
            seq_len_stream: seq_len_stream.into(),
            offset_per_idx,
            out_stream: out_stream.into(),
            id,
            context_info: Default::default(),
        };
        ctx.idx_stream.attach_receiver(&ctx);
        ctx.seq_len_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for CacheReadAddrGen {
    fn run(&mut self) {
        loop {
            let idx_elem = self.idx_stream.dequeue(&self.time);
            let seq_len_elem = self.seq_len_stream.dequeue(&self.time);

            match (idx_elem, seq_len_elem) {
                (Ok(idx_elem), Ok(seq_len_elem)) => match (idx_elem.data, seq_len_elem.data) {
                    (Elem::Val(idx_tile), Elem::Val(seq_len_tile)) => {
                        let idx_val = idx_tile.underlying.as_ref().unwrap()[[0, 0]];
                        let seq_len_val = seq_len_tile.underlying.as_ref().unwrap()[[0, 0]];

                        let start_time = self.time.tick();
                        for i in 0..(seq_len_val - 1) {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: start_time + i,
                                        data: Elem::Val(Tile::new(
                                            Array2::from_shape_vec(
                                                (1, 1),
                                                vec![idx_val * self.offset_per_idx + i as u64],
                                            )
                                            .unwrap()
                                            .to_shared(),
                                            8,
                                            false,
                                        )),
                                    },
                                )
                                .unwrap();
                        }
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: start_time + (seq_len_val - 1),
                                    data: Elem::ValStop(
                                        Tile::new(
                                            Array2::from_shape_vec(
                                                (1, 1),
                                                vec![
                                                    idx_val * self.offset_per_idx
                                                        + (seq_len_val - 1) as u64,
                                                ],
                                            )
                                            .unwrap()
                                            .to_shared(),
                                            8,
                                            false,
                                        ),
                                        1,
                                    ),
                                },
                            )
                            .unwrap();
                    }
                    (
                        Elem::ValStop(idx_tile, idx_stop_level),
                        Elem::ValStop(seq_len_tile, seq_len_stop_level),
                    ) => {
                        assert_eq!(idx_stop_level, seq_len_stop_level);

                        let idx_val = idx_tile.underlying.as_ref().unwrap()[[0, 0]];
                        let seq_len_val = seq_len_tile.underlying.as_ref().unwrap()[[0, 0]];

                        let start_time = self.time.tick();
                        for i in 0..(seq_len_val - 1) {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: start_time + i,
                                        data: Elem::Val(Tile::new(
                                            Array2::from_shape_vec(
                                                (1, 1),
                                                vec![idx_val * self.offset_per_idx + i as u64],
                                            )
                                            .unwrap()
                                            .to_shared(),
                                            8,
                                            false,
                                        )),
                                    },
                                )
                                .unwrap();
                        }
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: start_time + (seq_len_val - 1),
                                    data: Elem::ValStop(
                                        Tile::new(
                                            Array2::from_shape_vec(
                                                (1, 1),
                                                vec![
                                                    idx_val * self.offset_per_idx
                                                        + (seq_len_val - 1) as u64,
                                                ],
                                            )
                                            .unwrap()
                                            .to_shared(),
                                            8,
                                            false,
                                        ),
                                        idx_stop_level + 1,
                                    ),
                                },
                            )
                            .unwrap();
                    }
                    _ => {
                        panic!(
                            "CacheReadAddrGen {}: idx_stream and seq_len_stream must have the same shape",
                            self.id)
                    }
                },
                (Err(_), Err(_)) => {
                    return;
                }
                _ => {
                    panic!(
                        "CacheReadAddrGen {}: idx_stream and seq_len_stream must have the same shape",
                        self.id
                    );
                }
            }
        }
    }
}

#[context_macro]
pub struct FilterLastTile {
    seq_len_stream: ProfiledReceiver<Elem<Tile<u64>>>,
    out_stream: ProfiledSender<Elem<MultiHotN>>,
    id: u32,
}

impl FilterLastTile {
    pub fn new(
        seq_len_stream: Receiver<Elem<Tile<u64>>>,
        out_stream: Sender<Elem<MultiHotN>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            seq_len_stream: seq_len_stream.into(),
            out_stream: out_stream.into(),
            id,
            context_info: Default::default(),
        };
        ctx.seq_len_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for FilterLastTile {
    fn run(&mut self) {
        loop {
            match self.seq_len_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => {
                        let seq_len_val = data.underlying.as_ref().unwrap()[[0, 0]];

                        for _ in 0..(seq_len_val - 1) {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(MultiHotN::new(vec![false, true], false)),
                                    },
                                )
                                .unwrap();
                        }

                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: Elem::ValStop(
                                        MultiHotN::new(vec![true, false], false),
                                        1,
                                    ),
                                },
                            )
                            .unwrap();
                    }
                    Elem::ValStop(data, stop_level) => {
                        let seq_len_val = data.underlying.as_ref().unwrap()[[0, 0]];

                        for _ in 0..(seq_len_val - 1) {
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(MultiHotN::new(vec![false, true], false)),
                                    },
                                )
                                .unwrap();
                        }

                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: Elem::ValStop(
                                        MultiHotN::new(vec![true, false], false),
                                        stop_level + 1,
                                    ),
                                },
                            )
                            .unwrap();
                    }
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExpertAddrGen;
    use crate::{
        operator::flatmap::{CacheReadAddrGen, DynAddrGen, FilterLastTile},
        primitives::{elem::Elem, select::MultiHotN, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext};
    use ndarray::Array2;

    fn tolerance_fn<T: PartialEq>(a: &Elem<T>, b: &Elem<T>) -> bool {
        match (a, b) {
            (Elem::Val(a_tile), Elem::Val(b_tile)) => a_tile == b_tile,
            (Elem::ValStop(a_tile, a_level), Elem::ValStop(b_tile, b_level)) => {
                a_tile == b_tile && a_level == b_level
            }
            _ => false,
        }
    }

    #[test]
    fn test_expert_addr_gen() {
        let num_tile_per_expert = 3;

        let mut ctx = ProgramBuilder::default();

        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(
            || {
                vec![
                    Elem::Val(MultiHotN::new(
                        vec![false, false, true, false, false, false, false, false], // 2
                        false,
                    )),
                    Elem::Val(MultiHotN::new(
                        vec![false, true, false, false, false, false, false, false], // 1
                        false,
                    )),
                    Elem::Val(MultiHotN::new(
                        vec![false, false, false, true, false, false, false, false], // 3
                        false,
                    )),
                    Elem::Val(MultiHotN::new(
                        vec![false, false, false, false, false, false, false, true], // 7
                        false,
                    )),
                ]
                .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(ExpertAddrGen::<_>::new(
            in_data_rcv,
            out_data_snd,
            num_tile_per_expert,
            0,
            0,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![vec![0, 1, 2]; 4]
                    .into_iter()
                    .zip(vec![2, 1, 3, 7].into_iter())
                    .map(|(vec_addr, expert_i)| {
                        vec_addr
                            .iter()
                            .map(|addr| {
                                Elem::ValStop(
                                    Tile::new(
                                        Array2::from_shape_vec(
                                            (1, 1),
                                            vec![expert_i * num_tile_per_expert + *addr as u64],
                                        )
                                        .unwrap()
                                        .to_shared(),
                                        8,
                                        false,
                                    ),
                                    if *addr < num_tile_per_expert - 1 {
                                        1
                                    } else {
                                        2
                                    },
                                )
                            })
                            .collect::<Vec<Elem<Tile<u64>>>>()
                    })
                    .flatten()
            },
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    /// Helper: the `[1,1]` u64 tile a `DynAddrGen` emits for one address.
    fn addr_tile(addr: u64) -> Tile<u64> {
        Tile::new(
            Array2::from_shape_vec((1, 1), vec![addr])
                .unwrap()
                .to_shared(),
            8,
            false,
        )
    }

    /// `DynAddrGen` reduces to `ExpertAddrGen` for the identity view over a
    /// `[n, 1]` slab, so it must emit exactly what `test_expert_addr_gen`
    /// expects: `expert_idx * n + i` with the inner size-1 dim closing level 1
    /// and the last tile of each expert closing level 2.
    #[test]
    fn test_dyn_addr_gen_expert_equivalence() {
        // cargo test --package step_perf --lib -- operator::flatmap::tests::test_dyn_addr_gen_expert_equivalence --exact --show-output
        const NUM_TILE_PER_EXPERT: u64 = 3;

        let mut ctx = ProgramBuilder::default();

        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();

        let experts: Vec<usize> = vec![2, 1, 3, 7];
        let in_experts = experts.clone();
        ctx.add_child(GeneratorContext::new(
            move || {
                in_experts
                    .clone()
                    .into_iter()
                    .map(|e| {
                        let mut one_hot = vec![false; 8];
                        one_hot[e] = true;
                        Elem::Val(MultiHotN::new(one_hot, false))
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(DynAddrGen::<MultiHotN>::new(
            in_data_rcv,
            out_data_snd,
            vec![NUM_TILE_PER_EXPERT as usize, 1], // tensor_shape_tiled
            vec![1, 1],                            // stride
            vec![NUM_TILE_PER_EXPERT as usize, 1], // out_shape_tiled
            0,                                     // addr_base
            0,                                     // id
        ));

        // Observed output stream (12 elements), as `addr:stop_level` -- every
        // element carries a stop token here because the inner dim has size 1:
        //      6:1   7:1   8:2     <- expert 2, i.e. 2*3 + {0,1,2}
        //      3:1   4:1   5:2     <- expert 1
        //      9:1  10:1  11:2     <- expert 3
        //     21:1  22:1  23:2     <- expert 7
        let mut gold = vec![];
        for expert in experts {
            for i in 0..NUM_TILE_PER_EXPERT {
                gold.push(Elem::ValStop(
                    addr_tile(expert as u64 * NUM_TILE_PER_EXPERT + i),
                    if i < NUM_TILE_PER_EXPERT - 1 { 1 } else { 2 },
                ));
            }
        }

        ctx.add_child(ApproxCheckerContext::new(
            move || gold.clone().into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    /// A non-identity view: read a `[2,3]` tile grid transposed, i.e.
    /// `out_shape_tiled = [3,2]` with `stride = [1,3]`, so position `(i,j)` maps
    /// to tile `j*3 + i`. The input is a rank-0 index tile rather than a select,
    /// and each index steps past a whole `prod([2,3]) = 6` tile slab.
    #[test]
    fn test_dyn_addr_gen_strided_view() {
        // cargo test --package step_perf --lib -- operator::flatmap::tests::test_dyn_addr_gen_strided_view --exact --show-output
        const SLAB_TILES: u64 = 6; // prod(tensor_shape_tiled) = 2 * 3
        const ADDR_BASE: u64 = 100;

        let mut ctx = ProgramBuilder::default();

        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();

        let bases: Vec<u64> = vec![0, 2];
        let in_bases = bases.clone();
        ctx.add_child(GeneratorContext::new(
            move || {
                in_bases
                    .clone()
                    .into_iter()
                    .map(|b| Elem::Val(addr_tile(b)))
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(DynAddrGen::<Tile<u64>>::new(
            in_data_rcv,
            out_data_snd,
            vec![2, 3], // tensor_shape_tiled
            vec![1, 3], // stride: transposed read
            vec![3, 2], // out_shape_tiled
            ADDR_BASE,  // addr_base
            0,          // id
        ));

        // Row-major walk of the [3,2] output grid, each position mapped through
        // the stride. Every row end closes level 1; the last tile also closes
        // the whole grid at level 2.
        let view: Vec<(u64, Option<u32>)> = vec![
            (0, None),
            (3, Some(1)),
            (1, None),
            (4, Some(1)),
            (2, None),
            (5, Some(2)),
        ];

        // Observed output stream (12 elements), as `addr:stop_level` with `-` for
        // a plain `Val`. The addresses are non-monotonic because the stride
        // transposes the read:
        //     100:-  103:1  101:-  104:1  102:-  105:2   <- base 0, 100 + {0,3,1,4,2,5}
        //     112:-  115:1  113:-  116:1  114:-  117:2   <- base 2, 100 + 2*6 + same
        let mut gold = vec![];
        for base in bases {
            for (offset, stop) in view.iter() {
                let tile = addr_tile(ADDR_BASE + base * SLAB_TILES + offset);
                gold.push(match stop {
                    None => Elem::Val(tile),
                    Some(level) => Elem::ValStop(tile, *level),
                });
            }
        }

        ctx.add_child(ApproxCheckerContext::new(
            move || gold.clone().into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    /// A stop token on the input closes ranks above the address grid, so it is
    /// folded into the token that closes the grid: with `out_shape_tiled` of
    /// rank 2, an input `ValStop(_, 1)` turns that element's final level-2 token
    /// into a level-3 one. Non-final input elements are unaffected.
    #[test]
    fn test_dyn_addr_gen_input_stop_token() {
        // cargo test --package step_perf --lib -- operator::flatmap::tests::test_dyn_addr_gen_input_stop_token --exact --show-output
        const NUM_TILE: u64 = 2;

        let mut ctx = ProgramBuilder::default();

        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();

        // Two elements; the second ends the enclosing rank.
        ctx.add_child(GeneratorContext::new(
            || vec![Elem::Val(addr_tile(0)), Elem::ValStop(addr_tile(1), 1)].into_iter(),
            in_data_snd,
        ));

        ctx.add_child(DynAddrGen::<Tile<u64>>::new(
            in_data_rcv,
            out_data_snd,
            vec![NUM_TILE as usize, 1], // tensor_shape_tiled
            vec![1, 1],                 // stride
            vec![NUM_TILE as usize, 1], // out_shape_tiled
            0,                          // addr_base
            0,                          // id
        ));

        // Observed output stream (4 elements), as `addr:stop_level`:
        //     0:1  1:2     <- input Val(0):        grid closes at level 2
        //     2:1  3:3     <- input ValStop(1, 1): level 2 folded into level 3
        let gold = vec![
            // base 0, no input stop: the grid closes at level 2.
            Elem::ValStop(addr_tile(0), 1),
            Elem::ValStop(addr_tile(1), 2),
            // base 1, input stop of 1: the grid's level-2 token becomes 3.
            Elem::ValStop(addr_tile(2), 1),
            Elem::ValStop(addr_tile(3), 3),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || gold.clone().into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_cache_read_addr_gen() {
        // cargo test --package step_perf --lib -- operator::flatmap::tests::test_cache_read_addr_gen --exact --show-output
        let offset_per_idx = 16;

        let mut ctx = ProgramBuilder::default();

        let (idx_data_snd, idx_data_rcv) = ctx.unbounded();
        let (seq_len_data_snd, seq_len_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();

        // Idx
        ctx.add_child(GeneratorContext::new(
            || {
                vec![0, 3, 11].into_iter().map(|i| {
                    Elem::Val(Tile::new(
                        Array2::from_shape_vec((1, 1), vec![i]).unwrap().to_shared(),
                        8,
                        false,
                    ))
                })
            },
            idx_data_snd,
        ));

        // Seq len
        ctx.add_child(GeneratorContext::new(
            || {
                vec![2, 4, 3].into_iter().map(|i| {
                    Elem::Val(Tile::new(
                        Array2::from_shape_vec((1, 1), vec![i]).unwrap().to_shared(),
                        8,
                        false,
                    ))
                })
            },
            seq_len_data_snd,
        ));

        // Cache read addr gen
        ctx.add_child(CacheReadAddrGen::new(
            idx_data_rcv,
            seq_len_data_rcv,
            offset_per_idx,
            out_data_snd,
            0,
        ));

        let mut gold = vec![];

        for (idx, seq_len) in vec![(0, 2), (3, 4), (11, 3)].into_iter() {
            for i in 0..(seq_len - 1) {
                gold.push(Elem::Val(Tile::new(
                    Array2::from_shape_vec((1, 1), vec![idx * offset_per_idx + i as u64])
                        .unwrap()
                        .to_shared(),
                    8,
                    false,
                )));
            }
            gold.push(Elem::ValStop(
                Tile::new(
                    Array2::from_shape_vec(
                        (1, 1),
                        vec![idx * offset_per_idx + (seq_len - 1) as u64],
                    )
                    .unwrap()
                    .to_shared(),
                    8,
                    false,
                ),
                1,
            ));
        }

        ctx.add_child(ApproxCheckerContext::new(
            move || gold.clone().into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_filter_last_tile() {
        // cargo test --package step_perf --lib -- operator::flatmap::tests::test_filter_last_tile --exact --show-output

        let mut ctx = ProgramBuilder::default();

        let (seq_len_data_snd, seq_len_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();

        // Seq len
        ctx.add_child(GeneratorContext::new(
            || {
                vec![2, 4, 3].into_iter().map(|i| {
                    Elem::Val(Tile::new(
                        Array2::from_shape_vec((1, 1), vec![i]).unwrap().to_shared(),
                        8,
                        false,
                    ))
                })
            },
            seq_len_data_snd,
        ));

        // Cache read addr gen
        ctx.add_child(FilterLastTile::new(seq_len_data_rcv, out_data_snd, 0));

        let mut gold = vec![];

        for seq_len in vec![2, 4, 3].into_iter() {
            for _ in 0..(seq_len - 1) {
                gold.push(Elem::Val(MultiHotN::new(vec![false, true], false)));
            }
            gold.push(Elem::ValStop(MultiHotN::new(vec![true, false], false), 1));
        }

        ctx.add_child(ApproxCheckerContext::new(
            move || gold.clone().into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
