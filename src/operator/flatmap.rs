// This is an operator that will be abstracted as a FlatMap operator

use crate::primitives::elem::{Elem, StopType};
use crate::primitives::select::{MultiHotN, SelectAdapter};
use crate::primitives::tile::Tile;
use dam::context_tools::*;
use dam::types::DAMType;
use ndarray::Array2;

#[context_macro]
pub struct RetileStreamify<T: Clone> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Tile<T>>>,
    split_row: bool,
    filter_mask: bool,
    id: u32,
}

impl<T: Clone> RetileStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        split_row: bool,
        filter_mask: bool,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            split_row,
            filter_mask,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
    fn retile(&mut self, data: &Tile<T>, stop_level: Option<StopType>) {
        match &data.underlying {
            Some(arr) => {
                let offset = data.offset;

                let vec_iter = if self.split_row {
                    arr.rows().into_iter()
                } else {
                    arr.columns().into_iter()
                };

                let vec_iter_len = vec_iter.len();
                for (idx, row) in vec_iter.enumerate() {
                    let out_data = Tile::<T>::new_padded(
                        row.to_shared().insert_axis(ndarray::Axis(0)), // [N] => [1,N]
                        data.bytes_per_elem,
                        data.read_from_mu,
                        if idx + 1 <= offset { 1 } else { 0 },
                    );

                    // check whether this is the last value and set the stop level if needed
                    let elem = if stop_level.is_some() {
                        if (self.filter_mask && idx + 1 == offset) || (idx + 1 == vec_iter_len) {
                            Elem::ValStop(out_data, stop_level.unwrap())
                        } else {
                            Elem::Val(out_data)
                        }
                    } else {
                        Elem::Val(out_data)
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
                    if self.filter_mask && idx + 1 == offset {
                        break;
                    }
                }
            }
            None => {
                let offset = data.offset;

                let num_tiles = if self.split_row {
                    data.shape[0]
                } else {
                    data.shape[1]
                };

                let row_size = if self.split_row {
                    data.shape[1]
                } else {
                    data.shape[0]
                };

                for idx in 0..num_tiles {
                    let out_data = Tile::<T>::new_blank_padded(
                        vec![1, row_size],
                        data.bytes_per_elem,
                        data.read_from_mu,
                        if idx + 1 <= offset { 1 } else { 0 },
                    );

                    // check whether this is the last value and set the stop level if needed
                    let elem = if stop_level.is_some() {
                        if (self.filter_mask && idx + 1 == offset) || (idx + 1 == num_tiles) {
                            Elem::ValStop(out_data, stop_level.unwrap())
                        } else {
                            Elem::Val(out_data)
                        }
                    } else {
                        Elem::Val(out_data)
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
                    if self.filter_mask && idx + 1 == offset {
                        break;
                    }
                }
            }
        }
    }
}

impl<T: Clone> Context for RetileStreamify<T>
where
    Tile<T>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => {
                        self.retile(&data, None);
                    }
                    Elem::ValStop(data, s) => {
                        self.retile(&data, Some(s));
                    }
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[context_macro]
pub struct ExpertAddrGen<SEL: Clone + SelectAdapter> {
    in_stream: Receiver<Elem<SEL>>, // Index of the expert
    out_stream: Sender<Elem<Tile<u64>>>,
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
            in_stream,
            out_stream,
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

#[context_macro]
pub struct CacheReadAddrGen {
    idx_stream: Receiver<Elem<Tile<u64>>>, // Index of the request
    seq_len_stream: Receiver<Elem<Tile<u64>>>, // Sequence length
    offset_per_idx: u64,
    out_stream: Sender<Elem<Tile<u64>>>,
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
            idx_stream,
            seq_len_stream,
            offset_per_idx,
            out_stream,
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
    seq_len_stream: Receiver<Elem<Tile<u64>>>,
    out_stream: Sender<Elem<MultiHotN>>,
    id: u32,
}

impl FilterLastTile {
    pub fn new(
        seq_len_stream: Receiver<Elem<Tile<u64>>>,
        out_stream: Sender<Elem<MultiHotN>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            seq_len_stream,
            out_stream,
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
mod retile_tests {
    use super::RetileStreamify;
    use crate::{
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext};
    use ndarray::Array2;

    fn tolerance_fn(a: &Elem<Tile<i32>>, b: &Elem<Tile<i32>>) -> bool {
        match (a, b) {
            (Elem::Val(a_tile), Elem::Val(b_tile)) => a_tile == b_tile,
            (Elem::ValStop(a_tile, a_level), Elem::ValStop(b_tile, b_level)) => {
                a_tile == b_tile && a_level == b_level
            }
            _ => false,
        }
    }

    #[test]
    fn test_retile_col() {
        // [1,3] => [1,9]
        // [4,3] tile => [4,1] tile
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 8 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(1),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new(concatenated_array.to_shared(), 4, read_from_mu),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays_input: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((4, 1), vec![i as i32; 4]).unwrap())
            .collect();
        let arrays_output: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays_input, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays_output, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(RetileStreamify::<_>::new(
            in_data_rcv,
            out_data_snd,
            false,
            false,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_retile_row() {
        // [1,3] => [1,9]
        // [3,4] tile => [1,4] tile
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 8 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(0),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new(concatenated_array.to_shared(), 4, read_from_mu),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(RetileStreamify::<_>::new(
            in_data_rcv,
            out_data_snd,
            true,
            false,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_retile_row_filter() {
        // [1,3] => [1,7]
        // [3,4] tile => [1,4] tile (last tile is padded with 2 vectors)
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 6 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(0),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new_padded(concatenated_array.to_shared(), 4, read_from_mu, 1),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays_input: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let arrays_output: Vec<Array2<i32>> = (0..7)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays_input, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays_output, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(RetileStreamify::<_>::new(
            in_data_rcv,
            out_data_snd,
            true,
            true,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}

#[cfg(test)]
mod tests {
    use super::ExpertAddrGen;
    use crate::{
        operator::flatmap::{CacheReadAddrGen, FilterLastTile},
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
