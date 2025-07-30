use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::{select::SelectAdapter, tile::Tile};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use core::panic;
use dam::channel::PeekResult;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;

pub struct FlatReassembleConfig {
    pub switch_cycles: Vec<u64>,
    pub write_back_mu: bool,
}

#[context_macro]
pub struct FlatReassemble<E, A: DAMType, SELT: DAMType> {
    in_streams: Vec<Receiver<Elem<A>>>,
    sel_stream: Receiver<Elem<SELT>>,
    out_stream: Sender<Elem<A>>,
    reassemble_rank: StopType,
    config: FlatReassembleConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType + Bufferizable,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > FlatReassemble<E, A, SELT>
where
    Elem<A>: DAMType,
    Elem<SELT>: DAMType,
{
    pub fn new(
        in_streams: Vec<Receiver<Elem<A>>>,
        sel_stream: Receiver<Elem<SELT>>,
        out_stream: Sender<Elem<A>>,
        reassemble_rank: StopType,
        config: FlatReassembleConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_streams,
            sel_stream,
            out_stream,
            reassemble_rank,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_streams.iter().for_each(|s| s.attach_receiver(&ctx));
        ctx.sel_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    /// Helper function to calculate and increment load cycles for memory operations
    fn handle_load_cycles<T: Bufferizable>(
        &mut self,
        data_arrive_time: u64,
        data: &T,
        constant: Option<u64>,
    ) {
        let mut load_cycle = constant.unwrap_or(0);
        if data.read_from_mu() {
            load_cycle += div_ceil(data.size_in_bytes() as u64, PMU_BW);
        }
        self.time.advance((data_arrive_time + load_cycle).into());
    }

    fn handle_memory_writeback(&mut self, x: &A) {
        if self.config.write_back_mu {
            self.time
                .incr_cycles(div_ceil(x.size_in_bytes() as u64, PMU_BW));
        }
    }

    fn peek_all_streams(&mut self, select_vec: &[usize]) -> Vec<Option<ChannelElement<Elem<A>>>> {
        let mut peeked = vec![false; select_vec.len()];
        let mut num_peeked = 0;
        let mut peek_results = vec![None; select_vec.len()];

        while num_peeked < select_vec.len() {
            for (i, &idx) in select_vec.iter().enumerate() {
                if peeked[i] {
                    continue;
                }

                match self.in_streams[idx].peek() {
                    PeekResult::Something(elem) => {
                        peek_results[i] = Some(elem);
                        peeked[i] = true;
                        num_peeked += 1;
                    }
                    PeekResult::Nothing(_) => continue,
                    PeekResult::Closed => return vec![], // Signal that streams are closed
                }
            }
            if num_peeked < select_vec.len() {
                self.time.incr_cycles(1);
            }
        }

        peek_results
    }

    fn get_arrive_times(&self, peek_results: &[Option<ChannelElement<Elem<A>>>]) -> Vec<u64> {
        let mut data_arrive_times = vec![];
        peek_results.iter().for_each(|elem| {
            if let Some(ChannelElement { time: arrive, .. }) = elem {
                data_arrive_times.push(arrive.time());
            }
        });
        data_arrive_times
    }

    fn process_input_stream(&mut self, select_vec: &[usize], select_level: Option<u32>) {
        let addtional_rank = select_level.unwrap_or(0);
        let num_selected_streams = select_vec.len();
        // Peek the next wave of input elements
        let peek_results = self.peek_all_streams(select_vec);
        if peek_results.is_empty() {
            panic!("All input streams are closed or empty");
        }

        // Get the arrive time of each element in the peek_results
        let data_arrive_times = self.get_arrive_times(&peek_results);

        // Create idx vector based on the data_arrive_times
        let mut sorted_indices: Vec<usize> = (0..num_selected_streams).collect();
        sorted_indices.sort_by_key(|&i| data_arrive_times[i]);

        for &i in sorted_indices.iter() {
            let is_last_selected = sorted_indices.last() == Some(&i);
            let stream_idx = select_vec[i];
            let start_time = data_arrive_times[i];
            loop {
                // Peek next to the current stream
                match self.in_streams[stream_idx].peek_next(&self.time) {
                    Ok(ChannelElement {
                        time: _,
                        data: val_data,
                    }) => {
                        // Handle load cycles for the current element
                        match &val_data {
                            Elem::Val(x) => {
                                self.handle_load_cycles(
                                    data_arrive_times[i],
                                    x,
                                    Some(self.config.switch_cycles[stream_idx]),
                                );
                            }
                            Elem::ValStop(x, _) => {
                                self.handle_load_cycles(
                                    data_arrive_times[i],
                                    x,
                                    Some(self.config.switch_cycles[stream_idx]),
                                );
                            }
                        };

                        // Dequeue the current element
                        self.in_streams[stream_idx].dequeue(&self.time).unwrap();

                        // Enqueue the current element to the output stream
                        match &val_data {
                            Elem::Val(x) => {
                                self.handle_memory_writeback(x);
                                let updated_x: A =
                                    x.clone_with_updated_read_from_mu(self.config.write_back_mu);
                                let data = if self.reassemble_rank == 0 {
                                    if is_last_selected {
                                        Elem::ValStop(updated_x.clone(), 1 + addtional_rank)
                                    } else {
                                        Elem::Val(updated_x.clone())
                                    }
                                } else {
                                    Elem::Val(updated_x.clone())
                                };
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data,
                                        },
                                    )
                                    .unwrap();
                            }
                            Elem::ValStop(x, level) => {
                                self.handle_memory_writeback(x);
                                let updated_x: A =
                                    x.clone_with_updated_read_from_mu(self.config.write_back_mu);
                                let data = if self.reassemble_rank == 0 {
                                    if is_last_selected {
                                        Elem::ValStop(
                                            updated_x.clone(),
                                            *level + addtional_rank + 1,
                                        )
                                    } else {
                                        Elem::Val(updated_x.clone())
                                    }
                                } else {
                                    if is_last_selected && level >= &self.reassemble_rank {
                                        Elem::ValStop(
                                            updated_x.clone(),
                                            *level + addtional_rank + 1,
                                        )
                                    } else {
                                        Elem::ValStop(updated_x.clone(), *level)
                                    }
                                };
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data,
                                        },
                                    )
                                    .unwrap();
                            }
                        }

                        // Finally, break the current expert based on the rank
                        match val_data {
                            Elem::Val(_) => {
                                if self.reassemble_rank == 0 {
                                    // Logging
                                    dam::logging::log_event(&E::new(
                                        "FlatReassemble".to_string(),
                                        self.id,
                                        start_time,
                                        self.time.tick().time(),
                                        true,
                                    ))
                                    .unwrap();

                                    break;
                                }
                            }
                            Elem::ValStop(_, level) => {
                                if level >= self.reassemble_rank {
                                    // Logging
                                    dam::logging::log_event(&E::new(
                                        "FlatReassemble".to_string(),
                                        self.id,
                                        start_time,
                                        self.time.tick().time(),
                                        true,
                                    ))
                                    .unwrap();
                                    break;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        panic!("Stream {} is closed or empty", stream_idx);
                    }
                }
            }
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType + Bufferizable,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > Context for FlatReassemble<E, A, SELT>
where
    Elem<A>: DAMType,
    Elem<SELT>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.sel_stream.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: sel_data,
                }) => match sel_data {
                    Elem::Val(sel) => {
                        self.handle_load_cycles(self.time.tick().time(), &sel, None);
                        self.sel_stream.dequeue(&self.time).unwrap();
                        let select_vec = sel.to_sel_vec();
                        self.process_input_stream(&select_vec, None);
                    }
                    Elem::ValStop(sel, sel_level) => {
                        self.handle_load_cycles(self.time.tick().time(), &sel, None);
                        self.sel_stream.dequeue(&self.time).unwrap();
                        let select_vec = sel.to_sel_vec();
                        self.process_input_stream(&select_vec, Some(sel_level));
                    }
                },
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::select::MultiHotN;
    use crate::{
        operator::reassemble::{FlatReassemble, FlatReassembleConfig},
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext, PrinterContext};
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

    // Use the same index and output streams as input from `fn flat_partition_2d_multi_hot_rank_1()`
    #[test]
    fn flat_reassemble_2d_multi_hot_rank_1() {
        fn create_input_streams(
            arrays: &[Array2<i32>],
            read_from_mu: bool,
        ) -> Vec<Vec<Elem<Tile<i32>>>> {
            let mut input_streams: Vec<Vec<Elem<Tile<i32>>>> = vec![Vec::new(); 4];

            // Define the mapping of which arrays go to which output streams
            let stream_mappings = [
                vec![0, 1, 2],          // Stream 0: arrays 0, 1, 2S1 [1, 3]
                vec![0, 1, 2, 3, 4, 5], // Stream 1: arrays 0, 1, 2S1, 3, 4, 5S1 [2, 3]
                vec![3, 4, 5, 6, 7, 8], // Stream 2: arrays 3, 4, 5S1, 6, 7, 8S1 [2, 3]
                vec![6, 7, 8],          // Stream 3: arrays 6, 7, 8S1 [1, 3]
            ];

            for (stream_idx, array_indices) in stream_mappings.iter().enumerate() {
                for (pos, &array_idx) in array_indices.iter().enumerate() {
                    let tile = Tile::new(arrays[array_idx].clone().into(), 4, read_from_mu);

                    // Add ValStop at the end of each group of 3 elements
                    if (pos + 1) % 3 == 0 {
                        input_streams[stream_idx].push(Elem::ValStop(tile, 1));
                    } else {
                        input_streams[stream_idx].push(Elem::Val(tile));
                    }
                }
            }

            input_streams
        }

        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth = Vec::new();
            for i in 0..3 {
                for t in 0..2 {
                    for j in 0..3 {
                        let tile = Tile::new(arrays[i * 3 + j].clone().into(), 4, read_from_mu);
                        if i == 2 && t == 1 && j == 2 {
                            ground_truth.push(Elem::ValStop(tile, 3));
                        } else if i < 2 && t == 1 && j == 2 {
                            ground_truth.push(Elem::ValStop(tile, 2));
                        } else if t < 1 && j == 2 {
                            ground_truth.push(Elem::ValStop(tile, 1));
                        } else {
                            ground_truth.push(Elem::Val(tile));
                        }
                    }
                }
            }
            ground_truth
        }

        fn create_select_streams(read_from_mu: bool) -> Vec<Elem<MultiHotN>> {
            vec![
                Elem::Val(MultiHotN::new(vec![true, true, false, false], read_from_mu)),
                Elem::Val(MultiHotN::new(vec![false, true, true, false], read_from_mu)),
                Elem::ValStop(
                    MultiHotN::new(vec![false, false, true, true], read_from_mu),
                    1,
                ),
            ]
        } // [1, 3]

        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((2, 2), vec![i as i32; 4]).unwrap())
            .collect();
        let input_streams_data = create_input_streams(&arrays, true);
        let select_stream_data = create_select_streams(true);
        let ground_truth = create_ground_truth(&arrays, true);

        let mut ctx = ProgramBuilder::default();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        let (in_sel_snd, in_sel_rcv) = ctx.unbounded();
        let (exp1_snd, exp1_rcv) = ctx.unbounded();
        let (exp2_snd, exp2_rcv) = ctx.unbounded();
        let (exp3_snd, exp3_rcv) = ctx.unbounded();
        let (exp4_snd, exp4_rcv) = ctx.unbounded();

        let config = FlatReassembleConfig {
            switch_cycles: vec![1, 2, 3, 4],
            write_back_mu: true,
        };

        ctx.add_child(GeneratorContext::new(
            || input_streams_data[0].clone().into_iter(),
            exp1_snd,
        ));

        ctx.add_child(GeneratorContext::new(
            || input_streams_data[1].clone().into_iter(),
            exp2_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[2].clone().into_iter(),
            exp3_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[3].clone().into_iter(),
            exp4_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || select_stream_data.into_iter(),
            in_sel_snd,
        ));
        ctx.add_child(FlatReassemble::<SimpleEvent, _, _>::new(
            vec![exp1_rcv, exp2_rcv, exp3_rcv, exp4_rcv],
            in_sel_rcv,
            out_data_snd,
            1,
            config,
            0,
        )); // [1, 3, 2, 3]

        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        // println!("Expected output: {:?}", ground_truth);
        // ctx.add_child(PrinterContext::new(out_data_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flat_reassemble_1d_multi_hot_rank_0() {
        fn create_multi_hot_arrays<const N: usize>(
            sel: usize,
            length: usize,
            read_from_mu: bool,
        ) -> Vec<MultiHotN> {
            let mut multi_hot_arrays = Vec::new();
            for i in 0..length {
                let mut selection = vec![false; N];
                for j in 0..sel {
                    selection[(i + j) % N] = true;
                }
                // Convert Vec<bool> to [bool; N]
                multi_hot_arrays.push(MultiHotN::new(selection, read_from_mu));
            }
            multi_hot_arrays
        } // [9]

        fn create_input_streams<const N: usize>(
            arrays: &[Array2<i32>],
            multi_hot: &Vec<MultiHotN>,
            read_from_mu: bool,
        ) -> Vec<Vec<Elem<Tile<i32>>>> {
            let mut input_streams: Vec<Vec<Elem<Tile<i32>>>> = vec![Vec::new(); N];

            for (i, array_idx) in multi_hot.iter().enumerate() {
                for (j, &is_selected) in array_idx.iter().enumerate() {
                    if is_selected {
                        let tile = Tile::new(arrays[i].clone().into(), 4, read_from_mu);
                        input_streams[j].push(Elem::Val(tile));
                    }
                }
            }
            input_streams
        } // [Dyn]

        // If the tiles are different, then the ground truth should consider the timing.
        fn create_ground_truth(
            arrays: &[Array2<i32>],
            sel: usize,
            read_from_mu: bool,
        ) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth = Vec::new();
            for elem in arrays.iter() {
                for i in 0..sel {
                    let tile = Tile::new(elem.clone().into(), 4, read_from_mu);
                    if i == sel - 1 {
                        ground_truth.push(Elem::ValStop(tile, 1));
                    } else {
                        ground_truth.push(Elem::Val(tile));
                    }
                }
            }
            ground_truth
        }

        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((2, 2), vec![i as i32; 4]).unwrap())
            .collect();
        let multi_hot = create_multi_hot_arrays::<4>(2, 9, true);
        let input_streams_data = create_input_streams::<4>(&arrays, &multi_hot, true);
        let ground_truth = create_ground_truth(&arrays, 2, true);
        let select_stream_data: Vec<Elem<MultiHotN>> =
            multi_hot.iter().map(|m| Elem::Val(m.clone())).collect();
        let mut ctx = ProgramBuilder::default();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        let (in_sel_snd, in_sel_rcv) = ctx.unbounded();
        let (exp1_snd, exp1_rcv) = ctx.unbounded();
        let (exp2_snd, exp2_rcv) = ctx.unbounded();
        let (exp3_snd, exp3_rcv) = ctx.unbounded();
        let (exp4_snd, exp4_rcv) = ctx.unbounded();
        let config = FlatReassembleConfig {
            switch_cycles: vec![1, 2, 3, 4],
            write_back_mu: true,
        };
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[0].clone().into_iter(),
            exp1_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[1].clone().into_iter(),
            exp2_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[2].clone().into_iter(),
            exp3_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[3].clone().into_iter(),
            exp4_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || select_stream_data.into_iter(),
            in_sel_snd,
        ));
        ctx.add_child(FlatReassemble::<SimpleEvent, _, _>::new(
            vec![exp1_rcv, exp2_rcv, exp3_rcv, exp4_rcv],
            in_sel_rcv,
            out_data_snd,
            0,
            config,
            0,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        // println!("Expected output: {:?}", ground_truth);
        // ctx.add_child(PrinterContext::new(out_data_rcv));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
