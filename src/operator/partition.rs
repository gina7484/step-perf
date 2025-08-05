use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::{select::SelectAdapter, tile::Tile};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;
use std::panic;

pub struct FlatPartitionConfig {
    pub switch_cycles: Vec<u64>, // cycles between receiving
    pub write_back_mu: bool,     // Whether the output is written to a memory unit
}

#[context_macro]
pub struct FlatPartition<E, A: DAMType, SELT: DAMType> {
    in_stream: Receiver<Elem<A>>,
    sel_stream: Receiver<Elem<SELT>>,
    out_streams: Vec<Sender<Elem<A>>>,
    partition_rank: StopType,
    config: FlatPartitionConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > FlatPartition<E, A, SELT>
where
    Elem<A>: DAMType,
    Elem<SELT>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<A>>,
        sel_stream: Receiver<Elem<SELT>>,
        out_streams: Vec<Sender<Elem<A>>>,
        partition_rank: StopType,
        config: FlatPartitionConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            sel_stream,
            out_streams,
            partition_rank,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.sel_stream.attach_receiver(&ctx);
        for out in &ctx.out_streams {
            out.attach_sender(&ctx);
        }

        ctx
    }

    /// Helper function to calculate and increment load cycles for memory operations
    fn handle_load_cycles<T: Bufferizable>(&mut self, data: &T) {
        if data.read_from_mu() {
            let load_cycle = div_ceil(data.size_in_bytes() as u64, PMU_BW);
            self.time.incr_cycles(load_cycle);
        }
    }

    /// Helper function to calculate and increment write cycles based on expert indices
    fn handle_write_cycles<T: Bufferizable>(&mut self, select_vec: &[usize], data: &T) {
        let mut write_cycle = 0;

        // Find maximum switch cycle among selected experts
        // TODO: This could be optimized further
        for expert_idx in select_vec.iter() {
            if self.config.switch_cycles[*expert_idx] > write_cycle {
                write_cycle = self.config.switch_cycles[*expert_idx];
            }
        }

        // Add memory write back cycles if configured
        if self.config.write_back_mu {
            write_cycle += div_ceil(data.size_in_bytes() as u64, PMU_BW);
        }

        self.time.incr_cycles(write_cycle);
    }

    /// Helper function to enqueue data to all selected expert output streams
    fn enqueue_to_experts(&mut self, select_vec: &[usize], elem: Elem<A>) {
        for expert_idx in select_vec.iter() {
            self.out_streams[*expert_idx]
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem.clone(),
                    },
                )
                .unwrap();
        }
    }

    /// Process input stream elements with the given select vector
    fn process_input_stream(
        &mut self,
        select_vec: &[usize],
        expected_stop_level: Option<StopType>,
    ) {
        let mut start_time: Option<u64> = None;
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: val_data,
                }) => match val_data {
                    Elem::Val(x) => {
                        if start_time.is_none() {
                            start_time = Some(self.time.tick().time());
                        }
                        self.handle_load_cycles(&x);
                        self.in_stream.dequeue(&self.time).unwrap();
                        self.handle_write_cycles(select_vec, &x);

                        self.enqueue_to_experts(
                            select_vec,
                            Elem::Val(x.clone_with_updated_read_from_mu(self.config.write_back_mu)),
                        );

                        if self.partition_rank == 0 {
                            dam::logging::log_event(&E::new(
                                "FlatPartition".to_string(),
                                self.id,
                                start_time.unwrap(),
                                self.time.tick().time(),
                                true,
                            ))
                            .unwrap();

                            start_time = None;

                            break;
                        }
                    }
                    Elem::ValStop(x, stop_lev) => {
                        if start_time.is_none() {
                            start_time = Some(self.time.tick().time());
                        }
                        // Validate stop level based on context
                        if let Some(expected) = expected_stop_level.clone() {
                            if expected != stop_lev {
                                panic!("The expected stop level does not match the stop level in the select stream!");
                            }
                        } else if stop_lev > self.partition_rank {
                            println!(
                                "id {}: stop_lev in input {}, expected {:?}, partition_rank {}",
                                self.id, stop_lev, expected_stop_level, self.partition_rank
                            );
                            panic!("The stop level in the select stream is greater than the partition rank!");
                        }
                        // Determine output stop level
                        let output_stop_level = expected_stop_level
                            .map(|_| self.partition_rank)
                            .unwrap_or(stop_lev);

                        self.handle_load_cycles(&x);
                        self.in_stream.dequeue(&self.time).unwrap();
                        self.handle_write_cycles(select_vec, &x);
                        if output_stop_level == 0 {
                            self.enqueue_to_experts(
                                select_vec,
                                Elem::Val(
                                    x.clone_with_updated_read_from_mu(self.config.write_back_mu),
                                ),
                            );
                        } else {
                            self.enqueue_to_experts(
                                select_vec,
                                Elem::ValStop(
                                    x.clone_with_updated_read_from_mu(self.config.write_back_mu),
                                    output_stop_level,
                                ),
                            );
                        }
                        // Break if we've reached the partition rank
                        if stop_lev == self.partition_rank || expected_stop_level == Some(stop_lev)
                        {
                            dam::logging::log_event(&E::new(
                                "FlatPartition".to_string(),
                                self.id,
                                start_time.unwrap(),
                                self.time.tick().time(),
                                true,
                            ))
                            .unwrap();

                            start_time = None;

                            break;
                        }
                    }
                },
                Err(_) => {
                    let error_msg = if expected_stop_level.is_some() {
                        "The input stream lacks a stop token that corresponds to the stop token in the select stream!"
                    } else {
                        "Input stream ran out of things to dequeue during partition."
                    };
                    panic!("{}", error_msg);
                }
            }
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > Context for FlatPartition<E, A, SELT>
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
                        self.handle_load_cycles(&sel);
                        self.sel_stream.dequeue(&self.time).unwrap();
                        let select_vec = sel.to_sel_vec();
                        self.process_input_stream(&select_vec, None);
                    }
                    Elem::ValStop(sel, sel_level) => {
                        self.handle_load_cycles(&sel);
                        self.sel_stream.dequeue(&self.time).unwrap();
                        let select_vec = sel.to_sel_vec();
                        let expected_stop_level = sel_level + self.partition_rank;
                        self.process_input_stream(&select_vec, Some(expected_stop_level));
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
        operator::partition::{FlatPartition, FlatPartitionConfig},
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
    fn flat_partition_2d_multi_hot_rank_1() {
        fn create_ground_truth(
            arrays: &[Array2<i32>],
            read_from_mu: bool,
        ) -> Vec<Vec<Elem<Tile<i32>>>> {
            let mut ground_truth: Vec<Vec<Elem<Tile<i32>>>> = vec![Vec::new(); 4];

            // Define the mapping of which arrays go to which output streams
            let stream_mappings = [
                vec![0, 1, 2],          // Stream 0: arrays 0, 1, 2S1
                vec![0, 1, 2, 3, 4, 5], // Stream 1: arrays 0, 1, 2S1, 3, 4, 5S1
                vec![3, 4, 5, 6, 7, 8], // Stream 2: arrays 3, 4, 5S1, 6, 7, 8S1
                vec![6, 7, 8],          // Stream 3: arrays 6, 7, 8S1
            ];

            for (stream_idx, array_indices) in stream_mappings.iter().enumerate() {
                for (pos, &array_idx) in array_indices.iter().enumerate() {
                    let tile = Tile::new(arrays[array_idx].clone().into(), 4, read_from_mu);

                    // Add ValStop at the end of each group of 3 elements
                    if (pos + 1) % 3 == 0 {
                        ground_truth[stream_idx].push(Elem::ValStop(tile, 1));
                    } else {
                        ground_truth[stream_idx].push(Elem::Val(tile));
                    }
                }
            }

            ground_truth
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((2, 2), vec![i as i32; 4]).unwrap())
            .collect();
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        let in_read_from_mu = true;
        for (i, arr) in arrays.iter().enumerate() {
            let tile = Tile::new(arr.clone().into(), 4, in_read_from_mu);

            // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
            if i % 3 == 2 {
                if i == 8 {
                    in_stream_data.push(Elem::ValStop(tile, 2));
                } else {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                }
            } else {
                in_stream_data.push(Elem::Val(tile));
            }
        }
        // Step 3: Create a 3 rank-1 select stream with 2of4 multi-hot selections
        let select_read_from_mu = true;
        let select_stream_data = vec![
            Elem::Val(MultiHotN::new(
                vec![true, true, false, false],
                select_read_from_mu,
            )),
            Elem::Val(MultiHotN::new(
                vec![false, true, true, false],
                select_read_from_mu,
            )),
            Elem::ValStop(
                MultiHotN::new(vec![false, false, true, true], select_read_from_mu),
                1,
            ),
        ];

        // Step 4: Create the ground truth for 4 output streams
        let out_stream_data = create_ground_truth(&arrays, in_read_from_mu);

        // Step 5: Create 4 output streams
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (in_sel_snd, in_sel_rcv) = ctx.unbounded();
        let (exp1_snd, exp1_rcv) = ctx.unbounded();
        let (exp2_snd, exp2_rcv) = ctx.unbounded();
        let (exp3_snd, exp3_rcv) = ctx.unbounded();
        let (exp4_snd, exp4_rcv) = ctx.unbounded();

        // Step 6: Create the FlatPartitionConfig with 4 switch cycles and write_back_mu set to true
        let config = FlatPartitionConfig {
            switch_cycles: vec![1, 2, 3, 4],
            write_back_mu: true,
        };

        // Step 7: Create two GeneratorContexts for input and select streams
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || select_stream_data.into_iter(),
            in_sel_snd,
        ));

        // Step 8: Create the FlatPartition context
        ctx.add_child(FlatPartition::<SimpleEvent, _, _>::new(
            in_data_rcv,
            in_sel_rcv,
            vec![exp1_snd, exp2_snd, exp3_snd, exp4_snd],
            1, // partition_rank
            config,
            0, // id
        ));

        // Step 9: Create CheckerContexts for each output stream to verify the results
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[0].clone().into_iter(),
            exp1_rcv,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[1].clone().into_iter(),
            exp2_rcv,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[2].clone().into_iter(),
            exp3_rcv,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[3].clone().into_iter(),
            exp4_rcv,
            tolerance_fn,
        ));
        // Step 10: Initialize and run the context
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flat_partition_1d_multi_hot_rank_0() {
        fn create_ground_truth(
            arrays: &[Array2<i32>],
            multi_hot: &Vec<MultiHotN>,
            read_from_mu: bool,
        ) -> Vec<Vec<Elem<Tile<i32>>>> {
            let mut ground_truth: Vec<Vec<Elem<Tile<i32>>>> = vec![Vec::new(); multi_hot.len()];

            for (i, array_idx) in multi_hot.iter().enumerate() {
                for (j, &is_selected) in array_idx.iter().enumerate() {
                    if is_selected {
                        let tile = Tile::new(arrays[i].clone().into(), 4, read_from_mu);
                        ground_truth[j].push(Elem::Val(tile));
                    }
                }
            }
            ground_truth
        }

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
                multi_hot_arrays.push(MultiHotN::new(selection, read_from_mu));
            }
            multi_hot_arrays
        }

        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((2, 2), vec![i as i32; 4]).unwrap())
            .collect();
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        let in_read_from_mu = true;
        for (i, arr) in arrays.iter().enumerate() {
            let tile = Tile::new(arr.clone().into(), 4, in_read_from_mu);
            if i == 8 {
                in_stream_data.push(Elem::ValStop(tile, 1));
            } else {
                in_stream_data.push(Elem::Val(tile));
            }
        }

        let select_read_from_mu = true;
        let select_multi_hots = create_multi_hot_arrays::<4>(2, 9, select_read_from_mu);
        let mut select_stream_data: Vec<Elem<MultiHotN>> = Vec::new();
        for (i, multi_hot) in select_multi_hots.iter().enumerate() {
            if i == 8 {
                select_stream_data.push(Elem::ValStop(multi_hot.clone(), 1));
            } else {
                select_stream_data.push(Elem::Val(multi_hot.clone()));
            }
        }
        let out_stream_data = create_ground_truth(&arrays, &select_multi_hots, in_read_from_mu);
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (in_sel_snd, in_sel_rcv) = ctx.unbounded();
        let (exp1_snd, exp1_rcv) = ctx.unbounded();
        let (exp2_snd, exp2_rcv) = ctx.unbounded();
        let (exp3_snd, exp3_rcv) = ctx.unbounded();
        let (exp4_snd, exp4_rcv) = ctx.unbounded();
        let config = FlatPartitionConfig {
            switch_cycles: vec![1, 2, 3, 4],
            write_back_mu: true,
        };
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(GeneratorContext::new(
            || select_stream_data.into_iter(),
            in_sel_snd,
        ));
        ctx.add_child(FlatPartition::<SimpleEvent, _, _>::new(
            in_data_rcv,
            in_sel_rcv,
            vec![exp1_snd, exp2_snd, exp3_snd, exp4_snd],
            0, // partition_rank
            config,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[0].clone().into_iter(),
            exp1_rcv,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[1].clone().into_iter(),
            exp2_rcv,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[2].clone().into_iter(),
            exp3_rcv,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || out_stream_data[3].clone().into_iter(),
            exp4_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
