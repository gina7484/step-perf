use std::marker::PhantomData;
use std::panic;
use crate::utils::events::LoggableEventSimple;
use crate::memory::PMU_BW;
use dam::{context_tools::*, logging::LogEvent};
use crate::primitives::elem::{Elem, StopType, Bufferizable};
use crate::primitives::{tile::Tile, select::SelectAdapter};
use crate::utils::calculation::div_ceil;

pub struct FlatPartitionConfig {
    pub switch_cycles: Vec<u64>, // cycles between receiving
    pub write_back_mu: bool, // Whether the output is written to a memory unit
}

#[context_macro]
pub struct FlatPartition<E, A: DAMType, SELT: DAMType> {
    in_stream: Receiver<Elem<Tile<A>>>,
    sel_stream: Receiver<Elem<SELT>>,
    out_stream: Vec<Sender<Elem<Tile<A>>>>,
    partition_rank: StopType,
    config: FlatPartitionConfig, 
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > FlatPartition<E, A, SELT>
where
    Elem<Tile<A>>: DAMType,
    Elem<SELT>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<A>>>,
        sel_stream: Receiver<Elem<SELT>>,
        out_stream: Vec<Sender<Elem<Tile<A>>>>,
        partition_rank: StopType,
        config: FlatPartitionConfig,
    ) -> Self {
        let ctx = Self {
            in_stream,
            sel_stream,
            out_stream,
            partition_rank,
            config,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.sel_stream.attach_receiver(&ctx);
        for out in &ctx.out_stream {
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
    fn enqueue_to_experts(&mut self, select_vec: &[usize], elem: Elem<Tile<A>>) {
        for expert_idx in select_vec.iter() {
            self.out_stream[*expert_idx]
                .enqueue(&self.time, ChannelElement { 
                    time: self.time.tick(), 
                    data: elem.clone() 
                }).unwrap();
        }
    }

    /// Process input stream elements with the given select vector
    fn process_input_stream(&mut self, select_vec: &[usize], expected_stop_level: Option<StopType>) {
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data: val_data }) => match val_data {
                    Elem::Val(x) => {
                        self.handle_load_cycles(&x);
                        self.in_stream.dequeue(&self.time).unwrap();
                        self.handle_write_cycles(select_vec, &x);
                        self.enqueue_to_experts(select_vec, Elem::Val(x.clone()));

                        if self.partition_rank == 0 {
                            break;
                        }
                    }
                    Elem::ValStop(x, stop_lev) => {
                        // Validate stop level based on context
                        if let Some(expected) = expected_stop_level {
                            if expected != stop_lev {
                                panic!("The stop token ranks do not match between input stream and the select stream!");
                            }
                        } else if stop_lev > self.partition_rank {
                            panic!("The stop token ranks do not match between input stream and the select stream!");
                        }

                        // Break if we've reached the partition rank
                        if stop_lev == self.partition_rank {
                            break;
                        }

                        self.handle_load_cycles(&x);
                        self.in_stream.dequeue(&self.time).unwrap();
                        self.handle_write_cycles(select_vec, &x);
                        
                        // Determine output stop level
                        let output_stop_level = expected_stop_level
                            .map(|_| self.partition_rank)
                            .unwrap_or(stop_lev);
                        
                        self.enqueue_to_experts(select_vec, Elem::ValStop(x.clone(), output_stop_level));
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
        A: DAMType,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > Context for FlatPartition<E, A, SELT>
where
    Elem<Tile<A>>: DAMType,
    Elem<SELT>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.sel_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data: sel_data }) => match sel_data {
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
    use dam::simulation::ProgramBuilder;
    use ndarray::Array2;
    use dam::types::DAMType;
    use dam::context_tools::{Context, Receiver, Sender, ChannelElement};
    use crate::primitives::elem::Elem;
    use crate::primitives::tile::Tile;
    use crate::operator::partition::FlatPartition;

    #[test]
    fn flat_partition_1d_multi_hot() {
        // 1. Create 9 different ndarray::ArcArray2<T> with shape 2x2
        // 2. Create a 3x3 rank-2 data stream from these arrays
        // 3. Create a 3 rank-1 select stream with 2of4 multi-hot selections
        // 4. Create 4 output streams
        // 5. Create the ground truth for 4 output streams
        // 6. Create the FlatPartitionConfig with 4 switch cycles and write_back_mu set to true
        // 7. Create the FlatPartition context
        // 8. Run the context

        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Vec<Elem<Tile<i32>>>> {
            let mut ground_truth: Vec<Vec<Elem<Tile<i32>>>> = vec![Vec::new(); 4];
            
            // Define the mapping of which arrays go to which output streams
            let stream_mappings = [
                vec![0, 1, 2],  // Stream 0: arrays 0, 1, 2
                vec![0, 1, 2, 3, 4, 5],  // Stream 1: arrays 0, 1, 2, 3, 4, 5
                vec![3, 4, 5, 6, 7, 8],  // Stream 2: arrays 3, 4, 5, 6, 7, 8
                vec![6, 7, 8],  // Stream 3: arrays 6, 7, 8
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
        let in_read_from_mu = false;
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
        let select_read_from_mu = false;
        let select_stream_data = vec![
            Elem::Val(MultiHotN::new([true, true, false, false], select_read_from_mu)),
            Elem::Val(MultiHotN::new([false, true, true, false], select_read_from_mu)),
            Elem::ValStop(MultiHotN::new([false, false, true, true], select_read_from_mu), 1),
        ];

        // Step 4: Create the ground truth for 4 output streams
        let out_stream_data = create_ground_truth(&arrays, in_read_from_mu);

        // Step 5: Create 4 output streams
        // let mut ctx = ProgramBuilder::default();
        // let (exp1_snd, exp1_rcv) = ctx.unbounded();
        // let (exp2_snd, exp2_rcv) = ctx.unbounded();
        // let (exp3_snd, exp3_rcv) = ctx.unbounded();
        // let (exp4_snd, exp4_rcv) = ctx.unbounded();


    }
}