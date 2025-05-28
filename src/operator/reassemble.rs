use core::panic;
use std::marker::PhantomData;
use crate::utils::events::LoggableEventSimple;
use crate::memory::PMU_BW;
use dam::channel::PeekResult;
use dam::{context_tools::*, logging::LogEvent};
use crate::primitives::elem::{Elem, StopType, Bufferizable};
use crate::primitives::{tile::Tile, select::SelectAdapter};
use crate::utils::calculation::div_ceil;

pub struct FlatReassembleConfig {
    pub switch_cycles: Vec<u64>,
    pub write_back_mu: bool,
}

#[context_macro]
pub struct FlatReassemble<E, A: DAMType, SELT: DAMType> {
    in_streams: Vec<Receiver<Elem<Tile<A>>>>,
    sel_stream: Receiver<Elem<SELT>>,
    out_stream: Sender<Elem<Tile<A>>>,
    in_stream_rank: StopType,
    config: FlatReassembleConfig,
    _phantom: PhantomData<E>,
}


impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > FlatReassemble<E, A, SELT>
where 
    Elem<Tile<A>>: DAMType,
    Elem<SELT>: DAMType,
{
    pub fn new(
        in_streams: Vec<Receiver<Elem<Tile<A>>>>,
        sel_stream: Receiver<Elem<SELT>>,
        out_stream: Sender<Elem<Tile<A>>>,
        in_stream_rank: StopType,
        config: FlatReassembleConfig,
    ) -> Self {
        let ctx = Self {
            in_streams,
            sel_stream,
            out_stream,
            in_stream_rank,
            config,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_streams.iter().for_each(|s| s.attach_receiver(&ctx));
        ctx.sel_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    /// Helper function to calculate and increment load cycles for memory operations
    fn handle_load_cycles<T: Bufferizable>(&mut self, data: &T) {
        if data.read_from_mu() {
            let load_cycle = div_ceil(data.size_in_bytes() as u64, PMU_BW);
            self.time.incr_cycles(load_cycle);
        }
    }

    fn process_input_stream(&mut self, select_vec: &[usize], index_level: Option<StopType>) {
        'expert: loop {
            let peek_results = self.peek_all_streams(select_vec);
            
            if peek_results.is_empty() {
                return; // All streams closed
            }

            self.validate_peek_results(&peek_results);
            let data_ready_times = self.calculate_data_ready_times(&peek_results, select_vec);
            
            self.dequeue_streams_in_order(&data_ready_times, select_vec);
            self.advance_time_to_max_ready(&peek_results);
            
            if self.process_and_enqueue_outputs(&peek_results, select_vec, index_level) {
                break 'expert;
            }
        }
    }

    fn peek_all_streams(&mut self, select_vec: &[usize]) -> Vec<Option<ChannelElement<Elem<Tile<A>>>>> {
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
        }
        
        peek_results
    }

    fn validate_peek_results(&self, peek_results: &[Option<ChannelElement<Elem<Tile<A>>>>]) {
        let (data_arrive_times, all_data) = self.check_all_data(peek_results);
        let (stop_values, stop_arrive_times, all_stop) = self.check_all_stop(peek_results);
        
        if !all_data && !all_stop {
            panic!("Not all selected streams have data or stop tokens available");
        }
    }

    fn check_all_data(&self, peek_results: &[Option<ChannelElement<Elem<Tile<A>>>>]) -> (Vec<u64>, bool) {
        let mut data_arrive_times = vec![];
        let all_data = peek_results.iter().all(|t| {
            match t {
                Some(ChannelElement { time: arrive, data: Elem::Val(_) }) => {
                    data_arrive_times.push(arrive.time());
                    true
                }
                _ => false,
            }
        });
        (data_arrive_times, all_data)
    }

    fn check_all_stop(&self, peek_results: &[Option<ChannelElement<Elem<Tile<A>>>>]) -> (Vec<StopType>, Vec<u64>, bool) {
        let mut stop_values = vec![];
        let mut stop_arrive_times = vec![];
        let all_stop = peek_results.iter().all(|t| {
            match t {
                Some(ChannelElement { time: arrive, data: Elem::ValStop(_, level) }) => {
                    stop_arrive_times.push(arrive.time());
                    stop_values.push(*level);
                    true
                }
                _ => false,
            }
        });
        
        let uniform_stop = all_stop && stop_values.iter().all(|s| s == &stop_values[0]);
        (stop_values, stop_arrive_times, uniform_stop)
    }

    fn calculate_data_ready_times(&self, peek_results: &[Option<ChannelElement<Elem<Tile<A>>>>], select_vec: &[usize]) -> Vec<u64> {
        let mut data_ready_times = vec![];
        
        for (i, peek_elem) in peek_results.iter().enumerate() {
            if let Some(elem) = peek_elem {
                let stream_id = select_vec[i];
                let base_time = elem.time.time() + self.config.switch_cycles[stream_id];
                
                let ready_time = match &elem.data {
                    Elem::Val(x) | Elem::ValStop(x, _) => {
                        if x.read_from_mu() {
                            base_time + div_ceil(x.size_in_bytes() as u64, PMU_BW)
                        } else {
                            base_time
                        }
                    }
                };
                data_ready_times.push(ready_time);
            }
        }
        
        data_ready_times
    }

    fn dequeue_streams_in_order(&mut self, data_ready_times: &[u64], select_vec: &[usize]) {
        // Dequeue in ascending order of data ready times (FIFO scheduling)
        let mut sorted_indices: Vec<usize> = (0..data_ready_times.len()).collect();
        sorted_indices.sort_by_key(|&i| data_ready_times[i]);
        
        for &i in sorted_indices.iter() {
            self.in_streams[select_vec[i]].dequeue(&self.time).unwrap();
        }
    }

    fn advance_time_to_max_ready(&mut self, peek_results: &[Option<ChannelElement<Elem<Tile<A>>>>]) {
        let max_ready_time = peek_results
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|elem| elem.time.time())
            .max()
            .unwrap_or(0);
        
        self.time.advance(max_ready_time.into());
    }

    fn process_and_enqueue_outputs(
        &mut self, 
        peek_results: &[Option<ChannelElement<Elem<Tile<A>>>>], 
        select_vec: &[usize], 
        index_level: Option<StopType>
    ) -> bool {
        let additional_rank = index_level.unwrap_or(0);
        
        for i in 0..select_vec.len() {
            if let Some(elem) = &peek_results[i] {
                if self.process_single_element(elem, i, select_vec.len(), additional_rank) {
                    return true; // Break outer loop
                }
            }
        }
        false
    }

    fn process_single_element(
        &mut self, 
        elem: &ChannelElement<Elem<Tile<A>>>, 
        index: usize, 
        total_streams: usize, 
        additional_rank: StopType
    ) -> bool {
        match &elem.data {
            Elem::Val(x) => {
                self.handle_memory_writeback(x);
                self.enqueue_val_element(x, index, total_streams, 0)
            }
            Elem::ValStop(x, level) => {
                if self.in_stream_rank == 0 {
                    panic!("The in_stream_rank does not match the stop token level!");
                }
                self.handle_memory_writeback(x);
                self.enqueue_val_stop_element(x, *level, index, total_streams, additional_rank)
            }
        }
    }

    fn handle_memory_writeback(&mut self, x: &Tile<A>) {
        if self.config.write_back_mu {
            self.time.incr_cycles(div_ceil(x.size_in_bytes() as u64, PMU_BW));
        }
    }

    fn enqueue_val_element(
        &mut self, 
        x: &Tile<A>, 
        index: usize, 
        total_streams: usize, 
        additional_rank: StopType
    ) -> bool {
        if index == total_streams - 1 {
            // Last stream - always enqueue as ValStop
            self.out_stream.enqueue(
                &self.time, 
                ChannelElement { 
                    time: self.time.tick(), 
                    data: Elem::ValStop(x.clone(), additional_rank + 1) 
                }
            ).unwrap();
            
            self.in_stream_rank == 0
        } else {
            // Not last stream
            let data = if self.in_stream_rank == 0 {
                Elem::ValStop(x.clone(), additional_rank + 1)
            } else {
                Elem::Val(x.clone())
            };
            
            self.out_stream.enqueue(
                &self.time, 
                ChannelElement { time: self.time.tick(), data }
            ).unwrap();
            
            false
        }
    }

    fn enqueue_val_stop_element(
        &mut self, 
        x: &Tile<A>, 
        level: StopType, 
        index: usize, 
        total_streams: usize, 
        additional_rank: StopType
    ) -> bool {
        let base_rank = additional_rank + level;
        
        if index == total_streams - 1 {
            // Last stream
            self.out_stream.enqueue(
                &self.time, 
                ChannelElement { 
                    time: self.time.tick(), 
                    data: Elem::ValStop(x.clone(), base_rank + 1) 
                }
            ).unwrap();
            
            level == self.in_stream_rank
        } else {
            // Not last stream
            self.out_stream.enqueue(
                &self.time, 
                ChannelElement { 
                    time: self.time.tick(), 
                    data: Elem::Val(x.clone()) 
                }
            ).unwrap();
            
            false
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: DAMType,
        SELT: DAMType + SelectAdapter + Bufferizable,
    > Context for FlatReassemble<E, A, SELT>
where 
    Elem<Tile<A>>: DAMType,
    Elem<SELT>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.sel_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data: sel_data}) => match sel_data {
                    Elem::Val(sel) => {
                        self.handle_load_cycles(&sel);
                        self.sel_stream.dequeue(&self.time).unwrap();
                        let select_vec = sel.to_sel_vec();
                        self.process_input_stream(&select_vec, None);
                    }
                    Elem::ValStop(sel, sel_level ) => {
                        self.handle_load_cycles(&sel);
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
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext, PrinterContext};
    use ndarray::Array2;
    use crate::{
        primitives::{elem::Elem, tile::Tile},
        operator::reassemble::{FlatReassemble, FlatReassembleConfig},
        utils::events::DummyEvent
    };

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
        fn create_input_streams(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Vec<Elem<Tile<i32>>>> {
            let mut input_streams: Vec<Vec<Elem<Tile<i32>>>> = vec![Vec::new(); 4];
            
            // Define the mapping of which arrays go to which output streams
            let stream_mappings = [
                vec![0, 1, 2],  // Stream 0: arrays 0, 1, 2S1
                vec![0, 1, 2, 3, 4, 5],  // Stream 1: arrays 0, 1, 2S1, 3, 4, 5S1
                vec![3, 4, 5, 6, 7, 8],  // Stream 2: arrays 3, 4, 5S1, 6, 7, 8S1
                vec![6, 7, 8],  // Stream 3: arrays 6, 7, 8S1
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
            for (i, array) in arrays.iter().enumerate() {
                let tile = Tile::new(array.clone().into(), 4, read_from_mu);
                ground_truth.push(Elem::Val(tile.clone()));
                if i == arrays.len() - 1 {
                    ground_truth.push(Elem::ValStop(tile, 3));
                } else {
                    if (i + 1) % 3 == 0 {
                        ground_truth.push(Elem::ValStop(tile, 2));
                    } else {
                        ground_truth.push(Elem::ValStop(tile, 1));
                    }
                }
            }
            ground_truth
        }

        fn create_select_streams(read_from_mu: bool) -> Vec<Elem<MultiHotN<4>>> {
            vec![
                Elem::Val(MultiHotN::new([true, true, false, false], read_from_mu)),
                Elem::Val(MultiHotN::new([false, true, true, false], read_from_mu)),
                Elem::ValStop(MultiHotN::new([false, false, true, true], read_from_mu), 1),
            ]
        }

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
            exp1_snd
        ));

        ctx.add_child(GeneratorContext::new(
            || input_streams_data[1].clone().into_iter(),
            exp2_snd
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[2].clone().into_iter(),
            exp3_snd
        ));
        ctx.add_child(GeneratorContext::new(
            || input_streams_data[3].clone().into_iter(),
            exp4_snd
        ));
        ctx.add_child(GeneratorContext::new(
            || select_stream_data.into_iter(),
            in_sel_snd
        ));
        ctx.add_child(FlatReassemble::<DummyEvent, _, _>::new(
            vec![exp1_rcv, exp2_rcv, exp3_rcv, exp4_rcv],
            in_sel_rcv,
            out_data_snd,
            1,
            config,
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
