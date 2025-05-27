use core::panic;
use std::marker::PhantomData;
use crate::utils::events::LoggableEventSimple;
use crate::memory::PMU_BW;
use dam::channel::PeekResult;
use dam::{context_tools::*, logging::LogEvent};
use half::vec;
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
            // Use `peek` to check the presence of tokens in selected in_streams
            let mut peeked = vec![false; select_vec.len()];
            let mut num_peeked = 0;
            let mut peek_results = vec![None; select_vec.len()];
            while num_peeked < select_vec.len() {
                for (i, &idx) in select_vec.iter().enumerate() {
                    if peeked[i] {
                        continue; // Skip already peeked elements
                    }
                    match self.in_streams[idx].peek() {
                        PeekResult::Something(elem) => {
                            peek_results[i] = Some(elem);
                            peeked[i] = true;
                            num_peeked += 1;
                        }
                        PeekResult::Nothing(_) => {
                            continue;
                        }
                        PeekResult::Closed => {
                            // Handle closed stream, if necessary
                            return;
                        }
                    }
                }
                let mut data_arrive_times = vec![];
                let all_data = peek_results.iter().all(|t| {
                    match t {
                        Some(ChannelElement { time: arrive, data: Elem::Val(_)}) => {
                            data_arrive_times.push(arrive.time());
                            true
                        },
                        _ => false
                    }
                });

                let mut stop_values = vec![];
                let mut stop_arrive_times = vec![];
                let all_stop = peek_results.iter().all(|t| {
                    match t {
                        Some(ChannelElement { time: arrive, data: Elem::ValStop(_, level)}) => {
                            stop_arrive_times.push(arrive.time());
                            stop_values.push(*level);
                            true
                        },
                        _ => false
                    }
                });
                let all_stop = all_stop && stop_values.iter().all(|s| s == &stop_values[0]);

                if !all_data && !all_stop {
                    panic!("Not all selected streams have data or stop tokens available");
                }

                let mut data_ready_times = vec![];
                for (i, peek_elem) in peek_results.iter().enumerate() {
                    peek_elem.as_ref().map(|elem| {
                        let stream_id = select_vec[i];
                        let base_time = elem.time.time() + self.config.switch_cycles[stream_id];
                        match &elem.data {
                            Elem::Val(x) => {
                                if x.read_from_mu() {
                                    data_ready_times.push(base_time + div_ceil(x.size_in_bytes() as u64, PMU_BW));
                                } else{
                                    data_ready_times.push(base_time);
                                }
                            },
                            Elem::ValStop(x, _) => {
                                if x.read_from_mu() {
                                    data_ready_times.push(base_time + div_ceil(x.size_in_bytes() as u64, PMU_BW));
                                } else{
                                    data_ready_times.push(base_time);
                                }
                            }
                        }
                    });
                }
                // Dequeue in the ascending order of data ready times (FIFO scheuling)
                // TODO: Should we use ascending base time instead?
                let mut sorted_indices: Vec<usize> = (0..data_ready_times.len()).collect();
                sorted_indices.sort_by_key(|&i| data_ready_times[i]);
                let additional_rank = index_level.unwrap_or(0);
                for &i in sorted_indices.iter() {
                    self.in_streams[select_vec[i]].dequeue(&self.time).unwrap();
                }
                let max_arrive_time = *data_arrive_times.iter().max().unwrap_or(&0);
                self.time.advance(max_arrive_time.into());
                for i in 0..select_vec.len() {
                    let to_break = peek_results[i].as_ref().map(|elem| {
                        match &elem.data {
                            Elem::Val(x) => {
                                if self.config.write_back_mu {
                                    self.time.incr_cycles(div_ceil(x.size_in_bytes() as u64, PMU_BW));
                                }
                                if i == select_vec.len() - 1 {
                                    self.out_stream.enqueue(&self.time, ChannelElement { time: self.time.tick(), data: Elem::ValStop(x.clone(), additional_rank + 1)}).unwrap();
                                    if self.in_stream_rank == 0 {
                                        return true;
                                    }
                                } else {
                                    if self.in_stream_rank == 0 {
                                        self.out_stream.enqueue(&self.time, ChannelElement { time: self.time.tick(), data: Elem::ValStop(x.clone(), additional_rank + 1)}).unwrap();
                                    } else {
                                        self.out_stream.enqueue(&self.time, ChannelElement { time: self.time.tick(), data: Elem::Val(x.clone())}).unwrap();
                                    }
                                }
                                false
                            }
                            Elem::ValStop(x, level) => {
                                if self.in_stream_rank == 0 {
                                    panic!("The in_stream_rank does not match the stop token level!");
                                }
                                if self.config.write_back_mu {
                                    self.time.incr_cycles(div_ceil(x.size_in_bytes() as u64, PMU_BW));
                                }
                                let base_rank = additional_rank + level;
                                if i == select_vec.len() - 1 {
                                    self.out_stream.enqueue(&self.time, ChannelElement { time: self.time.tick(), data: Elem::ValStop(x.clone(), base_rank + 1)}).unwrap();
                                    if *level == self.in_stream_rank {
                                        return true;
                                    }
                                } else {
                                    self.out_stream.enqueue(&self.time, ChannelElement { time: self.time.tick(), data: Elem::Val(x.clone())}).unwrap();
                                }
                                false
                            }
                        }
                    });
                    if to_break.unwrap_or(false) {
                        break 'expert;
                    }
                }
            }
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