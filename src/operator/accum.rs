use std::{marker::PhantomData, sync::Arc};

use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

pub struct AccumConfig {
    pub compute_bw: u64,
    pub write_back_mu: bool,
}

#[context_macro]
pub struct Accum<E, T: DAMType, OT: DAMType> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Tile<OT>>>,
    func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
    rank: StopType,
    config: AccumConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Accum<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<OT>>>,
        func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
        init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
        rank: StopType,
        config: AccumConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            func,
            init_accum,
            rank,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    fn process_accum(&mut self, data: Tile<T>, accumulator: &mut Tile<OT>) {
        let load_cycles = if data.read_from_mu {
            div_ceil(data.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let (comp_cycles, out_tile) = (self.func)(
            &data,
            &accumulator,
            self.config.compute_bw,
            self.config.write_back_mu,
        );
        *accumulator = out_tile;

        let roofline_cycles = [load_cycles, comp_cycles].into_iter().max().unwrap_or(0);

        // increment cycles and dequeue inputs
        self.time.incr_cycles(roofline_cycles);

        self.in_stream.dequeue(&self.time).unwrap();
    }

    fn process_accum_init(&mut self, data: Tile<T>, accumulator: &mut Tile<OT>) -> Tile<OT> {
        let load_cycles = if data.read_from_mu {
            div_ceil(data.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let (comp_cycles, out_tile) = (self.func)(
            &data,
            &accumulator,
            self.config.compute_bw,
            self.config.write_back_mu,
        );
        *accumulator = (self.init_accum)();

        let store_cycles = if self.config.write_back_mu {
            div_ceil(accumulator.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
            .into_iter()
            .max()
            .unwrap_or(0);

        self.time.incr_cycles(roofline_cycles);
        self.in_stream.dequeue(&self.time).unwrap();

        // Logging
        dam::logging::log_event(&E::new(
            "Accum".to_string(),
            self.id,
            self.time.tick().time() - roofline_cycles,
            self.time.tick().time(),
            true,
        ))
        .unwrap();

        out_tile
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Context for Accum<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    fn run(&mut self) {
        let mut accumulator = (self.init_accum)();
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        self.process_accum(x, &mut accumulator);
                    }
                    Elem::ValStop(x, level) => {
                        if level < self.rank {
                            self.process_accum(x, &mut accumulator);
                        } else if level == self.rank {
                            let out_tile = self.process_accum_init(x, &mut accumulator);
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(out_tile),
                                    },
                                )
                                .unwrap();
                        } else {
                            let out_tile = self.process_accum_init(x, &mut accumulator);
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(out_tile, level - self.rank),
                                    },
                                )
                                .unwrap();
                        }
                    }
                },
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        functions::accum_fn,
        operator::accum::{Accum, AccumConfig},
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext, PrinterContext};
    use ndarray::Array2;
    use std::sync::Arc;

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
        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

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
            in_stream_data
        }

        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
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
        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((2, 2), vec![i as i32; 4]).unwrap())
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
        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            in_data_rcv,
            out_data_snd,
            Arc::new(accum_fn::retile_col),
            Arc::new(move || Tile::new_empty([2, 0], 4, read_from_mu)),
            1, // rank
            AccumConfig {
                compute_bw: 1000, // FLOPs per cycle
                write_back_mu: true,
            },
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
        // [1,3,3] => [1,3]
        // [1,4] tile => [3,4] tile
        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

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
            in_stream_data
        }

        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
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
        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            in_data_rcv,
            out_data_snd,
            Arc::new(accum_fn::retile_row),
            Arc::new(move || Tile::new_empty([0, 4], 4, read_from_mu)),
            1, // rank
            AccumConfig {
                compute_bw: 1000, // FLOPs per cycle (Currently unused)
                write_back_mu: true,
            },
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
