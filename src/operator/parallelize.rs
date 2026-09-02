use crate::operator::partition::FlatPartitionConfig;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;

#[context_macro]
pub struct Parallelize<E, A: DAMType> {
    in_stream: Receiver<Elem<A>>,
    out_streams: Vec<Sender<Elem<A>>>,
    parallelize_rank: StopType,
    output_dim: u32,
    config: FlatPartitionConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType,
    > Parallelize<E, A>
where
    Elem<A>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<A>>,
        out_streams: Vec<Sender<Elem<A>>>,
        parallelize_rank: StopType,
        output_dim: u32,
        config: FlatPartitionConfig,
        id: u32,
    ) -> Self {
        assert!(
            !out_streams.is_empty(),
            "Parallelize needs an output stream"
        );
        assert!(output_dim > 0, "Parallelize output_dim must be positive");
        assert_eq!(
            out_streams.len(),
            config.switch_cycles.len(),
            "Parallelize needs one switch-cycle value per output stream"
        );
        let ctx = Self {
            in_stream,
            out_streams,
            parallelize_rank,
            output_dim,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        for out in &ctx.out_streams {
            out.attach_sender(&ctx);
        }

        ctx
    }

    fn enqueue(&mut self, lane: usize, data: Elem<A>) {
        self.out_streams[lane]
            .enqueue(
                &self.time,
                ChannelElement {
                    time: self.time.tick() + self.config.switch_cycles[lane],
                    data,
                },
            )
            .unwrap();
    }

    fn flush_lane_group(
        &mut self,
        pending_terminals: &mut [Option<A>],
        final_value: A,
        stop_level: StopType,
    ) {
        let final_lane = self.out_streams.len() - 1;
        for lane in 0..final_lane {
            let value = pending_terminals[lane].take().unwrap_or_else(|| {
                panic!(
                    "Parallelize {} is missing the terminal value for lane {}",
                    self.id, lane
                )
            });
            self.enqueue(lane, Elem::ValStop(value, stop_level));
        }
        self.enqueue(final_lane, Elem::ValStop(final_value, stop_level));
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType,
    > Context for Parallelize<E, A>
where
    Elem<A>: DAMType,
{
    fn run(&mut self) {
        let par_factor = self.out_streams.len();
        let final_lane = par_factor - 1;
        let mut lane = 0usize;
        let mut count = 0u32;
        // The final lane reveals whether the current outer group ends at
        // S(r+1) or at a higher level. Retain only each earlier lane's final
        // value so every output lane receives that same canonical stop level.
        let mut pending_terminals: Vec<Option<A>> = (0..final_lane).map(|_| None).collect();

        loop {
            let element = match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => data,
                Err(_) => {
                    if lane != 0 || count != 0 || pending_terminals.iter().any(Option::is_some) {
                        panic!(
                            "Parallelize {} reached the end of an incomplete lane group \
                             (lane {}, count {}, output_dim {})",
                            self.id, lane, count, self.output_dim
                        );
                    }
                    return;
                }
            };

            match element {
                Elem::Val(value) if self.parallelize_rank == 0 => {
                    count += 1;
                    if count > self.output_dim {
                        panic!(
                            "Parallelize {} exceeded output_dim {} on lane {}",
                            self.id, self.output_dim, lane
                        );
                    }
                    if count == self.output_dim {
                        if lane == final_lane {
                            panic!(
                                "Parallelize {} expected a stop token after {} values \
                                 on final lane {}",
                                self.id, self.output_dim, lane
                            );
                        }
                        assert!(
                            pending_terminals[lane].replace(value).is_none(),
                            "Parallelize {} already has a pending terminal for lane {}",
                            self.id,
                            lane
                        );
                        lane += 1;
                        count = 0;
                    } else {
                        self.enqueue(lane, Elem::Val(value));
                    }
                }
                Elem::Val(value) => self.enqueue(lane, Elem::Val(value)),
                Elem::ValStop(value, stop_level) if self.parallelize_rank == 0 => {
                    count += 1;
                    if stop_level == 0 || count != self.output_dim || lane != final_lane {
                        panic!(
                            "Parallelize {} saw terminal S{} at lane {} count {}; \
                             expected final lane {} count {}",
                            self.id, stop_level, lane, count, final_lane, self.output_dim
                        );
                    }
                    self.flush_lane_group(&mut pending_terminals, value, stop_level);
                    lane = 0;
                    count = 0;
                }
                Elem::ValStop(value, stop_level) if stop_level < self.parallelize_rank => {
                    self.enqueue(lane, Elem::ValStop(value, stop_level));
                }
                Elem::ValStop(value, stop_level) if stop_level == self.parallelize_rank => {
                    count += 1;
                    if count > self.output_dim {
                        panic!(
                            "Parallelize {} exceeded output_dim {} on lane {}",
                            self.id, self.output_dim, lane
                        );
                    }
                    if count == self.output_dim {
                        if lane == final_lane {
                            panic!(
                                "Parallelize {} expected S(N), N > {}, at the end \
                                 of final lane {}",
                                self.id, self.parallelize_rank, lane
                            );
                        }
                        assert!(
                            pending_terminals[lane].replace(value).is_none(),
                            "Parallelize {} already has a pending terminal for lane {}",
                            self.id,
                            lane
                        );
                        lane += 1;
                        count = 0;
                    } else {
                        self.enqueue(lane, Elem::ValStop(value, stop_level));
                    }
                }
                Elem::ValStop(value, stop_level) => {
                    count += 1;
                    if count != self.output_dim || lane != final_lane {
                        panic!(
                            "Parallelize {} saw terminal S{} at lane {} count {}; \
                             expected final lane {} count {}",
                            self.id, stop_level, lane, count, final_lane, self.output_dim
                        );
                    }
                    self.flush_lane_group(&mut pending_terminals, value, stop_level);
                    lane = 0;
                    count = 0;
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Parallelize;
    use crate::{
        operator::partition::FlatPartitionConfig,
        operator::static_reassemble::StaticReassemble,
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext};

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
    fn parallelize_0d() {
        // cargo test --package step_perf --lib -- operator::parallelize::tests::parallelize_0d --exact --show-output
        type VT = u32;
        const READ_FROM_MU: bool = false;
        const DUMMY_ID: u32 = 0;
        let mut ctx = ProgramBuilder::default();

        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd0, out_data_rcv0) = ctx.unbounded();
        let (out_data_snd1, out_data_rcv1) = ctx.unbounded();
        let (out_data_snd2, out_data_rcv2) = ctx.unbounded();
        let (out_data_snd3, out_data_rcv3) = ctx.unbounded();
        use ndarray::ArcArray2;

        let tile1 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile2 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile3 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile4 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );

        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::Val(tile1),
                    Elem::Val(tile2),
                    Elem::Val(tile3),
                    Elem::ValStop(tile4, 1),
                ]
                .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(Parallelize::<SimpleEvent, _>::new(
            in_data_rcv,
            vec![out_data_snd0, out_data_snd1, out_data_snd2, out_data_snd3],
            0,
            1,
            FlatPartitionConfig {
                switch_cycles: vec![1; 4],
                write_back_mu: false,
            },
            DUMMY_ID,
        ));

        let tile1 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile2 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile3 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile4 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile1, 1)].into_iter(),
            out_data_rcv0,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile2, 1)].into_iter(),
            out_data_rcv1,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile3, 1)].into_iter(),
            out_data_rcv2,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile4, 1)].into_iter(),
            out_data_rcv3,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn parallelize_1d() {
        // cargo test --package step_perf --lib -- operator::parallelize::tests::parallelize_1d --exact --show-output
        type VT = u32;
        const READ_FROM_MU: bool = false;
        const DUMMY_ID: u32 = 0;
        let mut ctx = ProgramBuilder::default();

        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd0, out_data_rcv0) = ctx.unbounded();
        let (out_data_snd1, out_data_rcv1) = ctx.unbounded();
        let (out_data_snd2, out_data_rcv2) = ctx.unbounded();
        let (out_data_snd3, out_data_rcv3) = ctx.unbounded();
        use ndarray::ArcArray2;

        let tile1 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile2 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile3 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile4 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );

        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::ValStop(tile1, 1),
                    Elem::ValStop(tile2, 1),
                    Elem::ValStop(tile3, 1),
                    Elem::ValStop(tile4, 2),
                ]
                .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(Parallelize::<SimpleEvent, _>::new(
            in_data_rcv,
            vec![out_data_snd0, out_data_snd1, out_data_snd2, out_data_snd3],
            1,
            1,
            FlatPartitionConfig {
                switch_cycles: vec![1; 4],
                write_back_mu: false,
            },
            DUMMY_ID,
        ));

        let tile1 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile2 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile3 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile4 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile1, 2)].into_iter(),
            out_data_rcv0,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile2, 2)].into_iter(),
            out_data_rcv1,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile3, 2)].into_iter(),
            out_data_rcv2,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::ValStop(tile4, 2)].into_iter(),
            out_data_rcv3,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    fn scalar_tile(value: u32) -> Tile<u32> {
        use ndarray::ArcArray2;

        Tile::new(
            ArcArray2::from_shape_vec((1, 1), vec![value]).unwrap(),
            0,
            false,
        )
    }

    fn canonical_stream(shape: &[usize], values: Vec<u32>) -> Vec<Elem<Tile<u32>>> {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                let position = index + 1;
                let mut stride = 1usize;
                let mut stop_level = 0u32;
                for dim in shape.iter().rev() {
                    stride *= dim;
                    if position % stride != 0 {
                        break;
                    }
                    stop_level += 1;
                }
                let tile = scalar_tile(value);
                if stop_level == 0 {
                    Elem::Val(tile)
                } else {
                    Elem::ValStop(tile, stop_level)
                }
            })
            .collect()
    }

    #[test]
    fn parallelize_inner_rank_preserves_outer_stops_on_every_lane() {
        let mut ctx = ProgramBuilder::default();
        let (input_sender, input_receiver) = ctx.unbounded();
        let (lane0_sender, lane0_receiver) = ctx.unbounded();
        let (lane1_sender, lane1_receiver) = ctx.unbounded();

        let input = canonical_stream(&[2, 4, 3], (0..24).collect());
        ctx.add_child(GeneratorContext::new(
            move || input.into_iter(),
            input_sender,
        ));
        ctx.add_child(Parallelize::<SimpleEvent, _>::new(
            input_receiver,
            vec![lane0_sender, lane1_sender],
            1,
            2,
            FlatPartitionConfig {
                switch_cycles: vec![1; 2],
                write_back_mu: false,
            },
            0,
        ));

        let lane0_values = (0..6).chain(12..18).collect();
        let lane1_values = (6..12).chain(18..24).collect();
        let lane0 = canonical_stream(&[2, 2, 3], lane0_values);
        let lane1 = canonical_stream(&[2, 2, 3], lane1_values);
        ctx.add_child(ApproxCheckerContext::new(
            move || lane0.into_iter(),
            lane0_receiver,
            tolerance_fn,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            move || lane1.into_iter(),
            lane1_receiver,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn parallelize_then_reassemble_inner_rank_round_trips() {
        let mut ctx = ProgramBuilder::default();
        let (input_sender, input_receiver) = ctx.unbounded();
        let (lane0_sender, lane0_receiver) = ctx.unbounded();
        let (lane1_sender, lane1_receiver) = ctx.unbounded();
        let (output_sender, output_receiver) = ctx.unbounded();

        let input = canonical_stream(&[2, 4, 3], (0..24).collect());
        let expected = canonical_stream(&[2, 4, 3], (0..24).collect());
        ctx.add_child(GeneratorContext::new(
            move || input.into_iter(),
            input_sender,
        ));
        ctx.add_child(Parallelize::<SimpleEvent, _>::new(
            input_receiver,
            vec![lane0_sender, lane1_sender],
            1,
            2,
            FlatPartitionConfig {
                switch_cycles: vec![1; 2],
                write_back_mu: false,
            },
            0,
        ));
        ctx.add_child(StaticReassemble::<SimpleEvent, _>::new(
            vec![lane0_receiver, lane1_receiver],
            output_sender,
            1,
            FlatPartitionConfig {
                switch_cycles: vec![1; 2],
                write_back_mu: false,
            },
            1,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            move || expected.into_iter(),
            output_receiver,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
