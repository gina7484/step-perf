use crate::operator::partition::FlatPartitionConfig;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;

#[context_macro]
pub struct StaticReassemble<E, A: DAMType> {
    in_streams: Vec<Receiver<Elem<A>>>,
    out_stream: Sender<Elem<A>>,
    reassemble_rank: StopType,
    config: FlatPartitionConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType,
    > StaticReassemble<E, A>
where
    Elem<A>: DAMType,
{
    pub fn new(
        in_streams: Vec<Receiver<Elem<A>>>,
        out_stream: Sender<Elem<A>>,
        reassemble_rank: StopType,
        config: FlatPartitionConfig,
        id: u32,
    ) -> Self {
        assert!(
            !in_streams.is_empty(),
            "StaticReassemble needs an input stream"
        );
        assert_eq!(
            in_streams.len(),
            config.switch_cycles.len(),
            "StaticReassemble needs one switch-cycle value per input stream"
        );
        let ctx = Self {
            in_streams,
            out_stream,
            reassemble_rank,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        for in_stream in &ctx.in_streams {
            in_stream.attach_receiver(&ctx);
        }
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }

    fn enqueue(&mut self, lane: usize, data: Elem<A>) {
        self.out_stream
            .enqueue(
                &self.time,
                ChannelElement {
                    time: self.time.tick() + self.config.switch_cycles[lane],
                    data,
                },
            )
            .unwrap();
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        A: Bufferizable + DAMType,
    > Context for StaticReassemble<E, A>
where
    Elem<A>: DAMType,
{
    fn run(&mut self) {
        let par_factor = self.in_streams.len();
        let final_lane = par_factor - 1;

        loop {
            let mut expected_stop_level: Option<StopType> = None;
            for lane in 0..par_factor {
                loop {
                    match self.in_streams[lane].dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: val_data,
                        }) => match val_data {
                            Elem::Val(value) => self.enqueue(lane, Elem::Val(value)),
                            Elem::ValStop(value, stop_level) => {
                                if stop_level <= self.reassemble_rank {
                                    self.enqueue(lane, Elem::ValStop(value, stop_level));
                                    continue;
                                }

                                match expected_stop_level {
                                    Some(expected) if expected != stop_level => {
                                        panic!(
                                            "StaticReassemble {} saw S{} on lane {}, but \
                                             the lane group started with S{}",
                                            self.id, stop_level, lane, expected
                                        );
                                    }
                                    None => expected_stop_level = Some(stop_level),
                                    _ => {}
                                }

                                let output = if lane == final_lane {
                                    Elem::ValStop(value, stop_level)
                                } else if self.reassemble_rank == 0 {
                                    // Remove the lane-local S1 when joining the
                                    // innermost dimension.
                                    Elem::Val(value)
                                } else {
                                    // Keep boundaries below the reconstructed
                                    // axis, but remove this lane's closure of
                                    // the reconstructed axis and outer axes.
                                    Elem::ValStop(value, self.reassemble_rank)
                                };
                                self.enqueue(lane, output);
                                break;
                            }
                        },
                        Err(_) if lane == 0 => {
                            for remaining_lane in 1..par_factor {
                                if self.in_streams[remaining_lane].dequeue(&self.time).is_ok() {
                                    panic!(
                                        "StaticReassemble {} lane {} has data after \
                                         lane 0 ended",
                                        self.id, remaining_lane
                                    );
                                }
                            }
                            return;
                        }
                        Err(_) => {
                            panic!(
                                "StaticReassemble {} lane {} ended inside a lane group",
                                self.id, lane
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StaticReassemble;
    use crate::{
        operator::partition::FlatPartitionConfig,
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
    fn static_reassemble_0d() {
        // cargo test --package step_perf --lib -- operator::static_reassemble::tests::static_reassemble_0d --exact --show-output
        type VT = u32;
        const READ_FROM_MU: bool = false;
        const DUMMY_ID: u32 = 0;
        let mut ctx = ProgramBuilder::default();

        let (in_data_snd0, in_data_rcv0) = ctx.unbounded();
        let (in_data_snd1, in_data_rcv1) = ctx.unbounded();
        let (in_data_snd2, in_data_rcv2) = ctx.unbounded();
        let (in_data_snd3, in_data_rcv3) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        use ndarray::ArcArray2;

        let tile1 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        // Each rank-0 lane still has a terminal S1.
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile1, 1)].into_iter(),
            in_data_snd0,
        ));

        let tile2_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile2_clone, 1)].into_iter(),
            in_data_snd1,
        ));

        let tile3_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile3_clone, 1)].into_iter(),
            in_data_snd2,
        ));

        let tile4_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile4_clone, 1)].into_iter(),
            in_data_snd3,
        ));

        ctx.add_child(StaticReassemble::<SimpleEvent, _>::new(
            vec![in_data_rcv0, in_data_rcv1, in_data_rcv2, in_data_rcv3],
            out_data_snd,
            0,
            FlatPartitionConfig {
                switch_cycles: vec![1; 4],
                write_back_mu: false,
            },
            DUMMY_ID,
        ));

        // Expected output: all tiles merged back into single stream
        let tile1_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile2_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile3_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile4_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );

        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(tile1_exp),
                    Elem::Val(tile2_exp),
                    Elem::Val(tile3_exp),
                    Elem::ValStop(tile4_exp, 1),
                ]
                .into_iter()
            },
            out_data_rcv,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn static_reassemble_1d() {
        // cargo test --package step_perf --lib -- operator::static_reassemble::tests::static_reassemble_1d --exact --show-output
        type VT = u32;
        const READ_FROM_MU: bool = false;
        const DUMMY_ID: u32 = 0;
        let mut ctx = ProgramBuilder::default();

        let (in_data_snd0, in_data_rcv0) = ctx.unbounded();
        let (in_data_snd1, in_data_rcv1) = ctx.unbounded();
        let (in_data_snd2, in_data_rcv2) = ctx.unbounded();
        let (in_data_snd3, in_data_rcv3) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        use ndarray::ArcArray2;

        // Create tiles for input streams (inverse of parallelize_1d test)
        let tile1 = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        // Each [1, 1] lane closes both of its dimensions.
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile1, 2)].into_iter(),
            in_data_snd0,
        ));

        let tile2_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile2_clone, 2)].into_iter(),
            in_data_snd1,
        ));

        let tile3_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile3_clone, 2)].into_iter(),
            in_data_snd2,
        ));

        let tile4_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::ValStop(tile4_clone, 2)].into_iter(),
            in_data_snd3,
        ));

        ctx.add_child(StaticReassemble::<SimpleEvent, _>::new(
            vec![in_data_rcv0, in_data_rcv1, in_data_rcv2, in_data_rcv3],
            out_data_snd,
            1,
            FlatPartitionConfig {
                switch_cycles: vec![1; 4],
                write_back_mu: false,
            },
            DUMMY_ID,
        ));

        // Expected output: all tiles merged back with their ValStop markers
        let tile1_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (0..4).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile2_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile3_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        let tile4_exp = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );

        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::ValStop(tile1_exp, 1),
                    Elem::ValStop(tile2_exp, 1),
                    Elem::ValStop(tile3_exp, 1),
                    Elem::ValStop(tile4_exp, 2),
                ]
                .into_iter()
            },
            out_data_rcv,
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
    fn static_reassemble_inner_rank_restores_outer_stream() {
        let mut ctx = ProgramBuilder::default();
        let (lane0_sender, lane0_receiver) = ctx.unbounded();
        let (lane1_sender, lane1_receiver) = ctx.unbounded();
        let (output_sender, output_receiver) = ctx.unbounded();

        let lane0_values = (0..6).chain(12..18).collect();
        let lane1_values = (6..12).chain(18..24).collect();
        let lane0 = canonical_stream(&[2, 2, 3], lane0_values);
        let lane1 = canonical_stream(&[2, 2, 3], lane1_values);
        ctx.add_child(GeneratorContext::new(
            move || lane0.into_iter(),
            lane0_sender,
        ));
        ctx.add_child(GeneratorContext::new(
            move || lane1.into_iter(),
            lane1_sender,
        ));
        ctx.add_child(StaticReassemble::<SimpleEvent, _>::new(
            vec![lane0_receiver, lane1_receiver],
            output_sender,
            1,
            FlatPartitionConfig {
                switch_cycles: vec![1; 2],
                write_back_mu: false,
            },
            0,
        ));

        let output = canonical_stream(&[2, 4, 3], (0..24).collect());
        ctx.add_child(ApproxCheckerContext::new(
            move || output.into_iter(),
            output_receiver,
            tolerance_fn,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
