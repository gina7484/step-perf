use crate::operator::partition::FlatPartitionConfig;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use crate::trace::TracingSender as Sender;
use std::marker::PhantomData;

#[context_macro]
pub struct StaticReassemble<E, A: DAMType> {
    in_streams: Vec<Receiver<Elem<A>>>,
    out_stream: Sender<Elem<A>>,
    merge_rank: StopType,
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
        merge_rank: StopType,
        config: FlatPartitionConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_streams,
            out_stream,
            merge_rank,
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
        loop {
            for i in 0..par_factor {
                loop {
                    match self.in_streams[i].dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: val_data,
                        }) => match val_data {
                            Elem::Val(x) => {
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick() + self.config.switch_cycles[i],
                                            data: Elem::Val(x),
                                        },
                                    )
                                    .unwrap();
                                if self.merge_rank == 0 {
                                    break;
                                }
                            }
                            Elem::ValStop(x, stop_lev) => {
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick() + self.config.switch_cycles[i],
                                            data: Elem::ValStop(x, stop_lev),
                                        },
                                    )
                                    .unwrap();
                                if stop_lev == self.merge_rank {
                                    break;
                                } else if stop_lev > self.merge_rank {
                                    panic!("Stop level is greater than merge rank");
                                }
                            }
                        },
                        Err(_) => return,
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

        // Each input stream has one tile (inverse of parallelize_0d)
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(tile1)].into_iter(),
            in_data_snd0,
        ));

        let tile2_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (4..8).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(tile2_clone)].into_iter(),
            in_data_snd1,
        ));

        let tile3_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (8..12).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(tile3_clone)].into_iter(),
            in_data_snd2,
        ));

        let tile4_clone = Tile::<VT>::new(
            ArcArray2::from_shape_vec((2, 2), (12..16).collect()).unwrap(),
            2,
            READ_FROM_MU,
        );
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(tile4_clone)].into_iter(),
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
                    Elem::Val(tile4_exp),
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

        // Each input stream has one tile with ValStop at rank 1
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
}
