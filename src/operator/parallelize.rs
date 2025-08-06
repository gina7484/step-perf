use crate::memory::PMU_BW;
use crate::operator::partition::FlatPartitionConfig;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::{select::SelectAdapter, tile::Tile};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;
use std::panic;

#[context_macro]
pub struct Parallelize<E, A: DAMType> {
    in_stream: Receiver<Elem<A>>,
    out_streams: Vec<Sender<Elem<A>>>,
    partition_rank: StopType,
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
        partition_rank: StopType,
        config: FlatPartitionConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_streams,
            partition_rank,
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
        loop {
            for i in 0..par_factor {
                loop {
                    match self.in_stream.dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: val_data,
                        }) => match val_data {
                            Elem::Val(x) => {
                                self.out_streams[i]
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick() + self.config.switch_cycles[i],
                                            data: Elem::Val(x),
                                        },
                                    )
                                    .unwrap();
                                if self.partition_rank == 0 {
                                    break;
                                }
                            }
                            Elem::ValStop(x, stop_lev) => {
                                self.out_streams[i]
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick() + self.config.switch_cycles[i],
                                            data: Elem::ValStop(x, stop_lev),
                                        },
                                    )
                                    .unwrap();
                                if stop_lev == self.partition_rank {
                                    break;
                                } else if stop_lev > self.partition_rank {
                                    panic!("Stop level is greater than partition rank");
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
    use super::Parallelize;
    use crate::primitives::select::MultiHotN;
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
                    Elem::Val(tile4),
                ]
                .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(Parallelize::<SimpleEvent, _>::new(
            in_data_rcv,
            vec![out_data_snd0, out_data_snd1, out_data_snd2, out_data_snd3],
            0,
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
            move || vec![Elem::Val(tile1)].into_iter(),
            out_data_rcv0,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::Val(tile2)].into_iter(),
            out_data_rcv1,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::Val(tile3)].into_iter(),
            out_data_rcv2,
            tolerance_fn,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || vec![Elem::Val(tile4)].into_iter(),
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
                    Elem::ValStop(tile4, 1),
                ]
                .into_iter()
            },
            in_data_snd,
        ));

        ctx.add_child(Parallelize::<SimpleEvent, _>::new(
            in_data_rcv,
            vec![out_data_snd0, out_data_snd1, out_data_snd2, out_data_snd3],
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
}
