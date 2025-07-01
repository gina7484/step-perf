use crate::primitives::elem::{Elem, StopType};
use dam::context_tools::*;

#[context_macro]
pub struct Flatten<T: DAMType> {
    in_stream: Receiver<Elem<T>>,
    out_stream: Sender<Elem<T>>,
    min_rank: StopType,
    max_rank: StopType,
}

impl<T: DAMType> Flatten<T>
where
    Self: Context,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        out_stream: Sender<Elem<T>>,
        min_rank: StopType,
        max_rank: StopType,
    ) -> Self {
        assert!(min_rank < max_rank, "min_rank must be less than max_rank");
        let ctx = Self {
            in_stream,
            out_stream,
            min_rank,
            max_rank,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType> Context for Flatten<T> {
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: Elem::Val(x.clone()),
                                },
                            )
                            .unwrap();
                    }
                    Elem::ValStop(x, s) => {
                        let new_rank = if s <= self.min_rank {
                            s
                        } else if self.min_rank < s && s <= self.max_rank {
                            self.min_rank
                        } else {
                            s - (self.max_rank - self.min_rank)
                        };

                        let output_date = if new_rank == 0 {
                            Elem::Val(x.clone())
                        } else {
                            Elem::ValStop(x.clone(), new_rank)
                        };
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: output_date,
                                },
                            )
                            .unwrap();
                    }
                },
                Err(_) => return,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, GeneratorContext, PrinterContext,
        },
    };
    use ndarray::ArcArray;

    use crate::primitives::{buffer::Buffer, elem::Elem, tile::Tile};

    use super::Flatten;

    #[test]
    fn flatten_0_1() {
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 4
            ])
            .into_shape_with_order((3, 2, 4))
            .unwrap(),
        );
        ctx.add_child(GeneratorContext::new(
            move || {
                Buffer::new((*in_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(Flatten::new(in_rcv, out_snd, 0, 1));

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 4
            ])
            .into_shape_with_order((3, 8))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatten_0_1_size1() {
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 1
            ])
            .into_shape_with_order((3, 2, 1))
            .unwrap(),
        );
        ctx.add_child(GeneratorContext::new(
            move || {
                Buffer::new((*in_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(Flatten::new(in_rcv, out_snd, 0, 1));

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 1
            ])
            .into_shape_with_order((3, 2 * 1))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn flatten_1_3() {
        // (5, 2, 3, 2, 4) => (5, 2 * 3 * 2, 4)
        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                5 * 2 * 3 * 2 * 4
            ])
            .into_shape_with_order((5, 2, 3, 2, 4))
            .unwrap(),
        );
        ctx.add_child(GeneratorContext::new(
            move || {
                Buffer::new((*in_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            in_snd,
        ));

        ctx.add_child(Flatten::new(in_rcv, out_snd, 1, 3));

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                5 * 2 * 3 * 2 * 4
            ])
            .into_shape_with_order((5, 2 * 3 * 2, 4))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
