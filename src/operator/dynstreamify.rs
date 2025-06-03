use std::marker::PhantomData;

use dam::logging::LogEvent;
use dam::{context_tools::*, types::StaticallySized};
use ndarray::{IntoDimension, Ix2, IxDyn, IxDynImpl};

use crate::primitives::buffer::Buffer;
use crate::primitives::elem::Bufferizable;
use crate::{
    primitives::elem::{Elem, StopType},
    ramulator::access::MemoryData,
};

use crate::utils::events::LoggableEventSimple;

use crate::primitives::tile::Tile;

/// `rank`: rank of the buffers <br/>
/// `repeat_rank`: the rank of the substream in ref_stream that corresponds to a single buffer.
///                If you are only broadcasting a single rank, this is 1. If you want to add N
///                more dimensions through broadcasting each buffer, this is N.
#[context_macro]
pub struct DynStreamify<E: LoggableEventSimple, T: Bufferizable + Clone, R: Clone> {
    pub rank: StopType,
    pub repeat_rank: StopType,
    pub ref_stream: Receiver<Elem<R>>, // rank = repeat_rank + in_stream's rank
    pub in_stream: Receiver<Elem<Buffer<T>>>,
    pub out_stream: Sender<Elem<T>>,
    pub id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
        R: DAMType,
    > DynStreamify<E, T, R>
where
    Buffer<T>: DAMType,
{
    pub fn new(
        rank: StopType,
        repeat_rank: StopType,
        ref_stream: Receiver<Elem<R>>,
        in_stream: Receiver<Elem<Buffer<T>>>,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            rank,
            repeat_rank,
            ref_stream,
            in_stream,
            out_stream,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.ref_stream.attach_receiver(&ctx);
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
        R: DAMType,
    > Context for DynStreamify<E, T, R>
where
    Buffer<T>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: buff_elem,
                }) => {
                    let start_time = self.time.tick().time();
                    match buff_elem {
                        Elem::Val(buff) => {
                            loop {
                                match self.ref_stream.dequeue(&self.time) {
                                    Ok(ChannelElement {
                                        time: _,
                                        data: ref_elem,
                                    }) => match ref_elem {
                                        Elem::Val(_) => {
                                            let buff_clone = buff.clone();
                                            for elem in buff_clone.to_elem_iter() {
                                                self.out_stream
                                                    .enqueue(
                                                        &self.time,
                                                        ChannelElement {
                                                            time: self.time.tick(),
                                                            data: elem,
                                                        },
                                                    )
                                                    .unwrap();

                                                self.time.incr_cycles(1);
                                            }
                                        }
                                        Elem::ValStop(_, ref_stop_lev) => {
                                            let buff_clone = buff.clone();
                                            for elem in buff_clone.to_elem_iter() {
                                                match elem {
                                                    Elem::Val(tile) => {
                                                        self.out_stream
                                                            .enqueue(
                                                                &self.time,
                                                                ChannelElement {
                                                                    time: self.time.tick(),
                                                                    data: Elem::Val(tile),
                                                                },
                                                            )
                                                            .unwrap();

                                                        self.time.incr_cycles(1);
                                                    }
                                                    Elem::ValStop(tile, tile_stop_lev) => {
                                                        let new_stop_level =
                                                            if tile_stop_lev == self.rank {
                                                                tile_stop_lev + ref_stop_lev
                                                            } else {
                                                                tile_stop_lev
                                                            };
                                                        self.out_stream
                                                            .enqueue(
                                                                &self.time,
                                                                ChannelElement {
                                                                    time: self.time.tick(),
                                                                    data: Elem::ValStop(
                                                                        tile,
                                                                        new_stop_level,
                                                                    ),
                                                                },
                                                            )
                                                            .unwrap();

                                                        self.time.incr_cycles(1);
                                                    }
                                                }
                                            }
                                            if ref_stop_lev >= self.repeat_rank {
                                                // If the stop level is greater than or equal to the repeat rank,
                                                // we move on to the next buffer
                                                break;
                                            }
                                        }
                                    },
                                    Err(_) => {
                                        break;
                                    }
                                }
                            }
                        }
                        Elem::ValStop(buff, outer_stop_lev) => {
                            loop {
                                match self.ref_stream.dequeue(&self.time) {
                                    Ok(ChannelElement {
                                        time: _,
                                        data: ref_elem,
                                    }) => match ref_elem {
                                        Elem::Val(_) => {
                                            let buff_clone = buff.clone();
                                            for elem in buff_clone.to_elem_iter() {
                                                self.out_stream
                                                    .enqueue(
                                                        &self.time,
                                                        ChannelElement {
                                                            time: self.time.tick(),
                                                            data: elem,
                                                        },
                                                    )
                                                    .unwrap();

                                                self.time.incr_cycles(1);
                                            }
                                        }
                                        Elem::ValStop(_, ref_stop_lev) => {
                                            let buff_clone = buff.clone();
                                            for elem in buff_clone.to_elem_iter() {
                                                match elem {
                                                    Elem::Val(tile) => {
                                                        self.out_stream
                                                            .enqueue(
                                                                &self.time,
                                                                ChannelElement {
                                                                    time: self.time.tick(),
                                                                    data: Elem::Val(tile),
                                                                },
                                                            )
                                                            .unwrap();

                                                        self.time.incr_cycles(1);
                                                    }
                                                    Elem::ValStop(tile, tile_stop_lev) => {
                                                        let new_stop_level = if tile_stop_lev
                                                            == self.rank
                                                        {
                                                            if ref_stop_lev >= self.repeat_rank {
                                                                tile_stop_lev
                                                                    + ref_stop_lev
                                                                    + outer_stop_lev
                                                            } else {
                                                                tile_stop_lev + ref_stop_lev
                                                            }
                                                        } else {
                                                            tile_stop_lev
                                                        };
                                                        self.out_stream
                                                            .enqueue(
                                                                &self.time,
                                                                ChannelElement {
                                                                    time: self.time.tick(),
                                                                    data: Elem::ValStop(
                                                                        tile,
                                                                        new_stop_level,
                                                                    ),
                                                                },
                                                            )
                                                            .unwrap();

                                                        self.time.incr_cycles(1);
                                                    }
                                                }
                                            }
                                            if ref_stop_lev >= self.repeat_rank {
                                                // If the stop level is greater than or equal to the repeat rank,
                                                // we move on to the next buffer
                                                break;
                                            }
                                        }
                                    },
                                    Err(_) => {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    self.in_stream.dequeue(&self.time).unwrap();
                    dam::logging::log_event(&E::new(
                        self.id,
                        start_time,
                        self.time.tick().time(),
                        false,
                    ))
                    .unwrap();
                }
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, FunctionContext, GeneratorContext,
        },
    };
    use ndarray::{ArcArray, IxDyn};

    use super::Buffer;
    use crate::{
        operator::bufferize::Bufferize,
        primitives::{elem::Elem, tile::Tile},
        utils::events::{SimpleEvent, DUMMY_ID},
    };

    #[test]
    fn round_trip_test_3d() {
        // [2,|2,2] (in) -> [1,2,3,2,2]
        // [1, 2,3]   (ref)
        type VT = u32;
        type RT = u32;

        let mut ctx = ProgramBuilder::default();
        let (snd, rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = Arc::new(
            ArcArray::from_vec(tile_vec)
                .into_shape_with_order((2, 2))
                .unwrap(),
        );
        let arr_clone = arr.clone();
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::Val(Buffer::new((*arr_clone).clone().into_dyn(), 0)),
                    Elem::Val(Buffer::new((*arr_clone).clone().into_dyn(), 0)),
                ]
                .into_iter()
            },
            snd,
        ));

        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::Val(0),
                    Elem::Val(1),
                    Elem::ValStop(2, 1),
                    Elem::Val(0),
                    Elem::Val(1),
                    Elem::ValStop(2, 2),
                ]
                .into_iter()
            },
            ref_snd,
        ));

        ctx.add_child(super::DynStreamify::<SimpleEvent, _, _>::new(
            2, 1, ref_rcv, rcv, out_snd, DUMMY_ID,
        ));

        ctx.add_child(ApproxCheckerContext::new(
            move || {
                vec![
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
                    Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 4),
                ]
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
