use std::marker::PhantomData;
use std::thread::panicking;

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

/// `bufferized_rank`: Rank of the buffers in in_stream <br/><br/>
/// `repeat_rank`: The last (largest) rank that is expanded. <br/><br/>
///
/// `in_stream`: The last (`repeat_rank`+1) ranks' shape should all be 1. (i.e., rank 0 ~ rank `repeat_rank`)<br/>
///
/// `ref_stream`: The ranks higher than `repeat_rank` should be the same as `in_stream` (i.e., rank `repeat_rank+1`~) <br/>
///
/// `out_stream`: Same shape as the `ref_stream`
#[context_macro]
pub struct DynStreamify<E: LoggableEventSimple, T: Bufferizable + Clone, R: Clone> {
    pub in_stream: Receiver<Elem<Buffer<T>>>,
    pub bufferized_rank: StopType,
    pub repeat_rank: StopType,
    pub ref_stream: Receiver<Elem<R>>,
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
        in_stream: Receiver<Elem<Buffer<T>>>,
        bufferized_rank: StopType,
        repeat_rank: StopType,
        ref_stream: Receiver<Elem<R>>,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            bufferized_rank,
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
                            // This is when the input stream is a rank 0 stream (stream shape = [1])
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
                                            panic!("Unexpected stop token in reference stream: S({}). \
                                            Expected only value elements.", ref_stop_lev);
                                        }
                                    },
                                    Err(_) => return,
                                }
                            }
                        }
                        Elem::ValStop(buff, outer_stop_lev) => {
                            loop {
                                match self.ref_stream.dequeue(&self.time) {
                                    Ok(ChannelElement {
                                        time: _,
                                        data: ref_elem,
                                    }) => {
                                        match ref_elem {
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
                                                                == self.bufferized_rank
                                                            {
                                                                // last tile in the buffer
                                                                tile_stop_lev + ref_stop_lev
                                                            } else if tile_stop_lev
                                                                < self.bufferized_rank
                                                            {
                                                                tile_stop_lev
                                                            } else {
                                                                panic!("Larger stop token than the buffer's rank! \
                                                                Buffer rank: {}, Tile stop level: {}", self.bufferized_rank, tile_stop_lev);
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
                                                if ref_stop_lev >= self.repeat_rank + 1 {
                                                    // If the stop level is greater than or equal to the repeat rank,
                                                    // we move on to the next buffer
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    self.in_stream.dequeue(&self.time).unwrap();
                    dam::logging::log_event(&E::new(
                        "DynStreamify".to_string(),
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
            ApproxCheckerContext, CheckerContext, FunctionContext, GeneratorContext, PrinterContext,
        },
    };
    use ndarray::{ArcArray, IxDyn};

    use super::Buffer;
    use crate::{
        operator::bufferize::Bufferize,
        primitives::{
            elem::{Elem, StopType},
            tile::Tile,
        },
        utils::events::{SimpleEvent, DUMMY_ID},
    };

    #[test]
    fn round_trip_test_repeat_rank_0() {
        // [1] (in) buffer_shape = [2]
        // [4] (ref)
        // repeat_rank = 0
        // output = [4,2]
        type VT = u32;

        const REPEAT_RANK_PER_BUFFER: StopType = 0;
        const BUFFER_RANK: StopType = 1;

        let mut ctx = ProgramBuilder::default();
        let (snd, rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = false;
        const DUMMY_CREATION_TIME: u64 = 0;
        let tile_vec = vec![Tile::<VT>::new_blank(vec![2, 2], BYTES_PER_ELEM, READ_FROM_MU); 2];

        // =============== Input [1] ================
        // Create Buffer (each are a buffer of 2x2 tiles)
        let arr = Arc::new(
            ArcArray::from_vec(tile_vec)
                .into_shape_with_order((2,))
                .unwrap(),
        );
        let arr_clone = arr.clone();
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![Elem::Val(Buffer::new(
                    (*arr_clone).clone().into_dyn(),
                    DUMMY_CREATION_TIME,
                ))]
                .into_iter()
            },
            snd,
        ));

        // =============== Ref Stream [4] ================
        ctx.add_child(GeneratorContext::new(
            move || vec![Elem::Val(0), Elem::Val(0), Elem::Val(0), Elem::Val(0)].into_iter(),
            ref_snd,
        ));

        ctx.add_child(super::DynStreamify::<SimpleEvent, _, _>::new(
            rcv,
            BUFFER_RANK,
            REPEAT_RANK_PER_BUFFER,
            ref_rcv,
            out_snd,
            DUMMY_ID,
        ));

        // =============== Output Stream [4,2] ================
        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                2
            ])
            .into_shape_with_order((2,))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .chain(
                        Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                            .to_elem_iter()
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
                    .chain(
                        Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                            .to_elem_iter()
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
                    .chain(
                        Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                            .to_elem_iter()
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn round_trip_test_repeat_rank_1() {
        // [2,3,1,1] (in) buffer_shape = [2,2]
        // [2,3,2,4] (ref)
        // repeat_rank = 1
        // output = [2,3,2,4,2,2]
        type VT = u32;

        const REPEAT_RANK_PER_BUFFER: StopType = 1;
        const BUFFER_RANK: StopType = 2;

        let mut ctx = ProgramBuilder::default();
        let (snd, rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;
        let tile_vec = vec![Tile::<VT>::new_blank(vec![2, 2], BYTES_PER_ELEM, READ_FROM_MU); 2 * 2];

        // =============== Input [2,3,1,1] ================
        // Create 2x2 Buffers (each are a buffer of 2x2 tiles)
        let arr = Arc::new(
            ArcArray::from_vec(tile_vec)
                .into_shape_with_order((2, 2))
                .unwrap(),
        );
        let arr_clone = arr.clone();
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::ValStop(
                        Buffer::new((*arr_clone).clone().into_dyn(), DUMMY_CREATION_TIME),
                        2,
                    ),
                    Elem::ValStop(
                        Buffer::new((*arr_clone).clone().into_dyn(), DUMMY_CREATION_TIME),
                        2,
                    ),
                    Elem::ValStop(
                        Buffer::new((*arr_clone).clone().into_dyn(), DUMMY_CREATION_TIME),
                        3,
                    ),
                    Elem::ValStop(
                        Buffer::new((*arr_clone).clone().into_dyn(), DUMMY_CREATION_TIME),
                        2,
                    ),
                    Elem::ValStop(
                        Buffer::new((*arr_clone).clone().into_dyn(), DUMMY_CREATION_TIME),
                        2,
                    ),
                    Elem::ValStop(
                        Buffer::new((*arr_clone).clone().into_dyn(), DUMMY_CREATION_TIME),
                        3,
                    ),
                ]
                .into_iter()
            },
            snd,
        ));

        // =============== Ref Stream [2,3,2,4] ================
        let ref_arr = Arc::new(
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
                Buffer::new((*ref_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .chain(
                        Buffer::new((*ref_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                            .to_elem_iter()
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
            },
            ref_snd,
        ));

        ctx.add_child(super::DynStreamify::<SimpleEvent, _, _>::new(
            rcv,
            BUFFER_RANK,
            REPEAT_RANK_PER_BUFFER,
            ref_rcv,
            out_snd,
            DUMMY_ID,
        ));

        // =============== Output Stream [2,3,2,4,2,2] ================
        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    vec![2, 2],
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 2 * 4 * 2 * 2
            ])
            .into_shape_with_order((3, 2, 4, 2, 2))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .chain(
                        Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                            .to_elem_iter()
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
            },
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
