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
/// `repeat_rank`: The rank of the substream in ref_stream that corresponds to a single buffer.
///                If you are only broadcasting a single rank, this is 1. If you want to add N
///                more dimensions through broadcasting each buffer, this is N.  <br/><br/>
///
/// `in_stream`: `[D1,D2,...,DN]` <br/>
///
/// `ref_stream`:<br/>
///     `[D1,...,D(repeat_rank), D(repeat_rank+1), ..., D(repeat_rank+N)]` <br/>
///     `<-----repeated------>`  <br/><br/>
///
/// `out_stream`: <br/>
///     `[D1,..,D(buf_rank), D(buf_rank+1),..,D(buf_rank + rep_rank), .., D(buf_rank + rep_rank + N)]` <br/>
///     `<---- buffer ---->`  `<----repeated based on ref_stream----->`  `<-------in_stream shape------>`
#[context_macro]
pub struct DynStreamify<E: LoggableEventSimple, T: Bufferizable + Clone, R: Clone> {
    pub in_stream: Receiver<Elem<Buffer<T>>>,
    pub bufferized_rank: StopType, // rank of the buffers in in_stream
    pub repeat_rank: StopType, // rank of the substream in ref_stream that corresponds to a single buffer
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
                                                                // Add the stop level of the reference stream
                                                                // if it's the last tile in the buffer
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
                                                if ref_stop_lev == self.repeat_rank {
                                                    // If the stop level is greater than or equal to the repeat rank,
                                                    // we move on to the next buffer
                                                    break;
                                                } else if ref_stop_lev > self.repeat_rank {
                                                    panic!("Unexpected stop level in reference stream: {}", ref_stop_lev);
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
                                                            == self.bufferized_rank
                                                        {
                                                            if ref_stop_lev < self.repeat_rank {
                                                                tile_stop_lev + ref_stop_lev
                                                            } else if ref_stop_lev
                                                                == self.repeat_rank + outer_stop_lev
                                                            {
                                                                tile_stop_lev
                                                                    + ref_stop_lev
                                                                    + outer_stop_lev
                                                            } else {
                                                                panic!(
                                                                    "Unexpected stop level in reference stream: {} (Expected: {})",
                                                                    ref_stop_lev,
                                                                    self.repeat_rank + outer_stop_lev
                                                                );
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
        primitives::{
            elem::{Elem, StopType},
            tile::Tile,
        },
        utils::events::{SimpleEvent, DUMMY_ID},
    };

    #[test]
    fn round_trip_test_3d() {
        // [2,|2,2] (in)
        // [2,3] (ref)
        // repeat_rank = 1
        // output = [2,3,2,2]
        type VT = u32;

        const REPEAT_RANK_PER_BUFFER: StopType = 1;
        const BUFFER_RANK: StopType = 2;

        let mut ctx = ProgramBuilder::default();
        let (snd, rcv) = ctx.unbounded();
        let (ref_snd, ref_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = false;
        const DUMMY_CREATION_TIME: u64 = 0;
        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], BYTES_PER_ELEM, READ_FROM_MU),
            Tile::<VT>::new_blank(vec![2, 2], BYTES_PER_ELEM, READ_FROM_MU),
            Tile::<VT>::new_blank(vec![2, 2], BYTES_PER_ELEM, READ_FROM_MU),
            Tile::<VT>::new_blank(vec![2, 2], BYTES_PER_ELEM, READ_FROM_MU),
        ];

        // =============== Input [2,|2,2] ================
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
                    Elem::Val(Buffer::new(
                        (*arr_clone).clone().into_dyn(),
                        DUMMY_CREATION_TIME,
                    )),
                    Elem::Val(Buffer::new(
                        (*arr_clone).clone().into_dyn(),
                        DUMMY_CREATION_TIME,
                    )),
                ]
                .into_iter()
            },
            snd,
        ));

        // =============== Ref Stream [2,3] ================
        ctx.add_child(GeneratorContext::new(
            move || {
                vec![
                    Elem::Val(0),
                    Elem::Val(1),
                    Elem::ValStop(2, 1),
                    Elem::Val(0),
                    Elem::Val(1),
                    Elem::ValStop(2, 1),
                ]
                .into_iter()
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

        // =============== Output Stream [2,3,2,2] ================
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
                    Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
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
