use std::marker::PhantomData;

use dam::{context_tools::*, logging::LogEvent};

use crate::{
    primitives::{
        buffer::{Buffer, BufferizeError},
        elem::{Bufferizable, Elem, StopType},
        tile::Tile,
    },
    utils::events::LoggableEventSimple,
};

#[context_macro]
pub struct Bufferize<E, T: Clone> {
    in_stream: Receiver<Elem<T>>,
    out_stream: Sender<Elem<Buffer<T>>>,
    rank: StopType,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Clone + Bufferizable,
    > Bufferize<E, T>
where
    Elem<T>: DAMType,
    Buffer<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<T>>,
        out_stream: Sender<Elem<Buffer<T>>>,
        rank: StopType,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            rank,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Clone + Bufferizable,
    > Context for Bufferize<E, T>
where
    Elem<T>: DAMType,
    Buffer<T>: DAMType,
{
    fn run(&mut self) {
        loop {
            match Buffer::<T>::from_stream::<E>(
                &self.in_stream,
                &self.time,
                self.rank as usize,
                self.id,
            ) {
                Ok(buffer) => {
                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: Elem::Val(buffer),
                            },
                        )
                        .unwrap();
                }
                Err(BufferizeError::StopToken(buffer, stop_level)) => {
                    // Handle the following stop tokens
                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: Elem::ValStop(buffer, stop_level),
                            },
                        )
                        .unwrap();
                }
                Err(BufferizeError::Finished) => return,
                Err(BufferizeError::Incomplete) => {
                    panic!("Stream terminated, but buffer was incomplete")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{CheckerContext, FunctionContext, GeneratorContext},
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
        type VT = u32;

        let mut ctx = ProgramBuilder::default();

        let golden = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
        ];

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(|| golden.into_iter(), snd));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(Bufferize::<SimpleEvent, _>::new(rcv, out_snd, 1, DUMMY_ID));

        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = Arc::new(
            ArcArray::from_vec(tile_vec)
                .into_shape_with_order(3)
                .unwrap(),
        );
        let arr_clone = arr.clone();
        ctx.add_child(CheckerContext::new(
            move || {
                vec![
                    Elem::Val(Buffer::new((*arr_clone).clone().into_dyn(), 0)),
                    Elem::ValStop(Buffer::new((*arr_clone).clone().into_dyn(), 0), 1),
                ]
                .into_iter()
            },
            out_rcv,
        ));

        // let mut output_check = FunctionContext::new();
        // rcv.attach_receiver(&output_check);
        // output_check.set_run(move |time| {
        //     let buffer = Buffer::from_stream::<SimpleEvent>(&rcv, time, 3).unwrap();
        //     assert_eq!(buffer.shape(), tensor.shape());
        //     assert_eq!(buffer, tensor);
        // });
        // ctx.add_child(output_check);

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
