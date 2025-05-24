use std::marker::PhantomData;

use dam::{context_tools::*, logging::LogEvent};

use crate::{
    memory::events::LoggableEventSimple,
    primitives::{
        buffer::{Buffer, BufferizeError},
        elem::Elem,
        tile::Tile,
    },
};

#[context_macro]
pub struct Bufferize<E, T> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Buffer>>,
    rank: usize,
    _phantom: PhantomData<E>,
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Bufferize<E, T> {
    pub fn new(
        in_stream: Receiver<Elem<Tile>>,
        out_stream: Sender<Elem<Buffer>>,
        rank: usize,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            rank,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Context
    for Bufferize<E>
{
    fn run(&mut self) {
        loop {
            match Buffer::from_stream::<E>(&self.in_stream, &self.time, self.rank) {
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
                Err(BufferizeError::StopToken(x)) => {
                    // Handle the following stop tokens
                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: Elem::Stop((x - self.rank).try_into().unwrap_or_else(|_| {
                                    panic!("Error converting usize back into StopType!")
                                })),
                            },
                        )
                        .unwrap();
                }
                Err(BufferizeError::Finished) => return,
                Err(err @ BufferizeError::Incomplete) => panic!("{:?}", err),
            }
        }
    }
}
