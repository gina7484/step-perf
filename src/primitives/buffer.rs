use core::panic;

use dam::{
    channel::{ChannelElement, Receiver},
    logging::LogEvent,
    structures::TimeManager,
    types::DAMType,
};

use crate::memory::{data::Tile, events::LoggableEventSimple};
use thiserror::Error;

use super::elem::Elem;

#[derive(Error, Debug)]
pub enum BufferizeError {
    #[error("Stream was empty at start of bufferization")]
    Finished,
    #[error("Stream terminated, but buffer was incomplete")]
    Incomplete,
    #[error("we see stop token larger than rank")]
    StopToken(usize),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Buffer {
    buffer_shape: Vec<usize>,
    tile_shape: Tile,
}

impl DAMType for Buffer {
    fn dam_size(&self) -> usize {
        let buffer_size: usize = self.buffer_shape.iter().product();
        self.tile_shape.size_in_bytes() * buffer_size
    }
}

impl Buffer {
    pub fn from_stream<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
    >(
        stream: &Receiver<Elem<Tile>>,
        manager: &TimeManager,
        rank: usize,
    ) -> Result<Self, BufferizeError> {
        assert!(
            rank > 0,
            "Buffer::from_stream only operates on buffer of rank >= 1"
        );

        let mut tile_shape: Option<Tile> = None;
        let mut start_time: u64 = 0;
        let mut end_time: u64 = 0;

        let mut buffer = vec![];
        let mut tracked_shape_info: Vec<bool> = vec![];

        // a vector consisting of how many elements have been seen since the last stop token of rank K
        let mut shape_info = vec![0];
        loop {
            match stream.dequeue(manager) {
                Ok(ChannelElement { time: _time, data }) => match data {
                    Elem::Val(value) => {
                        if start_time == 0 {
                            start_time = manager.tick().time();
                            tile_shape = Some(value.clone());
                        } else {
                            if tile_shape.clone().unwrap() != value {
                                panic!("The tiles in a stream should all have the same spec");
                            }
                        }
                        end_time = manager.tick().time();

                        buffer.push(value);
                        if shape_info.len() == 1 {
                            shape_info[0] += 1;
                        }
                    }
                    Elem::Stop(st) => {
                        let st_as_usize: usize = st.try_into().unwrap_or_else(|_| {
                            panic!("Error converting a stop token into a usize!")
                        });

                        if st_as_usize == rank {
                            // log start_time ~ end_time
                            dam::logging::log_event(&E::new(start_time, end_time, false)).unwrap();
                            break;
                        } else if st_as_usize > rank {
                            return Err(BufferizeError::StopToken(st_as_usize));
                        }

                        // log stop token
                        dam::logging::log_event(&E::new(
                            manager.tick().time(),
                            manager.tick().time() + 1,
                            true,
                        ))
                        .unwrap();

                        if shape_info.len() == st_as_usize {
                            shape_info.push(1);
                            tracked_shape_info.push(true);
                        } else if shape_info.len() > st_as_usize
                            && (tracked_shape_info.len() <= st_as_usize)
                        {
                            shape_info[st_as_usize] += 1;
                        }
                    }
                },
                Err(_) if buffer.is_empty() => return Err(BufferizeError::Finished),
                Err(_) => return Err(BufferizeError::Incomplete),
            }
        }

        // At this point, we have a full "tensor"

        // Our shape info is also backwards because we keep pushing.
        shape_info.reverse();

        // println!("{:?}", shape_info);
        // println!("{:?}", buffer);

        Ok(Buffer {
            buffer_shape: shape_info,
            tile_shape: tile_shape.unwrap(),
        })
    }
}
