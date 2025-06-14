use core::panic;

use dam::{
    channel::{ChannelElement, Receiver},
    logging::LogEvent,
    structures::TimeManager,
    types::{DAMType, StaticallySized},
};
use ndarray::{ArcArray, Array, Dimension, IxDyn};

use crate::primitives::elem::StopType;
use crate::primitives::tile::Tile;
use crate::utils::events::LoggableEventSimple;
use thiserror::Error;

use super::{
    elem::{Bufferizable, Elem},
    tile,
};

#[derive(Error, Debug)]
pub enum BufferizeError<T> {
    #[error("Stream was empty at start of bufferization")]
    Finished,
    #[error("Stream terminated, but buffer was incomplete")]
    Incomplete,
    #[error("we see stop token larger than rank")]
    StopToken(Buffer<T>, StopType),
}

#[derive(Clone, Debug, Default)]
pub struct Buffer<T> {
    underlying: Option<ndarray::ArcArray<T, IxDyn>>,
    creation_time: u64,
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.underlying == other.underlying
        // creation_time is intentionally excluded
    }
}

impl<T: PartialEq> Buffer<T> {
    pub fn eq_with_time(&self, other: &Self) -> bool {
        self.underlying == other.underlying && self.creation_time == other.creation_time
    }
}

impl<T: Clone + Bufferizable> Buffer<T>
where
    Elem<T>: DAMType,
{
    pub fn new(arr: ndarray::ArcArray<T, IxDyn>, creation_time: u64) -> Self {
        Self {
            underlying: Some(arr),
            creation_time: creation_time,
        }
    }

    pub fn creation_time(&self) -> u64 {
        self.creation_time
    }

    pub fn from_stream<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
    >(
        stream: &Receiver<Elem<T>>,
        manager: &TimeManager,
        rank: usize,
        id: u32,
    ) -> Result<Self, BufferizeError<T>> {
        assert!(
            rank > 0,
            "Buffer::from_stream only operates on buffer of rank >= 1"
        );

        let mut creation_time = None;

        let mut buffer = vec![];
        let mut tracked_shape_info: Vec<bool> = vec![];

        // a vector consisting of how many elements have been seen since the last stop token of rank K
        let mut shape_info = vec![0];

        let mut stop_level = None;
        loop {
            match stream.dequeue(manager) {
                Ok(ChannelElement { time: _time, data }) => {
                    match data {
                        Elem::Val(value) => {
                            buffer.push(value);
                            if shape_info.len() == 1 {
                                shape_info[0] += 1;
                            }

                            if creation_time.is_none() {
                                // If it's the first element, set the creation time
                                creation_time = Some(manager.tick().time());
                            }
                            // As the compute node encodes the overhead to store data, we will not increment cycle here
                        }
                        Elem::ValStop(value, st) => {
                            if creation_time.is_none() {
                                // If it's the first element, set the creation time
                                creation_time = Some(manager.tick().time());
                            }

                            buffer.push(value);

                            let st_as_usize: usize = st.try_into().unwrap_or_else(|_| {
                                panic!("Error converting a stop token into a usize!")
                            });

                            if st_as_usize == rank {
                                shape_info[rank - 1] += 1;
                                break;
                            } else if st_as_usize > rank {
                                shape_info[rank - 1] += 1;
                                stop_level = Some(st_as_usize - rank);
                                break;
                            }

                            if shape_info.len() == st_as_usize {
                                shape_info[st_as_usize - 1] += 1;
                                shape_info.push(1);
                                tracked_shape_info.push(true);
                            } else if shape_info.len() > st_as_usize
                                && (tracked_shape_info.len() <= st_as_usize)
                            {
                                shape_info[st_as_usize] += 1;
                            }
                        }
                    }
                    manager.incr_cycles(1);
                }
                Err(_) if buffer.is_empty() => return Err(BufferizeError::Finished),
                Err(_) => return Err(BufferizeError::Incomplete),
            }
        }

        // At this point, we have a full "tensor"
        dam::logging::log_event(&E::new(
            "Bufferize".to_string(),
            id,
            creation_time.unwrap(),
            manager.tick().time(),
            false,
        ))
        .unwrap();

        // Our shape info is also backwards because we keep pushing.
        shape_info.reverse();

        let arc = ArcArray::from_shape_vec(shape_info, buffer)
            .expect("Unexpected mismatched shape when reading a stream into a buffer");

        match stop_level {
            Some(new_level) => Err(BufferizeError::StopToken(
                Buffer::new(arc, creation_time.unwrap()),
                new_level as StopType,
            )),
            None => Ok(Buffer::new(arc, creation_time.unwrap())),
        }
    }

    pub fn to_elem_iter<'a>(&'a self) -> impl Iterator<Item = Elem<T>> + 'a {
        let ndim = self.ndim();
        let mut previous_dim: Option<IxDyn> = None;
        let mut previous_data: Option<T> = None;
        self.indexed_iter()
            .enumerate()
            .flat_map(move |(i, (ind, val))| match &mut previous_dim {
                Some(prev) => {
                    let changed_index = outermost_diff_index(&ind, &prev);

                    let mut result = vec![];

                    // Enqueue the previous data with the proper stop token if necessary
                    if ndim - changed_index - 1 == 0 {
                        result.push(Elem::Val(previous_data.as_ref().unwrap().clone()));
                    } else {
                        result.push(Elem::ValStop(
                            previous_data.as_ref().unwrap().clone(),
                            (ndim - changed_index - 1) as StopType,
                        ));
                    }

                    let is_last = i == self.len() - 1;
                    if is_last {
                        // If it's the last element, enque because we don't have the next iteration to take care of this
                        result.push(Elem::ValStop(val.clone(), ndim as StopType));
                    } else {
                        previous_dim = Some(ind);
                        previous_data = Some(val.clone());
                    }
                    result
                }
                None => {
                    if self.underlying.as_ref().unwrap().len() == 1 {
                        // Single element buffer
                        vec![Elem::ValStop(val.clone(), 1)]
                    } else {
                        previous_dim = Some(ind);
                        previous_data = Some(val.clone());
                        vec![]
                    }
                }
            })
    }
}

impl<T> std::ops::Deref for Buffer<T> {
    type Target = ndarray::ArcArray<T, IxDyn>;

    fn deref(&self) -> &Self::Target {
        self.underlying
            .as_ref()
            .expect("Can't deref a null buffer!")
    }
}

impl<T> std::ops::DerefMut for Buffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.underlying
            .as_mut()
            .expect("Can't deref_mut a null buffer!")
    }
}

impl<T: StaticallySized> StaticallySized for Buffer<T> {
    const SIZE: usize = unimplemented!();
    // As the actual shape or size of a Buffer is not known in compile time,
    // we keep SIZE as unimplemented.
}

/// Calculates the first index where two dims differ.
fn outermost_diff_index(a: &IxDyn, b: &IxDyn) -> usize {
    a.as_array_view()
        .iter()
        .zip(b.as_array_view().iter())
        .enumerate()
        .find(|(_, (a_ind, b_ind))| a_ind != b_ind)
        .expect("The two inputs were identical!")
        .0
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
        primitives::{buffer, elem::Elem, tile::Tile},
        utils::events::{SimpleEvent, DUMMY_ID},
    };

    #[test]
    fn buffer_to_iter() {
        type VT = u32;
        let golden = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
        ];

        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = ArcArray::from_vec(tile_vec)
            .into_shape_with_order((2, 3))
            .unwrap();
        let tensor = Buffer::new(arr.into_dyn(), 0);
        let vec = tensor.to_elem_iter().collect::<Vec<_>>();
        assert_eq!(vec, golden);
    }

    #[test]
    fn buffer_to_iter_unit_dim() {
        type VT = u32;
        // 1 x 1 x 3
        let golden = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
        ];

        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = ArcArray::from_vec(tile_vec)
            .into_shape_with_order((1, 1, 3))
            .unwrap();
        let tensor = Buffer::new(arr.into_dyn(), 0);
        let vec = tensor.to_elem_iter().collect::<Vec<_>>();
        assert_eq!(vec, golden);
    }

    #[test]
    fn buffer_to_iter_3d() {
        type VT = u32;
        // 2 x 2 x 2
        let golden = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
        ];

        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = ArcArray::from_vec(tile_vec)
            .into_shape_with_order((2, 2, 2))
            .unwrap();
        let tensor = Buffer::new(arr.into_dyn(), 0);
        let vec = tensor.to_elem_iter().collect::<Vec<_>>();
        assert_eq!(vec, golden);
    }

    #[test]
    fn round_trip_test() {
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

        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = ArcArray::from_vec(tile_vec)
            .into_shape_with_order((2, 3))
            .unwrap();
        let tensor = Buffer::new(arr.into_dyn(), 0);
        let input_stream = tensor.to_elem_iter().collect::<Vec<_>>();

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(|| input_stream.into_iter(), snd));

        let mut output_check = FunctionContext::new();
        rcv.attach_receiver(&output_check);
        output_check.set_run(move |time| {
            let buffer = Buffer::from_stream::<SimpleEvent>(&rcv, time, 2, DUMMY_ID).unwrap();
            assert_eq!(buffer, tensor);
        });
        ctx.add_child(output_check);

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn round_trip_test_3d() {
        type VT = u32;

        let mut ctx = ProgramBuilder::default();

        // 2 x 2 x 3
        let tile_vec = vec![
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
            Tile::<VT>::new_blank(vec![2, 2], 2, false),
        ];

        let arr = ArcArray::from_vec(tile_vec)
            .into_shape_with_order((2, 2, 3))
            .unwrap();
        let tensor = Buffer::new(arr.into_dyn(), 1);
        let input_stream = tensor.to_elem_iter().collect::<Vec<_>>();

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(|| input_stream.into_iter(), snd));

        let mut output_check = FunctionContext::new();
        rcv.attach_receiver(&output_check);
        output_check.set_run(move |time| {
            let buffer = Buffer::from_stream::<SimpleEvent>(&rcv, time, 3, DUMMY_ID).unwrap();
            assert_eq!(buffer, tensor);
            assert!(buffer.eq_with_time(&tensor));
        });
        ctx.add_child(output_check);

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
