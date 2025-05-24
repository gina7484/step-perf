use core::panic;

use dam::{
    channel::{ChannelElement, Receiver},
    logging::LogEvent,
    structures::TimeManager,
    types::{DAMType, StaticallySized},
};
use ndarray::{ArcArray, Array, Dimension, IxDyn};

use crate::primitives::tile::Tile;
use crate::{memory::events::LoggableEventSimple, primitives::elem::StopType};
use thiserror::Error;

use super::{
    elem::{Bufferizable, Elem},
    tile,
};

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
pub struct Buffer<T> {
    underlying: Option<ndarray::ArcArray<T, IxDyn>>,
    creation_time: u64,
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

    pub fn from_stream<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
    >(
        stream: &Receiver<Elem<T>>,
        manager: &TimeManager,
        rank: usize,
    ) -> Result<Self, BufferizeError> {
        assert!(
            rank > 0,
            "Buffer::from_stream only operates on buffer of rank >= 1"
        );

        let mut creation_time = None;

        let mut buffer = vec![];
        let mut tracked_shape_info: Vec<bool> = vec![];

        // a vector consisting of how many elements have been seen since the last stop token of rank K
        let mut shape_info = vec![0];
        loop {
            match stream.dequeue(manager) {
                Ok(ChannelElement { time: _time, data }) => match data {
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
                        let st_as_usize: usize = st.try_into().unwrap_or_else(|_| {
                            panic!("Error converting a stop token into a usize!")
                        });

                        if st_as_usize == rank {
                            break;
                        } else if st_as_usize > rank {
                            return Err(BufferizeError::StopToken(st_as_usize));
                        }

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
        let arc = ArcArray::from_shape_vec(shape_info, buffer)
            .expect("Unexpected mismatched shape when reading a stream into a buffer");

        Ok(Buffer::new(arc, creation_time.unwrap()))
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
                    previous_dim = Some(ind);
                    previous_data = Some(val.clone());
                    vec![]
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
    use crate::primitives::{elem::Elem, tile::Tile};

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
}
