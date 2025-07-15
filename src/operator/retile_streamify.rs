use std::{marker::PhantomData, sync::Arc};

use crate::primitives::elem::{Elem, StopType};
use crate::primitives::tile::Tile;
use dam::context_tools::*;
use dam::types::DAMType;

#[context_macro]
pub struct RetileStreamify<T: Clone> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Tile<T>>>,
    split_row: bool,
    filter_mask: bool,
    id: u32,
}

impl<T: Clone> RetileStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        split_row: bool,
        filter_mask: bool,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            split_row,
            filter_mask,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
    fn retile(&mut self, data: &Tile<T>, stop_level: Option<StopType>) {
        match &data.underlying {
            Some(arr) => {
                let offset = data.offset;

                let vec_iter = if self.split_row {
                    arr.rows().into_iter()
                } else {
                    arr.columns().into_iter()
                };

                let vec_iter_len = vec_iter.len();
                for (idx, row) in vec_iter.enumerate() {
                    let out_data = Tile::<T>::new_padded(
                        row.to_shared().insert_axis(ndarray::Axis(0)), // [N] => [1,N]
                        data.bytes_per_elem,
                        data.read_from_mu,
                        if idx + 1 <= offset { 1 } else { 0 },
                    );

                    // check whether this is the last value and set the stop level if needed
                    let elem = if stop_level.is_some() {
                        if (self.filter_mask && idx + 1 == offset) || (idx + 1 == vec_iter_len) {
                            Elem::ValStop(out_data, stop_level.unwrap())
                        } else {
                            Elem::Val(out_data)
                        }
                    } else {
                        Elem::Val(out_data)
                    };

                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: elem,
                            },
                        )
                        .unwrap();
                    if self.filter_mask && idx + 1 == offset {
                        break;
                    }
                }
            }
            None => {
                let offset = data.offset;

                let num_tiles = if self.split_row {
                    data.shape[0]
                } else {
                    data.shape[1]
                };

                let row_size = if self.split_row {
                    data.shape[1]
                } else {
                    data.shape[0]
                };

                for idx in 0..num_tiles {
                    let out_data = Tile::<T>::new_blank_padded(
                        vec![1, row_size],
                        data.bytes_per_elem,
                        data.read_from_mu,
                        if idx + 1 <= offset { 1 } else { 0 },
                    );

                    // check whether this is the last value and set the stop level if needed
                    let elem = if stop_level.is_some() {
                        if (self.filter_mask && idx + 1 == offset) || (idx + 1 == num_tiles) {
                            Elem::ValStop(out_data, stop_level.unwrap())
                        } else {
                            Elem::Val(out_data)
                        }
                    } else {
                        Elem::Val(out_data)
                    };

                    self.out_stream
                        .enqueue(
                            &self.time,
                            ChannelElement {
                                time: self.time.tick(),
                                data: elem,
                            },
                        )
                        .unwrap();
                    if self.filter_mask && idx + 1 == offset {
                        break;
                    }
                }
            }
        }
    }
}

impl<T: Clone> Context for RetileStreamify<T>
where
    Tile<T>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: data_enum,
                }) => match data_enum {
                    Elem::Val(data) => {
                        self.retile(&data, None);
                    }
                    Elem::ValStop(data, s) => {
                        self.retile(&data, Some(s));
                    }
                },
                Err(_) => {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RetileStreamify;
    use crate::{
        functions::map_fn,
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext, PrinterContext};
    use ndarray::Array2;
    use std::sync::Arc;

    fn tolerance_fn(a: &Elem<Tile<i32>>, b: &Elem<Tile<i32>>) -> bool {
        match (a, b) {
            (Elem::Val(a_tile), Elem::Val(b_tile)) => a_tile == b_tile,
            (Elem::ValStop(a_tile, a_level), Elem::ValStop(b_tile, b_level)) => {
                a_tile == b_tile && a_level == b_level
            }
            _ => false,
        }
    }

    #[test]
    fn test_retile_col() {
        // [1,3] => [1,9]
        // [4,3] tile => [4,1] tile
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 8 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(1),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new(concatenated_array.to_shared(), 4, read_from_mu),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays_input: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((4, 1), vec![i as i32; 4]).unwrap())
            .collect();
        let arrays_output: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays_input, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays_output, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(RetileStreamify::<_>::new(
            in_data_rcv,
            out_data_snd,
            false,
            false,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_retile_row() {
        // [1,3] => [1,9]
        // [3,4] tile => [1,4] tile
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 8 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(0),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new(concatenated_array.to_shared(), 4, read_from_mu),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(RetileStreamify::<_>::new(
            in_data_rcv,
            out_data_snd,
            true,
            false,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_retile_row_filter() {
        // [1,3] => [1,7]
        // [3,4] tile => [1,4] tile (last tile is padded with 2 vectors)
        fn create_ground_truth(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for (i, arr) in arrays.iter().enumerate() {
                let tile = Tile::new(arr.clone().into(), 4, read_from_mu);

                // Add ValStop at indices 2, 5, 8 (end of each row in 3x3 grid)
                if i == 6 {
                    in_stream_data.push(Elem::ValStop(tile, 1));
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
            in_stream_data
        }

        fn create_input_data(arrays: &[Array2<i32>], read_from_mu: bool) -> Vec<Elem<Tile<i32>>> {
            let mut ground_truth_data: Vec<Elem<Tile<i32>>> = Vec::new();
            for i in 0..3 {
                let concatenated_array = ndarray::concatenate(
                    ndarray::Axis(0),
                    &[
                        arrays[i * 3].view(),
                        arrays[i * 3 + 1].view(),
                        arrays[i * 3 + 2].view(),
                    ],
                )
                .unwrap_or_else(|_| {
                    panic!("Failed to concatenate input data and accumulator data")
                });

                let elem = if i == 2 {
                    Elem::ValStop(
                        Tile::new_padded(concatenated_array.to_shared(), 4, read_from_mu, 1),
                        1,
                    )
                } else {
                    Elem::Val(Tile::new(concatenated_array.to_shared(), 4, read_from_mu))
                };

                ground_truth_data.push(elem);
            }
            ground_truth_data
        }

        // Step 1: Create 9 different ndarray::ArcArray2<T> with shape 2x2
        let arrays_input: Vec<Array2<i32>> = (0..9)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let arrays_output: Vec<Array2<i32>> = (0..7)
            .map(|i| Array2::from_shape_vec((1, 4), vec![i as i32; 4]).unwrap())
            .collect();
        let read_from_mu = true;
        // Step 2: Create a 3x3 rank-2 data stream from these arrays
        let in_stream_data = create_input_data(&arrays_input, read_from_mu);

        // Step 3: Create a ground truth for the output stream
        let ground_truth_data = create_ground_truth(&arrays_output, read_from_mu);

        // Step 4: Create the STeP program
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(RetileStreamify::<_>::new(
            in_data_rcv,
            out_data_snd,
            true,
            true,
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
