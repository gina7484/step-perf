use crate::primitives::elem::{Elem, StopType};
use crate::primitives::tile::Tile;
use dam::context_tools::*;
use dam::types::DAMType;

#[context_macro]
pub struct FlatmapFilterRowStreamify<T: Clone> {
    in_stream: Receiver<Elem<Tile<T>>>,      // tile shape: [R,C]
    mask_stream: Receiver<Elem<Tile<bool>>>, // tile shape: [R,1]
    out_stream: Sender<Elem<Tile<T>>>,       // tile shape: [1,C]
    id: u32,
}

impl<T: Clone> FlatmapFilterRowStreamify<T>
where
    Tile<T>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        mask_stream: Receiver<Elem<Tile<bool>>>,
        out_stream: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            mask_stream,
            out_stream,
            id,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.mask_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
    fn filter_streamify(
        &mut self,
        data: &Tile<T>,
        mask: &Tile<bool>,
        stop_level: Option<StopType>,
    ) {
        let mask_iter: ndarray::iter::LanesIter<'_, bool, ndarray::Dim<[usize; 1]>> =
            mask.underlying.as_ref().unwrap().rows().into_iter(); // [1] (bool) x R
        let mut count = 0;
        for mask_i in mask_iter {
            if mask_i[0] {
                count += 1;
            }
        }

        match &data.underlying {
            Some(arr) => {
                // split_row: [R,C] => [C] x R
                // Later if we want to do column-wise, we can do that here ([R,C] => [R] x C)
                // let vec_iter = if self.split_row {
                //     arr.rows().into_iter()
                // } else {
                //     arr.columns().into_iter()
                // };
                let vec_iter = arr.rows().into_iter(); // [C] x R
                let mask_iter = mask.underlying.as_ref().unwrap().rows().into_iter(); // [1] (bool) x R

                let mut curr_cnt = 0;
                for (row, mask_i) in vec_iter.zip(mask_iter) {
                    // Only process rows where the mask is true
                    if mask_i[0] {
                        let out_data = Tile::<T>::new_padded(
                            row.to_shared().insert_axis(ndarray::Axis(0)), // [N] => [1,N]
                            data.bytes_per_elem,
                            data.read_from_mu,
                            1,
                        );
                        curr_cnt += 1;

                        // check whether this is the last value and set the stop level if needed
                        let elem = if stop_level.is_some() && (curr_cnt == count) {
                            Elem::ValStop(out_data, stop_level.unwrap())
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
                    }
                }
            }
            None => {
                // let vec_size = if self.split_row {
                //     data.shape[1]
                // } else {
                //     data.shape[0]
                // };
                let vec_size = data.shape[1];

                for idx in 0..count {
                    let out_data = Tile::<T>::new_blank(
                        vec![1, vec_size],
                        data.bytes_per_elem,
                        data.read_from_mu,
                    );

                    // check whether this is the last value and set the stop level if needed
                    let elem = if stop_level.is_some() && (idx + 1 == count) {
                        Elem::ValStop(out_data, stop_level.unwrap())
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
                }
            }
        }
    }
}

impl<T: Clone> Context for FlatmapFilterRowStreamify<T>
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
                    Elem::Val(data) => match self.mask_stream.dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: mask_enum,
                        }) => match mask_enum {
                            Elem::Val(mask) => self.filter_streamify(&data, &mask, None),
                            Elem::ValStop(_, s) => {
                                panic!(
                                    "Stop token position mismatch: data(val), mask(valstop({}))",
                                    s
                                );
                            }
                        },
                        Err(_) => {
                            panic!("Mask stream terminated before data stream");
                        }
                    },
                    Elem::ValStop(data, s) => match self.mask_stream.dequeue(&self.time) {
                        Ok(ChannelElement {
                            time: _,
                            data: mask_enum,
                        }) => match mask_enum {
                            Elem::Val(_) => {
                                panic!(
                                    "Stop token position mismatch: data(valstop({})), mask(val)",
                                    s
                                );
                            }
                            Elem::ValStop(mask, mask_s) => {
                                assert_eq!(mask_s, s);
                                self.filter_streamify(&data, &mask, Some(s))
                            }
                        },
                        Err(_) => {
                            panic!("Mask stream terminated before data stream");
                        }
                    },
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
    use std::sync::Arc;

    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext},
    };
    use ndarray::ArcArray;

    use super::FlatmapFilterRowStreamify;
    use crate::primitives::{buffer::Buffer, tile::Tile};
    use crate::{functions::accum_fn::retile_row, operator::accum::AccumConfig};
    use crate::{
        operator::{accum::Accum, reshape::ReshapePadStream},
        utils::events::SimpleEvent,
    };

    #[test]
    fn roundtrip_test() {
        // Reshape: (1, 3, 9) => (1, 3, 3, 4), (1, 3, 3, 4)
        // * The last three vector tiles will be padded values

        // Accum (RetileRow): (1, 3, 3, 4) => (1, 3, 3)
        // * This will be done both for data and mask stream

        // FlatmapFilterRowStreamify: (1, 3, 3) => (1, 3, 9)

        type VT = u32;
        const BYTES_PER_ELEM: usize = 2;
        const READ_FROM_MU: bool = true;
        const DUMMY_CREATION_TIME: u64 = 0;

        let tile_shape: Vec<usize> = vec![1, 4];

        let mut ctx = ProgramBuilder::default();

        let (in_snd, in_rcv) = ctx.unbounded();
        let (mask_snd, mask_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();
        let (accum_data_snd, accum_data_rcv) = ctx.unbounded();
        let (accum_mask_snd, accum_mask_rcv) = ctx.unbounded();
        let (final_snd, final_rcv) = ctx.unbounded();

        let in_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 9
            ])
            .into_shape_with_order((3, 9))
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

        ctx.add_child(ReshapePadStream::new(
            in_rcv,
            out_snd,
            mask_snd,
            0,
            4,
            Some(Tile::new_blank_padded(
                tile_shape.clone(),
                BYTES_PER_ELEM,
                READ_FROM_MU,
                0,
            )),
            2,
            false,
            0,
        ));

        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            out_rcv,
            accum_data_snd,
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                retile_row(tile1, tile2, comp_bw, write_back_mu)
            }),
            Arc::new(move || Tile::new_empty([0, 4], BYTES_PER_ELEM, false)),
            1,
            AccumConfig {
                compute_bw: 1024,
                write_back_mu: false,
            },
            0,
        ));

        ctx.add_child(Accum::<SimpleEvent, _, _>::new(
            mask_rcv,
            accum_mask_snd,
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                retile_row(tile1, tile2, comp_bw, write_back_mu)
            }),
            Arc::new(move || Tile::new_empty([0, 1], BYTES_PER_ELEM, false)),
            1,
            AccumConfig {
                compute_bw: 1024,
                write_back_mu: false,
            },
            0,
        ));

        ctx.add_child(FlatmapFilterRowStreamify::new(
            accum_data_rcv,
            accum_mask_rcv,
            final_snd,
            0,
        ));

        let mut output_tile_vec = vec![];
        for _ in 0..3 {
            output_tile_vec.extend(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                9
            ]);
            output_tile_vec.extend(vec![
                Tile::<VT>::new_blank_padded(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU,
                    0
                );
                3
            ]);
        }

        let out_arr = Arc::new(
            ArcArray::from_vec(vec![
                Tile::<VT>::new_blank(
                    tile_shape.clone(),
                    BYTES_PER_ELEM,
                    READ_FROM_MU
                );
                3 * 9
            ])
            .into_shape_with_order((3, 9))
            .unwrap(),
        );
        ctx.add_child(ApproxCheckerContext::new(
            move || {
                Buffer::new((*out_arr).clone().into_dyn(), DUMMY_CREATION_TIME)
                    .to_elem_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
            },
            final_rcv,
            |x, y| x == y,
        ));
        // ctx.add_child(PrinterContext::new(out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
