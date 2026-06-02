use std::{marker::PhantomData, sync::Arc};

use ndarray::ArcArray;

use crate::memory::PMU_BW;
use crate::operator::accum::AccumConfig;
use crate::primitives::buffer::Buffer;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

/// Higher-order reduction operator.
///
/// Where [`Accum`](super::accum::Accum) reduces a stream onto a single
/// accumulator tile, `AccumBuff` reduces onto a *buffer* of accumulator tiles.
/// This expresses reductions such as `[I,K,J] -> [I,J]` or `[I,J,K] -> [J,K]`
/// where the reduced dimension is streamed outermost and the retained
/// dimensions form an `(N-1)`-dimensional accumulator buffer (`N == rank`).
///
/// Because the retained dimensions are streamed innermost in identical
/// row-major order on every reduction pass, a single flat index that
/// increments per tile and wraps modulo `product(buffer_shape)` always selects
/// the correct accumulator slot, regardless of `N`. `buffer_shape` only needs
/// to record the retained-dimension extents so the emitted [`Buffer`] carries
/// the correct shape.
#[context_macro]
pub struct AccumBuff<E, T: DAMType, OT: DAMType> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Buffer<Tile<OT>>>>,
    func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
    rank: StopType,
    /// Extents of the `(rank - 1)`-dimensional accumulator buffer. The number
    /// of accumulator slots is the product of these extents.
    buffer_shape: Vec<usize>,
    config: AccumConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > AccumBuff<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
    Buffer<Tile<OT>>: DAMType,
    Elem<Buffer<Tile<OT>>>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Buffer<Tile<OT>>>>,
        func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
        init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
        rank: StopType,
        buffer_shape: Vec<usize>,
        config: AccumConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            func,
            init_accum,
            rank,
            buffer_shape,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    /// Accumulate `data` into the slot at `index`. Used for elements that do
    /// not complete a reduction group.
    fn process_accum(&mut self, data: Tile<T>, accumulators: &mut Vec<Tile<OT>>, index: usize) {
        let mut load_cycles = div_ceil(accumulators[0].size_in_bytes() as u64, PMU_BW);
        let store_cycles = load_cycles;

        if data.read_from_mu {
            load_cycles = load_cycles + div_ceil(data.size_in_bytes() as u64, PMU_BW)
        }

        let (comp_cycles, out_tile) = (self.func)(
            &data,
            &accumulators[index],
            self.config.compute_bw,
            self.config.write_back_mu,
        );
        accumulators[index] = out_tile;

        let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
            .into_iter()
            .max()
            .unwrap_or(0);

        // increment cycles and dequeue inputs
        self.time.incr_cycles(roofline_cycles);

        self.in_stream.dequeue(&self.time).unwrap();
    }

    /// Accumulate the reduction-completing `data` into the slot at `index`,
    /// then snapshot the full accumulator buffer, re-initialize all slots, and
    /// return the completed buffer.
    fn process_accum_flush(
        &mut self,
        data: Tile<T>,
        accumulators: &mut Vec<Tile<OT>>,
        index: usize,
    ) -> Buffer<Tile<OT>> {
        let mut load_cycles = div_ceil(accumulators[0].size_in_bytes() as u64, PMU_BW);
        let store_cycles = load_cycles;

        if data.read_from_mu {
            load_cycles = load_cycles + div_ceil(data.size_in_bytes() as u64, PMU_BW)
        }

        let (comp_cycles, out_tile) = (self.func)(
            &data,
            &accumulators[index],
            self.config.compute_bw,
            self.config.write_back_mu,
        );
        accumulators[index] = out_tile;

        let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
            .into_iter()
            .max()
            .unwrap_or(0);

        self.time.incr_cycles(roofline_cycles);
        self.in_stream.dequeue(&self.time).unwrap();

        // Logging
        dam::logging::log_event(&E::new(
            "AccumBuff".to_string(),
            self.id,
            self.time.tick().time() - roofline_cycles,
            self.time.tick().time(),
            true,
        ))
        .unwrap();

        // Snapshot the completed slots and re-initialize the buffer in place.
        let n = accumulators.len();
        let slots: Vec<Tile<OT>> =
            std::mem::replace(accumulators, (0..n).map(|_| (self.init_accum)()).collect());

        let arr = ArcArray::from_shape_vec(self.buffer_shape.clone(), slots)
            .expect("AccumBuff: buffer_shape does not match the number of accumulator slots");

        Buffer::new(arr, self.time.tick().time())
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Context for AccumBuff<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
    Buffer<Tile<OT>>: DAMType,
    Elem<Buffer<Tile<OT>>>: DAMType,
{
    fn run(&mut self) {
        let n: usize = self.buffer_shape.iter().product();
        assert!(
            n > 0,
            "AccumBuff: buffer_shape must describe at least one accumulator slot"
        );
        let mut accumulators: Vec<Tile<OT>> = (0..n).map(|_| (self.init_accum)()).collect();
        let mut index: usize = 0;
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        self.process_accum(x, &mut accumulators, index);
                        index = (index + 1) % n;
                    }
                    Elem::ValStop(x, level) => {
                        if level < self.rank {
                            // Intermediate retained-dimension boundary: keep
                            // accumulating across the reduced dimension.
                            self.process_accum(x, &mut accumulators, index);
                            index = (index + 1) % n;
                        } else if level == self.rank {
                            let out_buffer = self.process_accum_flush(x, &mut accumulators, index);
                            index = 0;
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::Val(out_buffer),
                                    },
                                )
                                .unwrap();
                        } else {
                            let out_buffer = self.process_accum_flush(x, &mut accumulators, index);
                            index = 0;
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data: Elem::ValStop(out_buffer, level - self.rank),
                                    },
                                )
                                .unwrap();
                        }
                    }
                },
                Err(_) => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        functions::accum_fn,
        operator::{accum::AccumConfig, accum_buff::AccumBuff},
        primitives::{buffer::Buffer, elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext};
    use ndarray::{ArcArray, Array2};
    use std::sync::Arc;

    fn tolerance_fn(a: &Elem<Buffer<Tile<i32>>>, b: &Elem<Buffer<Tile<i32>>>) -> bool {
        match (a, b) {
            (Elem::Val(a_buf), Elem::Val(b_buf)) => a_buf == b_buf,
            (Elem::ValStop(a_buf, a_level), Elem::ValStop(b_buf, b_level)) => {
                a_buf == b_buf && a_level == b_level
            }
            _ => false,
        }
    }

    /// Build a 1x2 tile whose two elements are both `v`.
    fn scalar_tile(v: i32, read_from_mu: bool) -> Tile<i32> {
        Tile::new(
            Array2::from_shape_vec((1, 2), vec![v, v]).unwrap().into(),
            4,
            read_from_mu,
        )
    }

    fn zero_init(read_from_mu: bool) -> Tile<i32> {
        scalar_tile(0, read_from_mu)
    }

    fn run_reduction(
        in_stream_data: Vec<Elem<Tile<i32>>>,
        ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>>,
        rank: u32,
        buffer_shape: Vec<usize>,
        read_from_mu: bool,
    ) {
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(AccumBuff::<SimpleEvent, _, _>::new(
            in_data_rcv,
            out_data_snd,
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                accum_fn::add(tile1, tile2, comp_bw, write_back_mu, 0)
            }),
            Arc::new(move || zero_init(read_from_mu)),
            rank,
            buffer_shape,
            AccumConfig {
                compute_bw: 1000,
                write_back_mu: true,
            },
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

    /// `[I,K,J] -> [I,J]` reducing over K. rank = 2, buffer shape = [J].
    /// I = 2, K = 3, J = 2. tile(i,k,j) holds the scalar value i*100 + k*10 + j.
    #[test]
    fn test_accum_buff_rank2() {
        let read_from_mu = true;
        let (i_dim, k_dim, j_dim) = (2usize, 3usize, 2usize);

        // Input stream in row-major [I,K,J] order.
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        for i in 0..i_dim {
            for k in 0..k_dim {
                for j in 0..j_dim {
                    let v = (i * 100 + k * 10 + j) as i32;
                    let tile = scalar_tile(v, read_from_mu);
                    if j == j_dim - 1 {
                        // end of the innermost (J) run
                        if k == k_dim - 1 {
                            // also end of the reduced (K) run
                            if i == i_dim - 1 {
                                in_stream_data.push(Elem::ValStop(tile, 3));
                            } else {
                                in_stream_data.push(Elem::ValStop(tile, 2));
                            }
                        } else {
                            in_stream_data.push(Elem::ValStop(tile, 1));
                        }
                    } else {
                        in_stream_data.push(Elem::Val(tile));
                    }
                }
            }
        }

        // Ground truth: one [J] buffer per i. slot(i,j) = sum_k (i*100 + k*10 + j).
        let mut ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>> = Vec::new();
        for i in 0..i_dim {
            let slots: Vec<Tile<i32>> = (0..j_dim)
                .map(|j| {
                    let sum: i32 = (0..k_dim).map(|k| (i * 100 + k * 10 + j) as i32).sum();
                    scalar_tile(sum, read_from_mu)
                })
                .collect();
            let arr = ArcArray::from_shape_vec(vec![j_dim], slots).unwrap();
            let buffer = Buffer::new(arr, 0);
            if i == i_dim - 1 {
                ground_truth_data.push(Elem::ValStop(buffer, 1));
            } else {
                ground_truth_data.push(Elem::Val(buffer));
            }
        }

        run_reduction(
            in_stream_data,
            ground_truth_data,
            2,
            vec![j_dim],
            read_from_mu,
        );
    }

    /// `[I,J,K] -> [J,K]` reducing over the outermost I. rank = 3,
    /// buffer shape = [J,K]. I = 2, J = 2, K = 3.
    /// tile(i,j,k) holds the scalar value i*100 + j*10 + k.
    #[test]
    fn test_accum_buff_rank3() {
        let read_from_mu = true;
        let (i_dim, j_dim, k_dim) = (2usize, 2usize, 3usize);

        // Input stream in row-major [I,J,K] order.
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        for i in 0..i_dim {
            for j in 0..j_dim {
                for k in 0..k_dim {
                    let v = (i * 100 + j * 10 + k) as i32;
                    let tile = scalar_tile(v, read_from_mu);
                    if k == k_dim - 1 {
                        // end of innermost (K) run
                        if j == j_dim - 1 {
                            // also end of the middle (J) run
                            if i == i_dim - 1 {
                                // also end of the reduced (I) run, whole tensor
                                in_stream_data.push(Elem::ValStop(tile, 3));
                            } else {
                                in_stream_data.push(Elem::ValStop(tile, 2));
                            }
                        } else {
                            in_stream_data.push(Elem::ValStop(tile, 1));
                        }
                    } else {
                        in_stream_data.push(Elem::Val(tile));
                    }
                }
            }
        }

        // Ground truth: a single [J,K] buffer summed over I.
        // slot(j,k) = sum_i (i*100 + j*10 + k).
        let slots: Vec<Tile<i32>> = (0..j_dim)
            .flat_map(|j| {
                (0..k_dim).map(move |k| {
                    let sum: i32 = (0..i_dim).map(|i| (i * 100 + j * 10 + k) as i32).sum();
                    scalar_tile(sum, read_from_mu)
                })
            })
            .collect();
        let arr = ArcArray::from_shape_vec(vec![j_dim, k_dim], slots).unwrap();
        let buffer = Buffer::new(arr, 0);
        // End of the whole tensor (level 3) > rank (3)? No: level 3 == rank, but
        // the I run is the reduced dimension and there is no outer dimension, so
        // the single output buffer carries no further stop level.
        let ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>> = vec![Elem::Val(buffer)];

        run_reduction(
            in_stream_data,
            ground_truth_data,
            3,
            vec![j_dim, k_dim],
            read_from_mu,
        );
    }
}
