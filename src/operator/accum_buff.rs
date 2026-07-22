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
/// It expresses reductions such as `[I,K,J] -> [I,J]` (reduce K), `[I,J,K] ->
/// [J,K]` (reduce I), or `[I,K,J] -> [J]` (reduce both I and K), where one or
/// more reduced dimensions fold onto a retained-dimension accumulator buffer.
///
/// The input stream carries a stop level per dimension boundary, with level 1
/// innermost and higher levels further out. By stop level the dimensions form
/// three contiguous bands:
///
/// * **retained** (levels `1..=L`, where `L = buffer_shape.len()`): the
///   innermost dimensions; they become the accumulator-buffer slots.
/// * **reduced** (levels `L+1..=rank`): folded into the buffer. There are
///   `rank - L` of them, so a single-dimension reduction has `rank == L + 1`
///   and a multi-dimension reduction uses a larger `rank`.
/// * **passthrough** (levels `> rank`): outer dimensions streamed through
///   unchanged; each completed buffer is emitted as `ValStop(level - rank)`.
///
/// Because the retained dimensions are streamed innermost in identical
/// row-major order on every reduction pass, a single flat index that
/// increments per tile and wraps modulo `product(buffer_shape)` always selects
/// the correct accumulator slot, regardless of how many dimensions are reduced:
/// the index completes one full sweep of the buffer between successive
/// reduced-dimension boundaries, landing back on slot 0 each time. This holds
/// only when the retained dimensions are the innermost-contiguous band with the
/// reduced dimensions directly outside them; an interleaved layout (e.g.
/// reducing the middle `J` of `[I,J,K] -> [I,K]`) breaks the mapping.
/// `buffer_shape` records the retained-dimension extents so the emitted
/// [`Buffer`] carries the correct shape.
#[context_macro]
pub struct AccumBuff<E, T: DAMType, OT: DAMType> {
    in_stream: Receiver<Elem<Tile<T>>>,
    out_stream: Sender<Elem<Buffer<Tile<OT>>>>,
    func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>, // bytes, bytes, FLOPs per cycle -> cycles
    init_accum: Arc<dyn Fn(usize, usize) -> Tile<OT> + Sync + Send>,
    /// Stop level at which one complete reduction group flushes: the stop level
    /// of the outermost reduced dimension. Boundaries below `rank` keep
    /// accumulating; boundaries above it are passed through as
    /// `ValStop(level - rank)`.
    rank: StopType,
    /// Extents of the retained (innermost) dimensions, one entry per retained
    /// dimension. `buffer_shape.len()` is the number of retained dimensions and
    /// `rank - buffer_shape.len()` the number of reduced dimensions. The number
    /// of accumulator slots is the product of these extents.
    buffer_shape: Vec<usize>,
    /// Row extent of each accumulator slot tile, or `0` if the row dimension
    /// is dynamic and resolved from the first input tile of each reduction
    /// group.
    tile_row: usize,
    /// Column extent of each accumulator slot tile, or `0` if the column
    /// dimension is dynamic and resolved from the first input tile of each
    /// reduction group.
    tile_col: usize,
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
        init_accum: Arc<dyn Fn(usize, usize) -> Tile<OT> + Sync + Send>,
        rank: StopType,
        buffer_shape: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
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
            tile_row,
            tile_col,
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

        // Snapshot the completed slots, leaving `accumulators` empty. The empty
        // vec signals the run loop to re-resolve tile dimensions and rebuild the
        // buffer from the first input of the next reduction group.
        let slots: Vec<Tile<OT>> = std::mem::take(accumulators);

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
        // Accumulator slots, (re)built lazily at the start of each reduction
        // group. An empty vec means "unresolved": the next input's shape fills
        // in any dynamic (0) tile dimension. `process_accum_flush` empties the
        // vec via `std::mem::take`, so each new group re-resolves.
        let mut accumulators: Vec<Tile<OT>> = Vec::new();
        let mut index: usize = 0;
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => {
                    if accumulators.is_empty() {
                        // Start of a reduction group: resolve dynamic tile dims
                        // from this first input and build the buffer.
                        let (in_rows, in_cols) = match &data {
                            Elem::Val(t) | Elem::ValStop(t, _) => (t.shape[0], t.shape[1]),
                        };
                        let rows = if self.tile_row == 0 { in_rows } else { self.tile_row };
                        let cols = if self.tile_col == 0 { in_cols } else { self.tile_col };
                        accumulators = (0..n).map(|_| (self.init_accum)(rows, cols)).collect();
                    }
                    match data {
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
                                let out_buffer =
                                    self.process_accum_flush(x, &mut accumulators, index);
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
                                let out_buffer =
                                    self.process_accum_flush(x, &mut accumulators, index);
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
                    }
                }
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
        tile_row: usize,
        tile_col: usize,
        init_accum: Arc<dyn Fn(usize, usize) -> Tile<i32> + Send + Sync>,
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
            init_accum,
            rank,
            buffer_shape,
            tile_row,
            tile_col,
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
            1,
            2,
            Arc::new(move |_rows, _cols| zero_init(read_from_mu)),
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
            1,
            2,
            Arc::new(move |_rows, _cols| zero_init(read_from_mu)),
        );
    }

    /// `[I,K,J] -> [J]` reducing over BOTH the outer I and the middle K.
    /// rank = 3, buffer shape = [J]. I = 2, K = 3, J = 2.
    /// tile(i,k,j) holds the scalar value i*100 + k*10 + j.
    ///
    /// This exercises `rank > len(buffer_shape) + 1`: two dimensions (I and K)
    /// are reduced onto the single retained dimension J. Every intermediate
    /// stop (level 1 = end of J, level 2 = end of K) is `< rank`, so
    /// accumulation continues; only the level-3 end-of-tensor stop flushes.
    /// The input stream is byte-for-byte identical to `test_accum_buff_rank2`;
    /// only `rank` (3 vs 2) and the expected output differ.
    #[test]
    fn test_accum_buff_reduce_two_dims() {
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
                            // also end of the (reduced) K run
                            if i == i_dim - 1 {
                                // also end of the (reduced) I run, whole tensor
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

        // Ground truth: a single [J] buffer summed over BOTH I and K.
        // slot(j) = sum_{i,k} (i*100 + k*10 + j).
        let slots: Vec<Tile<i32>> = (0..j_dim)
            .map(|j| {
                let mut sum = 0i32;
                for i in 0..i_dim {
                    for k in 0..k_dim {
                        sum += (i * 100 + k * 10 + j) as i32;
                    }
                }
                scalar_tile(sum, read_from_mu)
            })
            .collect();
        let arr = ArcArray::from_shape_vec(vec![j_dim], slots).unwrap();
        let buffer = Buffer::new(arr, 0);
        let ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>> = vec![Elem::Val(buffer)];

        run_reduction(
            in_stream_data,
            ground_truth_data,
            3,
            vec![j_dim],
            1,
            2,
            Arc::new(move |_rows, _cols| zero_init(read_from_mu)),
        );
    }

    /// A `rows x cols` tile whose every element is `v`.
    fn filled_tile(rows: usize, cols: usize, v: i32, read_from_mu: bool) -> Tile<i32> {
        Tile::new(Array2::from_elem((rows, cols), v).into(), 4, read_from_mu)
    }

    /// Dynamic row (`tile_row = 0`) resolved from the first input, single
    /// group. Reduce K=2 onto a 2-slot buffer of `2x3` tiles.
    /// `tile(k,j)` holds `k*10 + j`. slot(j) = sum_k tile(k,j).
    #[test]
    fn test_accum_buff_dynamic_row_single_group() {
        let read_from_mu = true;
        let (rows, cols) = (2usize, 3usize);
        let (k_dim, j_dim) = (2usize, 2usize);

        // Row-major [K, J], J retained (buffer_shape=[J]), K reduced, rank=2.
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        for k in 0..k_dim {
            for j in 0..j_dim {
                let tile = filled_tile(rows, cols, (k * 10 + j) as i32, read_from_mu);
                if j == j_dim - 1 {
                    if k == k_dim - 1 {
                        in_stream_data.push(Elem::ValStop(tile, 2)); // end reduced == rank
                    } else {
                        in_stream_data.push(Elem::ValStop(tile, 1)); // end retained run
                    }
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
        }

        let slots: Vec<Tile<i32>> = (0..j_dim)
            .map(|j| {
                let sum: i32 = (0..k_dim).map(|k| (k * 10 + j) as i32).sum();
                filled_tile(rows, cols, sum, read_from_mu)
            })
            .collect();
        let arr = ArcArray::from_shape_vec(vec![j_dim], slots).unwrap();
        let ground_truth_data = vec![Elem::Val(Buffer::new(arr, 0))];

        run_reduction(
            in_stream_data,
            ground_truth_data,
            2,
            vec![j_dim],
            0, // tile_row dynamic
            cols,
            Arc::new(move |r, c| Tile::new_zero([r, c], 4, read_from_mu)),
        );
    }

    /// Per-group re-resolution: two passthrough groups whose tiles have
    /// DIFFERENT row counts. buffer_shape=[1], rank=2, reduced K=2,
    /// passthrough I=2, `tile_row = 0`. Group 0 tiles are `2x2`, group 1
    /// tiles are `3x2`. Each emitted buffer must be sized to its own group's
    /// first input. This PANICS under resolve-once semantics (adding a `3x2`
    /// tile into a `2x2` accumulator), so it pins per-group resolution.
    #[test]
    fn test_accum_buff_dynamic_row_per_group() {
        let read_from_mu = true;
        let cols = 2usize;
        let group_rows = [2usize, 3usize]; // I = 2 groups
        let k_dim = 2usize;

        // Row-major [I, K, retained(=1)]. Retained level 1, reduced K level 2,
        // passthrough I level 3, rank=2.
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        for i in 0..group_rows.len() {
            for k in 0..k_dim {
                let tile = filled_tile(group_rows[i], cols, (i * 10 + k) as i32, read_from_mu);
                if k == k_dim - 1 {
                    if i == group_rows.len() - 1 {
                        in_stream_data.push(Elem::ValStop(tile, 3)); // end passthrough
                    } else {
                        in_stream_data.push(Elem::ValStop(tile, 2)); // end reduced == rank
                    }
                } else {
                    in_stream_data.push(Elem::ValStop(tile, 1)); // end retained run
                }
            }
        }

        // Per group: slot0 = sum_k (i*10 + k), tile sized group_rows[i] x cols.
        let mut ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>> = Vec::new();
        for i in 0..group_rows.len() {
            let sum: i32 = (0..k_dim).map(|k| (i * 10 + k) as i32).sum();
            let slot = filled_tile(group_rows[i], cols, sum, read_from_mu);
            let arr = ArcArray::from_shape_vec(vec![1usize], vec![slot]).unwrap();
            let buffer = Buffer::new(arr, 0);
            if i == group_rows.len() - 1 {
                ground_truth_data.push(Elem::ValStop(buffer, 1)); // level 3 - rank 2
            } else {
                ground_truth_data.push(Elem::Val(buffer));
            }
        }

        run_reduction(
            in_stream_data,
            ground_truth_data,
            2,
            vec![1],
            0, // tile_row dynamic
            cols,
            Arc::new(move |r, c| Tile::new_zero([r, c], 4, read_from_mu)),
        );
    }
}
