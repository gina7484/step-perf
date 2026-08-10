use std::marker::PhantomData;

use ndarray::Array2;

use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

/// Which row-wise statistic the operator emits at the end of a reduction group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowStat {
    Mean,
    Var,
}

pub struct AccumRowStatConfig {
    pub compute_bw: u64,
    pub write_back_mu: bool,
    pub stat: RowStat,
    /// `Some(n)` for the `MeanStatic` / `VarStatic` frontend functions, whose
    /// element count per row is known at compile time. `None` for `MeanDyn` /
    /// `VarDyn`, which count at run time.
    pub count: Option<u64>,
}

/// Running Welford state for one reduction group, one entry per tile row.
struct GroupState {
    mean: Vec<f32>,
    m2: Vec<f32>,
    /// Elements folded into *each* row so far (identical across rows).
    n: u64,
    /// Non-padded row count, carried over from the most recent input tile.
    offset: usize,
    /// Cleared by the first blank (timing-only) tile in the group.
    has_data: bool,
    started: bool,
}

impl GroupState {
    fn new() -> Self {
        Self {
            mean: Vec::new(),
            m2: Vec::new(),
            n: 0,
            offset: 0,
            has_data: true,
            started: false,
        }
    }

    fn rows(&self) -> usize {
        self.mean.len()
    }

    fn reset(&mut self) {
        self.mean.clear();
        self.m2.clear();
        self.n = 0;
        self.offset = 0;
        self.has_data = true;
        self.started = false;
    }
}

/// Row-wise mean / variance over the innermost `rank` stream ranks.
///
/// Unlike [`crate::operator::accum::Accum`], the accumulator is *not* the
/// output: a `[R, C]` input stream collapses to a single `[R, 1]` tile per
/// reduction group, so the tile's columns are reduced alongside the stream
/// ranks. Variance also needs two running quantities and both statistics need a
/// finalizing divide, neither of which fits `Accum`'s
/// `func(&Tile, &Tile) -> Tile` fold.
///
/// The running state is Welford's, which keeps `m2 >= 0` by construction — a
/// near-constant row therefore cannot feed a negative variance into a
/// downstream `rsqrt`. The cycle model below instead prices the two-accumulator
/// (`sum`, `sumsq`) datapath the hardware would use; cycle counts are derived
/// from tile shapes and are independent of how the value is computed here.
#[context_macro]
pub struct AccumRowStat<E> {
    in_stream: Receiver<Elem<Tile<f32>>>,
    out_stream: Sender<Elem<Tile<f32>>>,
    rank: StopType,
    bytes_per_elem: usize,
    config: AccumRowStatConfig,
    id: u32,
    state: GroupState,
    _phantom: PhantomData<E>,
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> AccumRowStat<E>
where
    Elem<Tile<f32>>: DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<Tile<f32>>>,
        out_stream: Sender<Elem<Tile<f32>>>,
        rank: StopType,
        bytes_per_elem: usize,
        config: AccumRowStatConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            rank,
            bytes_per_elem,
            config,
            id,
            state: GroupState::new(),
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    /// Fold one input tile into the running statistics. Returns compute cycles.
    fn fold(&mut self, data: &Tile<f32>) -> u64 {
        assert_eq!(
            data.shape.len(),
            2,
            "AccumRowStat_{} expects 2D tiles",
            self.id
        );
        let rows = data.shape[0];
        let cols = data.shape[1];

        if self.state.started {
            assert_eq!(
                self.state.rows(),
                rows,
                "AccumRowStat_{} row count changed mid-reduction",
                self.id
            );
        } else {
            self.state.mean = vec![0.0; rows];
            self.state.m2 = vec![0.0; rows];
            self.state.started = true;
        }
        self.state.offset = data.offset;

        match &data.underlying {
            Some(arr) => {
                for i in 0..rows {
                    let mut n = self.state.n;
                    let mut mean = self.state.mean[i];
                    let mut m2 = self.state.m2[i];
                    for j in 0..cols {
                        let x = arr[[i, j]];
                        n += 1;
                        let delta = x - mean;
                        mean += delta / (n as f32);
                        m2 += delta * (x - mean);
                    }
                    self.state.mean[i] = mean;
                    self.state.m2[i] = m2;
                }
            }
            // Timing-only tile: its shape still says how many elements the group
            // covers, so keep counting and drop the functional result.
            None => self.state.has_data = false,
        }
        self.state.n += cols as u64;

        let flops = match self.config.stat {
            // sum += x
            RowStat::Mean => (rows * cols) as u64,
            // x*x, sum += x, sumsq += x*x
            RowStat::Var => 3 * (rows * cols) as u64,
        };
        div_ceil(flops, self.config.compute_bw)
    }

    /// Emit the group's statistic and reset for the next group.
    fn finalize(&mut self) -> (u64, Tile<f32>) {
        let rows = self.state.rows();
        let counted = self.state.n;
        // `*Static` honours its compile-time element count, `*Dyn` uses what it
        // counted. A mismatch means the lowering pass declared the wrong count.
        debug_assert!(
            self.config.count.is_none() || self.config.count == Some(counted),
            "AccumRowStat_{}: declared count {:?} but reduced {} elements per row",
            self.id,
            self.config.count,
            counted
        );
        let n = self.config.count.unwrap_or(counted);

        let out = if self.state.has_data && self.state.started && n > 0 && counted > 0 {
            let vals: Vec<f32> = (0..rows)
                .map(|i| match self.config.stat {
                    // Welford tracks `sum / counted`; rescaling recovers
                    // `sum / n`. When the counts agree the ratio is exactly 1.0
                    // in f32, so the static path adds no rounding.
                    RowStat::Mean => self.state.mean[i] * (counted as f32 / n as f32),
                    RowStat::Var => self.state.m2[i] / (n as f32),
                })
                .collect();
            Tile::new_padded(
                Array2::from_shape_vec((rows, 1), vals)
                    .expect("row statistic vector matches [rows, 1]")
                    .to_shared(),
                self.bytes_per_elem,
                self.config.write_back_mu,
                self.state.offset,
            )
        } else {
            Tile::new_blank_padded(
                vec![rows, 1],
                self.bytes_per_elem,
                self.config.write_back_mu,
                self.state.offset,
            )
        };

        let flops = match self.config.stat {
            // sum / n
            RowStat::Mean => rows as u64,
            // sum/n, mean*mean, sumsq/n, subtract
            RowStat::Var => 4 * rows as u64,
        };

        self.state.reset();
        (div_ceil(flops, self.config.compute_bw), out)
    }

    fn process(&mut self, data: Tile<f32>) {
        let load_cycles = if data.read_from_mu {
            div_ceil(data.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let comp_cycles = self.fold(&data);
        let roofline_cycles = [load_cycles, comp_cycles].into_iter().max().unwrap_or(0);

        self.time.incr_cycles(roofline_cycles);
        self.in_stream.dequeue(&self.time).unwrap();
    }

    fn process_and_finalize(&mut self, data: Tile<f32>) -> Tile<f32> {
        let load_cycles = if data.read_from_mu {
            div_ceil(data.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let fold_cycles = self.fold(&data);
        let (finalize_cycles, out_tile) = self.finalize();

        let store_cycles = if self.config.write_back_mu {
            div_ceil(out_tile.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let roofline_cycles = [load_cycles, fold_cycles + finalize_cycles, store_cycles]
            .into_iter()
            .max()
            .unwrap_or(0);

        self.time.incr_cycles(roofline_cycles);
        self.in_stream.dequeue(&self.time).unwrap();

        dam::logging::log_event(&E::new(
            "AccumRowStat".to_string(),
            self.id,
            self.time.tick().time() - roofline_cycles,
            self.time.tick().time(),
            true,
        ))
        .unwrap();

        out_tile
    }
}

impl<E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send> Context
    for AccumRowStat<E>
where
    Elem<Tile<f32>>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => {
                        self.process(x);
                    }
                    Elem::ValStop(x, level) => {
                        if level < self.rank {
                            self.process(x);
                        } else {
                            let out_tile = self.process_and_finalize(x);
                            let data = if level == self.rank {
                                Elem::Val(out_tile)
                            } else {
                                Elem::ValStop(out_tile, level - self.rank)
                            };
                            self.out_stream
                                .enqueue(
                                    &self.time,
                                    ChannelElement {
                                        time: self.time.tick(),
                                        data,
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
    use super::*;
    use crate::utils::events::SimpleEvent;
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::{ApproxCheckerContext, GeneratorContext};

    const BYTES: usize = 4;
    const COMPUTE_BW: u64 = 1000;

    fn tile(rows: usize, cols: usize, vals: Vec<f32>) -> Tile<f32> {
        Tile::new(
            Array2::from_shape_vec((rows, cols), vals)
                .unwrap()
                .to_shared(),
            BYTES,
            false,
        )
    }

    fn close(a: &Elem<Tile<f32>>, b: &Elem<Tile<f32>>) -> bool {
        let (a_tile, b_tile) = match (a, b) {
            (Elem::Val(x), Elem::Val(y)) => (x, y),
            (Elem::ValStop(x, xl), Elem::ValStop(y, yl)) if xl == yl => (x, y),
            _ => return false,
        };
        if a_tile.shape != b_tile.shape {
            return false;
        }
        match (&a_tile.underlying, &b_tile.underlying) {
            (Some(x), Some(y)) => x
                .iter()
                .zip(y.iter())
                .all(|(l, r)| (l - r).abs() < 1e-5 * r.abs().max(1.0)),
            (None, None) => true,
            _ => false,
        }
    }

    /// Drive one reduction group through the operator and return what it emits.
    fn run_group(
        input: Vec<Elem<Tile<f32>>>,
        expected: Vec<Elem<Tile<f32>>>,
        stat: RowStat,
        count: Option<u64>,
        rank: StopType,
    ) {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();

        ctx.add_child(GeneratorContext::new(|| input.into_iter(), in_snd));
        ctx.add_child(AccumRowStat::<SimpleEvent>::new(
            in_rcv,
            out_snd,
            rank,
            BYTES,
            AccumRowStatConfig {
                compute_bw: COMPUTE_BW,
                write_back_mu: false,
                stat,
                count,
            },
            0,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || expected.into_iter(),
            out_rcv,
            close,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    // A [2, 3] tile reduces to [2, 1]: the tile's columns collapse alongside the
    // stream rank, so row 0 averages [1,2,3] and row 1 averages [7,8,9].
    #[test]
    fn mean_static_single_tile_collapses_columns() {
        let input = vec![Elem::ValStop(
            tile(2, 3, vec![1., 2., 3., 7., 8., 9.]),
            1,
        )];
        let expected = vec![Elem::Val(tile(2, 1, vec![2., 8.]))];
        run_group(input, expected, RowStat::Mean, Some(3), 1);
    }

    // Three tiles in one reduction group: the count spans every tile's columns,
    // so the divisor is 3 tiles * 2 columns = 6, not the tile count.
    #[test]
    fn mean_static_reduces_across_tiles() {
        let input = vec![
            Elem::Val(tile(2, 2, vec![1., 2., 10., 20.])),
            Elem::Val(tile(2, 2, vec![3., 4., 30., 40.])),
            Elem::ValStop(tile(2, 2, vec![5., 6., 50., 60.]), 1),
        ];
        // row 0: (1+2+3+4+5+6)/6 = 3.5   row 1: (10+..+60)/6 = 35
        let expected = vec![Elem::Val(tile(2, 1, vec![3.5, 35.]))];
        run_group(input, expected, RowStat::Mean, Some(6), 1);
    }

    // MeanDyn counts as it goes and must agree with the declared-count variant.
    #[test]
    fn mean_dyn_matches_static() {
        let input = vec![
            Elem::Val(tile(2, 2, vec![1., 2., 10., 20.])),
            Elem::ValStop(tile(2, 2, vec![3., 4., 30., 40.]), 1),
        ];
        let expected = vec![Elem::Val(tile(2, 1, vec![2.5, 25.]))];
        run_group(input, expected, RowStat::Mean, None, 1);
    }

    // Population variance (divide by N, not N-1), matching native_layer_norm.
    // Row 0 = [1,2,3,4]: mean 2.5, var = (2.25+0.25+0.25+2.25)/4 = 1.25.
    // Row 1 = [2,4,6,8]: mean 5.0, var = (9+1+1+9)/4 = 5.
    #[test]
    fn var_static_is_population_variance() {
        let input = vec![Elem::ValStop(
            tile(2, 4, vec![1., 2., 3., 4., 2., 4., 6., 8.]),
            1,
        )];
        let expected = vec![Elem::Val(tile(2, 1, vec![1.25, 5.]))];
        run_group(input, expected, RowStat::Var, Some(4), 1);
    }

    #[test]
    fn var_dyn_matches_static() {
        let input = vec![
            Elem::Val(tile(1, 2, vec![1., 2.])),
            Elem::ValStop(tile(1, 2, vec![3., 4.]), 1),
        ];
        let expected = vec![Elem::Val(tile(1, 1, vec![1.25]))];
        run_group(input, expected, RowStat::Var, None, 1);
    }

    // A constant row must give exactly zero variance, never a small negative
    // that would turn into NaN under a downstream rsqrt.
    #[test]
    fn var_of_constant_row_is_exactly_zero() {
        let input = vec![Elem::ValStop(tile(1, 8, vec![1e4; 8]), 1)];
        let expected = vec![Elem::Val(tile(1, 1, vec![0.0]))];
        run_group(input, expected, RowStat::Var, Some(8), 1);
    }

    // A stop level above the operator's rank passes the remainder downstream,
    // and each inner group emits its own statistic.
    #[test]
    fn outer_stop_level_is_forwarded() {
        let input = vec![
            Elem::ValStop(tile(1, 2, vec![1., 3.]), 1),
            Elem::ValStop(tile(1, 2, vec![10., 30.]), 2),
        ];
        let expected = vec![
            Elem::Val(tile(1, 1, vec![2.])),
            Elem::ValStop(tile(1, 1, vec![20.]), 1),
        ];
        run_group(input, expected, RowStat::Mean, Some(2), 1);
    }

    // Timing-only tiles carry no data, so the group emits a blank [R, 1] tile
    // rather than fabricating a value.
    #[test]
    fn blank_tiles_produce_a_blank_output() {
        let input = vec![Elem::ValStop(
            Tile::<f32>::new_blank(vec![4, 16], BYTES, false),
            1,
        )];
        let expected = vec![Elem::Val(Tile::<f32>::new_blank(vec![4, 1], BYTES, false))];
        run_group(input, expected, RowStat::Mean, Some(16), 1);
    }

    // Padding lives in rows, and a row-wise reduction preserves rows, so the
    // input tile's offset must reach the output unchanged.
    #[test]
    fn offset_is_preserved() {
        let padded = Tile::new_padded(
            Array2::from_shape_vec((4, 2), vec![1.0f32; 8])
                .unwrap()
                .to_shared(),
            BYTES,
            false,
            2, // only the first 2 rows are real
        );
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();
        let input = vec![Elem::ValStop(padded, 1)];

        ctx.add_child(GeneratorContext::new(|| input.into_iter(), in_snd));
        ctx.add_child(AccumRowStat::<SimpleEvent>::new(
            in_rcv,
            out_snd,
            1,
            BYTES,
            AccumRowStatConfig {
                compute_bw: COMPUTE_BW,
                write_back_mu: false,
                stat: RowStat::Mean,
                count: Some(2),
            },
            0,
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || {
                vec![Elem::Val(Tile::new_padded(
                    Array2::from_shape_vec((4, 1), vec![1.0f32; 4])
                        .unwrap()
                        .to_shared(),
                    BYTES,
                    false,
                    2,
                ))]
                .into_iter()
            },
            out_rcv,
            |a: &Elem<Tile<f32>>, b: &Elem<Tile<f32>>| match (a, b) {
                (Elem::Val(x), Elem::Val(y)) => x.offset == y.offset && x.shape == y.shape,
                _ => false,
            },
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
