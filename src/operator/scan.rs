use std::{marker::PhantomData, sync::Arc};

use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

pub struct ScanConfig {
    pub compute_bw: u64,
    pub write_back_mu: bool,
    /// `true` emits the accumulator *after* folding the current input
    /// (`out[i] = x[0] op … op x[i]`); `false` emits it *before*
    /// (`out[0] = init`, `out[i] = x[0] op … op x[i-1]`).
    pub inclusive: bool,
}

/// One fold step: `(data, accumulator, flop/cycle, write_back_mu) -> (cycles, accumulator')`.
/// Same shape as the closures [`crate::operator::accum::Accum`] takes, so the
/// `accum_fn` library is reused verbatim.
type Fold<T, OT> = Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>;

/// Running scan over the innermost `rank` stream ranks.
///
/// Same fold as [`crate::operator::accum::Accum`], but the intermediate state is
/// emitted on *every* input rather than only at the end of a group. Two
/// consequences distinguish it from `Accum`:
///
/// - **The output keeps the input's shape.** `rank` says only when the
///   accumulator resets; it does not strip ranks off the output, so every
///   `Elem::ValStop(_, level)` is forwarded at the *same* `level` it arrived on
///   (`Accum` emits `level - rank`).
/// - **Every element is stored.** `write_back_mu` charges store cycles on each
///   input, not just at the group boundary.
///
/// With `fn2`/`in2_stream` present the state advances as
/// `acc = fn2(in2, fn1(in1, acc))`; the two folds are sequential because `fn2`
/// consumes `fn1`'s result, so their cycles add.
#[context_macro]
pub struct Scan<E, T: DAMType, OT: DAMType> {
    in1_stream: Receiver<Elem<Tile<T>>>,
    in2_stream: Option<Receiver<Elem<Tile<T>>>>,
    out_stream: Sender<Elem<Tile<OT>>>,
    fn1: Fold<T, OT>,
    fn2: Option<Fold<T, OT>>,
    init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
    rank: StopType,
    config: ScanConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Scan<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    pub fn new(
        in1_stream: Receiver<Elem<Tile<T>>>,
        in2_stream: Option<Receiver<Elem<Tile<T>>>>,
        out_stream: Sender<Elem<Tile<OT>>>,
        fn1: Fold<T, OT>,
        fn2: Option<Fold<T, OT>>,
        init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>,
        rank: StopType,
        config: ScanConfig,
        id: u32,
    ) -> Self {
        assert_eq!(
            in2_stream.is_some(),
            fn2.is_some(),
            "Scan_{}: input2 and fn2 must be supplied together",
            id
        );

        let ctx = Self {
            in1_stream,
            in2_stream,
            out_stream,
            fn1,
            fn2,
            init_accum,
            rank,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in1_stream.attach_receiver(&ctx);
        if let Some(in2_stream) = &ctx.in2_stream {
            in2_stream.attach_receiver(&ctx);
        }
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    /// Fold one input into the accumulator and return the tile to emit.
    ///
    /// `group_end` resets the accumulator afterwards. The fold still runs on
    /// that last input in exclusive mode even though the reset discards its
    /// result — the datapath folds every element unconditionally, so it is
    /// charged unconditionally too.
    fn process(
        &mut self,
        data1: Tile<T>,
        data2: Option<Tile<T>>,
        accumulator: &mut Tile<OT>,
        group_end: bool,
    ) -> Tile<OT> {
        let mut load_cycles = 0_u64;
        if data1.read_from_mu {
            load_cycles += div_ceil(data1.size_in_bytes() as u64, PMU_BW);
        }
        if let Some(data2) = &data2 {
            if data2.read_from_mu {
                load_cycles += div_ceil(data2.size_in_bytes() as u64, PMU_BW);
            }
        }

        // An exclusive scan emits the state as it was *before* this input.
        let pre_fold = if self.config.inclusive {
            None
        } else {
            Some(accumulator.clone())
        };

        let (mut comp_cycles, mut folded) = (self.fn1)(
            &data1,
            accumulator,
            self.config.compute_bw,
            self.config.write_back_mu,
        );
        if let (Some(fn2), Some(data2)) = (&self.fn2, &data2) {
            let (fn2_cycles, fn2_out) = fn2(
                data2,
                &folded,
                self.config.compute_bw,
                self.config.write_back_mu,
            );
            comp_cycles += fn2_cycles;
            folded = fn2_out;
        }

        let out_tile = pre_fold.unwrap_or_else(|| folded.clone());
        *accumulator = if group_end {
            (self.init_accum)()
        } else {
            folded
        };

        let store_cycles = if self.config.write_back_mu {
            div_ceil(out_tile.size_in_bytes() as u64, PMU_BW)
        } else {
            0
        };

        let roofline_cycles = [load_cycles, comp_cycles, store_cycles]
            .into_iter()
            .max()
            .unwrap_or(0);

        self.time.incr_cycles(roofline_cycles);

        self.in1_stream.dequeue(&self.time).unwrap();
        if let Some(in2_stream) = &self.in2_stream {
            in2_stream.dequeue(&self.time).unwrap();
        }

        dam::logging::log_event(&E::new(
            "Scan".to_string(),
            self.id,
            self.time.tick().time() - roofline_cycles,
            self.time.tick().time(),
            group_end,
        ))
        .unwrap();

        out_tile
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: DAMType,
        OT: DAMType,
    > Context for Scan<E, T, OT>
where
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
{
    fn run(&mut self) {
        let mut accumulator = (self.init_accum)();
        loop {
            // The two inputs advance in lockstep, so one outliving the other is a
            // malformed graph rather than a normal end-of-stream.
            let in1 = self.in1_stream.peek_next(&self.time);
            let in2 = self
                .in2_stream
                .as_ref()
                .map(|in2_stream| in2_stream.peek_next(&self.time));
            let (in1, in2) = match (in1, in2) {
                (Ok(ChannelElement { time: _, data: in1 }), None) => (in1, None),
                (
                    Ok(ChannelElement { time: _, data: in1 }),
                    Some(Ok(ChannelElement { time: _, data: in2 })),
                ) => (in1, Some(in2)),
                (Err(_), None) | (Err(_), Some(Err(_))) => return,
                _ => panic!("One stream closed earlier! Scan id: {}", self.id),
            };

            let (data1, level) = match in1 {
                Elem::Val(data1) => (data1, None),
                Elem::ValStop(data1, level) => (data1, Some(level)),
            };
            let data2 = match in2 {
                None => None,
                Some(Elem::Val(data2)) if level.is_none() => Some(data2),
                Some(Elem::ValStop(data2, level2)) if level == Some(level2) => Some(data2),
                Some(other) => panic!(
                    "The two input streams' shape don't match ({:?} != {:?})! Scan id: {}",
                    level, other, self.id
                ),
            };

            // `rank` gates only the accumulator reset: the output carries the
            // input's stop level through unchanged.
            let group_end = level.is_some_and(|level| level >= self.rank);
            let out_tile = self.process(data1, data2, &mut accumulator, group_end);

            let data = match level {
                None => Elem::Val(out_tile),
                Some(level) => Elem::ValStop(out_tile, level),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::accum_fn;
    use crate::utils::events::SimpleEvent;
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::GeneratorContext;
    use ndarray::Array2;
    use std::sync::Mutex;

    const ROWS: usize = 2;
    const COLS: usize = 2;
    const BYTES: usize = 4;
    const COMPUTE_BW: u64 = 1000;

    /// Collects everything the operator emits, so a test can assert on the whole
    /// output. A `CheckerContext` only walks the ground truth and would accept an
    /// output stream with extra elements on the end — for a scan, "one output per
    /// input" is the property under test, so the length has to be checked.
    #[context_macro]
    struct Collect {
        chan: Receiver<Elem<Tile<i32>>>,
        out: Arc<Mutex<Vec<Elem<Tile<i32>>>>>,
    }

    impl Collect {
        fn new(chan: Receiver<Elem<Tile<i32>>>, out: Arc<Mutex<Vec<Elem<Tile<i32>>>>>) -> Self {
            let ctx = Self {
                chan,
                out,
                context_info: Default::default(),
            };
            ctx.chan.attach_receiver(&ctx);
            ctx
        }
    }

    impl Context for Collect {
        fn run(&mut self) {
            loop {
                match self.chan.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => self.out.lock().unwrap().push(data),
                    Err(_) => return,
                }
                self.time.incr_cycles(1);
            }
        }
    }

    /// A `ROWS x COLS` tile filled with `v`.
    fn tile(v: i32) -> Tile<i32> {
        Tile::new(
            Array2::from_shape_vec((ROWS, COLS), vec![v; ROWS * COLS])
                .unwrap()
                .to_shared(),
            BYTES,
            false,
        )
    }

    /// `(fill value, stop level)` pairs -> a stream of tiles.
    fn stream(elems: &[(i32, Option<StopType>)]) -> Vec<Elem<Tile<i32>>> {
        elems
            .iter()
            .map(|(v, level)| match level {
                None => Elem::Val(tile(*v)),
                Some(level) => Elem::ValStop(tile(*v), *level),
            })
            .collect()
    }

    /// Drive `in1` (and `in2`, when given) through one `Scan` and return
    /// everything it emitted. `add` is used for both folds.
    fn run_scan(
        in1: Vec<Elem<Tile<i32>>>,
        in2: Option<Vec<Elem<Tile<i32>>>>,
        rank: StopType,
        inclusive: bool,
    ) -> Vec<Elem<Tile<i32>>> {
        let fn1: Fold<i32, i32> = Arc::new(|t1, t2, comp_bw, write_back_mu| {
            accum_fn::add(t1, t2, comp_bw, write_back_mu, 0)
        });
        let fn2: Option<Fold<i32, i32>> = in2.as_ref().map(|_| {
            let fold: Fold<i32, i32> = Arc::new(|t1, t2, comp_bw, write_back_mu| {
                accum_fn::add(t1, t2, comp_bw, write_back_mu, 0)
            });
            fold
        });

        let mut ctx = ProgramBuilder::default();
        let (in1_snd, in1_rcv) = ctx.unbounded();
        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(|| in1.into_iter(), in1_snd));

        let in2_rcv = match in2 {
            Some(in2) => {
                let (in2_snd, in2_rcv) = ctx.unbounded();
                ctx.add_child(GeneratorContext::new(|| in2.into_iter(), in2_snd));
                Some(in2_rcv)
            }
            None => None,
        };

        let collected = Arc::new(Mutex::new(Vec::new()));
        ctx.add_child(Scan::<SimpleEvent, _, _>::new(
            in1_rcv,
            in2_rcv,
            out_snd,
            fn1,
            fn2,
            Arc::new(|| Tile::new_zero([ROWS, COLS], BYTES, false)),
            rank,
            ScanConfig {
                compute_bw: COMPUTE_BW,
                write_back_mu: false,
                inclusive,
            },
            0,
        ));
        ctx.add_child(Collect::new(out_rcv, collected.clone()));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());

        let out = collected.lock().unwrap().clone();
        out
    }

    // One group of three: 1, 2, 3 -> running sums 1, 3, 6.
    #[test]
    fn test_inclusive_scan_emits_every_running_sum() {
        let out = run_scan(stream(&[(1, None), (2, None), (3, Some(1))]), None, 1, true);
        assert_eq!(out, stream(&[(1, None), (3, None), (6, Some(1))]));
    }

    // The same group, exclusive: the state *before* each input, so the `Zero`
    // init leads and the group total (6) is never emitted.
    #[test]
    fn test_exclusive_scan_emits_the_state_before_each_input() {
        let out = run_scan(
            stream(&[(1, None), (2, None), (3, Some(1))]),
            None,
            1,
            false,
        );
        assert_eq!(out, stream(&[(0, None), (1, None), (3, Some(1))]));
    }

    // rank=1 resets at every level-1 stop, so the second group starts from Zero
    // again. The output keeps the input's stop levels (1 then 2) — unlike Accum,
    // Scan does not consume ranks.
    #[test]
    fn test_scan_restarts_at_each_group() {
        let input = stream(&[
            (1, None),
            (2, None),
            (3, Some(1)),
            (4, None),
            (5, None),
            (6, Some(2)),
        ]);
        let out = run_scan(input, None, 1, true);
        assert_eq!(
            out,
            stream(&[
                (1, None),
                (3, None),
                (6, Some(1)),
                (4, None),
                (9, None),
                (15, Some(2)),
            ])
        );
    }

    // rank=2 spans both inner ranks: the level-1 stop is forwarded but does not
    // reset, so the running sum carries into the second group (6 + 4 = 10).
    #[test]
    fn test_rank2_scan_carries_across_the_inner_stop() {
        let input = stream(&[
            (1, None),
            (2, None),
            (3, Some(1)),
            (4, None),
            (5, None),
            (6, Some(2)),
        ]);
        let out = run_scan(input, None, 2, true);
        assert_eq!(
            out,
            stream(&[
                (1, None),
                (3, None),
                (6, Some(1)),
                (10, None),
                (15, None),
                (21, Some(2)),
            ])
        );
    }

    // Exclusive + reset: each group's first output is the `Zero` init again.
    #[test]
    fn test_exclusive_scan_restarts_at_each_group() {
        let input = stream(&[
            (1, None),
            (2, None),
            (3, Some(1)),
            (4, None),
            (5, None),
            (6, Some(2)),
        ]);
        let out = run_scan(input, None, 1, false);
        assert_eq!(
            out,
            stream(&[
                (0, None),
                (1, None),
                (3, Some(1)),
                (0, None),
                (4, None),
                (9, Some(2)),
            ])
        );
    }

    // `acc = fn2(in2, fn1(in1, acc))`, both folds `add`:
    //   0 -> 10+(0+1)=11 -> 20+(11+2)=33 -> 30+(33+3)=66
    #[test]
    fn test_two_inputs_chain_both_folds() {
        let out = run_scan(
            stream(&[(1, None), (2, None), (3, Some(1))]),
            Some(stream(&[(10, None), (20, None), (30, Some(1))])),
            1,
            true,
        );
        assert_eq!(out, stream(&[(11, None), (33, None), (66, Some(1))]));
    }
}
