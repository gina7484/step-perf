//! Gina's rank-delimited prefix scan.

use std::{marker::PhantomData, sync::Arc};

use dam::{context_tools::*, logging::LogEvent};

use crate::{
    memory::PMU_BW,
    primitives::{
        elem::{Bufferizable, Elem, StopType},
        tile::Tile,
    },
    trace::TracingSender,
    utils::{calculation::div_ceil, events::LoggableEventSimple},
};

pub struct ScanConfig {
    pub compute_bw: u64,
    pub write_back_mu: bool,
    pub inclusive: bool,
}

type Fold<T, OT> = Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>;

#[context_macro]
pub struct GinaScan<E, T: DAMType, OT: DAMType> {
    in1: Receiver<Elem<Tile<T>>>,
    in2: Option<Receiver<Elem<Tile<T>>>>,
    out: TracingSender<Elem<Tile<OT>>>,
    paired_out: Option<TracingSender<Elem<Tile<OT>>>>,
    ctr: Option<Receiver<Elem<Tile<u64>>>>,
    fn1: Fold<T, OT>,
    fn2: Option<Fold<T, OT>>,
    init: Arc<dyn Fn() -> Tile<OT> + Send + Sync>,
    rank: StopType,
    config: ScanConfig,
    id: u32,
    _event: PhantomData<E>,
}

impl<E, T: DAMType, OT: DAMType> GinaScan<E, T, OT>
where
    E: LoggableEventSimple + LogEvent + Sync + Send,
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
    Elem<Tile<u64>>: DAMType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in1: Receiver<Elem<Tile<T>>>,
        in2: Option<Receiver<Elem<Tile<T>>>>,
        out: TracingSender<Elem<Tile<OT>>>,
        paired_out: Option<TracingSender<Elem<Tile<OT>>>>,
        ctr: Option<Receiver<Elem<Tile<u64>>>>,
        fn1: Fold<T, OT>,
        fn2: Option<Fold<T, OT>>,
        init: Arc<dyn Fn() -> Tile<OT> + Send + Sync>,
        rank: StopType,
        config: ScanConfig,
        id: u32,
    ) -> Self {
        assert_eq!(
            in2.is_some(),
            fn2.is_some(),
            "Scan_{id}: input2 and fn2 must be supplied together"
        );
        assert!(rank > 0, "Scan_{id}: rank must be positive");
        let scan = Self {
            in1,
            in2,
            out,
            paired_out,
            ctr,
            fn1,
            fn2,
            init,
            rank,
            config,
            id,
            context_info: Default::default(),
            _event: PhantomData,
        };
        scan.in1.attach_receiver(&scan);
        if let Some(input) = &scan.in2 {
            input.attach_receiver(&scan);
        }
        if let Some(ctr) = &scan.ctr {
            ctr.attach_receiver(&scan);
        }
        scan.out.attach_sender(&scan);
        if let Some(output) = &scan.paired_out {
            output.attach_sender(&scan);
        }
        scan
    }
}

impl<E, T: DAMType, OT: DAMType> Context for GinaScan<E, T, OT>
where
    E: LoggableEventSimple + LogEvent + Sync + Send,
    Elem<Tile<T>>: DAMType,
    Elem<Tile<OT>>: DAMType,
    Elem<Tile<u64>>: DAMType,
{
    fn run(&mut self) {
        let mut accumulator = (self.init)();
        let mut group_open = false;
        let mut expected_steps = None;
        let mut observed_steps = 0_u64;
        loop {
            let first = self.in1.peek_next(&self.time);
            let second = self.in2.as_ref().map(|input| input.peek_next(&self.time));
            let (first, second) = match (first, second) {
                (Ok(first), None) => (first.data, None),
                (Ok(first), Some(Ok(second))) => (first.data, Some(second.data)),
                (Err(_), None) | (Err(_), Some(Err(_))) => return,
                _ => panic!(
                    "Scan_{}: paired streams closed at different positions",
                    self.id
                ),
            };

            if !group_open {
                if let Some(ctr) = &self.ctr {
                    let count = ctr.dequeue(&self.time).unwrap_or_else(|_| {
                        panic!("Scan_{}: missing trip count for group", self.id)
                    });
                    let tile = match count.data {
                        Elem::Val(tile) | Elem::ValStop(tile, _) => tile,
                    };
                    // Performance-only streams contain blank tiles. When the
                    // payload is present, make the lowering-only counter prove
                    // the same boundary as the logical stop stream.
                    expected_steps = tile
                        .underlying
                        .as_ref()
                        .and_then(|values| values.iter().next())
                        .copied();
                }
                observed_steps = 0;
                group_open = true;
            }

            let (data1, stop) = match first {
                Elem::Val(data) => (data, None),
                Elem::ValStop(data, level) => (data, Some(level)),
            };
            let data2 = match second {
                None => None,
                Some(Elem::Val(data)) if stop.is_none() => Some(data),
                Some(Elem::ValStop(data, level)) if stop == Some(level) => Some(data),
                Some(other) => panic!(
                    "Scan_{}: paired stop mismatch: {:?} != {:?}",
                    self.id, stop, other
                ),
            };

            let mut load_cycles = if data1.read_from_mu {
                div_ceil(data1.size_in_bytes() as u64, PMU_BW)
            } else {
                0
            };
            if let Some(data) = &data2 {
                if data.read_from_mu {
                    load_cycles += div_ceil(data.size_in_bytes() as u64, PMU_BW);
                }
            }

            let prior = accumulator.clone();
            let (mut compute_cycles, mut next) = (self.fn1)(
                &data1,
                &accumulator,
                self.config.compute_bw,
                self.config.write_back_mu,
            );
            if let (Some(function), Some(data)) = (&self.fn2, &data2) {
                let (cycles, value) = function(
                    data,
                    &next,
                    self.config.compute_bw,
                    self.config.write_back_mu,
                );
                compute_cycles += cycles;
                next = value;
            }

            let group_end = stop.is_some_and(|level| level >= self.rank);
            observed_steps += 1;
            if let Some(expected) = expected_steps {
                assert!(
                    (group_end && observed_steps == expected)
                        || (!group_end && observed_steps < expected),
                    "Scan_{}: counter says {} steps, but the rank-{} stop stream {} after step {}",
                    self.id,
                    expected,
                    self.rank,
                    if group_end { "ended" } else { "continued" },
                    observed_steps,
                );
            }
            let canonical = if self.paired_out.is_some() || self.config.inclusive {
                next.clone()
            } else {
                prior.clone()
            };
            let store_cycles = if self.config.write_back_mu {
                div_ceil(canonical.size_in_bytes() as u64, PMU_BW)
            } else {
                0
            };
            let cycles = load_cycles.max(compute_cycles).max(store_cycles);
            self.time.incr_cycles(cycles);
            self.in1.dequeue(&self.time).unwrap();
            if let Some(input) = &self.in2 {
                input.dequeue(&self.time).unwrap();
            }

            let attach_stop = |value| match stop {
                Some(level) => Elem::ValStop(value, level),
                None => Elem::Val(value),
            };
            self.out
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: attach_stop(canonical),
                    },
                )
                .unwrap();
            if let Some(output) = &self.paired_out {
                output
                    .enqueue(
                        &self.time,
                        ChannelElement {
                            time: self.time.tick(),
                            data: attach_stop(prior),
                        },
                    )
                    .unwrap();
            }

            dam::logging::log_event(&E::new(
                "Scan".to_string(),
                self.id,
                self.time.tick().time() - cycles,
                self.time.tick().time(),
                group_end,
            ))
            .unwrap();

            if group_end {
                accumulator = (self.init)();
                group_open = false;
                expected_steps = None;
            } else {
                accumulator = next;
            }
        }
    }
}
