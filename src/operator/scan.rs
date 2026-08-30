//! `Scan`: prefix-scan loop with a loop-carried value and TWO output taps.
//!
//! Folds `in1` (and, optionally, a second per-step operand `in2`) into a running
//! value via `func1` (then `func2`), emitting one result per step. Exposes both
//! taps the hardware lowering does:
//!   * `next_stream`  (stream_idx 0) -- the value AFTER folding this step
//!   * `prior_stream` (stream_idx 1) -- the value BEFORE folding this step
//! Both are always fully populated and position-aligned, one real tile per step.
//!
//! INVOCATION BOUNDARIES COME FROM THE STOP TOKENS, NOT FROM `ctr`.
//! The hardware `Select` needs `ctr` delivered up front to know an invocation's
//! trip count before the data arrives, but for FUNCTIONAL evaluation the input's
//! own stop token already says where the invocation ends: an element arriving as
//! `ValStop(x, s>=1)` closes the scanned axis, so it is the last step and the
//! running value resets afterwards. Driving off the tokens avoids having to
//! reproduce the (step, sub) interleave contract, and keeps the two taps exactly
//! aligned with the input stream. `ctr` is still DEQUEUED once per invocation so
//! its producer does not block forever.
//!
//! LIMITATION -- chunk_factor: with `chunk_factor = C > 1` the scanned axis is
//! split into C independent recurrences by strided dealing (chunk c owns stream
//! elements j where j mod C == c), each becoming its own Select context. This
//! implementation carries ONE running value, i.e. C = 1. Under C > 1 it would
//! fold all chunks into a single recurrence, which is WRONG. The constructor
//! asserts C <= 1 rather than silently producing bad numbers.
use std::{marker::PhantomData, sync::Arc};

use crate::primitives::elem::Elem;
use crate::primitives::tile::Tile;
use crate::trace::TracingSender as Sender;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};

pub struct ScanConfig {
    pub compute_bw: u64,
    pub write_back_mu: bool,
}

type FoldFn<T, OT> =
    Arc<dyn Fn(&Tile<OT>, &Tile<T>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>;

#[context_macro]
pub struct Scan<E, T: DAMType, OT: DAMType> {
    in1: Receiver<Elem<Tile<T>>>,
    in2: Option<Receiver<Elem<Tile<T>>>>,
    ctr: Receiver<Elem<Tile<u64>>>,
    next_stream: Sender<Elem<Tile<OT>>>,
    prior_stream: Sender<Elem<Tile<OT>>>,
    func1: FoldFn<T, OT>,
    func2: Option<FoldFn<T, OT>>,
    init: Arc<dyn Fn() -> Tile<OT> + Send + Sync>,
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
    Elem<Tile<u64>>: DAMType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in1: Receiver<Elem<Tile<T>>>,
        in2: Option<Receiver<Elem<Tile<T>>>>,
        ctr: Receiver<Elem<Tile<u64>>>,
        next_stream: Sender<Elem<Tile<OT>>>,
        prior_stream: Sender<Elem<Tile<OT>>>,
        func1: FoldFn<T, OT>,
        func2: Option<FoldFn<T, OT>>,
        init: Arc<dyn Fn() -> Tile<OT> + Send + Sync>,
        chunk_factor: u32,
        config: ScanConfig,
        id: u32,
    ) -> Self {
        assert!(
            chunk_factor <= 1,
            "Scan (step_perf functional): chunk_factor={} unsupported. C>1 needs C \
             independent running values dealt strided across the scanned axis; this \
             carries one, so it would silently fold all chunks together. Implement the \
             strided deal before enabling C>1.",
            chunk_factor
        );
        let ctx = Self {
            in1,
            in2,
            ctr,
            next_stream,
            prior_stream,
            func1,
            func2,
            init,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in1.attach_receiver(&ctx);
        if let Some(i2) = &ctx.in2 {
            i2.attach_receiver(&ctx);
        }
        ctx.ctr.attach_receiver(&ctx);
        ctx.next_stream.attach_sender(&ctx);
        ctx.prior_stream.attach_sender(&ctx);
        ctx
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
    Elem<Tile<u64>>: DAMType,
{
    fn run(&mut self) {
        let mut running: Tile<OT> = (self.init)();
        let mut new_invocation = true;
        // Drain one ctr tile up front so its producer is not blocked from the
        // start. Errors are ignored: ctr is not used for semantics here.
        let _ = self.ctr.dequeue(&self.time);
        loop {
            let (data, stop) = match self.in1.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    Elem::Val(x) => (x, 0u32),
                    Elem::ValStop(x, s) => (x, s),
                },
                Err(_) => {
                    if std::env::var("STEP_PERF_OP_TRACE").is_ok() { eprintln!("[SCAN {} exit: in1 closed]", self.id); }
                    return;
                }
            };

            // Reset the running value at an invocation boundary.
            //
            // NOTE on `ctr`: an earlier version dequeued one ctr tile per
            // invocation here. That over-read the channel -- invocation count
            // derived from stop tokens does not match the ctr tile count -- and
            // produced `DisconnectedReceiver`. `ctr` carries no information this
            // functional model needs (boundaries come from stop tokens), so it is
            // drained separately and defensively below instead.
            if new_invocation {
                running = (self.init)();
                new_invocation = false;
            }

            let prior = running.clone();

            let (c1, folded) = (self.func1)(
                &running,
                &data,
                self.config.compute_bw,
                self.config.write_back_mu,
            );
            running = folded;

            let mut cycles = c1;
            if let (Some(f2), Some(i2)) = (&self.func2, &self.in2) {
                let operand = match i2.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => match data {
                        Elem::Val(x) => x,
                        Elem::ValStop(x, _) => x,
                    },
                    Err(_) => {
                        if std::env::var("STEP_PERF_OP_TRACE").is_ok() { eprintln!("[SCAN {} exit: in2 closed EARLY]", self.id); }
                        return;
                    }
                };
                let (c2, folded2) = (f2)(
                    &running,
                    &operand,
                    self.config.compute_bw,
                    self.config.write_back_mu,
                );
                running = folded2;
                cycles = cycles.max(c2);
            }

            self.time.incr_cycles(cycles);

            let mk = |t: Tile<OT>| {
                if stop == 0 {
                    Elem::Val(t)
                } else {
                    Elem::ValStop(t, stop)
                }
            };
            // The `prior` tap may have NO consumer: the hardware lowering always
            // produces both taps, but a given STeP graph often reads only
            // `next` (stream_idx 0). Enqueueing to a receiverless channel errors,
            // so this must not unwrap -- doing so panicked the whole simulation
            // with `DisconnectedReceiver`. Dropping an unread tap is correct.
            let _ = self.prior_stream.enqueue(
                &self.time,
                ChannelElement { time: self.time.tick(), data: mk(prior) },
            );
            let _ = self.next_stream.enqueue(
                &self.time,
                ChannelElement { time: self.time.tick(), data: mk(running.clone()) },
            );

            dam::logging::log_event(&E::new(
                "Scan".to_string(),
                self.id,
                self.time.tick().time() - cycles,
                self.time.tick().time(),
                true,
            ))
            .unwrap();

            // A stop token on the scanned axis closes this invocation.
            if stop >= 1 {
                new_invocation = true;
            }
        }
    }
}
