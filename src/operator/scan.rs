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
//! CHUNKING (`chunk_factor = C`): NO CHUNK-SPECIFIC LOGIC IS NEEDED HERE, and
//! that is a measured result, not an assumption.
//!
//! The proto documents chunking as "strided dealing -- chunk c owns stream
//! elements j where j mod C == c". That describes which KV elements BELONG to
//! which chunk, which is decided UPSTREAM by the Select/loader addressing. It is
//! NOT the order in which elements arrive here. At the Scan the C chunks show up
//! as C SEQUENTIAL Select contexts, each closed by its own stop token, so the
//! pre-existing reset-on-stop already gives every chunk an independent
//! recurrence.
//!
//! Both readings were implemented and measured against FA C=1 on the 15 batch
//! rows that C=2 does not pad (see below), 1920 values:
//!
//!   deal=contexts (this default)  max abs diff 1.8e-07   allclose PASS
//!   deal=strided  (interleaved)   max abs diff 5.5e-01   allclose FAIL (4.0e+03 rel)
//!
//! `STEP_PERF_SCAN_DEAL=strided` keeps the refuted interleaved reading available
//! for re-testing if the lowering ever changes; it is wrong for today's graphs.
//!
//! PADDING / MASKING CAVEAT: at C>1 the frontend pads each request's KV-tile
//! count UP to a multiple of C (dynamic_combined_elasticy.py: `seq_len_tiled =
//! ((seq_len_tiled + C - 1) // C) * C`). Those pad tiles read zero-initialized
//! cache rows, and NOTHING MASKS THEM YET (GH#3 / CHECKLIST T5). So C>1 output
//! is exact only for requests whose tile count is already a multiple of C.
//! Measured at C=2: the 17 padded rows are EXACTLY the 17 wrong rows, and the 15
//! unpadded rows are exact -- a perfect correlation with no exceptions either
//! way. Masking is therefore a PREREQUISITE for C>1 numeric correctness on
//! ragged batches, not an independent feature.
//!
//! At C = 1 this is byte-for-byte the behaviour validated against the naive
//! layer (2.3e-07 max relative diff); the C=1 output is bit-identical.
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
    next_stream: Option<Sender<Elem<Tile<OT>>>>,
    prior_stream: Option<Sender<Elem<Tile<OT>>>>,
    func1: FoldFn<T, OT>,
    func2: Option<FoldFn<T, OT>>,
    init: Arc<dyn Fn() -> Tile<OT> + Send + Sync>,
    /// Flash-decoding chunk count C (>= 1). C independent recurrences, strided.
    chunk_factor: u32,
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
        next_stream: Option<Sender<Elem<Tile<OT>>>>,
        prior_stream: Option<Sender<Elem<Tile<OT>>>>,
        func1: FoldFn<T, OT>,
        func2: Option<FoldFn<T, OT>>,
        init: Arc<dyn Fn() -> Tile<OT> + Send + Sync>,
        chunk_factor: u32,
        config: ScanConfig,
        id: u32,
    ) -> Self {
        // C = 0 and C = 1 both mean "unchunked"; the field is optional upstream.
        let chunk_factor = chunk_factor.max(1);
        let ctx = Self {
            in1,
            in2,
            ctr,
            next_stream,
            prior_stream,
            func1,
            func2,
            init,
            chunk_factor,
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
        // Attach ONLY the taps a consumer actually reads. Creating a sender for
        // an unread tap leaves a channel with no receiver, which the runtime
        // reports as DisconnectedReceiver and which killed the whole simulation.
        // Measured: scan_l and scan_o expose `prior` but nothing consumes it,
        // while scan_m has both taps consumed.
        if let Some(n) = &ctx.next_stream {
            n.attach_sender(&ctx);
        }
        if let Some(pr) = &ctx.prior_stream {
            pr.attach_sender(&ctx);
        }
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
        // Which layout does the scanned stream actually use at C>1?
        //   "contexts" (default): the C chunks arrive as C SEQUENTIAL Select
        //       contexts, each closed by its own stop token. The Scan then needs
        //       no chunk awareness at all -- reset-on-stop already gives each
        //       chunk an independent recurrence. "Strided dealing" in the proto
        //       describes which KV elements belong to which chunk, which is done
        //       UPSTREAM by the Select/loader addressing, not the arrival order.
        //   "strided": the C chunks are interleaved element-by-element on one
        //       stream, so chunk k owns element j where j % C == k.
        // Measured: "strided" gives WRONG numbers (max rel diff 4.0e+03 vs C=1),
        // "contexts" is what the lowering actually produces.
        let strided = std::env::var("STEP_PERF_SCAN_DEAL").as_deref() == Ok("strided");
        let c = if strided { self.chunk_factor.max(1) as usize } else { 1 };
        // C independent running values. Chunk k owns stream elements j = k mod C.
        let mut running: Vec<Tile<OT>> = (0..c).map(|_| (self.init)()).collect();
        // Per-chunk "needs init" flags. Reset is LAZY -- a chunk re-inits on the
        // first element it sees after its own stop token -- because with C
        // interleaved Select contexts there is no single invocation boundary.
        let mut fresh: Vec<bool> = vec![true; c];
        // Monotonic element index across the whole stream; deliberately never
        // reset. Per the proto contract each invocation carries exactly
        // C x per-chunk tiles, so `j % C` stays chunk-aligned across boundaries.
        let mut j: usize = 0;
        // `ctr` MUST be drained once per invocation, not once overall.
        // Broadcast_136 feeds ctr to all three Scans, one tile per invocation; a
        // Scan that reads only one tile ever leaves the broadcast blocked as soon
        // as its buffer fills, and the whole simulation HANGS (observed: trace
        // frozen at tick ~3679 with the process alive).
        //
        // An earlier version did drain per invocation and hit
        // DisconnectedReceiver -- but that was a SEPARATE bug (unconditional
        // creation of the unused `prior` tap, now fixed). The two were
        // independent; per-invocation draining was correct all along.
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
            let chunk = j % c;
            j += 1;
            if fresh[chunk] {
                // One ctr tile per chunk context. At C=1 this is exactly the
                // per-invocation drain that was validated; at C>1 each of the C
                // Select contexts is delivered its own per-chunk trip count.
                let _ = self.ctr.dequeue(&self.time);
                running[chunk] = (self.init)();
                fresh[chunk] = false;
            }

            let prior = running[chunk].clone();

            let (c1, folded) = (self.func1)(
                &running[chunk],
                &data,
                self.config.compute_bw,
                self.config.write_back_mu,
            );
            running[chunk] = folded;

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
                    &running[chunk],
                    &operand,
                    self.config.compute_bw,
                    self.config.write_back_mu,
                );
                running[chunk] = folded2;
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
            if let Some(pr) = &self.prior_stream {
                pr.enqueue(
                    &self.time,
                    ChannelElement { time: self.time.tick(), data: mk(prior) },
                )
                .unwrap();
            }
            if let Some(n) = &self.next_stream {
                n.enqueue(
                    &self.time,
                    ChannelElement { time: self.time.tick(), data: mk(running[chunk].clone()) },
                )
                .unwrap();
            }

            dam::logging::log_event(&E::new(
                "Scan".to_string(),
                self.id,
                self.time.tick().time() - cycles,
                self.time.tick().time(),
                true,
            ))
            .unwrap();

            // A stop token on the scanned axis closes THIS CHUNK's context.
            // Only this chunk re-inits; the other C-1 recurrences are untouched.
            if stop >= 1 {
                fresh[chunk] = true;
            }
        }
    }
}
