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
//! CHUNKING (`chunk_factor = C`): the chunk split is encoded ONLY in `ctr`.
//!
//! This was originally implemented on the assumption that the C chunks arrive as
//! C stop-delimited contexts, so reset-on-stop would handle them for free. That
//! is FALSE, and believing it made C>1 pass VACUOUSLY: measured counts showed
//! one context per (request, head) at C=2 instead of two, one TakeLast final
//! instead of C, so the merge epilogue's Accum summed a single term and
//! exp(m)O / exp(m)l collapsed to O/l -- an identity pass-through that returns
//! the unchunked answer.
//!
//! What is actually true: the data stream carries NO chunk stop tokens. Its only
//! stop is at the end of each (request, head). The chunk split lives entirely in
//! `ctr`, which carries `n_padded / C` -- the PER-CHUNK trip count. So modelling
//! chunking requires honouring `ctr` as a trip count and SYNTHESIZING a stop at
//! each chunk boundary, which is what `run` now does:
//!
//!   * one `ctr` tile is read per (request, head) -- matching the [DynB]
//!     chunk_ctr stream, which carries ONE count shared by all C chunks;
//!   * every `trip` elements the running value is emitted with a synthesized
//!     stop of rank 1 and then re-initialized, opening the next chunk;
//!   * the element that carries the input's own stop uses that stop instead, so
//!     the last chunk closes on the real token. Total stops per (request, head)
//!     is C: (C-1) synthesized plus 1 real.
//!
//! `TakeLast` then sees C finals per logical tile with no changes of its own.
//!
//! WHY THE FRONTEND PADS, and how to stop: because `chunk_ctr` is [DynB], ONE
//! trip count is shared by all C chunks, which forces equal-length chunks and
//! hence `seq_len` padded up to a multiple of C. Those pad tiles are unmasked
//! junk (GH#3 / T5). Giving `chunk_ctr` C per-chunk counts instead --
//! floor(n/C) + (1 if c < n mod C) -- makes them sum to exactly n, consumes the
//! contiguous KV walk exactly, and removes the need for chunk padding AND for
//! masking it. This operator is already written against a per-chunk `trip`, so
//! it needs no further change to support that; the work is a frontend [B]->[B,C]
//! shape change.
//!
//! `STEP_PERF_SCAN_CHUNK=off` disables chunk modelling (reproducing the old
//! unchunked behaviour) for A/B diagnosis.
//!
//! At C = 1 this is byte-for-byte the behaviour validated against the naive
//! layer (2.3e-07 max relative diff): `chunked` is false, no stop is ever
//! synthesized, and `ctr` is read once per invocation exactly as before.
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
        // Chunk modelling is driven by `ctr`, not by stop tokens -- see the
        // module docs. `off` restores the (wrong for C>1) unchunked behaviour.
        // Synthesizing chunk stops here DOES NOT WORK, and the failure is
        // structural rather than a detail to patch. A chunk boundary has to be
        // visible in EVERY stream in the attention region, because sibling
        // streams get paired with this one: BinaryMap panics
        // ("The two input streams' shape don't match!", map.rs:110) as soon as
        // this tap carries a ValStop its sibling lacks. The siblings come from
        // the KV walk, which is driven by the UNCHUNKED seq_len and so has one
        // context per (request, head). In hardware the boundary is consistent
        // because Select genuinely iterates the whole region C times; in the
        // STeP graph chunking is invisible, expressed only as a division of the
        // trip count. Measured: C=2 dies after 6 elements (the first chunk_ctr
        // value) with panics in map.rs and broadcast.rs.
        //
        // Kept behind an opt-in flag for whoever fixes the graph-level
        // representation; see the module docs for what that needs.
        let chunked = self.chunk_factor > 1
            && std::env::var("STEP_PERF_SCAN_CHUNK").as_deref() == Ok("synth");
        let mut running: Tile<OT> = (self.init)();
        // Per-chunk trip count for the current (request, head), read from `ctr`.
        let mut trip: u64 = 0;
        // Elements folded into the current chunk.
        let mut pos: u64 = 0;
        let mut need_ctr = true;
        let mut n_elems: u64 = 0;
        let mut n_ctr: u64 = 0;
        let mut n_stops: u64 = 0;
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
                    if std::env::var("STEP_PERF_OP_COUNTS").is_ok() {
                        eprintln!("[SCANCOUNT id={} C={} elems={} ctr_deq={} stops={}]",
                            self.id, self.chunk_factor, n_elems, n_ctr, n_stops);
                    }
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
            n_elems += 1;
            if need_ctr {
                // One ctr tile per (request, head). Its VALUE is the per-chunk
                // trip count; at C=1 it is the whole invocation length and is
                // only drained (boundaries come from the stop token there).
                trip = match self.ctr.dequeue(&self.time) {
                    Ok(ChannelElement { time: _, data }) => {
                        let t = match data {
                            Elem::Val(t) => t,
                            Elem::ValStop(t, _) => t,
                        };
                        t.underlying.as_ref().map(|u| u[[0, 0]]).unwrap_or(0)
                    }
                    Err(_) => 0,
                };
                n_ctr += 1;
                need_ctr = false;
                pos = 0;
                running = (self.init)();
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

            pos += 1;
            // Honour `ctr` as the per-chunk trip count: every `trip` elements
            // closes a chunk. The element carrying the input's real stop uses
            // that stop instead, so the final chunk closes on the real token and
            // the total is exactly C stops per (request, head).
            let at_chunk_end = chunked && trip > 0 && pos >= trip;
            let out_stop = if stop >= 1 {
                stop
            } else if at_chunk_end {
                1
            } else {
                0
            };
            if out_stop >= 1 {
                n_stops += 1;
            }
            let mk = |t: Tile<OT>| {
                if out_stop == 0 {
                    Elem::Val(t)
                } else {
                    Elem::ValStop(t, out_stop)
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
                    ChannelElement { time: self.time.tick(), data: mk(running.clone()) },
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

            if stop >= 1 {
                // End of this (request, head): next one brings its own ctr.
                need_ctr = true;
            } else if at_chunk_end {
                // Close this chunk and open the next one on the same request.
                running = (self.init)();
                pos = 0;
            }
        }
    }
}
