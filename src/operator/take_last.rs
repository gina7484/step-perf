//! `TakeLast`: discard every real tile on a stream but the last, then close.
//!
//! The natural complement to `Scan`'s end-token design -- reads only the final
//! value a Scan invocation produces (FlashAttention's `l_final` / `O_final`,
//! used after the whole KV loop finishes).
//!
//! Semantics in the `Elem` model: the innermost sub-stream is delimited by a
//! `ValStop(x, s)` with `s >= 1`. Every `Val(x)` before it is an interior tile
//! and is DROPPED; the `ValStop` element is the one the loop already knows is
//! final, so it is forwarded with its rank decremented by one (the scanned axis
//! it closed is collapsed by this operator). A rank that decrements to 0 becomes
//! a plain `Val`.
//!
//! LIMITATION: this implements `keep_last = 1`, the unchunked case. With
//! `chunk_factor = C > 1` the STeP node declares `keep_last = C` because C
//! physical tiles (one per chunk recurrence) survive per logical tile, and the C
//! chunk contexts are interleaved on the stream. Handling that needs the
//! interleave pattern, which this does not yet model -- see the header note in
//! `proto_driver`'s TakeLast arm.
use crate::primitives::elem::Elem;
use crate::trace::TracingSender as Sender;
use dam::context_tools::*;

#[context_macro]
pub struct TakeLast<T: DAMType> {
    in_stream: Receiver<Elem<T>>,
    out_stream: Sender<Elem<T>>,
}

impl<T: DAMType> TakeLast<T>
where
    Self: Context,
{
    pub fn new(in_stream: Receiver<Elem<T>>, out_stream: Sender<Elem<T>>) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl<T: DAMType> Context for TakeLast<T> {
    fn run(&mut self) {
        let mut n_in: u64 = 0;
        let mut n_out: u64 = 0;
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    // Interior tile of the scanned axis: not the last, drop it.
                    Elem::Val(_x) => { n_in += 1; }
                    // Closes the scanned axis => this IS the final tile.
                    Elem::ValStop(x, s) => {
                        n_in += 1; n_out += 1;
                        // Forward the stop token UNCHANGED. The STeP node's own
                        // docstring says the scanned axis is collapsed while
                        // "keeping the same rank as the input" -- its stream shape
                        // is `shape[:-1] + (keep_last,)`, i.e. the last axis is
                        // REPLACED (extent keep_last) rather than removed. An
                        // earlier version decremented the rank, which corrupted
                        // every downstream Flatten/Map (observed as panics in
                        // flatten.rs and map.rs, and the final OffChipStore
                        // receiving 0 of its 4096 expected elements).
                        let out = Elem::ValStop(x, s);
                        self.out_stream
                            .enqueue(
                                &self.time,
                                ChannelElement {
                                    time: self.time.tick(),
                                    data: out,
                                },
                            )
                            .unwrap();
                    }
                },
                Err(_) => {
                    if std::env::var("STEP_PERF_OP_COUNTS").is_ok() {
                        eprintln!("[TAKELASTCOUNT in={} out={}]", n_in, n_out);
                    }
                    return;
                }
            }
        }
    }
}

// NOTE: no #[cfg(test)] block here -- this crate's TEST profile does not
// compile (pre-existing errors in streamify.rs, static_reassemble.rs and an
// unresolved dam::simulation::MongoOptionsBuilder import), so a unit test added
// here cannot be run without first repairing the whole test suite.
