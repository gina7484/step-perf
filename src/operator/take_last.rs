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
use crate::primitives::elem::{Elem, StopType};
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
        loop {
            match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: _, data }) => match data {
                    // Interior tile of the scanned axis: not the last, drop it.
                    Elem::Val(_x) => {}
                    // Closes the scanned axis => this IS the final tile.
                    Elem::ValStop(x, s) => {
                        let new_rank: StopType = s.saturating_sub(1);
                        let out = if new_rank == 0 {
                            Elem::Val(x)
                        } else {
                            Elem::ValStop(x, new_rank)
                        };
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
                    if std::env::var("STEP_PERF_OP_TRACE").is_ok() { eprintln!("[TAKELAST exit: input closed]"); }
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
