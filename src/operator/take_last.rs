//! Retain the last C tiles of each innermost stream, preserving its boundary.
use std::collections::VecDeque;
use crate::primitives::elem::Elem;
use crate::trace::TracingSender as Sender;
use dam::context_tools::*;

#[context_macro]
pub struct TakeLast<T: DAMType> {
    in_stream: Receiver<Elem<T>>,
    out_stream: Sender<Elem<T>>,
    keep: usize,
}

impl<T: DAMType> TakeLast<T>
where
    Self: Context,
{
    pub fn new(in_stream: Receiver<Elem<T>>, out_stream: Sender<Elem<T>>) -> Self {
        Self::new_keep(in_stream, out_stream, 1)
    }

    pub fn new_keep(in_stream: Receiver<Elem<T>>, out_stream: Sender<Elem<T>>, keep: usize) -> Self {
        assert!(keep > 0);
        let ctx = Self {
            keep,
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
        let mut pending = VecDeque::new();
        let mut n_in = 0;
        let mut n_out = 0;
        while let Ok(ChannelElement { data, .. }) = self.in_stream.dequeue(&self.time) {
            let (value, stop) = match data {
                Elem::Val(value) => (value, 0),
                Elem::ValStop(value, stop) => (value, stop),
            };
            n_in += 1;
            pending.push_back(value);
            if pending.len() > self.keep { pending.pop_front(); }
            if stop > 0 {
                assert_eq!(pending.len(), self.keep, "TakeLast stream shorter than keep count");
                while let Some(value) = pending.pop_front() {
                    let data = if pending.is_empty() { Elem::ValStop(value, stop) } else { Elem::Val(value) };
                    self.out_stream.enqueue(&self.time, ChannelElement {time: self.time.tick(), data}).unwrap();
                    n_out += 1;
                }
            }
        }
        assert!(pending.is_empty(), "TakeLast input ended without a stream boundary");
        if std::env::var("STEP_PERF_OP_COUNTS").is_ok() {
            eprintln!("[TAKELASTCOUNT keep={} in={} out={}]", self.keep, n_in, n_out);
        }
    }
}

// NOTE: no #[cfg(test)] block here -- this crate's TEST profile does not
// compile (pre-existing errors in streamify.rs, static_reassemble.rs and an
// unresolved dam::simulation::MongoOptionsBuilder import), so a unit test added
// here cannot be run without first repairing the whole test suite.
