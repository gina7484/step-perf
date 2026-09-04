//! Rank-1 strided permutation with explicit per-lane group boundaries.
use dam::context_tools::*;
use crate::{primitives::elem::Elem, trace::TracingSender};

fn unpack<T>(elem: Elem<T>) -> (T, u32) {
    match elem { Elem::Val(x) => (x, 0), Elem::ValStop(x, s) => (x, s) }
}
fn pack<T>(x: T, stop: u32) -> Elem<T> {
    if stop == 0 { Elem::Val(x) } else { Elem::ValStop(x, stop) }
}

#[context_macro]
pub struct ContextDeinterleave<T: DAMType> {
    input: Receiver<Elem<T>>,
    outputs: Vec<TracingSender<Elem<T>>>,
}
impl<T: DAMType> ContextDeinterleave<T> where Elem<T>: DAMType {
    pub fn new(input: Receiver<Elem<T>>, outputs: Vec<TracingSender<Elem<T>>>) -> Self {
        assert!(outputs.len() >= 2);
        let ctx = Self { input, outputs, context_info: Default::default() };
        ctx.input.attach_receiver(&ctx);
        for out in &ctx.outputs { out.attach_sender(&ctx); }
        ctx
    }
}
impl<T: DAMType> Context for ContextDeinterleave<T> where Elem<T>: DAMType {
    fn run(&mut self) {
        let mut group_open = false;
        loop {
            let mut round = Vec::with_capacity(self.outputs.len());
            let mut stop = 0;
            for lane in 0..self.outputs.len() {
                match self.input.dequeue(&self.time) {
                    Ok(elem) => {
                        let (value, s) = unpack(elem.data);
                        assert!(s == 0 || lane + 1 == self.outputs.len(),
                                "ContextDeinterleave: group length is not divisible by lanes");
                        round.push(value);
                        stop = s;
                    }
                    Err(_) => {
                        assert!(lane == 0 && !group_open,
                                "ContextDeinterleave: incomplete round or unterminated group");
                        return;
                    }
                }
            }
            // Hold one round so even early lanes receive the canonical stop.
            for (output, value) in self.outputs.iter().zip(round) {
                output.enqueue(&self.time, ChannelElement {
                    time: self.time.tick(), data: pack(value, stop),
                }).unwrap();
            }
            group_open = stop == 0;
        }
    }
}

#[context_macro]
pub struct ContextInterleave<T: DAMType> {
    inputs: Vec<Receiver<Elem<T>>>,
    output: TracingSender<Elem<T>>,
}
impl<T: DAMType> ContextInterleave<T> where Elem<T>: DAMType {
    pub fn new(inputs: Vec<Receiver<Elem<T>>>, output: TracingSender<Elem<T>>) -> Self {
        assert!(inputs.len() >= 2);
        let ctx = Self { inputs, output, context_info: Default::default() };
        for input in &ctx.inputs { input.attach_receiver(&ctx); }
        ctx.output.attach_sender(&ctx);
        ctx
    }
}
impl<T: DAMType> Context for ContextInterleave<T> where Elem<T>: DAMType {
    fn run(&mut self) {
        let mut group_open = false;
        loop {
            let round: Vec<_> = self.inputs.iter().map(|x| x.dequeue(&self.time)).collect();
            if round.iter().all(|x| x.is_err()) {
                assert!(!group_open, "ContextInterleave: unterminated group");
                return;
            }
            assert!(round.iter().all(|x| x.is_ok()), "ContextInterleave: unequal lane closure");
            let round: Vec<_> = round.into_iter().map(|x| unpack(x.unwrap().data)).collect();
            let stop = round[0].1;
            assert!(round.iter().all(|x| x.1 == stop), "ContextInterleave: lane stops disagree");
            let lanes = round.len();
            for (lane, (value, _)) in round.into_iter().enumerate() {
                self.output.enqueue(&self.time, ChannelElement {
                    time: self.time.tick(), data: pack(value, if lane + 1 == lanes { stop } else { 0 }),
                }).unwrap();
            }
            group_open = stop == 0;
        }
    }
}
