use dam::{simulation::ProgramBuilder,
          utility_contexts::{GeneratorContext, ApproxCheckerContext}};
use step_perf::{operator::context_lanes::{ContextDeinterleave, ContextInterleave},
                primitives::elem::Elem, trace::TracingSender};

#[test]
fn strided_lanes_each_receive_the_original_group_stop() {
    let mut b = ProgramBuilder::default();
    let (tx, rx) = b.unbounded();
    b.add_child(GeneratorContext::new(|| vec![Elem::Val(0u32), Elem::Val(1),
        Elem::Val(2), Elem::ValStop(3, 1), Elem::Val(10), Elem::ValStop(11, 3)].into_iter(), tx));
    let mut outputs = Vec::new();
    for lane in 0..2 {
        let (tx, rx) = b.unbounded();
        outputs.push(TracingSender::wrap(tx, 1, lane));
        b.add_child(ApproxCheckerContext::new(move || vec![Elem::Val(lane),
            Elem::ValStop(2 + lane, 1), Elem::ValStop(10 + lane, 3)].into_iter(),
            rx, |a, b| a == b));
    }
    b.add_child(ContextDeinterleave::new(rx, outputs));
    b.initialize(Default::default()).unwrap().run(Default::default());
}

#[test]
fn split_merge_roundtrip_preserves_values_and_stops() {
    for lanes in [2, 4, 8] {
        let expected: Vec<_> = (0..32u32).map(|x| {
            if x == 15 { Elem::ValStop(x, 1) }
            else if x == 31 { Elem::ValStop(x, 3) } else { Elem::Val(x) }
        }).collect();
        let mut b = ProgramBuilder::default();
        let (tx, rx) = b.unbounded();
        let input = expected.clone();
        b.add_child(GeneratorContext::new(move || input.into_iter(), tx));
        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        for lane in 0..lanes {
            let (tx, rx) = b.unbounded();
            outputs.push(TracingSender::wrap(tx, 1, lane));
            inputs.push(rx);
        }
        b.add_child(ContextDeinterleave::new(rx, outputs));
        let (tx, rx) = b.unbounded();
        b.add_child(ContextInterleave::new(inputs, TracingSender::wrap(tx, 2, 0)));
        b.add_child(ApproxCheckerContext::new(move || expected.into_iter(), rx, |a,b| a == b));
        b.initialize(Default::default()).unwrap().run(Default::default());
    }
}
