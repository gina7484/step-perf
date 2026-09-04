use std::sync::{Arc, Mutex};

use dam::{
    channel::Receiver, context_tools::*, simulation::ProgramBuilder,
    utility_contexts::GeneratorContext,
};
use ndarray::Array2;
use step_perf::{
    functions::accum_fn,
    operator::gina_scan::{GinaScan, ScanConfig},
    primitives::{elem::Elem, tile::Tile},
    trace::TracingSender,
    utils::events::SimpleEvent,
};

#[context_macro]
struct Collect {
    input: Receiver<Elem<Tile<i32>>>,
    values: Arc<Mutex<Vec<Elem<Tile<i32>>>>>,
}

impl Context for Collect {
    fn run(&mut self) {
        while let Ok(elem) = self.input.dequeue(&self.time) {
            self.values.lock().unwrap().push(elem.data);
        }
    }
}

fn tile(value: i32) -> Tile<i32> {
    Tile::new(Array2::from_elem((1, 1), value).to_shared(), 4, false)
}

fn run_scan(input: Vec<Elem<Tile<i32>>>, rank: u32, inclusive: bool) -> Vec<Elem<Tile<i32>>> {
    let mut builder = ProgramBuilder::default();
    let (input_tx, input_rx) = builder.unbounded();
    let (output_tx, output_rx) = builder.unbounded();
    builder.add_child(GeneratorContext::new(|| input.into_iter(), input_tx));

    let fold = Arc::new(|data: &Tile<i32>, accum: &Tile<i32>, bw, wb| {
        accum_fn::add(data, accum, bw, wb, 9)
    });
    builder.add_child(GinaScan::<SimpleEvent, _, _>::new(
        input_rx,
        None,
        TracingSender::wrap(output_tx, 7, 0),
        None,
        None,
        fold,
        None,
        Arc::new(|| Tile::new_zero([1, 1], 4, false)),
        rank,
        ScanConfig {
            compute_bw: 1,
            write_back_mu: false,
            inclusive,
        },
        7,
    ));

    let values = Arc::new(Mutex::new(Vec::new()));
    let collect = Collect {
        input: output_rx,
        values: values.clone(),
        context_info: Default::default(),
    };
    collect.input.attach_receiver(&collect);
    builder.add_child(collect);
    builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());
    Arc::try_unwrap(values).unwrap().into_inner().unwrap()
}

#[test]
fn gina_scan_inclusive_and_rank_reset_semantics() {
    let output = run_scan(
        vec![
            Elem::Val(tile(1)),
            Elem::ValStop(tile(2), 1),
            Elem::Val(tile(3)),
            Elem::ValStop(tile(4), 2),
        ],
        2,
        true,
    );
    assert_eq!(
        output,
        vec![
            Elem::Val(tile(1)),
            Elem::ValStop(tile(3), 1),
            Elem::Val(tile(6)),
            Elem::ValStop(tile(10), 2),
        ]
    );
}

#[test]
fn gina_scan_exclusive_emits_prior_state() {
    let output = run_scan(
        vec![
            Elem::Val(tile(1)),
            Elem::Val(tile(2)),
            Elem::ValStop(tile(3), 1),
        ],
        1,
        false,
    );
    assert_eq!(
        output,
        vec![
            Elem::Val(tile(0)),
            Elem::Val(tile(1)),
            Elem::ValStop(tile(3), 1)
        ]
    );
}

#[test]
fn last_returns_data_not_accumulator() {
    let (cycles, output) = accum_fn::last(&tile(7), &tile(99), 1, false, 5);
    assert_eq!(cycles, 0);
    assert_eq!(output, tile(7));
}

#[test]
fn gina_scan_two_inputs_apply_fn1_then_fn2() {
    let mut builder = ProgramBuilder::default();
    let (in1_tx, in1_rx) = builder.unbounded();
    let (in2_tx, in2_rx) = builder.unbounded();
    let (output_tx, output_rx) = builder.unbounded();
    builder.add_child(GeneratorContext::new(
        || vec![Elem::Val(tile(1)), Elem::ValStop(tile(2), 1)].into_iter(),
        in1_tx,
    ));
    builder.add_child(GeneratorContext::new(
        || vec![Elem::Val(tile(10)), Elem::ValStop(tile(20), 1)].into_iter(),
        in2_tx,
    ));
    let add1 = Arc::new(|data: &Tile<i32>, accum: &Tile<i32>, bw, wb| {
        accum_fn::add(data, accum, bw, wb, 9)
    });
    let add2 = Arc::new(|data: &Tile<i32>, accum: &Tile<i32>, bw, wb| {
        accum_fn::add(data, accum, bw, wb, 9)
    });
    builder.add_child(GinaScan::<SimpleEvent, _, _>::new(
        in1_rx,
        Some(in2_rx),
        TracingSender::wrap(output_tx, 7, 0),
        None,
        None,
        add1,
        Some(add2),
        Arc::new(|| Tile::new_zero([1, 1], 4, false)),
        1,
        ScanConfig {
            compute_bw: 1,
            write_back_mu: false,
            inclusive: true,
        },
        7,
    ));
    let values = Arc::new(Mutex::new(Vec::new()));
    let collect = Collect {
        input: output_rx,
        values: values.clone(),
        context_info: Default::default(),
    };
    collect.input.attach_receiver(&collect);
    builder.add_child(collect);
    builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());

    assert_eq!(
        Arc::try_unwrap(values).unwrap().into_inner().unwrap(),
        vec![Elem::Val(tile(11)), Elem::ValStop(tile(33), 1)]
    );
}

#[test]
fn gina_scan_paired_outputs_are_position_aligned_next_and_prior() {
    let mut builder = ProgramBuilder::default();
    let (input_tx, input_rx) = builder.unbounded();
    let (next_tx, next_rx) = builder.unbounded();
    let (prior_tx, prior_rx) = builder.unbounded();
    builder.add_child(GeneratorContext::new(
        || vec![Elem::Val(tile(1)), Elem::ValStop(tile(2), 1)].into_iter(),
        input_tx,
    ));
    let fold = Arc::new(|data: &Tile<i32>, accum: &Tile<i32>, bw, wb| {
        accum_fn::add(data, accum, bw, wb, 9)
    });
    builder.add_child(GinaScan::<SimpleEvent, _, _>::new(
        input_rx,
        None,
        TracingSender::wrap(next_tx, 7, 0),
        Some(TracingSender::wrap(prior_tx, 7, 1)),
        None,
        fold,
        None,
        Arc::new(|| Tile::new_zero([1, 1], 4, false)),
        1,
        ScanConfig {
            compute_bw: 1,
            write_back_mu: false,
            inclusive: true,
        },
        7,
    ));
    let next = Arc::new(Mutex::new(Vec::new()));
    let prior = Arc::new(Mutex::new(Vec::new()));
    let next_collect = Collect {
        input: next_rx,
        values: next.clone(),
        context_info: Default::default(),
    };
    next_collect.input.attach_receiver(&next_collect);
    builder.add_child(next_collect);
    let prior_collect = Collect {
        input: prior_rx,
        values: prior.clone(),
        context_info: Default::default(),
    };
    prior_collect.input.attach_receiver(&prior_collect);
    builder.add_child(prior_collect);
    builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());

    assert_eq!(
        Arc::try_unwrap(next).unwrap().into_inner().unwrap(),
        vec![Elem::Val(tile(1)), Elem::ValStop(tile(3), 1)]
    );
    assert_eq!(
        Arc::try_unwrap(prior).unwrap().into_inner().unwrap(),
        vec![Elem::Val(tile(0)), Elem::ValStop(tile(1), 1)]
    );
}
