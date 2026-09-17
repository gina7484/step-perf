use step_perf::operator::shuffle::{Shuffle, ShuffleConfig};
use step_perf::primitives::{elem::Elem, select::{MultiHotN, SelectAdapter}};
use dam::context_tools::*;
use step_perf::trace::TracingSender;
use step_perf::memory::PMU_BW;
use step_perf::{primitives::tile::Tile, utils::events::SimpleEvent};
use dam::{simulation::ProgramBuilder, utility_contexts::FunctionContext};
use ndarray::Array2;
use std::sync::{Arc, Mutex};

type Timed<T> = Vec<(u64, Elem<T>)>;

fn source<T: DAMType + 'static>(
    builder: &mut ProgramBuilder,
    sender: Sender<Elem<T>>,
    values: Vec<Elem<T>>,
) {
    let mut context = FunctionContext::new();
    sender.attach_sender(&context);
    context.set_run(move |time| {
        for value in values {
            sender
                .enqueue(time, ChannelElement::new(time.tick(), value))
                .unwrap();
        }
    });
    builder.add_child(context);
}

fn collect<T: DAMType + 'static>(
    builder: &mut ProgramBuilder,
    receiver: Receiver<Elem<T>>,
) -> Arc<Mutex<Timed<T>>> {
    let result = Arc::new(Mutex::new(Vec::new()));
    let collected = result.clone();
    let mut context = FunctionContext::new();
    receiver.attach_receiver(&context);
    context.set_run(move |time| {
        while let Ok(value) = receiver.dequeue(time) {
            collected
                .lock()
                .unwrap()
                .push((value.time.time(), value.data));
        }
    });
    builder.add_child(context);
    result
}

fn run(
    data: Vec<Elem<Tile<i32>>>,
    masks: Vec<Elem<MultiHotN>>,
    rank: u32,
    width: usize,
    write_back_mu: bool,
) -> (bool, Timed<Tile<i32>>, Timed<MultiHotN>) {
    let mut builder = ProgramBuilder::default();
    let (data_snd, data_rcv) = builder.unbounded();
    let (mask_snd, mask_rcv) = builder.unbounded();
    // Drain both complete outputs through small FIFOs to exercise backpressure.
    let (out_snd, out_rcv) = builder.bounded(1);
    let (index_snd, index_rcv) = builder.bounded(1);
    source(&mut builder, data_snd, data);
    source(&mut builder, mask_snd, masks);
    builder.add_child(Shuffle::<SimpleEvent, _>::new(
        data_rcv,
        mask_rcv,
        TracingSender::wrap(out_snd, 0, 0),
        TracingSender::wrap(index_snd, 0, 1),
        ShuffleConfig {
            rank,
            num_buckets: width,
            write_back_mu,
        },
        0,
    ));
    let data = collect(&mut builder, out_rcv);
    let masks = collect(&mut builder, index_rcv);
    let executed = builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());
    let data = data.lock().unwrap().clone();
    let masks = masks.lock().unwrap().clone();
    (executed.passed(), data, masks)
}

fn elem<T>(value: T, stop: u32) -> Elem<T> {
    if stop == 0 {
        Elem::Val(value)
    } else {
        Elem::ValStop(value, stop)
    }
}

fn tiles(values: &[(i32, u32)]) -> Vec<Elem<Tile<i32>>> {
    values
        .iter()
        .map(|&(value, stop)| {
            elem(
                Tile::new(Array2::from_elem((1, 2), value).into_shared(), 4, false),
                stop,
            )
        })
        .collect()
}

fn masks(values: &[(Vec<usize>, u32)], width: usize) -> Vec<Elem<MultiHotN>> {
    values
        .iter()
        .map(|(indices, stop)| {
            elem(
                MultiHotN::from_sel_vec(indices.clone(), width, false),
                *stop,
            )
        })
        .collect()
}

fn assert_values<T: DAMType + PartialEq>(actual: &Timed<T>, expected: Vec<Elem<T>>) {
    assert_eq!(
        actual
            .iter()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn supplied_rank_two_example() {
    let (passed, data, index) = run(
        tiles(&[
            (0, 0),
            (1, 1),
            (2, 0),
            (3, 1),
            (4, 0),
            (5, 1),
            (6, 0),
            (7, 2),
        ]),
        masks(
            &[
                (vec![1, 3], 0),
                (vec![1, 2], 0),
                (vec![2, 3], 0),
                (vec![1, 3], 1),
            ],
            4,
        ),
        2,
        4,
        false,
    );
    assert!(passed);
    assert_values(
        &data,
        tiles(&[
            (0, 0),
            (1, 1),
            (2, 0),
            (3, 1),
            (6, 0),
            (7, 2),
            (2, 0),
            (3, 1),
            (4, 0),
            (5, 2),
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (6, 0),
            (7, 3),
        ]),
    );
    assert_values(
        &index,
        masks(&[(vec![1], 0), (vec![2], 0), (vec![3], 1)], 4),
    );
}

#[test]
fn rank_one_resets_buckets_and_skips_dropped_items() {
    let (passed, data, index) = run(
        tiles(&[(0, 0), (1, 0), (2, 1), (3, 0), (4, 2)]),
        masks(
            &[
                (vec![2], 0),
                (vec![0], 0),
                (vec![], 1),
                (vec![2], 0),
                (vec![2], 2),
            ],
            3,
        ),
        1,
        3,
        false,
    );
    assert!(passed);
    assert_values(&data, tiles(&[(1, 1), (0, 2), (3, 0), (4, 3)]));
    assert_values(
        &index,
        masks(&[(vec![0], 0), (vec![2], 1), (vec![2], 2)], 3),
    );
}

#[test]
fn rank_three_preserves_inner_stops_and_outer_group_stops() {
    let (passed, data, index) = run(
        tiles(&[
            (0, 0),
            (1, 1),
            (2, 0),
            (3, 2),
            (4, 0),
            (5, 1),
            (6, 0),
            (7, 4),
        ]),
        masks(&[(vec![2], 0), (vec![0], 2)], 3),
        3,
        3,
        false,
    );
    assert!(passed);
    assert_values(
        &data,
        tiles(&[
            (4, 0),
            (5, 1),
            (6, 0),
            (7, 3),
            (0, 0),
            (1, 1),
            (2, 0),
            (3, 5),
        ]),
    );
    assert_values(&index, masks(&[(vec![0], 0), (vec![2], 2)], 3));
}

#[test]
fn writes_once_and_reads_each_duplicate_including_blank_tiles() {
    for blank in [false, true] {
        let tile = if blank {
            Tile::new_blank_padded(vec![1, (2 * PMU_BW / 4 + 1) as usize], 4, false, 0)
        } else {
            Tile::new_padded(Array2::from_elem((1, (2 * PMU_BW / 4 + 1) as usize), 9).into_shared(), 4, false, 0)
        };
        let (passed, single, _) = run(
            vec![elem(tile.clone(), 1)],
            masks(&[(vec![0], 1)], 2),
            1,
            2,
            false,
        );
        assert!(passed);
        let (passed, duplicated, index) = run(
            vec![elem(tile.clone(), 1)],
            masks(&[(vec![0, 1], 1)], 2),
            1,
            2,
            false,
        );
        assert!(passed);
        // A tile just over two PMU transfers takes 3 cycles. One write, then one read per selected bucket.
        // The input and output channels each add one cycle of latency.
        assert_eq!(single[0].0, 8);
        assert_eq!(
            duplicated.iter().map(|(time, _)| *time).collect::<Vec<_>>(),
            vec![8, 11]
        );
        assert_eq!(index[0].0, 5);
        assert_values(&duplicated, vec![elem(tile.clone(), 1), elem(tile, 2)]);
    }
}

#[test]
fn charges_input_memory_reads_and_optional_output_writeback() {
    let tile = Tile::<i32>::new_blank(vec![1, (2 * PMU_BW / 4 + 1) as usize], 4, true);
    let mask = MultiHotN::from_sel_vec(vec![0], 1, true);
    let (passed, data, index) =
        run(vec![elem(tile.clone(), 1)], vec![elem(mask, 1)], 1, 1, true);
    assert!(passed);
    // One mask read plus tile load, shared write, flush read and output write.
    assert_eq!(data[0].0, 2 + 1 + 4 * 3);
    assert_values(&data, vec![elem(tile, 2)]);
    assert_values(&index, masks(&[(vec![0], 1)], 1));
}

#[test]
fn rejects_entirely_empty_groups() {
    let (passed, data, index) = run(tiles(&[(0, 1)]), masks(&[(vec![], 1)], 2), 1, 2, false);
    assert!(!passed);
    assert!(data.is_empty() && index.is_empty());
}

#[test]
fn rejects_misaligned_or_truncated_streams() {
    let cases = [
        (tiles(&[(0, 2)]), masks(&[(vec![0], 2)], 1)),
        (tiles(&[(0, 0)]), masks(&[(vec![0], 1)], 1)),
        (tiles(&[(0, 1)]), masks(&[(vec![0], 0)], 1)),
        (tiles(&[(0, 2), (1, 2)]), masks(&[(vec![0], 1)], 1)),
        (tiles(&[(0, 2)]), masks(&[(vec![0], 1), (vec![0], 1)], 1)),
    ];
    for (data, index) in cases {
        assert!(!run(data, index, 2, 1, false).0);
    }
}

#[test]
fn rejects_wrong_mask_width() {
    assert!(!run(tiles(&[(0, 1)]), masks(&[(vec![0], 1)], 1), 1, 2, false).0);
}

#[test]
fn accepts_empty_streams() {
    let (passed, data, index) = run(vec![], vec![], 1, 1, false);
    assert!(passed);
    assert!(data.is_empty() && index.is_empty());
}
