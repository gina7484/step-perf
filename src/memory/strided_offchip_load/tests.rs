use super::*;

use std::sync::{Arc, Mutex};

use dam::simulation::ProgramBuilder;
use dam::structures::Time;
use dam::utility_contexts::{FunctionContext, GeneratorContext};
use ndarray::Array2;

use crate::ramulator::hbm_context::{HBMConfig, HBMContext, ReadBundle};
use crate::utils::events::SimpleEvent;

fn load(
    shape: [usize; 2],
    stride: [u64; 2],
    n_byte: usize,
    allocation_base: u64,
    request_bytes: u64,
) -> StridedOffChipLoad<SimpleEvent, f32> {
    let mut builder = ProgramBuilder::default();
    let (addr_snd, _) = builder.unbounded();
    let (_, resp_rcv) = builder.unbounded();
    let (_, raddr) = builder.unbounded();
    let (rdata, _) = builder.unbounded();
    StridedOffChipLoad::new(
        80,
        shape,
        stride,
        n_byte,
        allocation_base,
        request_bytes,
        3,
        addr_snd,
        resp_rcv,
        raddr,
        rdata,
        0,
    )
}

#[test]
fn generates_permuted_tile_offsets() {
    let load = load([2, 2], [1, 4], 4, 0, 4);
    assert_eq!(load.element_offsets(34), [34, 38, 35, 39]);
    assert_eq!(load.element_offsets(42), [42, 46, 43, 47]);
    assert_eq!(
        load.hbm_requests(&load.element_offsets(34)),
        [136, 152, 140, 156]
    );
    assert_eq!(
        load.hbm_requests(&load.element_offsets(42)),
        [168, 184, 172, 188]
    );
}

#[test]
fn generates_contiguous_tile_rows() {
    let load = load([2, 3], [8, 1], 4, 0, 16);
    let offsets = load.element_offsets(5);
    assert_eq!(offsets, [5, 6, 7, 13, 14, 15]);
    assert_eq!(load.hbm_requests(&offsets), [16, 48]);
}

#[test]
fn coalesces_blocks_within_each_tile() {
    let load = load([2, 2], [1, 4], 4, 0, 16);
    assert_eq!(load.hbm_requests(&load.element_offsets(34)), [128, 144]);
    assert_eq!(load.hbm_requests(&load.element_offsets(42)), [160, 176]);
    // Repeated tiles still issue reads; there is no cache across tiles.
    assert_eq!(load.hbm_requests(&load.element_offsets(34)), [128, 144]);
}

#[test]
fn zero_strides_preserve_repeated_positions_and_share_requests() {
    let load = load([2, 3], [0, 1], 4, 0, 4);
    let offsets = load.element_offsets(5);
    assert_eq!(offsets, [5, 6, 7, 5, 6, 7]);
    assert_eq!(load.hbm_requests(&offsets), [20, 24, 28]);
}

#[test]
fn covers_elements_crossing_request_boundaries() {
    let load = load([1, 2], [0, 1], 8, 6, 8);
    // Bytes 6..13 and 14..21 touch three aligned eight-byte blocks.
    assert_eq!(load.hbm_requests(&load.element_offsets(0)), [0, 8, 16]);
}

#[test]
fn preserves_allocation_bases_above_four_gibibytes() {
    let high = 1_u64 << 32;
    let load = load([2, 2], [1, 4], 4, high + 100, 4);
    assert_eq!(
        load.hbm_requests(&load.element_offsets(34)),
        [high + 236, high + 252, high + 240, high + 256],
    );
}

#[test]
#[should_panic(expected = "outside allocation")]
fn rejects_out_of_bounds_tiles() {
    load([2, 2], [1, 4], 4, 0, 4).element_offsets(76);
}

#[test]
#[should_panic(expected = "element offset overflow")]
fn rejects_element_offset_overflow() {
    load([2, 2], [u64::MAX, 1], 4, 0, 4).element_offsets(1);
}

#[test]
#[should_panic(expected = "byte address overflow")]
fn rejects_byte_address_overflow() {
    let load = load([1, 1], [0, 0], 4, u64::MAX - 2, 4);
    load.hbm_requests(&load.element_offsets(0));
}

#[test]
#[should_panic(expected = "tile dimensions must be positive")]
fn rejects_empty_tiles() {
    load([0, 2], [1, 4], 4, 0, 4);
}

fn base_token(base: u64, stop: Option<StopType>) -> Elem<Tile<u64>> {
    let tile = Tile::new(Array2::from_elem((1, 1), base).into_shared(), 8, false);
    match stop {
        Some(level) => Elem::ValStop(tile, level),
        None => Elem::Val(tile),
    }
}

#[test]
fn timing_only_load_issues_requests_and_preserves_stops() {
    let mut builder = ProgramBuilder::default();
    let (base_snd, base_rcv) = builder.unbounded();
    let (tile_snd, tile_rcv) = builder.unbounded();
    let (request_snd, request_rcv) = builder.unbounded::<ParAddrs>();
    let (response_snd, response_rcv) = builder.unbounded();
    let input = vec![
        base_token(34, None),
        base_token(42, Some(1)),
        base_token(34, Some(4)),
    ];
    builder.add_child(GeneratorContext::new(move || input.into_iter(), base_snd));
    builder.add_child(StridedOffChipLoad::<SimpleEvent, f32>::new(
        80,
        [2, 2],
        [1, 4],
        4,
        0,
        4,
        3,
        request_snd,
        response_rcv,
        base_rcv,
        tile_snd,
        0,
    ));

    // Capture actual requests and acknowledge them ten cycles after issue.
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded_requests = requests.clone();
    let mut memory = FunctionContext::new();
    request_rcv.attach_receiver(&memory);
    response_snd.attach_sender(&memory);
    memory.set_run(move |time| {
        while let Ok(request) = request_rcv.dequeue(time) {
            let ready = request.time + 10;
            recorded_requests
                .lock()
                .unwrap()
                .push((request.time.time(), request.data.addrs.clone()));
            for address in request.data.addrs {
                response_snd
                    .enqueue(
                        time,
                        ChannelElement {
                            time: ready,
                            data: address,
                        },
                    )
                    .unwrap();
            }
        }
    });
    builder.add_child(memory);

    let outputs = Arc::new(Mutex::new(Vec::new()));
    let recorded_outputs = outputs.clone();
    let mut sink = FunctionContext::new();
    tile_rcv.attach_receiver(&sink);
    sink.set_run(move |time| {
        while let Ok(output) = tile_rcv.dequeue(time) {
            recorded_outputs
                .lock()
                .unwrap()
                .push((output.time.time(), output.data));
        }
    });
    builder.add_child(sink);

    builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());

    let requests = requests.lock().unwrap();
    let batches: Vec<_> = requests
        .iter()
        .map(|(_, addresses)| addresses.clone())
        .collect();
    assert_eq!(
        batches,
        [
            vec![136, 152, 140],
            vec![156],
            vec![168, 184, 172],
            vec![188],
            vec![136, 152, 140],
            vec![156],
        ]
    );
    let outputs = outputs.lock().unwrap();
    assert_eq!(outputs.len(), 3);
    for (index, ((output_time, output), expected_stop)) in
        outputs.iter().zip([None, Some(1), Some(4)]).enumerate()
    {
        let (tile, stop) = match output {
            Elem::Val(tile) => (tile, None),
            Elem::ValStop(tile, level) => (tile, Some(*level)),
        };
        assert_eq!(tile.shape, [2, 2]);
        assert_eq!(tile.bytes_per_elem, 4);
        assert!(tile.underlying.is_none());
        assert_eq!(stop, expected_stop);
        let first_batch = requests[index * 2].0;
        let last_batch = requests[index * 2 + 1].0;
        assert_eq!(last_batch, first_batch + 1);
        assert!(
            *output_time >= last_batch + 10,
            "tile emitted before all reads completed"
        );
        if index == 1 {
            assert!(
                first_batch < outputs[0].0,
                "second tile did not overlap the first"
            );
        } else if index >= 2 {
            // Request and output arrival times both include one channel cycle.
            assert!(
                first_batch >= outputs[index - 2].0,
                "more than two tile buffers in use"
            );
        }
    }
}

#[test]
fn transposed_sliced_load_with_hbm_context() {
    // PYO3_PYTHON=/home/gina/miniconda3/envs/pytorch-step/bin/python \
    //  cargo test --offline --lib transposed_sliced_load_with_hbm_context -- --nocapture
    // Row-major A[8,8], viewed as A[::2, 1::2].T. Logical tiles visit the
    // top-left, top-right, bottom-left, and bottom-right of this 4x4 view.
    let bases = [1, 33, 5, 37];
    let stops = [None, Some(1), None, Some(2)];
    let expected_offsets = [
        [1, 17, 3, 19],
        [33, 49, 35, 51],
        [5, 21, 7, 23],
        [37, 53, 39, 55],
    ];
    let expected_requests = [[4096, 4160], [4224, 4288], [4112, 4176], [4240, 4304]];
    // One-cycle channels; one HBM channel accepts every two cycles and
    // returns each request after five cycles. Startup is zero.
    // T0 sends at 1,2; HBMContext sees 2,3; its channel accepts at 3,5.
    // Channel replies at 8,10 reach the load at 9,11; the tile is visible at 12.
    // T1 sends at 3,4 while T0 waits; its last reply reaches the load at 15.
    // T2 and T3 reuse the two buffers at 11 and 15, respectively.
    let expected_send_times = [[1, 2], [3, 4], [11, 12], [15, 16]];
    let expected_tile_times = [12, 16, 22, 26];

    let mut builder = ProgramBuilder::default();
    let (base_snd, base_rcv) = builder.unbounded();
    let (tile_snd, tile_rcv) = builder.unbounded();
    let (request_snd, request_rcv) = builder.unbounded();
    let (response_snd, response_rcv) = builder.unbounded();
    let mut hbm = HBMContext::new(
        &mut builder,
        HBMConfig {
            addr_offset: 16,
            channel_num: 1,
            per_channel_latency: 5,
            per_channel_init_interval: 2,
            per_channel_outstanding: 1,
            per_channel_start_up_time: 0,
        },
    );
    let traffic = hbm.traffic_stats();
    hbm.add_reader(ReadBundle::new(request_rcv, response_snd));
    builder.add_child(hbm);

    let input: Vec<_> = bases
        .into_iter()
        .zip(stops)
        .map(|(base, stop)| base_token(base, stop))
        .collect();
    builder.add_child(GeneratorContext::new(move || input.into_iter(), base_snd));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut load = StridedOffChipLoad::<SimpleEvent, f32>::new(
        64,
        [2, 2],
        [2, 16],
        4,
        4096,
        16,
        1,
        request_snd,
        response_rcv,
        base_rcv,
        tile_snd,
        0,
    );
    load.request_trace = Some(requests.clone());
    for (base, offsets) in bases.into_iter().zip(expected_offsets) {
        assert_eq!(load.element_offsets(base), offsets);
    }
    builder.add_child(load);

    let outputs = Arc::new(Mutex::new(Vec::new()));
    let recorded_outputs = outputs.clone();
    let mut sink = FunctionContext::new();
    tile_rcv.attach_receiver(&sink);
    sink.set_run(move |time| {
        while let Ok(output) = tile_rcv.dequeue(time) {
            recorded_outputs
                .lock()
                .unwrap()
                .push((output.time.time(), output.data));
        }
    });
    builder.add_child(sink);
    let executed = builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());
    assert!(executed.passed());

    let requests = requests.lock().unwrap();
    let outputs = outputs.lock().unwrap();
    println!("Allocation A: shape=[8,8], row-major, byte_base=4096, element_bytes=4");
    println!("View V=A[::2, 1::2].T: shape=[4,4], element_strides=[2,16]");
    println!("Tile layout: [T0 T1; T2 T3], each tile has shape [2,2]");
    println!("V element offsets:");
    for row in 0..4 {
        if row == 2 {
            println!("--------+--------");
        }
        println!(
            "{:3} {:3} | {:3} {:3}",
            1 + row * 2,
            17 + row * 2,
            33 + row * 2,
            49 + row * 2
        );
    }
    println!("V byte addresses:");
    for row in 0..4 {
        if row == 2 {
            println!("----------+----------");
        }
        println!(
            "{:4} {:4} | {:4} {:4}",
            4100 + row * 8,
            4164 + row * 8,
            4228 + row * 8,
            4292 + row * 8
        );
    }
    println!("HBM: request_bytes=16, channels=1, latency=5, init_interval=2, startup=0");
    println!(
        "Load: tile_buffers=2, par_dispatch=1; request and output channels each add one cycle"
    );
    for (index, (sent_at, addresses)) in requests.iter().enumerate() {
        println!(
            "request tile={} sent_at={} hbm_visible_at={} byte_addresses={addresses:?}",
            index / 2,
            sent_at,
            sent_at + 1
        );
    }
    for (index, (available_at, token)) in outputs.iter().enumerate() {
        let stop = match token {
            Elem::Val(_) => None,
            Elem::ValStop(_, level) => Some(*level),
        };
        println!(
            "tile={} base={} available_at={} stop={stop:?}",
            index, bases[index], available_at
        );
    }
    println!("HBM read_bytes={}", traffic.read_bytes());

    assert_eq!(requests.len(), 8);
    assert_eq!(outputs.len(), 4);
    for tile in 0..4 {
        for request in 0..2 {
            assert_eq!(
                requests[tile * 2 + request],
                (
                    expected_send_times[tile][request],
                    vec![expected_requests[tile][request]],
                )
            );
        }
        let (available_at, token) = &outputs[tile];
        assert_eq!(*available_at, expected_tile_times[tile]);
        let (payload, stop) = match token {
            Elem::Val(payload) => (payload, None),
            Elem::ValStop(payload, level) => (payload, Some(*level)),
        };
        assert_eq!(payload.shape, [2, 2]);
        assert_eq!(payload.bytes_per_elem, 4);
        assert!(payload.underlying.is_none());
        assert_eq!(stop, stops[tile]);
    }
    assert_eq!(traffic.read_bytes(), 8 * 16);
    assert_eq!(traffic.write_bytes(), 0);
}

struct LoadTrace {
    requests: Vec<(u64, Vec<u64>)>,
    outputs: Vec<(u64, Option<StopType>)>,
}

// Each tuple starts with its channel availability cycle. Every tile has two
// elements and issues two four-byte block reads in one batch.
fn run_scripted_load(
    bases: Vec<(u64, u64, Option<StopType>)>,
    responses: Vec<(u64, u64)>,
) -> LoadTrace {
    let mut builder = ProgramBuilder::default();
    let (base_snd, base_rcv) = builder.unbounded();
    let (tile_snd, tile_rcv) = builder.unbounded();
    let (request_snd, request_rcv) = builder.unbounded();
    let (response_snd, response_rcv) = builder.unbounded();

    let mut source = FunctionContext::new();
    base_snd.attach_sender(&source);
    source.set_run(move |time| {
        for (at, base, stop) in bases {
            base_snd
                .enqueue(
                    time,
                    ChannelElement {
                        time: Time::new(at),
                        data: base_token(base, stop),
                    },
                )
                .unwrap();
        }
    });
    builder.add_child(source);

    let mut request_sink = FunctionContext::new();
    request_rcv.attach_receiver(&request_sink);
    request_sink.set_run(move |time| while request_rcv.dequeue(time).is_ok() {});
    builder.add_child(request_sink);

    let mut memory = FunctionContext::new();
    response_snd.attach_sender(&memory);
    memory.set_run(move |time| {
        for (at, address) in responses {
            response_snd
                .enqueue(
                    time,
                    ChannelElement {
                        time: Time::new(at),
                        data: address,
                    },
                )
                .unwrap();
        }
    });
    builder.add_child(memory);

    let requests = Arc::new(Mutex::new(Vec::new()));
    let mut load = StridedOffChipLoad::<SimpleEvent, f32>::new(
        8,
        [1, 2],
        [0, 1],
        4,
        0,
        4,
        2,
        request_snd,
        response_rcv,
        base_rcv,
        tile_snd,
        0,
    );
    load.request_trace = Some(requests.clone());
    builder.add_child(load);

    let outputs = Arc::new(Mutex::new(Vec::new()));
    let recorded_outputs = outputs.clone();
    let mut sink = FunctionContext::new();
    tile_rcv.attach_receiver(&sink);
    sink.set_run(move |time| {
        while let Ok(output) = tile_rcv.dequeue(time) {
            let stop = match output.data {
                Elem::Val(_) => None,
                Elem::ValStop(_, stop) => Some(stop),
            };
            recorded_outputs
                .lock()
                .unwrap()
                .push((output.time.time(), stop));
        }
    });
    builder.add_child(sink);
    let executed = builder
        .initialize(Default::default())
        .unwrap()
        .run(Default::default());
    assert!(executed.passed());

    let requests = requests.lock().unwrap().clone();
    let outputs = outputs.lock().unwrap().clone();
    LoadTrace { requests, outputs }
}

#[test]
fn preserves_tile_order_with_out_of_order_responses() {
    let trace = run_scripted_load(
        vec![(1, 0, None), (1, 2, Some(1)), (1, 4, Some(2))],
        // T1 finishes at 6, T0 at 10, and T2 at 15.
        vec![(5, 12), (6, 8), (9, 4), (10, 0), (14, 20), (15, 16)],
    );
    // T1 retains its buffer until it can be emitted after T0. T2 waits for a
    // free buffer at 10 even though T1's responses have all arrived by 6.
    assert_eq!(
        trace.requests,
        [(1, vec![0, 4]), (2, vec![8, 12]), (10, vec![16, 20])]
    );
    assert_eq!(trace.outputs, [(11, None), (11, Some(1)), (16, Some(2))]);
}

#[test]
fn counts_shared_block_responses_once_per_tile() {
    let trace = run_scripted_load(
        vec![(1, 0, None), (1, 0, Some(1)), (1, 2, Some(2))],
        // T0 and T1 read the same blocks. Each needs its own pair of replies.
        vec![(5, 4), (6, 0), (9, 4), (10, 0), (13, 12), (14, 8)],
    );
    assert_eq!(
        trace.requests,
        [(1, vec![0, 4]), (2, vec![0, 4]), (6, vec![8, 12])]
    );
    assert_eq!(trace.outputs, [(7, None), (11, Some(1)), (15, Some(2))]);
}

#[test]
fn emits_completed_tile_while_next_base_is_delayed() {
    let trace = run_scripted_load(
        vec![(1, 0, None), (30, 2, Some(2))],
        vec![(5, 0), (6, 4), (34, 8), (35, 12)],
    );
    assert_eq!(trace.requests, [(1, vec![0, 4]), (30, vec![8, 12])]);
    // T0 leaves at 7 without waiting for T1's base at 30. T1 still drains
    // correctly after the input stream closes.
    assert_eq!(trace.outputs, [(7, None), (36, Some(2))]);
}
