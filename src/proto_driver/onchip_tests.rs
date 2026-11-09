//! One test per on-chip memory rule in `docs/resource_counting.md`. Each test
//! feeds hand-made f32 tiles into one operator and checks the bytes the run
//! reports for it. Inputs repeat a size, so a sum in place of a max shows up.
use super::proto_headers::graph_proto::{
    self as pb, AccumFunc, DataType, ElemtoElemFunc, InitFunc, Operation,
};
use super::*;
use crate::primitives::buffer::Buffer;
use crate::primitives::elem::{Elem, StopType};
use ndarray::{ArcArray, ArcArray2, IxDyn};

const F32_BYTES: usize = 4;

fn tile_type(rows: u32, cols: u32) -> Option<DataType> {
    Some(DataType {
        r#type: Some(Type::F32(f32_tile(rows, cols))),
    })
}

fn buffer_type(rows: u32, cols: u32) -> Option<DataType> {
    Some(DataType {
        r#type: Some(Type::Buffer(pb::Buffer {
            r#type: Some(buffer::Type::F32(f32_tile(rows, cols))),
        })),
    })
}

fn f32_tile(rows: u32, cols: u32) -> pb::F32 {
    pb::F32 {
        row_size: Some(pb::f32::RowSize::RowStatic(rows)),
        col_size: Some(pb::f32::ColSize::ColStatic(cols)),
    }
}

fn tile(shape: [usize; 2]) -> Tile<f32> {
    Tile::new(ArcArray2::from_elem(shape, 1.0), F32_BYTES, false)
}

/// One `shape` tile per entry, closed by the entry's stop level.
fn tiles(shape: [usize; 2], stops: &[Option<StopType>]) -> Vec<Elem<Tile<f32>>> {
    stops
        .iter()
        .map(|stop| with_stop(tile(shape), *stop))
        .collect()
}

fn with_stop<T>(value: T, stop: Option<StopType>) -> Elem<T> {
    match stop {
        Some(level) => Elem::ValStop(value, level),
        None => Elem::Val(value),
    }
}

fn buffer_of(count: usize, shape: [usize; 2]) -> Buffer<Tile<f32>> {
    let tiles = (0..count).map(|_| tile(shape)).collect();
    Buffer::new(ArcArray::from_shape_vec(IxDyn(&[count]), tiles).unwrap(), 0)
}

fn add() -> Option<AccumFunc> {
    Some(AccumFunc {
        accum_fn: Some(accum_func::AccumFn::Add(pb::Add {})),
    })
}

fn zero() -> Option<InitFunc> {
    Some(InitFunc {
        init_fn: Some(init_func::InitFn::Zero(pb::Zero {})),
    })
}

fn operation(id: u32, op_type: OpType) -> Operation {
    Operation {
        id,
        dtype_bytes: F32_BYTES as u32,
        op_type: Some(op_type),
        ..Default::default()
    }
}

fn hbm() -> HBMConfig {
    HBMConfig {
        addr_offset: 32,
        channel_num: 1,
        per_channel_latency: 1,
        per_channel_init_interval: 1,
        per_channel_outstanding: 1,
        per_channel_start_up_time: 1,
    }
}

/// Feed each `(channel, producer id, elements)` input, drain each
/// `(channel, op id)` output, run `operators`, and return their on-chip bytes.
macro_rules! onchip_bytes {
    (
        operators: $operators:expr,
        inputs: [$(($in_channel:ident, $in_id:expr, $in_values:expr)),* $(,)?],
        outputs: [$(($out_channel:ident, $out_id:expr)),* $(,)?] $(,)?
    ) => {{
        let mut builder = ProgramBuilder::default();
        let mut channels = ChannelMapCollection::default();
        $(
            let sender = channels.$in_channel.get_sender($in_id, None, &mut builder, Some(1));
            let values = $in_values;
            builder.add_child(GeneratorContext::new(move || values.into_iter(), sender));
        )*
        $(
            let receiver = channels.$out_channel.get_receiver($out_id, None, &mut builder, Some(1));
            builder.add_child(ConsumerContext::new(receiver));
        )*
        let (_, onchip) = build_from_proto(
            ProgramGraph {
                name: "onchip".into(),
                operators: $operators,
            },
            &mut channels,
            &mut builder,
            &hbm(),
            &SimConfig {
                channel_depth: Some(1),
                config_dict: HashMap::new(),
            },
            None,
        );
        assert!(builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default())
            .passed());
        onchip.bytes_by_op()
    }};
}

#[test]
fn linear_load_counts_two_of_its_tiles() {
    // Six 2x8 f32 tiles of 64 bytes each.
    let load = operation(
        1,
        OpType::LinearOffChipLoad(pb::LinearOffChipLoad {
            tensor_shape_tiled: vec![2, 3],
            stride: vec![3, 1],
            out_shape_tiled: vec![2, 3],
            tile_row: 2,
            tile_col: 8,
            dtype: tile_type(2, 8),
            par_dispatch: 1,
            ..Default::default()
        }),
    );
    let bytes = onchip_bytes!(operators: vec![load], inputs: [], outputs: [(tile_f32, 1)]);
    assert_eq!(bytes, HashMap::from([(1, 128)]));
}

#[test]
fn linear_load_ref_counts_two_of_its_tiles() {
    // Each reference token loads three 2x8 f32 tiles of 64 bytes each.
    let load = operation(
        1,
        OpType::LinearOffChipLoadRef(pb::LinearOffChipLoadRef {
            ref_id: 0,
            tensor_shape_tiled: vec![1, 3],
            stride: vec![3, 1],
            out_shape_tiled: vec![1, 3],
            tile_row: 2,
            tile_col: 8,
            dtype: tile_type(2, 8),
            ref_dtype: tile_type(1, 1),
            par_dispatch: 1,
            ..Default::default()
        }),
    );
    let bytes = onchip_bytes!(
        operators: vec![load],
        inputs: [(tile_f32, 0, tiles([1, 1], &[None, None]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 128)]));
}

#[test]
fn strided_load_counts_two_of_its_tiles() {
    // Three base addresses, each reading one 2x8 f32 tile of 64 bytes.
    let base_addresses: Vec<Elem<Tile<u64>>> = [0, 16, 32]
        .into_iter()
        .map(|base| Elem::Val(Tile::new(ArcArray2::from_elem([1, 1], base), 8, false)))
        .collect();
    let load = operation(
        1,
        OpType::StridedOffChipLoad(pb::StridedOffChipLoad {
            raddr_id: 0,
            allocation_elements: 64,
            tile_shape: vec![2, 8],
            tile_stride: vec![8, 1],
            par_dispatch: 1,
            dtype: tile_type(2, 8),
            ..Default::default()
        }),
    );
    let bytes = onchip_bytes!(
        operators: vec![load],
        inputs: [(tile_u64, 0, base_addresses)],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 128)]));
}

#[test]
fn bufferize_counts_its_largest_buffer() {
    // A buffer of three 64-byte tiles, then a buffer of two.
    let bufferize = operation(
        1,
        OpType::Bufferize(pb::Bufferize {
            input_id: 0,
            rank: 1,
            dtype: tile_type(2, 8),
            ..Default::default()
        }),
    );
    let bytes = onchip_bytes!(
        operators: vec![bufferize],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, None, Some(1), None, Some(2)]))],
        outputs: [(buff_tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 192)]));
}

#[test]
fn accum_buffer_counts_its_output_buffer() {
    // Two 2x8 f32 accumulator slots, flushed twice.
    let accum = operation(
        1,
        OpType::AccumBuffer(pb::AccumBuffer {
            input_id: 0,
            func: add(),
            init_func: zero(),
            rank: 1,
            buffer_shape: vec![2],
            tile_row: 2,
            tile_col: 8,
            compute_bw: 1,
            dtype_a: tile_type(2, 8),
            dtype_b: buffer_type(2, 8),
            ..Default::default()
        }),
    );
    let bytes = onchip_bytes!(
        operators: vec![accum],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, Some(1), None, Some(2)]))],
        outputs: [(buff_tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 128)]));
}

fn binary_map(
    func: elemto_elem_func::ElemElemFn,
    a: [u32; 2],
    b: [u32; 2],
    out: [u32; 2],
) -> Operation {
    operation(
        1,
        OpType::Binarymap(pb::BinaryMap {
            input_id1: 0,
            input_id2: 2,
            func: Some(ElemtoElemFunc {
                elem_elem_fn: Some(func),
            }),
            compute_bw: 1,
            dtype_a: tile_type(a[0], a[1]),
            dtype_b: tile_type(b[0], b[1]),
            dtype_out: tile_type(out[0], out[1]),
            ..Default::default()
        }),
    )
}

#[test]
fn matmul_counts_both_input_tiles() {
    // Each call holds a 2x8 tile of 64 bytes and an 8x4 tile of 128 bytes.
    let matmul = binary_map(
        elemto_elem_func::ElemElemFn::Matmul(pb::Matmul {
            weight_transposed: false,
        }),
        [2, 8],
        [8, 4],
        [2, 4],
    );
    let bytes = onchip_bytes!(
        operators: vec![matmul],
        inputs: [
            (tile_f32, 0, tiles([2, 8], &[None, Some(1)])),
            (tile_f32, 2, tiles([8, 4], &[None, Some(1)])),
        ],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 192)]));
}

#[test]
fn elementwise_binary_map_counts_nothing() {
    let add = binary_map(
        elemto_elem_func::ElemElemFn::Add(pb::Add {}),
        [2, 8],
        [2, 8],
        [2, 8],
    );
    let bytes = onchip_bytes!(
        operators: vec![add],
        inputs: [
            (tile_f32, 0, tiles([2, 8], &[None, Some(1)])),
            (tile_f32, 2, tiles([2, 8], &[None, Some(1)])),
        ],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

fn accum(shape: [u32; 2]) -> Operation {
    operation(
        1,
        OpType::Accum(pb::Accum {
            input_id: 0,
            func: add(),
            init_func: zero(),
            tile_row: shape[0],
            tile_col: shape[1],
            rank: 1,
            compute_bw: 1,
            dtype_a: tile_type(shape[0], shape[1]),
            dtype_b: tile_type(shape[0], shape[1]),
            ..Default::default()
        }),
    )
}

#[test]
fn accum_counts_its_output_tile() {
    // Two sums of two 2x8 f32 tiles, each output 64 bytes.
    let bytes = onchip_bytes!(
        operators: vec![accum([2, 8])],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, Some(1), None, Some(2)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 64)]));
}

#[test]
fn accum_with_a_one_by_one_output_counts_nothing() {
    let bytes = onchip_bytes!(
        operators: vec![accum([1, 1])],
        inputs: [(tile_f32, 0, tiles([1, 1], &[None, Some(1), None, Some(2)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

fn scan(shape: [u32; 2]) -> Operation {
    operation(
        1,
        OpType::Scan(pb::Scan {
            input_id1: 0,
            fn1: add(),
            init_func: zero(),
            tile_row: shape[0],
            tile_col: shape[1],
            rank: 1,
            inclusive: true,
            compute_bw: 1,
            dtype_a: tile_type(shape[0], shape[1]),
            dtype_b: tile_type(shape[0], shape[1]),
            ..Default::default()
        }),
    )
}

#[test]
fn scan_counts_its_output_tile() {
    // Every 2x8 f32 prefix sum is 64 bytes.
    let bytes = onchip_bytes!(
        operators: vec![scan([2, 8])],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, Some(1), None, Some(2)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 64)]));
}

#[test]
fn scan_with_a_one_by_one_output_counts_nothing() {
    let bytes = onchip_bytes!(
        operators: vec![scan([1, 1])],
        inputs: [(tile_f32, 0, tiles([1, 1], &[None, Some(1), None, Some(2)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

fn repeat_static(dtype: Option<DataType>) -> Operation {
    operation(
        1,
        OpType::RepeatStatic(pb::RepeatStatic {
            input_id: 0,
            repeat_factor: vec![3],
            dtype,
            ..Default::default()
        }),
    )
}

#[test]
fn repeat_static_counts_the_tile_it_repeats() {
    // Two 2x8 f32 tiles of 64 bytes, each repeated three times.
    let bytes = onchip_bytes!(
        operators: vec![repeat_static(tile_type(2, 8))],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, Some(1)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 64)]));
}

#[test]
fn repeat_static_of_a_buffer_counts_nothing() {
    let buffers = vec![
        with_stop(buffer_of(2, [2, 8]), None),
        with_stop(buffer_of(2, [2, 8]), Some(1)),
    ];
    let bytes = onchip_bytes!(
        operators: vec![repeat_static(buffer_type(2, 8))],
        inputs: [(buff_tile_f32, 0, buffers)],
        outputs: [(buff_tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

#[test]
fn repeat_static_of_one_by_one_tiles_counts_nothing() {
    let bytes = onchip_bytes!(
        operators: vec![repeat_static(tile_type(1, 1))],
        inputs: [(tile_f32, 0, tiles([1, 1], &[None, Some(1)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

fn repeat_ref(dtype: Option<DataType>) -> Operation {
    operation(
        1,
        OpType::RepeatRef(pb::RepeatRef {
            input_id: 0,
            ref_id: 2,
            dtype,
            ref_dtype: tile_type(1, 1),
            rank: 0,
            expanded_rank_cnt: 1,
            ..Default::default()
        }),
    )
}

/// Two references per input value: the second closes that value's repeats.
fn references() -> Vec<Elem<Tile<f32>>> {
    tiles([1, 1], &[None, Some(1), None, Some(2)])
}

#[test]
fn repeat_ref_counts_the_tile_it_repeats() {
    // Two 2x8 f32 tiles of 64 bytes, each repeated twice.
    let bytes = onchip_bytes!(
        operators: vec![repeat_ref(tile_type(2, 8))],
        inputs: [
            (tile_f32, 0, tiles([2, 8], &[None, Some(1)])),
            (tile_f32, 2, references()),
        ],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 64)]));
}

#[test]
fn repeat_ref_of_a_buffer_counts_nothing() {
    let buffers = vec![
        with_stop(buffer_of(2, [2, 8]), None),
        with_stop(buffer_of(2, [2, 8]), Some(1)),
    ];
    let bytes = onchip_bytes!(
        operators: vec![repeat_ref(buffer_type(2, 8))],
        inputs: [(buff_tile_f32, 0, buffers), (tile_f32, 2, references())],
        outputs: [(buff_tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

fn promote_outer(dtype: Option<DataType>) -> Operation {
    operation(
        1,
        OpType::PromoteOuter(pb::PromoteOuter {
            input_id: 0,
            dtype,
            ..Default::default()
        }),
    )
}

#[test]
fn promote_outer_counts_the_values_it_passes() {
    // Three 2x8 f32 tiles of 64 bytes each.
    let bytes = onchip_bytes!(
        operators: vec![promote_outer(tile_type(2, 8))],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, None, Some(1)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 64)]));
}

#[test]
fn promote_outer_counts_a_multi_hot_at_one_byte_per_candidate() {
    let masks: Vec<Elem<MultiHotN>> = (0..2)
        .map(|_| Elem::Val(MultiHotN::new(vec![true, false, true, false], false)))
        .collect();
    let dtype = Some(DataType {
        r#type: Some(Type::MultiHot(pb::MultiHot { width: 4 })),
    });
    let bytes = onchip_bytes!(
        operators: vec![promote_outer(dtype)],
        inputs: [(multihot, 0, masks)],
        outputs: [(multihot, 1)],
    );
    assert_eq!(bytes, HashMap::from([(1, 4)]));
}

#[test]
fn promote_outer_of_one_by_one_tiles_counts_nothing() {
    let bytes = onchip_bytes!(
        operators: vec![promote_outer(tile_type(1, 1))],
        inputs: [(tile_f32, 0, tiles([1, 1], &[None, None, Some(1)]))],
        outputs: [(tile_f32, 1)],
    );
    assert_eq!(bytes, HashMap::new());
}

#[test]
fn each_run_reports_only_its_own_operators() {
    let first = onchip_bytes!(
        operators: vec![accum([2, 8])],
        inputs: [(tile_f32, 0, tiles([2, 8], &[None, Some(1)]))],
        outputs: [(tile_f32, 1)],
    );
    let mut promote = promote_outer(tile_type(2, 4));
    promote.id = 5;
    let second = onchip_bytes!(
        operators: vec![promote],
        inputs: [(tile_f32, 0, tiles([2, 4], &[None, Some(1)]))],
        outputs: [(tile_f32, 5)],
    );
    assert_eq!(first, HashMap::from([(1, 64)]));
    assert_eq!(second, HashMap::from([(5, 32)]));
}
