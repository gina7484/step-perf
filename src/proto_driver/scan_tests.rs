use super::proto_headers::graph_proto::{self as pb, AccumFunc, DataType, InitFunc, Operation};
use super::*;
use crate::primitives::elem::{Elem, StopType};
use dam::context_tools::*;
use std::sync::Mutex;

#[context_macro]
struct Collect<T: DAMType> {
    input: Receiver<T>,
    output: Arc<Mutex<Vec<T>>>,
}

impl<T: DAMType> Collect<T> {
    fn new(input: Receiver<T>, output: Arc<Mutex<Vec<T>>>) -> Self {
        let ctx = Self {
            input,
            output,
            context_info: Default::default(),
        };
        ctx.input.attach_receiver(&ctx);
        ctx
    }
}

impl<T: DAMType> Context for Collect<T> {
    fn run(&mut self) {
        while let Ok(element) = self.input.dequeue(&self.time) {
            self.output.lock().unwrap().push(element.data);
        }
    }
}

fn stream<T: Clone>(
    values: &[(T, Option<StopType>)],
    shape: [usize; 2],
    bytes: usize,
) -> Vec<Elem<Tile<T>>> {
    values
        .iter()
        .map(|(value, stop)| {
            let tile = Tile::new(
                ndarray::ArcArray2::from_elem(shape, value.clone()),
                bytes,
                false,
            );
            match stop {
                Some(level) => Elem::ValStop(tile, *level),
                None => Elem::Val(tile),
            }
        })
        .collect()
}

fn counter_type() -> Type {
    Type::U64(pb::U64 {
        row_size: Some(pb::u64::RowSize::RowStatic(1)),
        col_size: Some(pb::u64::ColSize::ColStatic(1)),
    })
}

fn scan_operation(
    input: Type,
    output: Option<Type>,
    fold: accum_func::AccumFn,
    rank: u32,
    inclusive: bool,
) -> Operation {
    let bytes = match output.as_ref().unwrap_or(&input) {
        Type::U64(_) | Type::I64(_) => 8,
        Type::Bf16(_) => 2,
        _ => 4,
    };
    let initializer = if matches!(fold, accum_func::AccumFn::Max(_)) {
        init_func::InitFn::NegInf(pb::NegInf {})
    } else {
        init_func::InitFn::Zero(pb::Zero {})
    };
    Operation {
        id: 1,
        dtype_bytes: bytes,
        op_type: Some(OpType::Scan(pb::Scan {
            input_id1: 0,
            dtype_a: Some(DataType {
                r#type: Some(input),
            }),
            dtype_b: output.map(|kind| DataType { r#type: Some(kind) }),
            fn1: Some(AccumFunc {
                accum_fn: Some(fold),
            }),
            init_func: Some(InitFunc {
                init_fn: Some(initializer),
            }),
            tile_row: 1,
            tile_col: 1,
            rank,
            inclusive,
            compute_bw: 1,
            ..Default::default()
        })),
        ..Default::default()
    }
}

fn count_operation(input: Type, rank: u32) -> Operation {
    Operation {
        id: 1,
        dtype_bytes: 8,
        op_type: Some(OpType::Accum(pb::Accum {
            input_id: 0,
            dtype_a: Some(DataType {
                r#type: Some(input),
            }),
            dtype_b: Some(DataType {
                r#type: Some(counter_type()),
            }),
            func: Some(AccumFunc {
                accum_fn: Some(accum_func::AccumFn::Increment(pb::Increment {})),
            }),
            init_func: Some(InitFunc {
                init_fn: Some(init_func::InitFn::Zero(pb::Zero {})),
            }),
            tile_row: 1,
            tile_col: 1,
            rank,
            compute_bw: 1,
            ..Default::default()
        })),
        ..Default::default()
    }
}

// Feed explicit tokens through the protobuf driver and collect the full output,
// including its length and stop levels. Capacity one also exercises backpressure.
macro_rules! check_operator {
    ($input_channel:ident, $output_channel:ident, $operation:expr, $input:expr, $expected:expr) => {{
        let mut builder = ProgramBuilder::default();
        let mut channels = ChannelMapCollection::default();
        let sender = channels
            .$input_channel
            .get_sender(0, None, &mut builder, Some(1));
        let input = $input;
        builder.add_child(GeneratorContext::new(move || input.into_iter(), sender));
        let receiver = channels
            .$output_channel
            .get_receiver(1, None, &mut builder, Some(1));
        let output = Arc::new(Mutex::new(Vec::new()));
        builder.add_child(Collect::new(receiver, output.clone()));
        build_from_proto(
            ProgramGraph {
                name: "scan_count".into(),
                operators: vec![$operation],
            },
            &mut channels,
            &mut builder,
            &HBMConfig {
                addr_offset: 32,
                channel_num: 1,
                per_channel_latency: 1,
                per_channel_init_interval: 1,
                per_channel_outstanding: 1,
                per_channel_start_up_time: 1,
            },
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
        assert_eq!(*output.lock().unwrap(), $expected);
    }};
}

#[test]
fn increment_scan_enumerates_and_accum_counts_each_rank() {
    // Unequal groups exercise resets without a statically known group length.
    macro_rules! check_counting {
        ($channel:ident, $kind:expr, $value:expr, $bytes:expr) => {{
            let input = stream(
                &[
                    ($value, None),
                    ($value, Some(1)),
                    ($value, None),
                    ($value, None),
                    ($value, Some(2)),
                ],
                [4, 8],
                $bytes,
            );
            for rank in [1, 2] {
                for inclusive in [false, true] {
                    let mut indices = if rank == 1 {
                        vec![0, 1, 0, 1, 2]
                    } else {
                        vec![0, 1, 2, 3, 4]
                    };
                    if inclusive {
                        indices.iter_mut().for_each(|value| *value += 1);
                    }
                    let values: Vec<_> = indices
                        .into_iter()
                        .zip([None, Some(1), None, None, Some(2)])
                        .collect();
                    check_operator!(
                        $channel,
                        tile_u64,
                        scan_operation(
                            $kind,
                            Some(counter_type()),
                            accum_func::AccumFn::Increment(pb::Increment {}),
                            rank,
                            inclusive
                        ),
                        input.clone(),
                        stream(&values, [1, 1], 8)
                    );
                }
                let counts = if rank == 1 {
                    vec![(2_u64, None), (3, Some(1))]
                } else {
                    vec![(5, None)]
                };
                check_operator!(
                    $channel,
                    tile_u64,
                    count_operation($kind, rank),
                    input.clone(),
                    stream(&counts, [1, 1], 8)
                );
            }
        }};
    }
    check_counting!(tile_f32, Type::F32(pb::F32::default()), f32::NAN, 4);
    check_counting!(tile_f32, Type::Bf16(pb::Bf16::default()), -13.0_f32, 2);
    check_counting!(tile_u64, Type::U64(pb::U64::default()), 21_u64, 8);
    check_counting!(tile_i64, Type::I64(pb::I64::default()), -99_i64, 8);
    check_counting!(tile_bool, Type::Bool(pb::Bool::default()), false, 1);
}

#[test]
fn increment_counts_inputs_without_payload_data() {
    let input: Vec<Elem<Tile<f32>>> = vec![
        Elem::ValStop(Tile::new_blank(vec![0, 8], 4, false), 1),
        Elem::ValStop(Tile::new_blank(vec![7, 8], 4, false), 2),
    ];
    check_operator!(
        tile_f32,
        tile_u64,
        scan_operation(
            Type::F32(pb::F32::default()),
            Some(counter_type()),
            accum_func::AccumFn::Increment(pb::Increment {}),
            1,
            false
        ),
        input.clone(),
        stream(&[(0, Some(1)), (0, Some(2))], [1, 1], 8)
    );
    check_operator!(
        tile_f32,
        tile_u64,
        count_operation(Type::F32(pb::F32::default()), 1),
        input,
        stream(&[(1, None), (1, Some(1))], [1, 1], 8)
    );
}

#[test]
fn integer_scan_adds_and_resets_with_exact_values() {
    let large = (1_u64 << 54) + 1;
    check_operator!(
        tile_u64,
        tile_u64,
        scan_operation(
            counter_type(),
            Some(counter_type()),
            accum_func::AccumFn::Add(pb::Add {}),
            1,
            true
        ),
        stream(&[(large, None), (1, Some(1)), (3, Some(2))], [1, 1], 8),
        stream(
            &[(large, None), (large + 1, Some(1)), (3, Some(2))],
            [1, 1],
            8
        )
    );
    check_operator!(
        tile_i64,
        tile_i64,
        scan_operation(
            Type::I64(pb::I64::default()),
            Some(Type::I64(pb::I64::default())),
            accum_func::AccumFn::Add(pb::Add {}),
            1,
            false
        ),
        stream(&[(-7_i64, None), (4, Some(1)), (-9, Some(2))], [1, 1], 8),
        stream(&[(0_i64, None), (-7, Some(1)), (0, Some(2))], [1, 1], 8)
    );
}

#[test]
fn integer_max_scan_uses_the_smallest_value_as_its_identity() {
    check_operator!(
        tile_i64,
        tile_i64,
        scan_operation(
            Type::I64(pb::I64::default()),
            Some(Type::I64(pb::I64::default())),
            accum_func::AccumFn::Max(pb::Max {}),
            1,
            true
        ),
        stream(&[(-7_i64, None), (-4, Some(1)), (-9, Some(2))], [1, 1], 8),
        stream(&[(-7_i64, None), (-4, Some(1)), (-9, Some(2))], [1, 1], 8)
    );
}

#[test]
fn scan_accepts_legacy_protos_without_an_output_type() {
    check_operator!(
        tile_f32,
        tile_f32,
        scan_operation(
            Type::F32(pb::F32::default()),
            None,
            accum_func::AccumFn::Add(pb::Add {}),
            1,
            false
        ),
        stream(
            &[
                (21.0_f32, None),
                (22.0, Some(1)),
                (23.0, None),
                (24.0, None),
                (25.0, Some(2))
            ],
            [1, 1],
            4
        ),
        stream(
            &[
                (0.0_f32, None),
                (21.0, Some(1)),
                (0.0, None),
                (23.0, None),
                (47.0, Some(2))
            ],
            [1, 1],
            4
        )
    );
}
