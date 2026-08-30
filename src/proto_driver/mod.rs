pub mod configs;
pub mod proto_headers;

use crate::functions;
use crate::memory::dyn_linear_offchip_load::DynLinearOffChipLoad;
use crate::memory::dyn_offchip_store::DynOffChipStore;
use crate::memory::linear_offchip_load_ref::LinearOffChipLoadRef;
use crate::memory::metadata_gen::MetadataGen;
use crate::memory::random_offchip_load::RandomOffChipLoad;
use crate::memory::random_offchip_store::RandomOffChipStore;
use crate::operator::accum_buff::AccumBuff;
use crate::operator::take_last::TakeLast;
use crate::operator::scan::{Scan, ScanConfig};
use crate::operator::eager_merge::EagerMerge;
use crate::operator::expand::ExpandRef;
use crate::operator::flatmap_decomp::{
    FlatmapCounter, FlatmapFilterRowStreamify, FlatmapRowStreamify,
};
use crate::operator::parallelize::Parallelize;
use crate::primitives::select::MultiHotN;
use std::collections::HashMap;
use std::time::Instant;

use crate::operator::accum::{Accum, AccumConfig};
use crate::operator::broadcast::BroadcastContext;
use crate::operator::bufferize::Bufferize;
use crate::operator::dynstreamify::DynStreamify;
use crate::operator::flatmap::{CacheReadAddrGen, ExpertAddrGen, FilterLastTile, RetileStreamify};
use crate::operator::flatten::Flatten;
use crate::operator::map::{
    BinaryMapMultiHot, UnaryMap, UnaryMapConfig, UnaryMapMultiHot, UnaryMapToMultiHot,
};
use crate::operator::map_accum::BinaryMapAccum;
use crate::operator::partition::{FlatPartition, FlatPartitionConfig};
use crate::operator::promote::{Promote, PromoteOuter};
use crate::operator::reassemble::{FlatReassemble, FlatReassembleConfig};
use crate::operator::reshape::{Reshape, ReshapeNoPadStream, ReshapePadStream};
use crate::operator::static_reassemble::StaticReassemble;
use crate::operator::streamify::{StaticStreamify, Streamify};
use crate::proto_driver::proto_headers::graph_proto::map_accum_func;
use crate::utils::select_npy::read_multihot_elem_from_npy_iter;
use dam::simulation::{
    DotConvertible, LogFilterKind, LoggingOptions, ProgramBuilder,
    RunOptionsBuilder,
};
use dam::utility_contexts::{ConsumerContext, GeneratorContext, PrinterContext};
use std::sync::Arc;
use std::usize;

use crate::build_sim::channel::ChannelMapCollection;
use crate::memory::linear_offchip_load::LinearOffChipLoad;
use crate::memory::offchip_store::OffChipStore;
use crate::operator::{
    map::BinaryMap,
    repeat::{RepeatRef, RepeatStatic},
};
use crate::primitives::tile::Tile;
use crate::proto_driver::configs::SimConfig;
use crate::proto_driver::proto_headers::graph_proto::{
    accum_func, buffer, data_type::Type, elemto_elem_func, init_func, operation::OpType,
    ProgramGraph,
};
use crate::ramulator::hbm_context::{HBMConfig, HBMContext, ReadBundle, WriteBundle};
use crate::utils::{
    cast::{to_u64_vec, to_usize_vec},
    events::SimpleEvent,
};

/// Wraps `ProgramBuilder::add_child`, additionally recording the node (type
/// name + current proto op id + DAM id) and the channels captured for it into
/// the optional graph dump (see `crate::utils::graph_dump`). When dumping is
/// disabled this is just a plain `add_child`.
macro_rules! add_child {
    ($builder:expr, $node:expr) => {{
        let __child = $node;
        $crate::utils::graph_dump::record_node(&__child);
        $builder.add_child(__child);
    }};
}

// channel_depth will be set from sim_config.channel_depth
macro_rules! make_flatmap_counter {
    ($collection:expr, $operation: expr, $flatmap_counter: expr, $type:ident, $builder:expr, $channel_depth:expr) => {
        let rcv = $collection.$type.get_receiver(
            $flatmap_counter.input_id,
            $flatmap_counter.stream_idx,
            $builder,
            $channel_depth,
        );
        let snd = $collection
            .$type
            .get_sender($operation.id, None, $builder, $channel_depth);
        add_child!($builder, FlatmapCounter::new(rcv, snd, $operation.id));
    };
}

macro_rules! make_broadcast {
    ($collection:expr, $operation: expr, $broadcast: expr, $type:ident, $builder:expr, $channel_depth:expr) => {
        let rcv = $collection.$type.get_receiver(
            $broadcast.input_id,
            $broadcast.stream_idx,
            $builder,
            $channel_depth,
        );
        let mut broadcast_node = BroadcastContext::new(rcv);
        for stream_idx in 0..$broadcast.num_consumers {
            let snd = $collection.$type.get_sender(
                $operation.id,
                Some(stream_idx),
                $builder,
                $channel_depth,
            );
            broadcast_node.add_target(snd);
        }

        add_child!($builder, broadcast_node);
    };
}

macro_rules! make_linear_offchip_load_ref {
    ($collection:expr, $operation: expr, $dyn_offchip_load: expr,$hbm_config: expr,
     $type_ref:ident, $type:ident, $n_bytes: expr,$mem_context: expr, $builder:expr, $channel_depth:expr) => {
        let ref_rcv = $collection.$type_ref.get_receiver(
            $dyn_offchip_load.ref_id,
            $dyn_offchip_load.ref_stream_idx,
            $builder,
            $channel_depth,
        );

        let snd = $collection
            .$type
            .get_sender($operation.id, None, $builder, $channel_depth);

        let (addr_snd, addr_rcv) = $builder.unbounded();
        let (resp_snd, resp_rcv) = $builder.unbounded();

        add_child!(
            $builder,
            LinearOffChipLoadRef::<SimpleEvent, _, _>::new(
                to_usize_vec($dyn_offchip_load.tensor_shape_tiled),
                to_usize_vec($dyn_offchip_load.stride),
                to_usize_vec($dyn_offchip_load.out_shape_tiled),
                $dyn_offchip_load.npy_path,
                $dyn_offchip_load.tile_row as usize,
                $dyn_offchip_load.tile_col as usize,
                $n_bytes,
                0,
                $hbm_config.addr_offset,
                $dyn_offchip_load.par_dispatch as usize,
                ref_rcv,
                addr_snd,
                resp_rcv,
                snd,
                $dyn_offchip_load.transposed,
                $operation.id,
                $dyn_offchip_load.trigger_rank,
            )
        );
        $mem_context.add_reader(ReadBundle {
            addr: addr_rcv,
            resp: resp_snd,
        });
    };
}

fn get_chan_depth(
    custom_depth_chan: &HashMap<u32, usize>,
    id: u32,
    base_depth: Option<usize>,
) -> Option<usize> {
    if custom_depth_chan.contains_key(&id) {
        Some(custom_depth_chan[&id])
    } else {
        base_depth
    }
}

/// Does any operator read `scan_id`'s PRIOR tap (stream_idx 1)?
///
/// The Scan lowering always produces both taps, but most graphs read only
/// `next`: scan_l and scan_o expose `prior` and nothing consumes it, while
/// scan_m has both consumed. Building a sender for an unread tap leaves a
/// channel with no receiver, which the runtime reports as DisconnectedReceiver
/// and which aborts the whole simulation.
fn scan_prior_is_consumed(step_graph: &ProgramGraph, scan_id: u32) -> bool {
    step_graph.operators.iter().any(|o| {
        let t = format!("{:?}", o);
        [
            format!("input_id: {}, stream_idx: Some(1)", scan_id),
            format!("input_id1: {}, stream_idx1: Some(1)", scan_id),
            format!("input_id2: Some({}), stream_idx2: Some(1)", scan_id),
        ]
        .iter()
        .any(|pat| t.contains(pat.as_str()))
    })
}

fn build_from_proto<'a>(
    step_graph: &ProgramGraph,
    channel_map_collection: &mut ChannelMapCollection<'a>,
    builder: &mut ProgramBuilder<'a>,
    hbm_config: &HBMConfig,
    sim_config: &SimConfig,
    dump_prefix: Option<String>,
) {
    let channel_depth = sim_config.channel_depth;
    let mut mem_context = HBMContext::new(builder, hbm_config.clone());

    // Use a regular variable instead of a const, since sim_config.mock_bf16 is not a constant
    let f32_bytes: usize = if sim_config.mock_bf16 { 2 } else { 4 }; // we will use this to mimic bfloat16

    // Graph dump (file 1: proto operators). Built here, before the loop below.
    // `begin()` arms the per-node channel capture used by the `add_child!`
    // macro and the `channel.rs` hooks.
    let mut proto_dump = String::new();
    if dump_prefix.is_some() {
        for operation in &step_graph.operators {
            proto_dump.push_str(&format!("processing {:?}\n\n", operation));
        }
        crate::utils::graph_dump::begin();
    }

    // Clones rather than moving: `step_graph` is a `&ProgramGraph` here because
    // the caller still needs it after the run, for
    // `crate::trace::write_graph_json_if_requested`.
    for operation in step_graph.operators.clone() {
        crate::utils::graph_dump::set_current_op(operation.id);
        // if operation.id == 23 || operation.id == 24 || operation.id == 25 {
        //     println!("processing {:?}\n", operation);
        // }
        match operation.op_type.clone().unwrap() {
            OpType::Unarymap(unarymap) => match (
                unarymap.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                unarymap.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                (Type::F32(_), Type::F32(_)) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<f32>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Silu(silu) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::silu(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Exp(exp) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::exp(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Pow2(pow2) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::pow2(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Rsqrt(rsqrt) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::rsqrt(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::RowWiseSum(row_wise_sum) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::row_wise_sum(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::RowWiseMax(row_wise_max) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::row_wise_max(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::MulConstant(mul_constant) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::mul_constant(
                                    tile,
                                    mul_constant.constant_float.unwrap() as f32,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::AddConstant(add_constant) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::add_constant(
                                    tile,
                                    add_constant.constant_float.unwrap() as f32,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::BroadcastRows(broadcast_rows) => {
                            let row_size = broadcast_rows.row_size as usize;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::broadcast_rows(
                                    tile,
                                    row_size,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::Pow(pow) => {
                            let exponent = pow.exponent;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::pow(tile, exponent, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Tanh(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::tanh(tile, comp_bw, write_back_mu)
                            })
                        }
                        e => {
                            panic!("Unsupported unary map function type {:?}", e)
                        }
                    };

                    add_child!(
                        builder,
                        UnaryMap::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            map_fn,
                            UnaryMapConfig {
                                compute_bw: unarymap.compute_bw as u64,
                                write_back_mu: unarymap.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (Type::U64(_), Type::F32(_)) => {
                    let rcv = channel_map_collection.tile_u64.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<u64>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::MaskRow(mask_row) => {
                            let mock_bf16 = sim_config.mock_bf16.clone();
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::mask_row(
                                    tile,
                                    unarymap.write_back_mu,
                                    mask_row.row as usize,
                                    mask_row.col as usize,
                                    mock_bf16,
                                )
                            })
                        }
                        _ => {
                            panic!("Unsupported unary map function type")
                        }
                    };

                    add_child!(
                        builder,
                        UnaryMap::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            map_fn,
                            UnaryMapConfig {
                                compute_bw: unarymap.compute_bw as u64,
                                write_back_mu: unarymap.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (Type::U64(_), Type::U64(_)) => {
                    let rcv = channel_map_collection.tile_u64.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_u64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<u64>, u64, bool) -> (u64, Tile<u64>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::MulConstant(mul_constant) => {
                            Arc::new(move |tile1, comp_bw, write_back_mu| {
                                functions::map_fn::mul_constant(
                                    tile1,
                                    mul_constant.constant.unwrap() as u64,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::AddConstant(add_constant) => {
                            Arc::new(move |tile1, comp_bw, write_back_mu| {
                                functions::map_fn::add_constant(
                                    tile1,
                                    add_constant.constant.unwrap() as u64,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::SubConstant(sub_constant) => {
                            Arc::new(move |tile1, comp_bw, write_back_mu| {
                                functions::map_fn::sub_constant(
                                    tile1,
                                    sub_constant.constant.unwrap() as u64,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::ToConstInt(to_const_int) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::to_const_int(
                                    tile,
                                    to_const_int.constant as u64,
                                    write_back_mu,
                                )
                            })
                        }
                        _ => {
                            panic!("Unsupported unary map function type")
                        }
                    };

                    add_child!(
                        builder,
                        UnaryMap::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            map_fn,
                            UnaryMapConfig {
                                compute_bw: unarymap.compute_bw as u64,
                                write_back_mu: unarymap.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (Type::MultiHot(_), Type::U64(_)) => {
                    let rcv = channel_map_collection.multihot.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_u64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&MultiHotN, u64, bool) -> (u64, Tile<u64>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::SelectToScalar(select_to_scalar) => {
                            Arc::new(move |multihot, comp_bw, write_back_mu| {
                                functions::map_fn::select_to_scalar(
                                    multihot,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::MultihotToU64(_) => {
                            Arc::new(move |multihot, comp_bw, write_back_mu| {
                                functions::map_fn::multihot_to_u64(multihot, comp_bw, write_back_mu)
                            })
                        }
                        _ => {
                            panic!("Unsupported unary map function type")
                        }
                    };

                    add_child!(
                        builder,
                        UnaryMapMultiHot::<SimpleEvent, _>::new(
                            rcv,
                            snd,
                            map_fn,
                            UnaryMapConfig {
                                compute_bw: unarymap.compute_bw as u64,
                                write_back_mu: unarymap.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (Type::U64(_), Type::MultiHot(_)) => {
                    let rcv = channel_map_collection.tile_u64.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.multihot.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<u64>, u64, bool) -> (u64, MultiHotN) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::U64ToMultihot(u64_to_multihot) => {
                            let width = u64_to_multihot.width as usize;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::u64_to_multihot(
                                    tile,
                                    width,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        _ => {
                            panic!("Unsupported unary map function type")
                        }
                    };

                    add_child!(
                        builder,
                        UnaryMapToMultiHot::<SimpleEvent, _>::new(
                            rcv,
                            snd,
                            map_fn,
                            UnaryMapConfig {
                                compute_bw: unarymap.compute_bw as u64,
                                write_back_mu: unarymap.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (_, _) => panic!("Unsupported data types for UnaryMap operation yet"),
            },
            OpType::Binarymap(binary_map) => match (
                binary_map.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                binary_map.dtype_b.clone().unwrap().r#type.clone().unwrap(),
                binary_map
                    .dtype_out
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap(),
            ) {
                (Type::F32(_), Type::F32(_), Type::F32(_)) => {
                    // create
                    let rcv1 = channel_map_collection.tile_f32.get_receiver(
                        binary_map.input_id1,
                        binary_map.stream_idx1,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id1,
                            channel_depth,
                        ),
                    );
                    let rcv2 = channel_map_collection.tile_f32.get_receiver(
                        binary_map.input_id2,
                        binary_map.stream_idx2,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id2,
                            channel_depth,
                        ),
                    );
                    let snd = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Matmul(matmul) => {
                            let weight_transposed = matmul.weight_transposed;
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::matmul(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    weight_transposed,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::DynMatmul(matmul) => {
                            let weight_transposed = matmul.weight_transposed;
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::matmul(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    weight_transposed,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::Mul(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::mul(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::RowWiseAppend(row_wise_append) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::row_wise_append(tile1, tile2, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::ColWiseAppend(col_wise_append) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::col_wise_append(tile1, tile2, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Div(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::div(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Add(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::add(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Sub(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::sub(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Max(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::max(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        e => {
                            panic!("Unsupported binary map function type {:?}", e)
                        }
                    };
                    add_child!(
                        builder,
                        BinaryMap::<SimpleEvent, _, _, _>::new(
                            rcv1,
                            rcv2,
                            snd,
                            map_fn,
                            binary_map.compute_bw as u64,
                            binary_map.write_back_mu,
                            operation.id,
                        )
                    );
                }
                (Type::U64(_), Type::U64(_), Type::U64(_)) => {
                    // create
                    let rcv1 = channel_map_collection.tile_u64.get_receiver(
                        binary_map.input_id1,
                        binary_map.stream_idx1,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id1,
                            channel_depth,
                        ),
                    );
                    let rcv2 = channel_map_collection.tile_u64.get_receiver(
                        binary_map.input_id2,
                        binary_map.stream_idx2,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id2,
                            channel_depth,
                        ),
                    );
                    let snd = channel_map_collection.tile_u64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<u64>, &Tile<u64>, u64, bool) -> (u64, Tile<u64>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::CacheWriteAddrGen(cache_write_addr_gen) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::cache_write_addr_gen(
                                    tile1,
                                    tile2,
                                    cache_write_addr_gen.offset_per_idx,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::Add(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::add(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        e => {
                            panic!("Unsupported binary map function type {:?}", e)
                        }
                    };
                    add_child!(
                        builder,
                        BinaryMap::<SimpleEvent, _, _, _>::new(
                            rcv1,
                            rcv2,
                            snd,
                            map_fn,
                            binary_map.compute_bw as u64,
                            binary_map.write_back_mu,
                            operation.id,
                        )
                    );
                }
                (Type::U64(_), Type::U64(_), Type::MultiHot(_)) => {
                    // create
                    let rcv1 = channel_map_collection.tile_u64.get_receiver(
                        binary_map.input_id1,
                        binary_map.stream_idx1,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id1,
                            channel_depth,
                        ),
                    );
                    let rcv2 = channel_map_collection.tile_u64.get_receiver(
                        binary_map.input_id2,
                        binary_map.stream_idx2,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id2,
                            channel_depth,
                        ),
                    );
                    let snd = channel_map_collection.multihot.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<u64>, &Tile<u64>, u64, bool) -> (u64, MultiHotN) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::IsEqual(is_equal) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::is_equal_scalar(tile1, tile2, write_back_mu)
                            })
                        }
                        e => {
                            panic!("Unsupported binary map function type {:?}", e)
                        }
                    };
                    add_child!(
                        builder,
                        BinaryMapMultiHot::<SimpleEvent, _, _>::new(
                            rcv1,
                            rcv2,
                            snd,
                            map_fn,
                            binary_map.compute_bw as u64,
                            binary_map.write_back_mu,
                            operation.id,
                        )
                    );
                }

                (Type::F32(_), Type::U64(_), Type::F32(_)) => {
                    // create
                    let rcv1 = channel_map_collection.tile_f32.get_receiver(
                        binary_map.input_id1,
                        binary_map.stream_idx1,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id1,
                            channel_depth,
                        ),
                    );
                    let rcv2 = channel_map_collection.tile_u64.get_receiver(
                        binary_map.input_id2,
                        binary_map.stream_idx2,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id2,
                            channel_depth,
                        ),
                    );
                    let snd = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<f32>, &Tile<u64>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::SetOffset(set_offset) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::set_offset(tile1, tile2, write_back_mu)
                            })
                        }
                        e => {
                            panic!("Unsupported binary map function type {:?}", e)
                        }
                    };
                    add_child!(
                        builder,
                        BinaryMap::<SimpleEvent, _, _, _>::new(
                            rcv1,
                            rcv2,
                            snd,
                            map_fn,
                            binary_map.compute_bw as u64,
                            binary_map.write_back_mu,
                            operation.id,
                        )
                    );
                }
                data_types => panic!(
                    "Unsupported data types for BinaryMap operation {:?}",
                    data_types
                ),
            },
            OpType::BinarymapAccum(binary_map_accum) => match (
                binary_map_accum
                    .dtype_a
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap(),
                binary_map_accum
                    .dtype_b
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap(),
            ) {
                (Type::F32(_), Type::F32(_)) => {
                    // create
                    let in1_stream = channel_map_collection.tile_f32.get_receiver(
                        binary_map_accum.input_id1,
                        binary_map_accum.stream_idx1,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map_accum.input_id1,
                            channel_depth,
                        ),
                    );
                    let in2_stream = channel_map_collection.tile_f32.get_receiver(
                        binary_map_accum.input_id2,
                        binary_map_accum.stream_idx2,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map_accum.input_id2,
                            channel_depth,
                        ),
                    );
                    let out_stream = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<f32>, &Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>)
                            + Send
                            + Sync,
                    > = match binary_map_accum.func.unwrap().map_accum_fn.unwrap() {
                        map_accum_func::MapAccumFn::Matmul(matmul) => {
                            let weight_transposed = matmul.weight_transposed;
                            Arc::new(move |tile1, tile2, accumulator, comp_bw, write_back_mu| {
                                functions::map_accum_fn::matmul(
                                    tile1,
                                    tile2,
                                    accumulator,
                                    comp_bw,
                                    write_back_mu,
                                    weight_transposed,
                                )
                            })
                        }
                        map_accum_func::MapAccumFn::DynMatmul(matmul) => {
                            let weight_transposed = matmul.weight_transposed;
                            Arc::new(move |tile1, tile2, accumulator, comp_bw, write_back_mu| {
                                functions::map_accum_fn::dyn_matmul(
                                    tile1,
                                    tile2,
                                    accumulator,
                                    comp_bw,
                                    write_back_mu,
                                    weight_transposed,
                                )
                            })
                        }
                        e => {
                            panic!("Unsupported binary map accumulation function type {:?}", e)
                        }
                    };

                    let tile_row = binary_map_accum.tile_row as usize;
                    let tile_col = binary_map_accum.tile_col as usize;

                    add_child!(
                        builder,
                        BinaryMapAccum::<SimpleEvent, _, _>::new(
                            in1_stream,
                            in2_stream,
                            out_stream,
                            map_fn,
                            Arc::new(move || {
                                Tile::new_zero(
                                    [tile_row, tile_col],
                                    f32_bytes,
                                    binary_map_accum.write_back_mu,
                                )
                            }),
                            binary_map_accum.rank,
                            binary_map_accum.compute_bw as u64,
                            binary_map_accum.write_back_mu,
                            operation.id,
                        )
                    );
                }
                (_, _) => todo!(),
            },
            OpType::LinearOffChipLoad(linear_off_chip_load) => {
                match linear_off_chip_load
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let on_chip_snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            LinearOffChipLoad::<SimpleEvent, _>::new(
                                to_usize_vec(linear_off_chip_load.tensor_shape_tiled),
                                to_usize_vec(linear_off_chip_load.stride),
                                to_usize_vec(linear_off_chip_load.out_shape_tiled),
                                linear_off_chip_load.npy_path,
                                linear_off_chip_load.tile_row as usize,
                                linear_off_chip_load.tile_col as usize,
                                f32_bytes,
                                0,
                                hbm_config.addr_offset,
                                linear_off_chip_load.par_dispatch as usize,
                                addr_snd,
                                resp_rcv,
                                on_chip_snd,
                                linear_off_chip_load.transposed,
                                operation.id,
                            )
                        );

                        mem_context.add_reader(ReadBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    Type::U64(_) => {
                        let on_chip_snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            LinearOffChipLoad::<SimpleEvent, _>::new(
                                to_usize_vec(linear_off_chip_load.tensor_shape_tiled),
                                to_usize_vec(linear_off_chip_load.stride),
                                to_usize_vec(linear_off_chip_load.out_shape_tiled),
                                linear_off_chip_load.npy_path,
                                linear_off_chip_load.tile_row as usize,
                                linear_off_chip_load.tile_col as usize,
                                std::mem::size_of::<u64>(),
                                0,
                                hbm_config.addr_offset,
                                linear_off_chip_load.par_dispatch as usize,
                                addr_snd,
                                resp_rcv,
                                on_chip_snd,
                                linear_off_chip_load.transposed,
                                operation.id,
                            )
                        );

                        mem_context.add_reader(ReadBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    _ => todo!(),
                }
            }
            OpType::DynLinearOffChipLoad(dyn_linear_off_chip_load) => {
                match dyn_linear_off_chip_load
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let on_chip_snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            DynLinearOffChipLoad::<SimpleEvent, _>::new(
                                dyn_linear_off_chip_load.shape_path,
                                dyn_linear_off_chip_load.npy_path,
                                dyn_linear_off_chip_load.tile_row as usize,
                                dyn_linear_off_chip_load.tile_col as usize,
                                f32_bytes,
                                0,
                                hbm_config.addr_offset,
                                dyn_linear_off_chip_load.par_dispatch as usize,
                                addr_snd,
                                resp_rcv,
                                on_chip_snd,
                                operation.id,
                            )
                        );

                        mem_context.add_reader(ReadBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    _ => todo!(),
                }
            }
            OpType::DynOffChipStore(dyn_off_chip_store) => {
                match dyn_off_chip_store
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let on_chip_rcv = channel_map_collection.tile_f32.get_receiver(
                            dyn_off_chip_store.input_id,
                            dyn_off_chip_store.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                dyn_off_chip_store.input_id,
                                channel_depth,
                            ),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            DynOffChipStore::<SimpleEvent, _>::new(
                                dyn_off_chip_store.shape_path,
                                dyn_off_chip_store.tile_row as usize,
                                dyn_off_chip_store.tile_col as usize,
                                dyn_off_chip_store.store_path,
                                0,
                                hbm_config.addr_offset,
                                dyn_off_chip_store.par_dispatch as usize,
                                on_chip_rcv,
                                addr_snd,
                                resp_rcv,
                                operation.id,
                            )
                        );

                        mem_context.add_writer(WriteBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    _ => todo!(),
                }
            }
            OpType::OffChipStore(off_chip_store) => {
                match off_chip_store
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let on_chip_rcv = channel_map_collection.tile_f32.get_receiver(
                            off_chip_store.input_id,
                            off_chip_store.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                off_chip_store.input_id,
                                channel_depth,
                            ),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            OffChipStore::<SimpleEvent, _>::new(
                                to_usize_vec(off_chip_store.tensor_shape_tiled),
                                off_chip_store.tile_row as usize,
                                off_chip_store.tile_col as usize,
                                off_chip_store.store_path,
                                0,
                                hbm_config.addr_offset,
                                off_chip_store.par_dispatch as usize,
                                on_chip_rcv,
                                addr_snd,
                                resp_rcv,
                                operation.id,
                            )
                        );

                        mem_context.add_writer(WriteBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    _ => todo!(),
                }
            }
            OpType::RandomOffChipStore(random_off_chip_store) => {
                match random_off_chip_store
                    .wdata_dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let waddr = channel_map_collection.tile_u64.get_receiver(
                            random_off_chip_store.waddr_id,
                            random_off_chip_store.waddr_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                random_off_chip_store.waddr_id,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    random_off_chip_store.waddr_id,
                                    channel_depth,
                                ),
                            ),
                        );
                        let wdata = channel_map_collection.tile_f32.get_receiver(
                            random_off_chip_store.wdata_id,
                            random_off_chip_store.wdata_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                random_off_chip_store.wdata_id,
                                channel_depth,
                            ),
                        );

                        let wack = match random_off_chip_store.has_done_stream {
                            true => Some(channel_map_collection.bool.get_sender(
                                operation.id,
                                None,
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            )),
                            false => None,
                        };
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            RandomOffChipStore::<SimpleEvent, _>::new(
                                to_usize_vec(random_off_chip_store.tensor_shape_tiled),
                                random_off_chip_store.npy_path,
                                random_off_chip_store.tile_row as usize,
                                random_off_chip_store.tile_col as usize,
                                f32_bytes,
                                0,
                                hbm_config.addr_offset,
                                random_off_chip_store.par_dispatch as usize,
                                addr_snd,
                                resp_rcv,
                                waddr,
                                wdata,
                                wack,
                                operation.id,
                                random_off_chip_store.ack_based_on_waddr,
                                random_off_chip_store.transposed,
                            )
                        );

                        mem_context.add_writer(WriteBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    _ => todo!(),
                }
            }
            OpType::RandomOffChipLoad(random_off_chip_load) => {
                match random_off_chip_load
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let raddr = channel_map_collection.tile_u64.get_receiver(
                            random_off_chip_load.raddr_id,
                            random_off_chip_load.raddr_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                random_off_chip_load.raddr_id,
                                channel_depth,
                            ),
                        );
                        let on_chip_snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        add_child!(
                            builder,
                            RandomOffChipLoad::<SimpleEvent, _>::new(
                                to_usize_vec(random_off_chip_load.tensor_shape_tiled),
                                random_off_chip_load.npy_path,
                                random_off_chip_load.tile_row as usize,
                                random_off_chip_load.tile_col as usize,
                                f32_bytes,
                                0,
                                hbm_config.addr_offset,
                                random_off_chip_load.par_dispatch as usize,
                                addr_snd,
                                resp_rcv,
                                raddr,
                                on_chip_snd,
                                random_off_chip_load.transposed,
                                operation.id,
                                random_off_chip_load.track_traffic,
                            )
                        );

                        mem_context.add_reader(ReadBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    _ => todo!(),
                }
            }
            OpType::ExpandRef(expand_ref) => {
                match (
                    expand_ref.dtype.clone().unwrap().r#type.clone().unwrap(),
                    expand_ref
                        .ref_dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                ) {
                    (Type::F32(_), Type::F32(_)) => {
                        let in_rcv = channel_map_collection.tile_f32.get_receiver(
                            expand_ref.input_id,
                            expand_ref.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                expand_ref.input_id,
                                channel_depth,
                            ),
                        );
                        let ref_rcv = channel_map_collection.tile_f32.get_receiver(
                            expand_ref.ref_id,
                            expand_ref.ref_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                expand_ref.ref_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            ExpandRef::<_, _>::new(
                                in_rcv,
                                ref_rcv,
                                expand_ref.expand_rank,
                                snd,
                                operation.id,
                            )
                        );
                    }
                    (Type::U64(_), Type::U64(_)) => {
                        let in_rcv = channel_map_collection.tile_u64.get_receiver(
                            expand_ref.input_id,
                            expand_ref.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                expand_ref.input_id,
                                channel_depth,
                            ),
                        );
                        let ref_rcv = channel_map_collection.tile_u64.get_receiver(
                            expand_ref.ref_id,
                            expand_ref.ref_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                expand_ref.ref_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            ExpandRef::<_, _>::new(
                                in_rcv,
                                ref_rcv,
                                expand_ref.expand_rank,
                                snd,
                                operation.id,
                            )
                        );
                    }
                    e => panic!("Unsupported data type for ExpandRef operation {:?}", e),
                }
            }
            OpType::RepeatStatic(repeat_static) => {
                match repeat_static.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            repeat_static.input_id,
                            repeat_static.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_static.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatStatic::<_>::new(rcv, repeat_static.repeat_factor as usize, snd,)
                        );
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            repeat_static.input_id,
                            repeat_static.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_static.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatStatic::<_>::new(rcv, repeat_static.repeat_factor as usize, snd,)
                        );
                    }
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_)),
                    }) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            repeat_static.input_id,
                            repeat_static.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_static.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.buff_tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatStatic::<_>::new(rcv, repeat_static.repeat_factor as usize, snd,)
                        );
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            repeat_static.input_id,
                            repeat_static.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_static.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.multihot.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatStatic::<_>::new(rcv, repeat_static.repeat_factor as usize, snd,)
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for RepeatStatic operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::RepeatRef(repeat_ref) => {
                match (
                    repeat_ref.dtype.clone().unwrap().r#type.clone().unwrap(),
                    repeat_ref
                        .ref_dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                ) {
                    (Type::F32(_), Type::F32(_)) => {
                        let in_rcv = channel_map_collection.tile_f32.get_receiver(
                            repeat_ref.input_id,
                            repeat_ref.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_ref.input_id,
                                channel_depth,
                            ),
                        );
                        let ref_rcv = channel_map_collection.tile_f32.get_receiver(
                            repeat_ref.ref_id,
                            repeat_ref.ref_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_ref.ref_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatRef::<_, _>::new(
                                in_rcv,
                                ref_rcv,
                                snd,
                                repeat_ref.rank,
                                operation.id,
                            )
                        );
                    }
                    (Type::U64(_), Type::U64(_)) => {
                        let in_rcv = channel_map_collection.tile_u64.get_receiver(
                            repeat_ref.input_id,
                            repeat_ref.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_ref.input_id,
                                channel_depth,
                            ),
                        );
                        let ref_rcv = channel_map_collection.tile_u64.get_receiver(
                            repeat_ref.ref_id,
                            repeat_ref.ref_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_ref.ref_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatRef::<_, _>::new(
                                in_rcv,
                                ref_rcv,
                                snd,
                                repeat_ref.rank,
                                operation.id,
                            )
                        );
                    }
                    (
                        Type::Buffer(proto_headers::graph_proto::Buffer {
                            r#type: Some(buffer::Type::F32(_)),
                        }),
                        Type::F32(_),
                    ) => {
                        let in_rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            repeat_ref.input_id,
                            repeat_ref.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_ref.input_id,
                                channel_depth,
                            ),
                        );
                        let ref_rcv = channel_map_collection.tile_f32.get_receiver(
                            repeat_ref.ref_id,
                            repeat_ref.ref_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                repeat_ref.ref_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.buff_tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RepeatRef::<_, _>::new(
                                in_rcv,
                                ref_rcv,
                                snd,
                                repeat_ref.rank,
                                operation.id,
                            )
                        );
                    }
                    e => panic!("Unsupported data type for RepeatRef operation {:?}", e),
                }
            }
            OpType::Broadcast(broadcast) => {
                match broadcast.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            tile_f32,
                            builder,
                            channel_depth
                        );
                    }
                    Type::U64(_) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            tile_u64,
                            builder,
                            channel_depth
                        );
                    }
                    Type::MultiHot(_) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            multihot,
                            builder,
                            channel_depth
                        );
                    }
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_)),
                    }) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            buff_tile_f32,
                            builder,
                            channel_depth
                        );
                    }
                    Type::ScalarU64(_) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            u64,
                            builder,
                            channel_depth
                        );
                    }
                    dtype => panic!("Unsupported data type for Broadcast operation {:?}", dtype),
                }
            }
            OpType::FlatPartition(flat_partition) => {
                match flat_partition
                    .input_dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let input_rcv = channel_map_collection.tile_f32.get_receiver(
                            flat_partition.input_id,
                            flat_partition.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flat_partition.input_id,
                                channel_depth,
                            ),
                        );
                        let mut snd_list = vec![];
                        for i in 0..flat_partition.num_consumers {
                            snd_list.push(channel_map_collection.tile_f32.get_sender(
                                operation.id,
                                Some(i),
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            ));
                        }

                        match flat_partition
                            .control_dtype
                            .clone()
                            .unwrap()
                            .r#type
                            .clone()
                            .unwrap()
                        {
                            Type::MultiHot(multi_hot) => {
                                let control_rcv = channel_map_collection.multihot.get_receiver(
                                    flat_partition.control_id,
                                    flat_partition.control_stream_idx,
                                    builder,
                                    channel_depth,
                                );
                                add_child!(
                                    builder,
                                    FlatPartition::<SimpleEvent, _, _>::new(
                                        input_rcv,
                                        control_rcv,
                                        snd_list,
                                        flat_partition.partition_rank,
                                        FlatPartitionConfig {
                                            switch_cycles: to_u64_vec(flat_partition.switch_cycles),
                                            write_back_mu: flat_partition.write_back_mu,
                                        },
                                        operation.id,
                                    )
                                )
                            }
                            dtype => panic!("Unsupported data type {:?}", dtype),
                        }
                    }
                    Type::U64(_) => {
                        let input_rcv = channel_map_collection.tile_u64.get_receiver(
                            flat_partition.input_id,
                            flat_partition.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flat_partition.input_id,
                                channel_depth,
                            ),
                        );
                        let mut snd_list = vec![];
                        for i in 0..flat_partition.num_consumers {
                            snd_list.push(channel_map_collection.tile_u64.get_sender(
                                operation.id,
                                Some(i),
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            ));
                        }

                        match flat_partition
                            .control_dtype
                            .clone()
                            .unwrap()
                            .r#type
                            .clone()
                            .unwrap()
                        {
                            Type::MultiHot(multi_hot) => {
                                let control_rcv = channel_map_collection.multihot.get_receiver(
                                    flat_partition.control_id,
                                    flat_partition.control_stream_idx,
                                    builder,
                                    channel_depth,
                                );
                                add_child!(
                                    builder,
                                    FlatPartition::<SimpleEvent, _, _>::new(
                                        input_rcv,
                                        control_rcv,
                                        snd_list,
                                        flat_partition.partition_rank,
                                        FlatPartitionConfig {
                                            switch_cycles: to_u64_vec(flat_partition.switch_cycles),
                                            write_back_mu: flat_partition.write_back_mu,
                                        },
                                        operation.id,
                                    )
                                )
                            }
                            dtype => panic!("Unsupported data type {:?}", dtype),
                        }
                    }
                    Type::MultiHot(_) => {
                        let input_rcv = channel_map_collection.multihot.get_receiver(
                            flat_partition.input_id,
                            flat_partition.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flat_partition.input_id,
                                channel_depth,
                            ),
                        );
                        let mut snd_list = vec![];
                        for i in 0..flat_partition.num_consumers {
                            snd_list.push(channel_map_collection.multihot.get_sender(
                                operation.id,
                                Some(i),
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            ));
                        }

                        match flat_partition
                            .control_dtype
                            .clone()
                            .unwrap()
                            .r#type
                            .clone()
                            .unwrap()
                        {
                            Type::MultiHot(multi_hot) => {
                                let control_rcv = channel_map_collection.multihot.get_receiver(
                                    flat_partition.control_id,
                                    flat_partition.control_stream_idx,
                                    builder,
                                    channel_depth,
                                );
                                add_child!(
                                    builder,
                                    FlatPartition::<SimpleEvent, _, _>::new(
                                        input_rcv,
                                        control_rcv,
                                        snd_list,
                                        flat_partition.partition_rank,
                                        FlatPartitionConfig {
                                            switch_cycles: to_u64_vec(flat_partition.switch_cycles),
                                            write_back_mu: flat_partition.write_back_mu,
                                        },
                                        operation.id,
                                    )
                                )
                            }
                            dtype => panic!("Unsupported data type {:?}", dtype),
                        }
                    }
                    dtype => panic!("Unsupported data type {:?}", dtype),
                }
            }
            OpType::FlatReassemble(reassemble) => {
                match reassemble
                    .input_dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(f32) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in reassemble
                            .input_id_list
                            .into_iter()
                            .zip(reassemble.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.tile_f32.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        match reassemble
                            .control_dtype
                            .clone()
                            .unwrap()
                            .r#type
                            .clone()
                            .unwrap()
                        {
                            Type::MultiHot(multi_hot) => {
                                let control_rcv = channel_map_collection.multihot.get_receiver(
                                    reassemble.control_id,
                                    reassemble.control_stream_idx,
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        reassemble.control_id,
                                        channel_depth,
                                    ),
                                );
                                add_child!(
                                    builder,
                                    FlatReassemble::<SimpleEvent, _, _>::new(
                                        rcv_list,
                                        control_rcv,
                                        snd,
                                        reassemble.reassemble_rank,
                                        FlatReassembleConfig {
                                            switch_cycles: to_u64_vec(reassemble.switch_cycles),
                                            write_back_mu: reassemble.write_back_mu,
                                        },
                                        operation.id,
                                    )
                                )
                            }
                            dtype => panic!("Unsupported data type {:?}", dtype),
                        }
                    }
                    Type::MultiHot(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in reassemble
                            .input_id_list
                            .into_iter()
                            .zip(reassemble.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.multihot.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let snd = channel_map_collection.multihot.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        match reassemble
                            .control_dtype
                            .clone()
                            .unwrap()
                            .r#type
                            .clone()
                            .unwrap()
                        {
                            Type::MultiHot(multi_hot) => {
                                let control_rcv = channel_map_collection.multihot.get_receiver(
                                    reassemble.control_id,
                                    reassemble.control_stream_idx,
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        reassemble.control_id,
                                        channel_depth,
                                    ),
                                );
                                add_child!(
                                    builder,
                                    FlatReassemble::<SimpleEvent, _, _>::new(
                                        rcv_list,
                                        control_rcv,
                                        snd,
                                        reassemble.reassemble_rank,
                                        FlatReassembleConfig {
                                            switch_cycles: to_u64_vec(reassemble.switch_cycles),
                                            write_back_mu: reassemble.write_back_mu,
                                        },
                                        operation.id,
                                    )
                                )
                            }
                            dtype => panic!("Unsupported data type {:?}", dtype),
                        }
                    }
                    dtype => panic!("Unsupported data type {:?}", dtype),
                }
            }
            OpType::StaticReassemble(static_reassemble) => {
                match static_reassemble
                    .input_dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in static_reassemble
                            .input_id_list
                            .into_iter()
                            .zip(static_reassemble.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.tile_f32.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            StaticReassemble::<SimpleEvent, _>::new(
                                rcv_list,
                                snd,
                                static_reassemble.merge_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(static_reassemble.switch_cycles),
                                    write_back_mu: static_reassemble.write_back_mu,
                                },
                                operation.id,
                            )
                        );
                    }
                    Type::U64(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in static_reassemble
                            .input_id_list
                            .into_iter()
                            .zip(static_reassemble.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.tile_u64.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            StaticReassemble::<SimpleEvent, _>::new(
                                rcv_list,
                                snd,
                                static_reassemble.merge_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(static_reassemble.switch_cycles),
                                    write_back_mu: static_reassemble.write_back_mu,
                                },
                                operation.id,
                            )
                        );
                    }
                    Type::MultiHot(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in static_reassemble
                            .input_id_list
                            .into_iter()
                            .zip(static_reassemble.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.multihot.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let snd = channel_map_collection.multihot.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            StaticReassemble::<SimpleEvent, _>::new(
                                rcv_list,
                                snd,
                                static_reassemble.merge_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(static_reassemble.switch_cycles),
                                    write_back_mu: static_reassemble.write_back_mu,
                                },
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for StaticReassemble operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::Parallelize(parallelize) => {
                match parallelize
                    .input_dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(f32) => {
                        let input_rcv = channel_map_collection.tile_f32.get_receiver(
                            parallelize.input_id,
                            parallelize.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                parallelize.input_id,
                                channel_depth,
                            ),
                        );
                        let mut snd_list = vec![];
                        for i in 0..parallelize.num_consumers {
                            snd_list.push(channel_map_collection.tile_f32.get_sender(
                                operation.id,
                                Some(i),
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            ));
                        }
                        add_child!(
                            builder,
                            Parallelize::<SimpleEvent, _>::new(
                                input_rcv,
                                snd_list,
                                parallelize.parallelize_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(parallelize.switch_cycles),
                                    write_back_mu: parallelize.write_back_mu,
                                },
                                operation.id,
                            )
                        )
                    }
                    Type::MultiHot(_) => {
                        let input_rcv = channel_map_collection.multihot.get_receiver(
                            parallelize.input_id,
                            parallelize.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                parallelize.input_id,
                                channel_depth,
                            ),
                        );
                        let mut snd_list = vec![];
                        for i in 0..parallelize.num_consumers {
                            snd_list.push(channel_map_collection.multihot.get_sender(
                                operation.id,
                                Some(i),
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            ));
                        }
                        add_child!(
                            builder,
                            Parallelize::<SimpleEvent, _>::new(
                                input_rcv,
                                snd_list,
                                parallelize.parallelize_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(parallelize.switch_cycles),
                                    write_back_mu: parallelize.write_back_mu,
                                },
                                operation.id,
                            )
                        )
                    }
                    Type::U64(u64) => {
                        let input_rcv = channel_map_collection.tile_u64.get_receiver(
                            parallelize.input_id,
                            parallelize.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                parallelize.input_id,
                                channel_depth,
                            ),
                        );
                        let mut snd_list = vec![];
                        for i in 0..parallelize.num_consumers {
                            snd_list.push(channel_map_collection.tile_u64.get_sender(
                                operation.id,
                                Some(i),
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    operation.id,
                                    channel_depth,
                                ),
                            ));
                        }
                        add_child!(
                            builder,
                            Parallelize::<SimpleEvent, _>::new(
                                input_rcv,
                                snd_list,
                                parallelize.parallelize_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(parallelize.switch_cycles),
                                    write_back_mu: parallelize.write_back_mu,
                                },
                                operation.id,
                            )
                        )
                    }
                    dtype => panic!("Unsupported data type {:?}", dtype),
                }
            }
            OpType::Promote(promote) => {
                match promote.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(f32) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            promote.input_id,
                            promote.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                promote.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(builder, Promote::new(rcv, snd, promote.promote_rank));
                    }
                    dtype => panic!("Unsupported data type {:?}", dtype),
                }
            }
            OpType::PromoteOuter(promote_outer) => {
                match promote_outer.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(f32) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            promote_outer.input_id,
                            promote_outer.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                promote_outer.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(builder, PromoteOuter::new(rcv, snd));
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            promote_outer.input_id,
                            promote_outer.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                promote_outer.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(builder, PromoteOuter::new(rcv, snd));
                    }
                    Type::Bool(_) => {
                        let rcv = channel_map_collection.tile_bool.get_receiver(
                            promote_outer.input_id,
                            promote_outer.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                promote_outer.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_bool.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(builder, PromoteOuter::new(rcv, snd));
                    }
                    dtype => panic!("Unsupported data type {:?}", dtype),
                }
            }
            OpType::ConsumerContext(consumer_context) => {
                match consumer_context
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    Type::ScalarU64(_) => {
                        let rcv = channel_map_collection.u64.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    Type::ScalarBool(_) => {
                        let rcv = channel_map_collection.bool.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    Type::Bool(_) => {
                        let rcv = channel_map_collection.tile_bool.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_)),
                    }) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, ConsumerContext::new(rcv));
                    }
                    dtype => panic!(
                        "Unsupported data type for ConsumerContext operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::PrinterContext(printer_context) => {
                match printer_context
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, PrinterContext::new(rcv));
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, PrinterContext::new(rcv));
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, PrinterContext::new(rcv));
                    }
                    Type::Bool(_) => {
                        let rcv = channel_map_collection.tile_bool.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, PrinterContext::new(rcv));
                    }
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_)),
                    }) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, PrinterContext::new(rcv));
                    }
                    dtype => panic!(
                        "Unsupported data type for PrinterContext operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::Bufferize(bufferize) => {
                match bufferize.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            bufferize.input_id,
                            bufferize.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                bufferize.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.buff_tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Bufferize::<SimpleEvent, _>::new(
                                rcv,
                                snd,
                                bufferize.rank,
                                operation.id,
                            )
                        );
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            bufferize.input_id,
                            bufferize.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                bufferize.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.buff_tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Bufferize::<SimpleEvent, _>::new(
                                rcv,
                                snd,
                                bufferize.rank,
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!("Unsupported data type for Bufferize operation {:?}", dtype),
                }
            }
            OpType::Streamify(streamify) => {
                match streamify.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            streamify.input_id,
                            streamify.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Streamify::<SimpleEvent, _>::new(
                                to_usize_vec(streamify.repeat_factor),
                                streamify.rank,
                                rcv,
                                snd,
                                operation.id,
                            )
                        );
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.buff_tile_u64.get_receiver(
                            streamify.input_id,
                            streamify.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Streamify::<SimpleEvent, _>::new(
                                to_usize_vec(streamify.repeat_factor),
                                streamify.rank,
                                rcv,
                                snd,
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!("Unsupported data type for Streamify operation {:?}", dtype),
                }
            }
            OpType::StaticStreamify(static_streamify) => {
                match static_streamify
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            static_streamify.input_id,
                            static_streamify.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                static_streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            StaticStreamify::<SimpleEvent, _>::new(
                                to_usize_vec(static_streamify.stride),
                                to_usize_vec(static_streamify.out_shape),
                                rcv,
                                snd,
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for StaticStreamify operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::DynStreamify(dyn_streamify) => {
                match (
                    dyn_streamify
                        .input_dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                    dyn_streamify
                        .ref_dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                ) {
                    (Type::F32(_), Type::F32(_)) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            dyn_streamify.input_id,
                            dyn_streamify.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                dyn_streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let ref_rcv = channel_map_collection.tile_f32.get_receiver(
                            dyn_streamify.ref_id,
                            dyn_streamify.ref_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                dyn_streamify.ref_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            DynStreamify::<SimpleEvent, _, _>::new(
                                rcv,
                                dyn_streamify.bufferized_rank,
                                dyn_streamify.repeat_rank,
                                ref_rcv,
                                snd,
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for DynStreamify operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::LinearOffChipLoadRef(linear_offchip_load_ref) => {
                match (
                    linear_offchip_load_ref
                        .dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                    linear_offchip_load_ref
                        .ref_dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                ) {
                    (Type::F32(_), Type::F32(_)) => {
                        make_linear_offchip_load_ref!(
                            channel_map_collection,
                            operation,
                            linear_offchip_load_ref,
                            hbm_config,
                            tile_f32,
                            tile_f32,
                            f32_bytes,
                            mem_context,
                            builder,
                            channel_depth
                        );
                    }
                    (
                        Type::F32(_),
                        Type::Buffer(proto_headers::graph_proto::Buffer {
                            r#type: Some(buffer::Type::F32(_)),
                        }),
                    ) => {
                        make_linear_offchip_load_ref!(
                            channel_map_collection,
                            operation,
                            linear_offchip_load_ref,
                            hbm_config,
                            buff_tile_f32,
                            tile_f32,
                            f32_bytes,
                            mem_context,
                            builder,
                            channel_depth
                        );
                    }
                    (Type::F32(_), Type::MultiHot(_)) => {
                        make_linear_offchip_load_ref!(
                            channel_map_collection,
                            operation,
                            linear_offchip_load_ref,
                            hbm_config,
                            multihot,
                            tile_f32,
                            f32_bytes,
                            mem_context,
                            builder,
                            channel_depth
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for LinearOffChipLoadRef operation {:?}",
                        dtype
                    ),
                }
            }
            // TakeLast: keep only the final tile of the scanned axis. Needed for
            // the FlashAttention decode graph (l_final / O_final taps); step_perf
            // previously had NO arm for this op, so it fell through to
            // `_ => todo!()` and any FA graph panicked with "not yet implemented"
            // under functional_sim.
            //
            // LIMITATION: implements keep_last = 1 (unchunked). With
            // chunk_factor = C > 1 the STeP node declares keep_last = C and the C
            // chunk contexts interleave on the stream; that is not modelled yet.
            // Validate C=1 against the naive layer before trusting C>1 output.
            OpType::TakeLast(take_last) => {
                if std::env::var("STEP_PERF_OP_TRACE").is_ok() {
                    eprintln!("[BUILD TakeLast id={} in={} idx={:?}]",
                        operation.id, take_last.input_id, take_last.stream_idx);
                }
                match take_last.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            take_last.input_id,
                            take_last.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                take_last.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(builder, TakeLast::new(rcv, snd));
                    }
                    _ => todo!("TakeLast: only F32 is wired"),
                }
            }
            // Scan: prefix-scan loop with a loop-carried value and TWO output taps
            // (stream 0 = "next"/after the fold, stream 1 = "prior"/before).
            // step_perf had NO arm for this op, so every FlashAttention graph
            // panicked "not yet implemented" under functional_sim.
            //
            // Invocation boundaries come from the input's STOP TOKENS, not from
            // `ctr` -- see the operator's header. `ctr` is still dequeued once per
            // invocation so its producer does not block.
            //
            // chunk_factor > 1 is REJECTED by the operator rather than silently
            // folding all C chunk recurrences into one. Validate C=1 FA against
            // the naive layer before trusting anything here.
            OpType::Scan(scan) => {
                if std::env::var("STEP_PERF_OP_TRACE").is_ok() {
                    eprintln!("[BUILD Scan id={} in1={} in2={:?} ctr={} ctr_idx={:?}]",
                        operation.id, scan.input_id1, scan.input_id2, scan.ctr_id, scan.ctr_stream_idx);
                }
                let (ta, tb) = (
                    scan.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                    scan.dtype_b.clone().unwrap().r#type.clone().unwrap(),
                );
                match (ta, tb) {
                    (Type::F32(_), Type::F32(_)) => {
                        let in1 = channel_map_collection.tile_f32.get_receiver(
                            scan.input_id1,
                            scan.stream_idx1,
                            builder,
                            get_chan_depth(&sim_config.config_dict, scan.input_id1, channel_depth),
                        );
                        let in2 = scan.input_id2.map(|id2| {
                            channel_map_collection.tile_f32.get_receiver(
                                id2,
                                scan.stream_idx2,
                                builder,
                                get_chan_depth(&sim_config.config_dict, id2, channel_depth),
                            )
                        });
                        let ctr = channel_map_collection.tile_u64.get_receiver(
                            scan.ctr_id,
                            scan.ctr_stream_idx,
                            builder,
                            get_chan_depth(&sim_config.config_dict, scan.ctr_id, channel_depth),
                        );
                        // Build a tap ONLY if some operator reads it. An unread
                        // tap becomes a channel with a sender and no receiver,
                        // which the runtime reports as DisconnectedReceiver and
                        // which aborts the simulation. scan_l / scan_o expose
                        // `prior` but nothing consumes it.
                        let consumed = |idx: u32| -> bool {
                            step_graph.operators.iter().any(|o| {
                                let t = format!("{:?}", o);
                                t.contains(&format!("input_id1: {}", operation.id))
                                    || t.contains(&format!("input_id: {}", operation.id))
                                    || t.contains(&format!("input_id2: {}", operation.id))
                            }) && step_graph.operators.iter().any(|o| {
                                let t = format!("{:?}", o);
                                t.contains(&format!("stream_idx: Some({})", idx))
                                    || t.contains(&format!("stream_idx1: Some({})", idx))
                                    || t.contains(&format!("stream_idx2: Some({})", idx))
                            })
                        };
                        let _ = &consumed;
                        let next_snd = Some(channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            Some(0),
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        ));
                        let prior_snd = if scan_prior_is_consumed(step_graph, operation.id) {
                            Some(channel_map_collection.tile_f32.get_sender(
                                operation.id,
                                Some(1),
                                builder,
                                get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                            ))
                        } else {
                            None
                        };

                        fn pick(
                            f: elemto_elem_func::ElemElemFn,
                        ) -> Arc<
                            dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>)
                                + Send
                                + Sync,
                        > {
                            match f {
                                elemto_elem_func::ElemElemFn::Add(_) => Arc::new(
                                    move |a, b, bw, w| functions::map_fn::add(a, b, bw, w),
                                ),
                                elemto_elem_func::ElemElemFn::Mul(_) => Arc::new(
                                    move |a, b, bw, w| functions::map_fn::mul(a, b, bw, w),
                                ),
                                elemto_elem_func::ElemElemFn::Max(_) => Arc::new(
                                    move |a, b, bw, w| functions::map_fn::max(a, b, bw, w),
                                ),
                                other => todo!("Scan: fold fn {:?} not wired", other),
                            }
                        }

                        let f1 = pick(scan.func1.clone().unwrap().elem_elem_fn.unwrap());
                        let f2 = scan
                            .func2
                            .clone()
                            .and_then(|f| f.elem_elem_fn)
                            .map(pick);

                        let tile_row = scan.tile_row as usize;
                        let tile_col = scan.tile_col as usize;
                        let wbm = scan.write_back_mu;
                        let init: Arc<dyn Fn() -> Tile<f32> + Send + Sync> =
                            if sim_config.functional_sim {
                                match scan.init_func.clone().unwrap().init_fn.unwrap() {
                                    init_func::InitFn::Zero(_) => Arc::new(move || {
                                        Tile::new_zero([tile_row, tile_col], f32_bytes, wbm)
                                    }),
                                    init_func::InitFn::Empty(_) => Arc::new(move || {
                                        Tile::new_empty([tile_row, tile_col], f32_bytes, wbm)
                                    }),
                                    other => todo!("Scan: init {:?} not wired", other),
                                }
                            } else {
                                Arc::new(move || {
                                    Tile::new_blank(vec![tile_row, tile_col], f32_bytes, wbm)
                                })
                            };

                        add_child!(
                            builder,
                            Scan::<SimpleEvent, f32, f32>::new(
                                in1,
                                in2,
                                ctr,
                                next_snd,
                                prior_snd,
                                f1,
                                f2,
                                init,
                                // step-perf bundles its OWN copy of the proto
                                // (step-perf/step_perf_ir/) and that copy PREDATES
                                // the flash-decoding `chunk_factor` field, so it is
                                // not visible here at all. Passing 1 is therefore
                                // the only option -- but it means a C>1 graph is
                                // INDISTINGUISHABLE from C=1 to step_perf and would
                                // be scanned as unchunked, producing confidently
                                // WRONG numbers. Regenerate step-perf's proto before
                                // validating any chunked graph on this path.
                                1,
                                ScanConfig {
                                    compute_bw: scan.compute_bw as u64,
                                    write_back_mu: scan.write_back_mu,
                                },
                                operation.id,
                            )
                        );
                    }
                    other => todo!("Scan: dtype pair {:?} not wired", other),
                }
            }
            OpType::Flatten(flatten) => {
                match flatten.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            flatten.input_id,
                            flatten.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flatten.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Flatten::new(rcv, snd, flatten.min_rank, flatten.max_rank,)
                        );
                    }
                    Type::Bool(_) => {
                        let rcv = channel_map_collection.tile_bool.get_receiver(
                            flatten.input_id,
                            flatten.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flatten.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_bool.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Flatten::new(rcv, snd, flatten.min_rank, flatten.max_rank,)
                        );
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            flatten.input_id,
                            flatten.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flatten.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.multihot.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Flatten::new(rcv, snd, flatten.min_rank, flatten.max_rank,)
                        );
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            flatten.input_id,
                            flatten.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flatten.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            Flatten::new(rcv, snd, flatten.min_rank, flatten.max_rank,)
                        );
                    }

                    dtype => panic!("Unsupported data type for Flatten operation {:?}", dtype),
                }
            }
            OpType::SelectGen(select_gen) => match select_gen.is_multihot {
                true => {
                    let snd = channel_map_collection.multihot.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    add_child!(
                        builder,
                        GeneratorContext::new(
                            move || {
                                read_multihot_elem_from_npy_iter::<i64>(&select_gen.npy_path)
                                    .unwrap()
                            },
                            snd.into_sender(),
                        )
                    );
                }
                false => todo!("Add the same version for IndexN"),
            },
            OpType::Accum(accum) => match (
                accum.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                accum.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                (Type::F32(_), Type::F32(_)) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
                        accum.input_id,
                        accum.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, accum.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let func: Arc<
                        dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match accum.func.unwrap().accum_fn.unwrap() {
                        accum_func::AccumFn::Add(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::add(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        accum_func::AccumFn::RetileRow(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::retile_row(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        accum_func::AccumFn::RetileCol(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::retile_col(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        _ => todo!(),
                    };

                    let tile_row = accum.tile_row as usize;
                    let tile_col = accum.tile_col as usize;

                    let init_accum: Arc<dyn Fn() -> Tile<f32> + Send + Sync> = if sim_config
                        .functional_sim
                    {
                        match accum.init_func.unwrap().init_fn.unwrap() {
                            init_func::InitFn::Zero(_zero) => Arc::new(move || {
                                Tile::new_zero([tile_row, tile_col], f32_bytes, accum.write_back_mu)
                            }),
                            init_func::InitFn::Empty(_empty) => Arc::new(move || {
                                Tile::new_empty(
                                    [tile_row, tile_col],
                                    f32_bytes,
                                    accum.write_back_mu,
                                )
                            }),
                            init_func::InitFn::DynEmpty(_) => Arc::new(move || {
                                // DynEmpty means the row or the column size is known at run-time.
                                // Therefore, we will use the size of the first tile and keep the initial accumulator as [0,0]
                                Tile::new_empty([0, 0], f32_bytes, accum.write_back_mu)
                            }),
                            _ => todo!(),
                        }
                    } else {
                        Arc::new(move || {
                            Tile::new_blank(
                                vec![tile_row, tile_col],
                                f32_bytes,
                                accum.write_back_mu,
                            )
                        })
                    };

                    add_child!(
                        builder,
                        Accum::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            func,
                            init_accum,
                            accum.rank,
                            AccumConfig {
                                compute_bw: accum.compute_bw as u64,
                                write_back_mu: accum.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (Type::F32(_), Type::U64(_)) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
                        accum.input_id,
                        accum.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, accum.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_u64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let func: Arc<
                        dyn Fn(&Tile<f32>, &Tile<u64>, u64, bool) -> (u64, Tile<u64>) + Send + Sync,
                    > = match accum.func.unwrap().accum_fn.unwrap() {
                        accum_func::AccumFn::SignalReqAllRead(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::signal_req_all_read(
                                    tile1,
                                    tile2,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        _ => todo!(),
                    };

                    let tile_row = accum.tile_row as usize;
                    let tile_col = accum.tile_col as usize;

                    let init_accum = Arc::new(move || {
                        Tile::new_blank(vec![tile_row, tile_col], 8, accum.write_back_mu)
                    });

                    add_child!(
                        builder,
                        Accum::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            func,
                            init_accum,
                            accum.rank,
                            AccumConfig {
                                compute_bw: accum.compute_bw as u64,
                                write_back_mu: accum.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                (Type::Bool(_), Type::Bool(_)) => {
                    let rcv = channel_map_collection.tile_bool.get_receiver(
                        accum.input_id,
                        accum.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, accum.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_bool.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let func: Arc<
                        dyn Fn(&Tile<bool>, &Tile<bool>, u64, bool) -> (u64, Tile<bool>)
                            + Send
                            + Sync,
                    > = match accum.func.unwrap().accum_fn.unwrap() {
                        accum_func::AccumFn::RetileRow(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::retile_row(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        accum_func::AccumFn::RetileCol(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::retile_col(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        _ => todo!(),
                    };

                    let tile_row = accum.tile_row as usize;
                    let tile_col = accum.tile_col as usize;

                    // As the boolean tiles are used for masking, we need to functionally simulate them.
                    let init_accum: Arc<dyn Fn() -> Tile<bool> + Send + Sync> =
                        match accum.init_func.unwrap().init_fn.unwrap() {
                            init_func::InitFn::Empty(_empty) => Arc::new(move || {
                                Tile::new_empty([tile_row, tile_col], 1, accum.write_back_mu)
                            }),
                            init_func::InitFn::DynEmpty(_) => Arc::new(move || {
                                // DynEmpty means the row or the column size is known at run-time.
                                // Therefore, we will use the size of the first tile and keep the initial accumulator as [0,0]
                                Tile::new_empty([0, 0], 1, accum.write_back_mu)
                            }),
                            _ => todo!(),
                        };
                    // if sim_config.functional_sim {
                    //     match accum.init_func.unwrap().init_fn.unwrap() {
                    //         init_func::InitFn::Empty(_empty) => Arc::new(move || {
                    //             Tile::new_empty([tile_row, tile_col], 1, accum.write_back_mu)
                    //         }),
                    //         _ => todo!(),
                    //     }
                    // } else {
                    //     Arc::new(move || {
                    //         Tile::new_blank(vec![tile_row, tile_col], 1, accum.write_back_mu)
                    //     })
                    // };

                    add_child!(
                        builder,
                        Accum::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            func,
                            init_accum,
                            accum.rank,
                            AccumConfig {
                                compute_bw: accum.compute_bw as u64,
                                write_back_mu: accum.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                _ => todo!(),
            },
            OpType::AccumBuffer(accum) => match (
                accum.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                accum.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                (
                    Type::F32(_),
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_)),
                    }),
                ) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
                        accum.input_id,
                        accum.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, accum.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.buff_tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let func: Arc<
                        dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match accum.func.unwrap().accum_fn.unwrap() {
                        accum_func::AccumFn::Add(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::add(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    operation.id,
                                )
                            })
                        }
                        _ => todo!(),
                    };

                    let tile_row = accum.tile_row as usize;
                    let tile_col = accum.tile_col as usize;

                    let init_accum: Arc<dyn Fn() -> Tile<f32> + Send + Sync> = if sim_config
                        .functional_sim
                    {
                        match accum.init_func.unwrap().init_fn.unwrap() {
                            init_func::InitFn::Zero(_zero) => Arc::new(move || {
                                Tile::new_zero([tile_row, tile_col], f32_bytes, accum.write_back_mu)
                            }),
                            _ => todo!(),
                        }
                    } else {
                        Arc::new(move || {
                            Tile::new_blank(
                                vec![tile_row, tile_col],
                                f32_bytes,
                                accum.write_back_mu,
                            )
                        })
                    };

                    add_child!(
                        builder,
                        AccumBuff::<SimpleEvent, _, _>::new(
                            rcv,
                            // get_sender returns a TracingSender on this branch.
                            snd.into_sender(),
                            func,
                            init_accum,
                            accum.rank,
                            to_usize_vec(accum.buffer_shape),
                            AccumConfig {
                                compute_bw: accum.compute_bw as u64,
                                write_back_mu: accum.write_back_mu,
                            },
                            operation.id,
                        )
                    );
                }
                _ => todo!(),
            },
            OpType::RetileStreamify(retile_streamify) => {
                match retile_streamify
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            retile_streamify.input_id,
                            retile_streamify.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                retile_streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            RetileStreamify::<_>::new(
                                rcv,
                                snd,
                                retile_streamify.split_row,
                                retile_streamify.filter_mask,
                                retile_streamify.chunk as usize, // chunk size (default: 1 for backward compatibility)
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for RetileStreamify operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::FlatmapFilterRowStreamify(flatmap_filter_row_streamify) => {
                match flatmap_filter_row_streamify
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            flatmap_filter_row_streamify.input_id,
                            flatmap_filter_row_streamify.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flatmap_filter_row_streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        if let Some(mask_id) = flatmap_filter_row_streamify.mask_id {
                            let mask_rcv = channel_map_collection.tile_bool.get_receiver(
                                mask_id,
                                flatmap_filter_row_streamify.mask_stream_idx,
                                builder,
                                get_chan_depth(&sim_config.config_dict, mask_id, channel_depth),
                            );
                            add_child!(
                                builder,
                                FlatmapFilterRowStreamify::<_>::new(
                                    rcv,
                                    mask_rcv,
                                    snd,
                                    operation.id,
                                )
                            );
                        } else {
                            add_child!(
                                builder,
                                FlatmapRowStreamify::<_>::new(rcv, snd, operation.id,)
                            );
                        }
                    }
                    dtype => panic!(
                        "Unsupported data type for FlatmapFilterRowStreamify operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::MetadataGen(metadata_gen) => {
                let snd = channel_map_collection.tile_u64.get_sender(
                    operation.id,
                    None,
                    builder,
                    get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                );
                match metadata_gen.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::U64(_) => {
                        add_child!(
                            builder,
                            MetadataGen::<u64>::new(metadata_gen.npy_path, snd, operation.id,)
                        );
                    }
                    Type::ScalarU64(_) => {
                        add_child!(
                            builder,
                            MetadataGen::<u64>::new(metadata_gen.npy_path, snd, operation.id,)
                        );
                    }
                    Type::ScalarI64(_) => {
                        add_child!(
                            builder,
                            MetadataGen::<i64>::new(metadata_gen.npy_path, snd, operation.id,)
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for MetadataGen operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::ExpertAddrGen(expert_addr_gen) => {
                match expert_addr_gen
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            expert_addr_gen.input_id,
                            expert_addr_gen.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                expert_addr_gen.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            ExpertAddrGen::<_>::new(
                                rcv,
                                snd,
                                expert_addr_gen.num_tile_per_expert as u64,
                                expert_addr_gen.expert_addr_base as u64,
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for ExpertAddrGen operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::CacheReadAddrGen(cache_read_addr_gen) => {
                let idx_rcv = channel_map_collection.tile_u64.get_receiver(
                    cache_read_addr_gen.idx_id,
                    cache_read_addr_gen.idx_stream_idx,
                    builder,
                    get_chan_depth(
                        &sim_config.config_dict,
                        cache_read_addr_gen.idx_id,
                        channel_depth,
                    ),
                );
                let seq_len_rcv = channel_map_collection.tile_u64.get_receiver(
                    cache_read_addr_gen.seq_len_id,
                    cache_read_addr_gen.seq_len_stream_idx,
                    builder,
                    get_chan_depth(
                        &sim_config.config_dict,
                        cache_read_addr_gen.seq_len_id,
                        channel_depth,
                    ),
                );
                let snd = channel_map_collection.tile_u64.get_sender(
                    operation.id,
                    None,
                    builder,
                    get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                );
                add_child!(
                    builder,
                    CacheReadAddrGen::new(
                        idx_rcv,
                        seq_len_rcv,
                        cache_read_addr_gen.offset_per_idx,
                        snd,
                        operation.id,
                    )
                );
            }
            OpType::FilterLastTile(filter_last_tile) => {
                let seq_len_rcv = channel_map_collection.tile_u64.get_receiver(
                    filter_last_tile.seq_len_id,
                    filter_last_tile.seq_len_stream_idx,
                    builder,
                    get_chan_depth(
                        &sim_config.config_dict,
                        filter_last_tile.seq_len_id,
                        channel_depth,
                    ),
                );
                let snd = channel_map_collection.multihot.get_sender(
                    operation.id,
                    None,
                    builder,
                    get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                );
                add_child!(builder, FilterLastTile::new(seq_len_rcv, snd, operation.id));
            }
            OpType::Reshape(reshape) => {
                match reshape.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            reshape.input_id,
                            reshape.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                reshape.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );

                        match reshape.pad_func {
                            Some(pad_func) => {
                                let tile_row = reshape.tile_row.unwrap() as usize;
                                let tile_col = reshape.tile_col.unwrap() as usize;

                                let pad_val = match pad_func.init_fn.unwrap() {
                                    init_func::InitFn::Zero(_zero) => {
                                        if sim_config.functional_sim {
                                            Tile::new_zero_padded(
                                                [tile_row, tile_col],
                                                f32_bytes,
                                                reshape.write_back_mu,
                                                0,
                                            )
                                        } else {
                                            Tile::new_blank_padded(
                                                vec![tile_row, tile_col],
                                                f32_bytes,
                                                reshape.write_back_mu,
                                                0,
                                            )
                                        }
                                    }
                                    _ => todo!(),
                                };
                                add_child!(
                                    builder,
                                    Reshape::new(
                                        rcv,
                                        snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        Some(pad_val),
                                        reshape.input_stream_rank,
                                        reshape.add_outer_dim,
                                        operation.id,
                                    )
                                );
                            }
                            None => {
                                add_child!(
                                    builder,
                                    Reshape::new(
                                        rcv,
                                        snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        None,
                                        reshape.input_stream_rank,
                                        reshape.add_outer_dim,
                                        operation.id,
                                    )
                                );
                            }
                        }
                    }
                    dtype => panic!("Unsupported data type for Reshape operation {:?}", dtype),
                }
            }
            OpType::ReshapePadStream(reshape) => {
                match reshape.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            reshape.input_id,
                            reshape.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                reshape.input_id,
                                channel_depth,
                            ),
                        );
                        let padding_value = match reshape.pad_func {
                            Some(pad_func) => {
                                let tile_row = reshape.tile_row.unwrap() as usize;
                                let tile_col = reshape.tile_col.unwrap() as usize;

                                let pad_val = match pad_func.init_fn.unwrap() {
                                    init_func::InitFn::Zero(_zero) => {
                                        if sim_config.functional_sim {
                                            Tile::new_zero_padded(
                                                [tile_row, tile_col],
                                                f32_bytes,
                                                reshape.write_back_mu,
                                                0,
                                            )
                                        } else {
                                            Tile::new_blank_padded(
                                                vec![tile_row, tile_col],
                                                f32_bytes,
                                                reshape.write_back_mu,
                                                0,
                                            )
                                        }
                                    }
                                    _ => todo!(),
                                };
                                Some(pad_val)
                            }
                            None => None,
                        };
                        match reshape.have_pad_stream {
                            true => {
                                let mask_snd = channel_map_collection.tile_bool.get_sender(
                                    operation.id,
                                    Some(1),
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );
                                let snd = channel_map_collection.tile_f32.get_sender(
                                    operation.id,
                                    Some(0),
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );

                                add_child!(
                                    builder,
                                    ReshapePadStream::new(
                                        rcv,
                                        snd,
                                        mask_snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        padding_value,
                                        reshape.input_stream_rank,
                                        false,
                                        operation.id,
                                    )
                                );
                            }
                            false => {
                                let snd = channel_map_collection.tile_f32.get_sender(
                                    operation.id,
                                    None,
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );
                                add_child!(
                                    builder,
                                    ReshapeNoPadStream::new(
                                        rcv,
                                        snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        padding_value,
                                        reshape.input_stream_rank,
                                        false,
                                        operation.id,
                                    )
                                );
                            }
                        }
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            reshape.input_id,
                            reshape.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                reshape.input_id,
                                channel_depth,
                            ),
                        );
                        let padding_value = match reshape.pad_func {
                            Some(pad_func) => {
                                let tile_row = reshape.tile_row.unwrap() as usize;
                                let tile_col = reshape.tile_col.unwrap() as usize;

                                let pad_val = match pad_func.init_fn.unwrap() {
                                    init_func::InitFn::Zero(_zero) => {
                                        if sim_config.functional_sim {
                                            Tile::new_zero_padded(
                                                [tile_row, tile_col],
                                                8,
                                                reshape.write_back_mu,
                                                0,
                                            )
                                        } else {
                                            Tile::new_blank_padded(
                                                vec![tile_row, tile_col],
                                                8,
                                                reshape.write_back_mu,
                                                0,
                                            )
                                        }
                                    }
                                    _ => todo!(),
                                };
                                Some(pad_val)
                            }
                            None => None,
                        };
                        match reshape.have_pad_stream {
                            true => {
                                let mask_snd = channel_map_collection.tile_bool.get_sender(
                                    operation.id,
                                    Some(1),
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );
                                let snd = channel_map_collection.tile_u64.get_sender(
                                    operation.id,
                                    Some(0),
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );

                                add_child!(
                                    builder,
                                    ReshapePadStream::new(
                                        rcv,
                                        snd,
                                        mask_snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        padding_value,
                                        reshape.input_stream_rank,
                                        false,
                                        operation.id,
                                    )
                                );
                            }
                            false => {
                                let snd = channel_map_collection.tile_u64.get_sender(
                                    operation.id,
                                    None,
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );
                                add_child!(
                                    builder,
                                    ReshapeNoPadStream::new(
                                        rcv,
                                        snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        padding_value,
                                        reshape.input_stream_rank,
                                        false,
                                        operation.id,
                                    )
                                );
                            }
                        }
                    }
                    Type::MultiHot(multihot) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            reshape.input_id,
                            reshape.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                reshape.input_id,
                                channel_depth,
                            ),
                        );
                        let padding_value = match reshape.pad_func {
                            Some(pad_func) => {
                                panic!(
                                    "Padding value not supported for MultiHot in ReshapePadStream"
                                )
                            }
                            None => None,
                        };
                        match reshape.have_pad_stream {
                            true => {
                                let mask_snd = channel_map_collection.tile_bool.get_sender(
                                    operation.id,
                                    Some(1),
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );
                                let snd = channel_map_collection.multihot.get_sender(
                                    operation.id,
                                    Some(0),
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );

                                add_child!(
                                    builder,
                                    ReshapePadStream::new(
                                        rcv,
                                        snd,
                                        mask_snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        padding_value,
                                        reshape.input_stream_rank,
                                        false,
                                        operation.id,
                                    )
                                );
                            }
                            false => {
                                let snd = channel_map_collection.multihot.get_sender(
                                    operation.id,
                                    None,
                                    builder,
                                    get_chan_depth(
                                        &sim_config.config_dict,
                                        operation.id,
                                        channel_depth,
                                    ),
                                );
                                add_child!(
                                    builder,
                                    ReshapeNoPadStream::new(
                                        rcv,
                                        snd,
                                        reshape.split_dim as usize,
                                        reshape.chunk_size as usize,
                                        padding_value,
                                        reshape.input_stream_rank,
                                        false,
                                        operation.id,
                                    )
                                );
                            }
                        }
                    }
                    dtype => panic!(
                        "Unsupported data type for ReshapePadStream operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::EagerMerge(eager_merge) => {
                match eager_merge.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in eager_merge
                            .input_id_list
                            .into_iter()
                            .zip(eager_merge.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.tile_f32.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let sel_snd = channel_map_collection.multihot.get_sender(
                            operation.id,
                            Some(1),
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            Some(0),
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            EagerMerge::new(
                                rcv_list,
                                sel_snd,
                                snd,
                                eager_merge.input_rank,
                                operation.id,
                            )
                        );
                    }
                    Type::U64(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in eager_merge
                            .input_id_list
                            .into_iter()
                            .zip(eager_merge.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.tile_u64.get_receiver(
                                rcv_id,
                                if stream_idx < 0 {
                                    None
                                } else {
                                    Some(stream_idx as u32)
                                },
                                builder,
                                get_chan_depth(&sim_config.config_dict, rcv_id, channel_depth),
                            );
                            rcv_list.push(rcv);
                        }

                        let sel_snd = channel_map_collection.multihot.get_sender(
                            operation.id,
                            Some(1),
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let snd = channel_map_collection.tile_u64.get_sender(
                            operation.id,
                            Some(0),
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        add_child!(
                            builder,
                            EagerMerge::new(
                                rcv_list,
                                sel_snd,
                                snd,
                                eager_merge.input_rank,
                                operation.id,
                            )
                        );
                    }
                    dtype => panic!("Unsupported data type for EagerMerge operation {:?}", dtype),
                }
            }
            OpType::FlatmapCounter(flatmap_counter) => {
                match flatmap_counter
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    Type::U64(_) => {
                        make_flatmap_counter!(
                            channel_map_collection,
                            operation,
                            flatmap_counter,
                            tile_u64,
                            builder,
                            channel_depth
                        );
                    }
                    dtype => panic!(
                        "Unsupported data type for FlatmapCounter operation {:?}",
                        dtype
                    ),
                }
            }
            _ => todo!(),
        }
    }

    // The HBM context is added after the loop, so clear the op id to avoid
    // mislabeling it with the last operation's id.
    crate::utils::graph_dump::clear_current_op();
    add_child!(builder, mem_context);

    if let Some(prefix) = dump_prefix {
        let proto_path = format!("{}.proto.txt", prefix);
        let nodes_path = format!("{}.nodes.txt", prefix);
        let nodes_dump = crate::utils::graph_dump::render_nodes();
        match std::fs::write(&proto_path, &proto_dump) {
            Ok(()) => println!("[graph dump] wrote {}", proto_path),
            Err(e) => eprintln!("[graph dump] failed to write {}: {}", proto_path, e),
        }
        match std::fs::write(&nodes_path, &nodes_dump) {
            Ok(()) => println!("[graph dump] wrote {}", nodes_path),
            Err(e) => eprintln!("[graph dump] failed to write {}: {}", nodes_path, e),
        }
        crate::utils::graph_dump::end();
    }
}

pub fn parse_proto<'a>(
    step_graph: ProgramGraph,
    logging: bool,
    hbm_config: HBMConfig,
    sim_config: SimConfig,
    db_name: Option<String>,
    dump_prefix: Option<String>,
) -> (bool, u64, std::time::Duration) {
    let mut builder = ProgramBuilder::default();
    let mut channel_map_collection = ChannelMapCollection::default();
    if std::env::var("STEP_PERF_TRACE").is_ok() || std::env::var("STEP_PERF_GRAPH_JSON").is_ok() {
        crate::trace::reset_trace_registry();
    }

    build_from_proto(
        &step_graph,
        &mut channel_map_collection,
        &mut builder,
        &hbm_config,
        &sim_config,
        dump_prefix,
    );

    if let Err(e) = crate::trace::write_graph_json_if_requested(&step_graph) {
        eprintln!("[step_perf] graph export failed: {e}");
    }

    let initialized = builder.initialize(Default::default()).unwrap();
    let run_options = match logging {
        true => {
            let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
                dam::logging::LogFilter::Some([SimpleEvent::NAME.to_owned()].into()),
                // dam::logging::LogFilter::AllowAll,
            ));
            run_options.build().unwrap()
        }
        false => Default::default(),
    };

    // println!("{}", initialized.to_dot_string());

    let start = Instant::now();
    let executed = initialized.run(run_options);
    let duration = start.elapsed();

    println!("Duration: {:?}", duration);

    let cycles = executed.elapsed_cycles().unwrap();
    let passed = executed.passed();
    (passed, cycles, duration)
}
