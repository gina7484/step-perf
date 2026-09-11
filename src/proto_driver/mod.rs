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
use crate::operator::counter::Counter;
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
use crate::operator::accum_row_stat::{AccumRowStat, AccumRowStatConfig, RowStat};
use crate::operator::broadcast::BroadcastContext;
use crate::operator::bufferize::Bufferize;
use crate::operator::dynstreamify::DynStreamify;
use crate::operator::flatmap::{
    CacheReadAddrGen, DynAddrGen, ExpertAddrGen, FilterLastTile, RetileStreamify,
};
use crate::operator::flatten::Flatten;
use crate::operator::map::{
    BinaryMapMultiHot, UnaryMap, UnaryMapConfig, UnaryMapMultiHot, UnaryMapToMultiHot,
};
use crate::operator::map_accum::BinaryMapAccum;
use crate::operator::partition::{FlatPartition, FlatPartitionConfig};
use crate::operator::promote::{Promote, PromoteOuter};
use crate::operator::reassemble::{FlatReassemble, FlatReassembleConfig};
use crate::operator::reshape::{Reshape, ReshapeNoPadStream, ReshapePadStream};
use crate::operator::scan::{Scan, ScanConfig};
use crate::operator::static_reassemble::StaticReassemble;
use crate::operator::streamify::{StaticStreamify, Streamify};
use crate::proto_driver::proto_headers::graph_proto::map_accum_func;
use crate::utils::select_npy::read_multihot_elem_from_npy_iter;
use dam::simulation::{
    DotConvertible, LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder,
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
    file_printer::FilePrinterContext,
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
                $dyn_offchip_load.simulate_ramulator,
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

/// Recognize the accumulation functions that need [`AccumRowStat`] instead of
/// [`Accum`]. Returns the statistic and the number of elements reduced per row,
/// where `None` means the operator counts them at run time.
fn row_stat_of(accum_fn: &accum_func::AccumFn) -> Option<(RowStat, Option<u64>)> {
    match accum_fn {
        accum_func::AccumFn::MeanStatic(mean) => Some((RowStat::Mean, Some(mean.count))),
        accum_func::AccumFn::VarStatic(var) => Some((RowStat::Var, Some(var.count))),
        accum_func::AccumFn::MeanDyn(_) => Some((RowStat::Mean, None)),
        accum_func::AccumFn::VarDyn(_) => Some((RowStat::Var, None)),
        _ => None,
    }
}

/// Build one of [`Scan`]'s fold closures over `f32` tiles (the channel bf16 also
/// rides on). Mirrors the fold list in the `OpType::Accum` arm, which `Scan`
/// reuses verbatim — the two operators differ in *when* they emit, not in how
/// they fold.
fn scan_fold_f32(
    accum_fn: accum_func::AccumFn,
    id: u32,
) -> Arc<dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>) + Send + Sync> {
    match accum_fn {
        accum_func::AccumFn::Add(_) => Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
            functions::accum_fn::add(tile1, tile2, comp_bw, write_back_mu, id)
        }),
        accum_func::AccumFn::Mul(_) => Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
            functions::accum_fn::mul(tile1, tile2, comp_bw, write_back_mu, id)
        }),
        accum_func::AccumFn::Max(_) => Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
            functions::accum_fn::max(tile1, tile2, comp_bw, write_back_mu, id)
        }),
        accum_func::AccumFn::Last(_) => Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
            functions::accum_fn::last(tile1, tile2, comp_bw, write_back_mu, id)
        }),
        accum_func::AccumFn::RetileRow(_) => {
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                functions::accum_fn::retile_row(tile1, tile2, comp_bw, write_back_mu, id)
            })
        }
        accum_func::AccumFn::RetileCol(_) => {
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                functions::accum_fn::retile_col(tile1, tile2, comp_bw, write_back_mu, id)
            })
        }
        e => panic!("Unsupported scan function type {:?}", e),
    }
}

fn build_from_proto<'a>(
    step_graph: ProgramGraph,
    channel_map_collection: &mut ChannelMapCollection<'a>,
    builder: &mut ProgramBuilder<'a>,
    hbm_config: &HBMConfig,
    sim_config: &SimConfig,
    dump_prefix: Option<String>,
) {
    let channel_depth = sim_config.channel_depth;
    let mut mem_context = HBMContext::new(builder, hbm_config.clone());

    // Graph dump (file 1: proto operators). Built here, before the loop below
    // consumes `step_graph.operators`. `begin()` arms the per-node channel
    // capture used by the `add_child!` macro and the `channel.rs` hooks.
    let mut proto_dump = String::new();
    if dump_prefix.is_some() {
        for operation in &step_graph.operators {
            proto_dump.push_str(&format!("processing {:?}\n\n", operation));
        }
        crate::utils::graph_dump::begin();
    }

    for operation in step_graph.operators {
        crate::utils::graph_dump::set_current_op(operation.id);
        // if operation.id == 23 || operation.id == 24 || operation.id == 25 {
        //     println!("processing {:?}\n", operation);
        // }

        let dtype_bytes: usize = operation.dtype_bytes as usize; // we will use this to mimic bfloat16

        match operation.op_type.clone().unwrap() {
            OpType::Unarymap(unarymap) => match (
                unarymap.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                unarymap.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                // bf16 is modelled as Tile<f32> on the tile_f32 channel, so bf16->bf16
                // tile-preserving unary ops (silu, transpose, exp, …) share this arm.
                (Type::F32(_), Type::F32(_)) | (Type::Bf16(_), Type::Bf16(_)) => {
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
                        elemto_elem_func::ElemElemFn::Sigmoid(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::sigmoid(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::ZerosLike(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::zeros_like(tile, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Transpose(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::transpose(tile, comp_bw, write_back_mu)
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
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::mask_row(
                                    tile,
                                    unarymap.write_back_mu,
                                    mask_row.row as usize,
                                    mask_row.col as usize,
                                    dtype_bytes,
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
                        elemto_elem_func::ElemElemFn::FloorDivideConstant(fd) => {
                            let divisor = u64::try_from(fd.constant.expect("integer divisor")).expect("positive divisor");
                            assert!(divisor > 0);
                            Arc::new(move |tile, bw, write_back| {
                                functions::map_fn::floor_divide_scalar(tile, divisor, bw, write_back)
                            })
                        }
                        elemto_elem_func::ElemElemFn::RemainderConstant(rem) => {
                            let divisor = u64::try_from(rem.constant).expect("positive divisor");
                            assert!(divisor > 0);
                            Arc::new(move |tile, bw, write_back| {
                                functions::map_fn::remainder_u64(tile, divisor, bw, write_back)
                            })
                        }
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
                        elemto_elem_func::ElemElemFn::IndexToMultihot(index_to_multihot) => {
                            let num_classes = index_to_multihot.num_classes as usize;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::index_to_multihot(
                                    tile,
                                    num_classes,
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
                // index_to_multihot: i64 index tile -> multihot vector (e.g. topk indices).
                (Type::I64(_), Type::MultiHot(_)) => {
                    let rcv = channel_map_collection.tile_i64.get_receiver(
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
                        dyn Fn(&Tile<i64>, u64, bool) -> (u64, MultiHotN) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::IndexToMultihot(index_to_multihot) => {
                            let num_classes = index_to_multihot.num_classes as usize;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::index_to_multihot(
                                    tile,
                                    num_classes,
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
                // index_to_multihot: f32 index tile -> multihot vector.
                (Type::F32(_), Type::MultiHot(_)) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
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
                        dyn Fn(&Tile<f32>, u64, bool) -> (u64, MultiHotN) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::IndexToMultihot(index_to_multihot) => {
                            let num_classes = index_to_multihot.num_classes as usize;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::index_to_multihot(
                                    tile,
                                    num_classes,
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
                // index_to_multihot: bf16 index tile -> multihot vector. bf16 is
                // modelled as Tile<f32> on the tile_f32 channel (e.g. SimpleMoe select).
                (Type::Bf16(_), Type::MultiHot(_)) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
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
                        dyn Fn(&Tile<f32>, u64, bool) -> (u64, MultiHotN) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::IndexToMultihot(index_to_multihot) => {
                            let num_classes = index_to_multihot.num_classes as usize;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::index_to_multihot(
                                    tile,
                                    num_classes,
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
                // _to_copy: f32 -> bf16 cast. bf16 is modelled as Tile<f32>, so both
                // sides use the tile_f32 channel; the cast only changes bytes_per_elem.
                (Type::F32(_), Type::Bf16(_)) => {
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
                        elemto_elem_func::ElemElemFn::F32ToBf16(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::f32_bf16(tile, comp_bw, write_back_mu)
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
                // _to_copy: bf16 -> f32 cast (both use tile_f32 channel).
                (Type::Bf16(_), Type::F32(_)) => {
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
                        elemto_elem_func::ElemElemFn::Bf16ToF32(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::bf16_f32(tile, comp_bw, write_back_mu)
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
                // Integer (i64) unary ops: clamp, floor_divide, empty_like.
                (Type::I64(_), Type::I64(_)) => {
                    let rcv = channel_map_collection.tile_i64.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_i64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<i64>, u64, bool) -> (u64, Tile<i64>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Clamp(clamp) => {
                            let min = clamp
                                .min
                                .map(|v| v as i64)
                                .or(clamp.min_float.map(|v| v as i64));
                            let max = clamp
                                .max
                                .map(|v| v as i64)
                                .or(clamp.max_float.map(|v| v as i64));
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::clamp(tile, min, max, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::FloorDivideConstant(fd) => {
                            let divisor = fd
                                .constant
                                .map(|v| v as i64)
                                .or(fd.constant_float.map(|v| v as i64))
                                .expect("floor_divide requires a scalar divisor");
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::floor_divide_scalar(
                                    tile,
                                    divisor,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::RemainderConstant(rem) => {
                            let divisor = rem.constant;
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::remainder_scalar(
                                    tile,
                                    divisor,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::MulConstant(mul_constant) => {
                            let constant = mul_constant
                                .constant
                                .map(|value| value as i64)
                                .or(mul_constant.constant_float.map(|value| value as i64))
                                .expect("integer multiply requires a scalar constant");
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::mul_constant(
                                    tile,
                                    constant,
                                    comp_bw,
                                    write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::EmptyLike(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::empty_like(tile, comp_bw, write_back_mu)
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
                // ge.Scalar: i64 input -> bool output.
                (Type::I64(_), Type::Bool(_)) => {
                    let rcv = channel_map_collection.tile_i64.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_bool.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<i64>, u64, bool) -> (u64, Tile<bool>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::GeScalar(ge) => {
                            let scalar = ge
                                .constant
                                .map(|v| v as i64)
                                .or(ge.constant_float.map(|v| v as i64))
                                .expect("ge.Scalar requires a scalar operand");
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::ge_scalar(tile, scalar, comp_bw, write_back_mu)
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
                // Boolean unary ops: bitwise_not.
                (Type::Bool(_), Type::Bool(_)) => {
                    let rcv = channel_map_collection.tile_bool.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_bool.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<bool>, u64, bool) -> (u64, Tile<bool>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::BitwiseNot(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::bitwise_not(tile, comp_bw, write_back_mu)
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
                // _to_copy: f32 -> bool cast.
                (Type::F32(_), Type::Bool(_)) => {
                    let rcv = channel_map_collection.tile_f32.get_receiver(
                        unarymap.input_id,
                        unarymap.stream_idx,
                        builder,
                        get_chan_depth(&sim_config.config_dict, unarymap.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_bool.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<f32>, u64, bool) -> (u64, Tile<bool>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::F32ToBool(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::f32_bool(tile, comp_bw, write_back_mu)
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
                // _to_copy: i64 -> f32 cast.
                (Type::I64(_), Type::F32(_)) => {
                    let rcv = channel_map_collection.tile_i64.get_receiver(
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
                        dyn Fn(&Tile<i64>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::I64ToF32(_) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::i64_f32(tile, comp_bw, write_back_mu)
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
                (Type::I64(_), Type::U64(_)) => {
                    let rcv = channel_map_collection.tile_i64.get_receiver(
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
                        dyn Fn(&Tile<i64>, u64, bool) -> (u64, Tile<u64>) + Send + Sync,
                    > = match unarymap.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::I64ToU64(_) => {
                            Arc::new(functions::map_fn::i64_u64)
                        }
                        e => panic!("Unsupported unary map function type {:?}", e),
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
                // bf16 is modelled as Tile<f32>; all three sides use the tile_f32
                // channel (e.g. the bf16 expert matmuls in the MoE graph).
                (Type::F32(_), Type::F32(_), Type::F32(_))
                | (Type::Bf16(_), Type::Bf16(_), Type::Bf16(_)) => {
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
                        elemto_elem_func::ElemElemFn::Div(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::div(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Sub(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::sub(tile1, tile2, comp_bw, write_back_mu)
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
                (Type::I64(_), Type::I64(_), Type::I64(_)) => {
                    let rcv1 = channel_map_collection.tile_i64.get_receiver(
                        binary_map.input_id1,
                        binary_map.stream_idx1,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id1,
                            channel_depth,
                        ),
                    );
                    let rcv2 = channel_map_collection.tile_i64.get_receiver(
                        binary_map.input_id2,
                        binary_map.stream_idx2,
                        builder,
                        get_chan_depth(
                            &sim_config.config_dict,
                            binary_map.input_id2,
                            channel_depth,
                        ),
                    );
                    let snd = channel_map_collection.tile_i64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<i64>, &Tile<i64>, u64, bool) -> (u64, Tile<i64>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Gather(_) => {
                            Arc::new(move |source, index, comp_bw, write_back_mu| {
                                functions::map_fn::gather(source, index, comp_bw, write_back_mu)
                            })
                        }
                        elemto_elem_func::ElemElemFn::Add(_) => {
                            Arc::new(move |lhs, rhs, comp_bw, write_back_mu| {
                                functions::map_fn::add(lhs, rhs, comp_bw, write_back_mu)
                            })
                        }
                        e => panic!("Unsupported binary map function type {:?}", e),
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
                (Type::I64(_), Type::U64(_), Type::I64(_)) => {
                    let rcv1 = channel_map_collection.tile_i64.get_receiver(
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
                    let snd = channel_map_collection.tile_i64.get_sender(
                        operation.id,
                        None,
                        builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<i64>, &Tile<u64>, u64, bool) -> (u64, Tile<i64>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Gather(_) => {
                            Arc::new(move |source, index, comp_bw, write_back_mu| {
                                functions::map_fn::gather(source, index, comp_bw, write_back_mu)
                            })
                        }
                        e => panic!("Unsupported binary map function type {:?}", e),
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
                (Type::F32(_), Type::I64(_), Type::F32(_))
                | (Type::Bf16(_), Type::I64(_), Type::Bf16(_)) => {
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
                    let rcv2 = channel_map_collection.tile_i64.get_receiver(
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
                        dyn Fn(&Tile<f32>, &Tile<i64>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Mask(mask) => {
                            Arc::new(move |data, count, comp_bw, write_back_mu| {
                                functions::map_fn::mask(
                                    data, count, mask.row, mask.val, comp_bw, write_back_mu,
                                )
                            })
                        }
                        elemto_elem_func::ElemElemFn::Gather(_) => {
                            Arc::new(move |source, index, comp_bw, write_back_mu| {
                                functions::map_fn::gather(source, index, comp_bw, write_back_mu)
                            })
                        }
                        e => panic!("Unsupported binary map function type {:?}", e),
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

                (Type::F32(_), Type::U64(_), Type::F32(_))
                | (Type::Bf16(_), Type::U64(_), Type::Bf16(_)) => {
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
                        elemto_elem_func::ElemElemFn::Mask(mask) => {
                            Arc::new(move |data, count, comp_bw, write_back_mu| {
                                functions::map_fn::mask(
                                    data, count, mask.row, mask.val, comp_bw, write_back_mu,
                                )
                            })
                        }
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
                // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                (Type::F32(_), Type::F32(_)) | (Type::Bf16(_), Type::Bf16(_)) => {
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
                                    dtype_bytes,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel; only
                    // dtype_bytes (2) differs, and that comes from operation.dtype_bytes.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                dtype_bytes,
                                0,
                                hbm_config.addr_offset,
                                linear_off_chip_load.par_dispatch as usize,
                                linear_off_chip_load.simulate_ramulator,
                                addr_snd,
                                resp_rcv,
                                on_chip_snd,
                                linear_off_chip_load.transposed,
                                linear_off_chip_load.add_outer_singular_dim,
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
                                linear_off_chip_load.simulate_ramulator,
                                addr_snd,
                                resp_rcv,
                                on_chip_snd,
                                linear_off_chip_load.transposed,
                                linear_off_chip_load.add_outer_singular_dim,
                                operation.id,
                            )
                        );

                        mem_context.add_reader(ReadBundle {
                            addr: addr_rcv,
                            resp: resp_snd,
                        });
                    }
                    Type::I64(_) => {
                        let on_chip_snd = channel_map_collection.tile_i64.get_sender(
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
                                std::mem::size_of::<i64>(),
                                0,
                                hbm_config.addr_offset,
                                linear_off_chip_load.par_dispatch as usize,
                                linear_off_chip_load.simulate_ramulator,
                                addr_snd,
                                resp_rcv,
                                on_chip_snd,
                                linear_off_chip_load.transposed,
                                linear_off_chip_load.add_outer_singular_dim,
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
                                dtype_bytes,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                dtype_bytes,
                                random_off_chip_store.base_addr_byte,
                                hbm_config.addr_offset,
                                random_off_chip_store.par_dispatch as usize,
                                addr_snd,
                                resp_rcv,
                                waddr,
                                wdata,
                                wack,
                                operation.id,
                                random_off_chip_store.ack_based_on_waddr,
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
                    // bf16 tiles ride the f32 channel map; the element width
                    // comes from `dtype_bytes`, same as every other op.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                dtype_bytes,
                                random_off_chip_load.base_addr_byte,
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
                    Type::I64(_) => {
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
                        let on_chip_snd = channel_map_collection.tile_i64.get_sender(
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
                                dtype_bytes,
                                random_off_chip_load.base_addr_byte,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    (Type::I64(_), Type::U64(_)) => {
                        let in_rcv = channel_map_collection.tile_i64.get_receiver(
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
                        let snd = channel_map_collection.tile_i64.get_sender(
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
                        // bf16 is modelled as Tile<f32> on the (buff_)tile_f32 channels.
                        Type::Buffer(proto_headers::graph_proto::Buffer {
                            r#type: Some(buffer::Type::F32(_) | buffer::Type::Bf16(_)),
                        }),
                        Type::F32(_) | Type::Bf16(_),
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            tile_f32,
                            builder,
                            channel_depth
                        );
                    }
                    Type::I64(_) => {
                        make_broadcast!(
                            channel_map_collection,
                            operation,
                            broadcast,
                            tile_i64,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                        flat_partition.sel_npy_path.clone(),
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
                                        flat_partition.sel_npy_path.clone(),
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
                                        flat_partition.sel_npy_path.clone(),
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                        reassemble.sel_npy_path.clone(),
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
                                        reassemble.sel_npy_path.clone(),
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
                                static_reassemble.reassemble_rank,
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
                                static_reassemble.reassemble_rank,
                                FlatPartitionConfig {
                                    switch_cycles: to_u64_vec(static_reassemble.switch_cycles),
                                    write_back_mu: static_reassemble.write_back_mu,
                                },
                                operation.id,
                            )
                        );
                    }
                    Type::I64(_) => {
                        let mut rcv_list = vec![];
                        for (rcv_id, stream_idx) in static_reassemble
                            .input_id_list
                            .into_iter()
                            .zip(static_reassemble.input_stream_idx_list.into_iter())
                        {
                            let rcv = channel_map_collection.tile_i64.get_receiver(
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

                        let snd = channel_map_collection.tile_i64.get_sender(
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
                                static_reassemble.reassemble_rank,
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
                                static_reassemble.reassemble_rank,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                parallelize.output_dim,
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
                                parallelize.output_dim,
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
                                parallelize.output_dim,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    // Select (multihot) stream, e.g. the broadcast routing mask in the
                    // MoE control-flow path.
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            promote_outer.input_id,
                            promote_outer.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                promote_outer.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.multihot.get_sender(
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    // Buffer of bf16 is modelled as Buffer<Tile<f32>> on the buff_tile_f32 channel.
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_) | buffer::Type::Bf16(_)),
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
            OpType::FilePrinterContext(file_printer_context) => {
                match file_printer_context
                    .dtype
                    .clone()
                    .unwrap()
                    .r#type
                    .clone()
                    .unwrap()
                {
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            file_printer_context.input_id,
                            file_printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, FilePrinterContext::new(rcv, operation.id));
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            file_printer_context.input_id,
                            file_printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, FilePrinterContext::new(rcv, operation.id));
                    }
                    Type::I64(_) => {
                        let rcv = channel_map_collection.tile_i64.get_receiver(
                            file_printer_context.input_id,
                            file_printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, FilePrinterContext::new(rcv, operation.id));
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            file_printer_context.input_id,
                            file_printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, FilePrinterContext::new(rcv, operation.id));
                    }
                    Type::Bool(_) => {
                        let rcv = channel_map_collection.tile_bool.get_receiver(
                            file_printer_context.input_id,
                            file_printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, FilePrinterContext::new(rcv, operation.id));
                    }
                    // Buffer of bf16 is modelled as Buffer<Tile<f32>> on the buff_tile_f32 channel.
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_) | buffer::Type::Bf16(_)),
                    }) => {
                        let rcv = channel_map_collection.buff_tile_f32.get_receiver(
                            file_printer_context.input_id,
                            file_printer_context.stream_idx,
                            builder,
                            None,
                        );
                        add_child!(builder, FilePrinterContext::new(rcv, operation.id));
                    }
                    dtype => panic!(
                        "Unsupported data type for FilePrinterContext operation {:?}",
                        dtype
                    ),
                }
            }
            OpType::Bufferize(bufferize) => {
                match bufferize.dtype.clone().unwrap().r#type.clone().unwrap() {
                    // bf16 is modelled as Tile<f32>; a buffered bf16 tile lives on the
                    // buff_tile_f32 channel (same as f32).
                    Type::F32(_) | Type::Bf16(_) => {
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
                    // bf16 is modelled as Buffer<Tile<f32>> / Tile<f32> on the *_f32 channels.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    // bf16 is modelled as Tile<f32>; the buffered input lives on the
                    // buff_tile_f32 channel and the streamed output on tile_f32.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel, so
                    // f32/bf16 combine freely for both the loaded and ref dtypes.
                    (Type::F32(_) | Type::Bf16(_), Type::F32(_) | Type::Bf16(_)) => {
                        make_linear_offchip_load_ref!(
                            channel_map_collection,
                            operation,
                            linear_offchip_load_ref,
                            hbm_config,
                            tile_f32,
                            tile_f32,
                            dtype_bytes,
                            mem_context,
                            builder,
                            channel_depth
                        );
                    }
                    (
                        Type::F32(_) | Type::Bf16(_),
                        Type::Buffer(proto_headers::graph_proto::Buffer {
                            r#type: Some(buffer::Type::F32(_) | buffer::Type::Bf16(_)),
                        }),
                    ) => {
                        make_linear_offchip_load_ref!(
                            channel_map_collection,
                            operation,
                            linear_offchip_load_ref,
                            hbm_config,
                            buff_tile_f32,
                            tile_f32,
                            dtype_bytes,
                            mem_context,
                            builder,
                            channel_depth
                        );
                    }
                    (Type::F32(_) | Type::Bf16(_), Type::MultiHot(_)) => {
                        make_linear_offchip_load_ref!(
                            channel_map_collection,
                            operation,
                            linear_offchip_load_ref,
                            hbm_config,
                            multihot,
                            tile_f32,
                            dtype_bytes,
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
            OpType::Flatten(flatten) => {
                match flatten.dtype.clone().unwrap().r#type.clone().unwrap() {
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    Type::I64(_) => {
                        let rcv = channel_map_collection.tile_i64.get_receiver(
                            flatten.input_id,
                            flatten.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                flatten.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_i64.get_sender(
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
                            snd,
                        )
                    );
                }
                false => todo!("Add the same version for IndexN"),
            },
            OpType::Accum(accum) => match (
                accum.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                accum.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                (Type::F32(_), Type::F32(_)) | (Type::Bf16(_), Type::Bf16(_)) => {
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
                    let accum_fn_pb = accum.func.clone().unwrap().accum_fn.unwrap();

                    // The row statistics collapse each tile's columns alongside
                    // the reduced ranks, so the accumulator is not the output
                    // and `Accum`'s fold cannot express them. They get their own
                    // operator; every other function falls through to `Accum`.
                    if let Some((stat, count)) = row_stat_of(&accum_fn_pb) {
                        add_child!(
                            builder,
                            AccumRowStat::<SimpleEvent>::new(
                                rcv,
                                snd,
                                accum.rank,
                                dtype_bytes,
                                AccumRowStatConfig {
                                    compute_bw: accum.compute_bw as u64,
                                    write_back_mu: accum.write_back_mu,
                                    stat,
                                    count,
                                },
                                operation.id,
                            )
                        );
                    } else {
                        let func: Arc<
                            dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>)
                                + Send
                                + Sync,
                        > = match accum_fn_pb {
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
                            accum_func::AccumFn::Max(_) => {
                                Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                    functions::accum_fn::max(
                                        tile1,
                                        tile2,
                                        comp_bw,
                                        write_back_mu,
                                        operation.id,
                                    )
                                })
                            }
                            accum_func::AccumFn::Last(_) => {
                                Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                    functions::accum_fn::last(
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

                        let init_accum: Arc<dyn Fn() -> Tile<f32> + Send + Sync> = match accum
                            .init_func
                            .unwrap()
                            .init_fn
                            .unwrap()
                        {
                            init_func::InitFn::Zero(_zero) => Arc::new(move || {
                                Tile::new_zero(
                                    [tile_row, tile_col],
                                    dtype_bytes,
                                    accum.write_back_mu,
                                )
                            }),
                            init_func::InitFn::NegInf(_) => Arc::new(move || {
                                Tile::new_neg_inf(
                                    [tile_row, tile_col],
                                    dtype_bytes,
                                    accum.write_back_mu,
                                )
                            }),
                            init_func::InitFn::Empty(_empty) => Arc::new(move || {
                                Tile::new_empty(
                                    [tile_row, tile_col],
                                    dtype_bytes,
                                    accum.write_back_mu,
                                )
                            }),
                            init_func::InitFn::DynEmpty(_) => Arc::new(move || {
                                // DynEmpty means the row or the column size is known at run-time.
                                // Therefore, we will use the size of the first tile and keep the initial accumulator as [0,0]
                                Tile::new_empty([0, 0], dtype_bytes, accum.write_back_mu)
                            }),
                            _ => todo!(),
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
                }
                (Type::U64(_), Type::U64(_)) => {
                    let rcv = channel_map_collection.tile_u64.get_receiver(
                        accum.input_id, accum.stream_idx, builder,
                        get_chan_depth(&sim_config.config_dict, accum.input_id, channel_depth),
                    );
                    let snd = channel_map_collection.tile_u64.get_sender(
                        operation.id, None, builder,
                        get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                    );
                    let func: Arc<
                        dyn Fn(&Tile<u64>, &Tile<u64>, u64, bool) -> (u64, Tile<u64>) + Send + Sync,
                    > = match accum.func.unwrap().accum_fn.unwrap() {
                        accum_func::AccumFn::Add(_) => Arc::new(move |tile, state, bw, write_back| {
                            functions::accum_fn::add(tile, state, bw, write_back, operation.id)
                        }),
                        other => panic!("Unsupported u64 accumulation function {:?}", other),
                    };
                    let tile_row = accum.tile_row as usize;
                    let tile_col = accum.tile_col as usize;
                    let init_accum: Arc<dyn Fn() -> Tile<u64> + Send + Sync> =
                        match accum.init_func.unwrap().init_fn.unwrap() {
                            init_func::InitFn::Zero(_) => Arc::new(move || {
                                Tile::new_zero([tile_row, tile_col], 8, accum.write_back_mu)
                            }),
                            other => panic!("Unsupported u64 accumulation initializer {:?}", other),
                        };
                    add_child!(builder, Accum::<SimpleEvent, _, _>::new(
                        rcv, snd, func, init_accum, accum.rank,
                        AccumConfig {
                            compute_bw: accum.compute_bw as u64,
                            write_back_mu: accum.write_back_mu,
                        },
                        operation.id,
                    ));
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
            OpType::Scan(scan) => {
                match scan.dtype_a.clone().unwrap().r#type.clone().unwrap() {
                    // The proto carries a single dtype: input1, input2 and the
                    // output all share it. bf16 is modelled as Tile<f32> on the
                    // tile_f32 channel, so it rides this arm too.
                    Type::F32(_) | Type::Bf16(_) => {
                        let rcv1 = channel_map_collection.tile_f32.get_receiver(
                            scan.input_id1,
                            scan.stream_idx1,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                scan.input_id1,
                                channel_depth,
                            ),
                        );
                        let rcv2 = if let Some(input_id2) = scan.input_id2 {
                            Some(channel_map_collection.tile_f32.get_receiver(
                                input_id2,
                                scan.stream_idx2,
                                builder,
                                get_chan_depth(
                                    &sim_config.config_dict,
                                    input_id2,
                                    channel_depth,
                                ),
                            ))
                        } else {
                            None
                        };
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );

                        let fn1 = scan_fold_f32(
                            scan.fn1.clone().unwrap().accum_fn.unwrap(),
                            operation.id,
                        );
                        let fn2 = scan
                            .fn2
                            .clone()
                            .map(|fn2| scan_fold_f32(fn2.accum_fn.unwrap(), operation.id));

                        let tile_row = scan.tile_row as usize;
                        let tile_col = scan.tile_col as usize;
                        let write_back_mu = scan.write_back_mu;

                        let init_accum: Arc<dyn Fn() -> Tile<f32> + Send + Sync> =
                            match scan.init_func.unwrap().init_fn.unwrap() {
                                init_func::InitFn::Zero(_zero) => Arc::new(move || {
                                    Tile::new_zero(
                                        [tile_row, tile_col],
                                        dtype_bytes,
                                        write_back_mu,
                                    )
                                }),
                                init_func::InitFn::NegInf(_) => Arc::new(move || {
                                    Tile::new_neg_inf(
                                        [tile_row, tile_col],
                                        dtype_bytes,
                                        write_back_mu,
                                    )
                                }),
                                init_func::InitFn::Empty(_empty) => Arc::new(move || {
                                    Tile::new_empty(
                                        [tile_row, tile_col],
                                        dtype_bytes,
                                        write_back_mu,
                                    )
                                }),
                                init_func::InitFn::DynEmpty(_) => Arc::new(move || {
                                    // DynEmpty means the row or the column size is known at
                                    // run-time, so the first fold sizes the accumulator.
                                    Tile::new_empty([0, 0], dtype_bytes, write_back_mu)
                                }),
                            };

                        add_child!(
                            builder,
                            Scan::<SimpleEvent, _, _>::new(
                                rcv1,
                                rcv2,
                                snd,
                                fn1,
                                fn2,
                                init_accum,
                                scan.rank,
                                ScanConfig {
                                    compute_bw: scan.compute_bw as u64,
                                    write_back_mu: scan.write_back_mu,
                                    inclusive: scan.inclusive,
                                },
                                operation.id,
                            )
                        );
                    }
                    e => panic!("Unsupported data type {:?} for Scan", e),
                }
            }
            OpType::AccumBuffer(accum) => match (
                accum.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                accum.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                // bf16 is modelled as Tile<f32> / Buffer<Tile<f32>> on the *_f32 channels
                // (e.g. the bf16 expert-output accumulation buffer in the MoE graph).
                (
                    Type::F32(_) | Type::Bf16(_),
                    Type::Buffer(proto_headers::graph_proto::Buffer {
                        r#type: Some(buffer::Type::F32(_) | buffer::Type::Bf16(_)),
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

                    let init_accum: Arc<dyn Fn(usize, usize) -> Tile<f32> + Send + Sync> =
                        match accum.init_func.unwrap().init_fn.unwrap() {
                            init_func::InitFn::Zero(_zero) => Arc::new(move |rows, cols| {
                                Tile::new_zero([rows, cols], dtype_bytes, accum.write_back_mu)
                            }),
                            _ => todo!(),
                        };

                    add_child!(
                        builder,
                        AccumBuff::<SimpleEvent, _, _>::new(
                            rcv,
                            snd,
                            func,
                            init_accum,
                            accum.rank,
                            to_usize_vec(accum.buffer_shape),
                            tile_row,
                            tile_col,
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                    Type::I64(_) => {
                        let rcv = channel_map_collection.tile_i64.get_receiver(
                            retile_streamify.input_id,
                            retile_streamify.stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                retile_streamify.input_id,
                                channel_depth,
                            ),
                        );
                        let snd = channel_map_collection.tile_i64.get_sender(
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
                                retile_streamify.chunk as usize,
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
            OpType::DynAddrGen(dyn_addr_gen) => {
                // The view walk is dtype-independent; only how the base index is
                // read off one input element differs, so each arm just picks the
                // channel the input stream lives on.
                macro_rules! make_dyn_addr_gen {
                    ($chan:ident) => {{
                        let rcv = channel_map_collection.$chan.get_receiver(
                            dyn_addr_gen.input_id,
                            dyn_addr_gen.input_stream_idx,
                            builder,
                            get_chan_depth(
                                &sim_config.config_dict,
                                dyn_addr_gen.input_id,
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
                            DynAddrGen::<_>::new(
                                rcv,
                                snd,
                                dyn_addr_gen
                                    .tensor_shape_tiled
                                    .iter()
                                    .map(|x| *x as usize)
                                    .collect(),
                                dyn_addr_gen.stride.iter().map(|x| *x as usize).collect(),
                                dyn_addr_gen
                                    .out_shape_tiled
                                    .iter()
                                    .map(|x| *x as usize)
                                    .collect(),
                                dyn_addr_gen.addr_base,
                                operation.id,
                            )
                        );
                    }};
                }

                match dyn_addr_gen.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::MultiHot(_) => make_dyn_addr_gen!(multihot),
                    Type::U64(_) => make_dyn_addr_gen!(tile_u64),
                    Type::I64(_) => make_dyn_addr_gen!(tile_i64),
                    dtype => panic!(
                        "Unsupported data type for DynAddrGen operation {:?}",
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                    init_func::InitFn::Zero(_zero) => Tile::new_zero_padded(
                                        [tile_row, tile_col],
                                        dtype_bytes,
                                        reshape.write_back_mu,
                                        0,
                                    ),
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
                    // bf16 is modelled as Tile<f32> on the tile_f32 channel.
                    Type::F32(_) | Type::Bf16(_) => {
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
                                    init_func::InitFn::Zero(_zero) => Tile::new_zero_padded(
                                        [tile_row, tile_col],
                                        dtype_bytes,
                                        reshape.write_back_mu,
                                        0,
                                    ),
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
                                        Tile::new_zero_padded(
                                            [tile_row, tile_col],
                                            dtype_bytes,
                                            reshape.write_back_mu,
                                            0,
                                        )
                                        // Tile::new_blank_padded(
                                        //     vec![tile_row, tile_col],
                                        //     dtype_bytes,
                                        //     reshape.write_back_mu,
                                        //     0,
                                        // )
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
                    // bf16 tiles ride the f32 channel map, same as every other op.
                    Type::F32(_) | Type::Bf16(_) => {
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
            OpType::Counter(counter) => {
                let sender = channel_map_collection.tile_u64.get_sender(
                    operation.id,
                    None,
                    builder,
                    get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                );
                add_child!(builder, Counter::new(counter.count, sender, operation.id));
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
    build_from_proto(
        step_graph,
        &mut channel_map_collection,
        &mut builder,
        &hbm_config,
        &sim_config,
        dump_prefix,
    );

    let initialized = builder.initialize(Default::default()).unwrap();
    let run_options = match logging {
        true => {
            let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
                dam::logging::LogFilter::Some([SimpleEvent::NAME.to_owned()].into()),
                // dam::logging::LogFilter::AllowAll,
            ));
            let run_options = run_options.logging(LoggingOptions::Mongo(
                MongoOptionsBuilder::default()
                    .db(db_name.unwrap_or("sim_default_name".to_string()))
                    .uri("mongodb://127.0.0.1:27017".to_string())
                    .build()
                    .unwrap(),
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
