pub mod configs;
pub mod proto_headers;

use crate::functions;
use crate::memory::dyn_offchip_load::DynOffChipLoad;
use crate::memory::metadata_gen::MetadataGen;
use crate::memory::random_offchip_load::RandomOffChipLoad;
use crate::memory::random_offchip_store::RandomOffChipStore;
use crate::operator::eager_merge::EagerMerge;
use crate::operator::expand::ExpandRef;
use crate::operator::parallelize::Parallelize;
use std::collections::HashMap;
use std::time::Instant;

use crate::operator::accum::{Accum, AccumConfig};
use crate::operator::broadcast::BroadcastContext;
use crate::operator::bufferize::Bufferize;
use crate::operator::dynstreamify::DynStreamify;
use crate::operator::flatmap::{CacheReadAddrGen, ExpertAddrGen, FilterLastTile, RetileStreamify};
use crate::operator::flatten::Flatten;
use crate::operator::map::{UnaryMap, UnaryMapConfig};
use crate::operator::map_accum::BinaryMapAccum;
use crate::operator::partition::{FlatPartition, FlatPartitionConfig};
use crate::operator::promote::Promote;
use crate::operator::reassemble::{FlatReassemble, FlatReassembleConfig};
use crate::operator::reshape::Reshape;
use crate::operator::streamify::Streamify;
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
use crate::memory::offchip_load::OffChipLoad;
use crate::memory::offchip_store::OffChipStore;
use crate::operator::{map::BinaryMap, repeat::RepeatStatic};
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

// channel_depth will be set from sim_config.channel_depth

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

        $builder.add_child(broadcast_node);
    };
}

macro_rules! make_dyn_offchip_load {
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

        $builder.add_child(DynOffChipLoad::<SimpleEvent, _, _>::new(
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
            $operation.id,
        ));
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

fn build_from_proto<'a>(
    step_graph: ProgramGraph,
    channel_map_collection: &mut ChannelMapCollection<'a>,
    builder: &mut ProgramBuilder<'a>,
    hbm_config: &HBMConfig,
    sim_config: &SimConfig,
) {
    let channel_depth = sim_config.channel_depth;
    let mut mem_context = HBMContext::new(builder, hbm_config.clone());

    // Use a regular variable instead of a const, since sim_config.mock_bf16 is not a constant
    let f32_bytes: usize = if sim_config.mock_bf16 { 2 } else { 4 }; // we will use this to mimic bfloat16

    for operation in step_graph.operators {
        // if operation.id == 336 || operation.id == 272 || operation.id == 721 {
        //     println!("processing {:?}\n", operation);
        // }
        // println!("processing {:?}\n", operation);

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
                        elemto_elem_func::ElemElemFn::RowWiseSum(row_wise_sum) => {
                            Arc::new(move |tile, comp_bw, write_back_mu| {
                                functions::map_fn::row_wise_sum(tile, comp_bw, write_back_mu)
                            })
                        }
                        _ => {
                            panic!("Unsupported unary map function type")
                        }
                    };

                    builder.add_child(UnaryMap::<SimpleEvent, _, _>::new(
                        rcv,
                        snd,
                        map_fn,
                        UnaryMapConfig {
                            compute_bw: unarymap.compute_bw as u64,
                            write_back_mu: unarymap.write_back_mu,
                        },
                        operation.id,
                    ));
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
                        elemto_elem_func::ElemElemFn::Div(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::div(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        _ => {
                            panic!("Unsupported binary map function type")
                        }
                    };
                    builder.add_child(BinaryMap::<SimpleEvent, _, _, _>::new(
                        rcv1,
                        rcv2,
                        snd,
                        map_fn,
                        binary_map.compute_bw as u64,
                        binary_map.write_back_mu,
                        operation.id,
                    ));
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
                        _ => {
                            panic!("Unsupported binary map function type")
                        }
                    };
                    builder.add_child(BinaryMap::<SimpleEvent, _, _, _>::new(
                        rcv1,
                        rcv2,
                        snd,
                        map_fn,
                        binary_map.compute_bw as u64,
                        binary_map.write_back_mu,
                        operation.id,
                    ));
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
                        _ => {
                            panic!("Unsupported binary map function type")
                        }
                    };
                    builder.add_child(BinaryMap::<SimpleEvent, _, _, _>::new(
                        rcv1,
                        rcv2,
                        snd,
                        map_fn,
                        binary_map.compute_bw as u64,
                        binary_map.write_back_mu,
                        operation.id,
                    ));
                }
                _ => panic!("Unsupported data types for BinaryMap operation"),
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
                        _ => {
                            panic!("Unsupported binary map accumulation function type",)
                        }
                    };

                    let tile_row = binary_map_accum.tile_row as usize;
                    let tile_col = binary_map_accum.tile_col as usize;

                    builder.add_child(BinaryMapAccum::<SimpleEvent, _, _>::new(
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
                    ));
                }
                (_, _) => todo!(),
            },
            OpType::OffChipLoad(off_chip_load) => {
                match off_chip_load.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let on_chip_snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        builder.add_child(OffChipLoad::<SimpleEvent, _>::new(
                            to_usize_vec(off_chip_load.tensor_shape_tiled),
                            to_usize_vec(off_chip_load.stride),
                            to_usize_vec(off_chip_load.out_shape_tiled),
                            off_chip_load.npy_path,
                            off_chip_load.tile_row as usize,
                            off_chip_load.tile_col as usize,
                            f32_bytes,
                            0,
                            hbm_config.addr_offset,
                            off_chip_load.par_dispatch as usize,
                            addr_snd,
                            resp_rcv,
                            on_chip_snd,
                            operation.id,
                        ));

                        mem_context.add_reader(ReadBundle {
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

                        builder.add_child(OffChipStore::<SimpleEvent, _>::new(
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
                        ));

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
                        let wack = channel_map_collection.bool.get_sender(
                            operation.id,
                            None,
                            builder,
                            get_chan_depth(&sim_config.config_dict, operation.id, channel_depth),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        builder.add_child(RandomOffChipStore::<SimpleEvent, _>::new(
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
                        ));

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

                        builder.add_child(RandomOffChipLoad::<SimpleEvent, _>::new(
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
                            operation.id,
                        ));

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
                        builder.add_child(ExpandRef::<_, _>::new(
                            in_rcv,
                            ref_rcv,
                            expand_ref.expand_rank,
                            snd,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for ExpandRef operation"),
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
                        builder.add_child(RepeatStatic::<_>::new(
                            rcv,
                            repeat_static.repeat_factor as usize,
                            snd,
                        ));
                    }
                    _ => panic!("Unsupported data type for RepeatStatic operation"),
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
                    _ => panic!("Unsupported data type for Broadcast operation"),
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
                                builder.add_child(FlatPartition::<SimpleEvent, _, _>::new(
                                    input_rcv,
                                    control_rcv,
                                    snd_list,
                                    flat_partition.partition_rank,
                                    FlatPartitionConfig {
                                        switch_cycles: to_u64_vec(flat_partition.switch_cycles),
                                        write_back_mu: flat_partition.write_back_mu,
                                    },
                                    operation.id,
                                ))
                            }
                            _ => panic!("Unsupported data type"),
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
                                builder.add_child(FlatPartition::<SimpleEvent, _, _>::new(
                                    input_rcv,
                                    control_rcv,
                                    snd_list,
                                    flat_partition.partition_rank,
                                    FlatPartitionConfig {
                                        switch_cycles: to_u64_vec(flat_partition.switch_cycles),
                                        write_back_mu: flat_partition.write_back_mu,
                                    },
                                    operation.id,
                                ))
                            }
                            _ => panic!("Unsupported data type"),
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
                                builder.add_child(FlatPartition::<SimpleEvent, _, _>::new(
                                    input_rcv,
                                    control_rcv,
                                    snd_list,
                                    flat_partition.partition_rank,
                                    FlatPartitionConfig {
                                        switch_cycles: to_u64_vec(flat_partition.switch_cycles),
                                        write_back_mu: flat_partition.write_back_mu,
                                    },
                                    operation.id,
                                ))
                            }
                            _ => panic!("Unsupported data type"),
                        }
                    }
                    _ => panic!("Unsupported data type"),
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
                                builder.add_child(FlatReassemble::<SimpleEvent, _, _>::new(
                                    rcv_list,
                                    control_rcv,
                                    snd,
                                    reassemble.reassemble_rank,
                                    FlatReassembleConfig {
                                        switch_cycles: to_u64_vec(reassemble.switch_cycles),
                                        write_back_mu: reassemble.write_back_mu,
                                    },
                                    operation.id,
                                ))
                            }
                            _ => panic!("Unsupported data type"),
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
                                builder.add_child(FlatReassemble::<SimpleEvent, _, _>::new(
                                    rcv_list,
                                    control_rcv,
                                    snd,
                                    reassemble.reassemble_rank,
                                    FlatReassembleConfig {
                                        switch_cycles: to_u64_vec(reassemble.switch_cycles),
                                        write_back_mu: reassemble.write_back_mu,
                                    },
                                    operation.id,
                                ))
                            }
                            _ => panic!("Unsupported data type"),
                        }
                    }
                    _ => panic!("Unsupported data type"),
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
                        builder.add_child(Parallelize::<SimpleEvent, _>::new(
                            input_rcv,
                            snd_list,
                            parallelize.parallelize_rank,
                            FlatPartitionConfig {
                                switch_cycles: to_u64_vec(parallelize.switch_cycles),
                                write_back_mu: parallelize.write_back_mu,
                            },
                            operation.id,
                        ))
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
                        builder.add_child(Parallelize::<SimpleEvent, _>::new(
                            input_rcv,
                            snd_list,
                            parallelize.parallelize_rank,
                            FlatPartitionConfig {
                                switch_cycles: to_u64_vec(parallelize.switch_cycles),
                                write_back_mu: parallelize.write_back_mu,
                            },
                            operation.id,
                        ))
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
                        builder.add_child(Parallelize::<SimpleEvent, _>::new(
                            input_rcv,
                            snd_list,
                            parallelize.parallelize_rank,
                            FlatPartitionConfig {
                                switch_cycles: to_u64_vec(parallelize.switch_cycles),
                                write_back_mu: parallelize.write_back_mu,
                            },
                            operation.id,
                        ))
                    }
                    _ => panic!("Unsupported data type"),
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
                        builder.add_child(Promote::new(rcv, snd, promote.promote_rank));
                    }
                    _ => panic!("Unsupported data type"),
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
                        builder.add_child(ConsumerContext::new(rcv));
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        builder.add_child(ConsumerContext::new(rcv));
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        builder.add_child(ConsumerContext::new(rcv));
                    }
                    Type::ScalarU64(_) => {
                        let rcv = channel_map_collection.u64.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        builder.add_child(ConsumerContext::new(rcv));
                    }
                    Type::ScalarBool(_) => {
                        let rcv = channel_map_collection.bool.get_receiver(
                            consumer_context.input_id,
                            consumer_context.stream_idx,
                            builder,
                            None,
                        );
                        builder.add_child(ConsumerContext::new(rcv));
                    }
                    _ => panic!("Unsupported data type for ConsumerContext operation"),
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
                        builder.add_child(PrinterContext::new(rcv));
                    }
                    Type::U64(_) => {
                        let rcv = channel_map_collection.tile_u64.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        builder.add_child(PrinterContext::new(rcv));
                    }
                    Type::MultiHot(_) => {
                        let rcv = channel_map_collection.multihot.get_receiver(
                            printer_context.input_id,
                            printer_context.stream_idx,
                            builder,
                            None,
                        );
                        builder.add_child(PrinterContext::new(rcv));
                    }
                    _ => panic!("Unsupported data type for PrinterContext operation"),
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
                        builder.add_child(Bufferize::<SimpleEvent, _>::new(
                            rcv,
                            snd,
                            bufferize.rank,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for Bufferize operation"),
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
                        builder.add_child(Streamify::<SimpleEvent, _>::new(
                            to_usize_vec(streamify.repeat_factor),
                            streamify.rank,
                            rcv,
                            snd,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for Streamify operation"),
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
                        builder.add_child(DynStreamify::<SimpleEvent, _, _>::new(
                            rcv,
                            dyn_streamify.bufferized_rank,
                            dyn_streamify.repeat_rank,
                            ref_rcv,
                            snd,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for DynStreamify operation"),
                }
            }
            OpType::DynOffChipLoad(dyn_offchip_load) => {
                match (
                    dyn_offchip_load
                        .dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                    dyn_offchip_load
                        .ref_dtype
                        .clone()
                        .unwrap()
                        .r#type
                        .clone()
                        .unwrap(),
                ) {
                    (Type::F32(_), Type::F32(_)) => {
                        make_dyn_offchip_load!(
                            channel_map_collection,
                            operation,
                            dyn_offchip_load,
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
                        make_dyn_offchip_load!(
                            channel_map_collection,
                            operation,
                            dyn_offchip_load,
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
                        make_dyn_offchip_load!(
                            channel_map_collection,
                            operation,
                            dyn_offchip_load,
                            hbm_config,
                            multihot,
                            tile_f32,
                            f32_bytes,
                            mem_context,
                            builder,
                            channel_depth
                        );
                    }
                    _ => panic!("Unsupported data type for DynOffChipLoad operation"),
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
                        builder.add_child(Flatten::new(
                            rcv,
                            snd,
                            flatten.min_rank,
                            flatten.max_rank,
                        ));
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
                        builder.add_child(Flatten::new(
                            rcv,
                            snd,
                            flatten.min_rank,
                            flatten.max_rank,
                        ));
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
                        builder.add_child(Flatten::new(
                            rcv,
                            snd,
                            flatten.min_rank,
                            flatten.max_rank,
                        ));
                    }

                    _ => panic!("Unsupported data type for Flatten operation"),
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
                    builder.add_child(GeneratorContext::new(
                        move || {
                            read_multihot_elem_from_npy_iter::<i64>(&select_gen.npy_path).unwrap()
                        },
                        snd,
                    ));
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
                                functions::accum_fn::add(tile1, tile2, comp_bw, write_back_mu)
                            })
                        }
                        accum_func::AccumFn::RetileRow(_) => {
                            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                                functions::accum_fn::retile_row(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
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

                    builder.add_child(Accum::<SimpleEvent, _, _>::new(
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

                    builder.add_child(Accum::<SimpleEvent, _, _>::new(
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
                    ));
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
                        builder.add_child(RetileStreamify::<_>::new(
                            rcv,
                            snd,
                            retile_streamify.split_row,
                            retile_streamify.filter_mask,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for RetileStreamify operation"),
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
                    Type::ScalarU64(_) => {
                        builder.add_child(MetadataGen::<u64>::new(
                            metadata_gen.npy_path,
                            snd,
                            operation.id,
                        ));
                    }
                    Type::ScalarI64(_) => {
                        builder.add_child(MetadataGen::<i64>::new(
                            metadata_gen.npy_path,
                            snd,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for MetadataGen operation"),
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
                        builder.add_child(ExpertAddrGen::<_>::new(
                            rcv,
                            snd,
                            expert_addr_gen.num_tile_per_expert as u64,
                            expert_addr_gen.expert_addr_base as u64,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for ExpertAddrGen operation"),
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
                builder.add_child(CacheReadAddrGen::new(
                    idx_rcv,
                    seq_len_rcv,
                    cache_read_addr_gen.offset_per_idx,
                    snd,
                    operation.id,
                ));
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
                builder.add_child(FilterLastTile::new(seq_len_rcv, snd, operation.id));
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
                                builder.add_child(Reshape::new(
                                    rcv,
                                    snd,
                                    reshape.split_dim as usize,
                                    reshape.chunk_size as usize,
                                    Some(pad_val),
                                    reshape.input_stream_rank,
                                    reshape.add_outer_dim,
                                    operation.id,
                                ));
                            }
                            None => {
                                builder.add_child(Reshape::new(
                                    rcv,
                                    snd,
                                    reshape.split_dim as usize,
                                    reshape.chunk_size as usize,
                                    None,
                                    reshape.input_stream_rank,
                                    reshape.add_outer_dim,
                                    operation.id,
                                ));
                            }
                        }
                    }
                    _ => panic!("Unsupported data type for Reshape operation"),
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
                        builder.add_child(EagerMerge::new(
                            rcv_list,
                            sel_snd,
                            snd,
                            eager_merge.input_rank,
                            operation.id,
                        ));
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
                        builder.add_child(EagerMerge::new(
                            rcv_list,
                            sel_snd,
                            snd,
                            eager_merge.input_rank,
                            operation.id,
                        ));
                    }
                    _ => panic!("Unsupported data type for EagerMerge operation"),
                }
            }
            _ => todo!(),
        }
    }

    builder.add_child(mem_context);
}

pub fn parse_proto<'a>(
    step_graph: ProgramGraph,
    logging: bool,
    hbm_config: HBMConfig,
    sim_config: SimConfig,
    db_name: Option<String>,
) -> (bool, u64, std::time::Duration) {
    let mut builder = ProgramBuilder::default();
    let mut channel_map_collection = ChannelMapCollection::default();
    build_from_proto(
        step_graph,
        &mut channel_map_collection,
        &mut builder,
        &hbm_config,
        &sim_config,
    );

    let initialized = builder.initialize(Default::default()).unwrap();
    let run_options = match logging {
        true => {
            let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
                // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
                dam::logging::LogFilter::AllowAll,
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
