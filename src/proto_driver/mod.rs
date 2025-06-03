pub mod proto_headers;

use crate::functions;
use dam::simulation::{
    DotConvertible, LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder,
    RunOptionsBuilder,
};
use std::sync::Arc;

use crate::build_sim::channel::ChannelMapCollection;
use crate::memory::offchip_load::OffChipLoad;
use crate::memory::offchip_store::OffChipStore;
use crate::operator::{map::BinaryMap, repeat::RepeatStatic};
use crate::primitives::tile::Tile;
use crate::proto_driver::proto_headers::graph_proto::{
    data_type::Type, elemto_elem_func, operation::OpType, ProgramGraph,
};
use crate::ramulator::hbm_context::{HBMConfig, HBMContext, ReadBundle, WriteBundle};
use crate::utils::{cast::to_usize_vec, events::SimpleEvent};

fn build_from_proto<'a>(
    step_graph: ProgramGraph,
    channel_map_collection: &mut ChannelMapCollection<'a>,
    builder: &mut ProgramBuilder<'a>,
    hbm_config: &HBMConfig,
) {
    let mut mem_context = HBMContext::new(builder, hbm_config.clone());

    for operation in step_graph.operators {
        println!("processing {:?}\n", operation);
        match operation.op_type.clone().unwrap() {
            OpType::Binarymap(binary_map) => match (
                binary_map.dtype_a.clone().unwrap().r#type.clone().unwrap(),
                binary_map.dtype_b.clone().unwrap().r#type.clone().unwrap(),
            ) {
                (Type::F32(_), Type::F32(_)) => {
                    // create
                    let rcv1 = channel_map_collection.tile_f32.get_receiver(
                        binary_map.input_id1,
                        binary_map.stream_idx1,
                        builder,
                        Some(1),
                    );
                    let rcv2 = channel_map_collection.tile_f32.get_receiver(
                        binary_map.input_id2,
                        binary_map.stream_idx2,
                        builder,
                        Some(1),
                    );
                    let snd = channel_map_collection.tile_f32.get_sender(
                        operation.id,
                        None,
                        builder,
                        Some(1),
                    );
                    let map_fn: Arc<
                        dyn Fn(&Tile<f32>, &Tile<f32>, u64, bool) -> (u64, Tile<f32>) + Send + Sync,
                    > = match binary_map.func.unwrap().elem_elem_fn.unwrap() {
                        elemto_elem_func::ElemElemFn::Matmul(matmul) => {
                            Arc::new(|tile1, tile2, comp_bw, write_back_mu| {
                                functions::map_fn::matmul(
                                    tile1,
                                    tile2,
                                    comp_bw,
                                    write_back_mu,
                                    false,
                                )
                            })
                        }
                    };
                    builder.add_child(BinaryMap::<SimpleEvent, _, _>::new(
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
            OpType::OffChipLoad(off_chip_load) => {
                match off_chip_load.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let on_chip_snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            Some(1),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        builder.add_child(OffChipLoad::<SimpleEvent, _>::new(
                            to_usize_vec(off_chip_load.tensor_shape_tiled),
                            to_usize_vec(off_chip_load.stride),
                            to_usize_vec(off_chip_load.out_shape_tiled),
                            Some(off_chip_load.npy_path),
                            off_chip_load.tile_row as usize,
                            off_chip_load.tile_col as usize,
                            4,
                            0,
                            hbm_config.addr_offset,
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
                            Some(1),
                        );
                        let (addr_snd, addr_rcv) = builder.unbounded();
                        let (resp_snd, resp_rcv) = builder.unbounded();

                        builder.add_child(OffChipStore::<SimpleEvent, _>::new(
                            to_usize_vec(off_chip_store.tensor_shape_tiled),
                            off_chip_store.tile_row as usize,
                            off_chip_store.tile_col as usize,
                            Some(off_chip_store.store_path),
                            0,
                            hbm_config.addr_offset,
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
            OpType::RepeatStatic(repeat_static) => {
                match repeat_static.dtype.clone().unwrap().r#type.clone().unwrap() {
                    Type::F32(_) => {
                        let rcv = channel_map_collection.tile_f32.get_receiver(
                            repeat_static.input_id,
                            repeat_static.stream_idx,
                            builder,
                            Some(1),
                        );
                        let snd = channel_map_collection.tile_f32.get_sender(
                            operation.id,
                            None,
                            builder,
                            Some(1),
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
        }
    }

    builder.add_child(mem_context);
}

pub fn parse_proto<'a>(
    step_graph: ProgramGraph,
    logging: bool,
    hbm_config: HBMConfig,
) -> (bool, u64) {
    let mut builder = ProgramBuilder::default();
    let mut channel_map_collection = ChannelMapCollection::default();
    build_from_proto(
        step_graph,
        &mut channel_map_collection,
        &mut builder,
        &hbm_config,
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
                    .db("test_sim".to_string())
                    .uri("mongodb://127.0.0.1:27017".to_string())
                    .build()
                    .unwrap(),
            ));
            run_options.build().unwrap()
        }
        false => Default::default(),
    };

    println!("{}", initialized.to_dot_string());
    let executed = initialized.run(run_options);

    let cycles = executed.elapsed_cycles().unwrap();
    let passed = executed.passed();
    (passed, cycles)
}
