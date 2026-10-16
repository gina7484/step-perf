use crate::memory::PMU_BW;
use crate::operator::partition::FlatPartitionConfig;
use crate::primitives::elem::{Bufferizable, Elem, EmptyFiber, StopType};
use crate::primitives::select::{MultiHotN, SelectAdapter};
use crate::utils::calculation::div_ceil;
use dam::{context_tools::*, simulation::ProgramBuilder};

#[derive(Clone, Debug, Default)]
struct Bucket<A>(Vec<Elem<A>>);

impl<A: DAMType> DAMType for Bucket<A> {
    fn dam_size(&self) -> usize {
        self.0.iter().map(DAMType::dam_size).sum()
    }
}

/// Collect one rank-(m+1) group, then let every branch drain independently.
/// Bucket handoffs carry ownership of the group's storage. Acknowledgements
/// prevent collection of the next group until every bucket has drained.
#[context_macro]
pub struct StructuredPartition<A: DAMType> {
    input: Receiver<Elem<A>>,
    control: Receiver<Elem<MultiHotN>>,
    buckets: Vec<Sender<Bucket<A>>>,
    drained: Vec<Receiver<()>>,
    rank: StopType,
    sel_npy_path: String,
    routing_table: Option<Vec<MultiHotN>>,
}

#[context_macro]
struct DrainBucket<A: DAMType> {
    input: Receiver<Bucket<A>>,
    output: Sender<Elem<A>>,
    drained: Sender<()>,
    switch_cycles: u64,
    write_back_mu: bool,
}

impl<A: DAMType + Bufferizable> Context for DrainBucket<A> {
    fn run(&mut self) {
        while let Ok(bucket) = self.input.dequeue(&self.time) {
            for elem in bucket.data.0 {
                let (value, stop) = elem.into_parts();
                let store = if self.write_back_mu {
                    div_ceil(value.size_in_bytes() as u64, PMU_BW)
                } else {
                    0
                };
                self.time.incr_cycles(self.switch_cycles + store);
                self.output
                    .enqueue(
                        &self.time,
                        ChannelElement::new(
                            self.time.tick(),
                            Elem::with_stop(
                                value.clone_with_updated_read_from_mu(self.write_back_mu),
                                stop,
                            ),
                        ),
                    )
                    .unwrap();
            }
            self.drained
                .enqueue(&self.time, ChannelElement::new(self.time.tick(), ()))
                .unwrap();
        }
    }
}

impl<A: DAMType + Bufferizable + EmptyFiber + 'static> StructuredPartition<A> {
    pub fn new(
        builder: &mut ProgramBuilder<'_>,
        input: Receiver<Elem<A>>,
        control: Receiver<Elem<MultiHotN>>,
        outputs: Vec<Sender<Elem<A>>>,
        rank: StopType,
        config: FlatPartitionConfig,
        sel_npy_path: String,
    ) -> Self {
        assert!(!outputs.is_empty(), "StructuredPartition needs an output");
        assert_eq!(outputs.len(), config.switch_cycles.len());
        let mut buckets = Vec::new();
        let mut drained = Vec::new();
        for (output, switch_cycles) in outputs.into_iter().zip(config.switch_cycles) {
            let (send_bucket, receive_bucket) = builder.bounded(1);
            let (send_ack, receive_ack) = builder.bounded(1);
            let writer = DrainBucket {
                input: receive_bucket,
                output,
                drained: send_ack,
                switch_cycles,
                write_back_mu: config.write_back_mu,
                context_info: Default::default(),
            };
            writer.input.attach_receiver(&writer);
            writer.output.attach_sender(&writer);
            writer.drained.attach_sender(&writer);
            builder.add_child(writer);
            buckets.push(send_bucket);
            drained.push(receive_ack);
        }
        let ctx = Self {
            input,
            control,
            buckets,
            drained,
            rank,
            sel_npy_path,
            routing_table: None,
            context_info: Default::default(),
        };
        ctx.input.attach_receiver(&ctx);
        ctx.control.attach_receiver(&ctx);
        for sender in &ctx.buckets {
            sender.attach_sender(&ctx);
        }
        for receiver in &ctx.drained {
            receiver.attach_receiver(&ctx);
        }
        ctx
    }

    fn read_selections(&self) -> Option<Vec<MultiHotN>> {
        let file = match std::fs::File::open(&self.sel_npy_path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
            Err(error) => panic!("Could not read routing tensor: {error}"),
        };
        let npy = npyz::NpyFile::new(file).expect("invalid routing .npy file");
        assert_eq!(
            npy.order(),
            npyz::Order::C,
            "routing tensor must use C order"
        );
        let width = *npy
            .shape()
            .last()
            .expect("routing tensor needs a candidate dimension") as usize;
        assert_eq!(
            width,
            self.buckets.len(),
            "routing tensor width must match outputs"
        );
        let values: Vec<i64> = npy
            .into_vec()
            .expect("routing tensor must contain int64 values");
        Some(
            values
                .chunks(width)
                .map(|row| MultiHotN::new(row.iter().map(|x| *x != 0).collect(), false))
                .collect(),
        )
    }
}

impl<A: DAMType + Bufferizable + EmptyFiber + 'static> Context for StructuredPartition<A> {
    fn init(&mut self) {
        // npy header parsing needs the host stack, not DAM's coroutine stack.
        self.routing_table = self.read_selections();
    }

    fn run(&mut self) {
        let width = self.buckets.len();
        let mut buckets = vec![Vec::new(); width];
        let mut group_open = false;
        let mut blank_mode = None;
        let table = self.routing_table.take();
        let mut row = 0;
        while let Ok(selection) = self.control.dequeue(&self.time) {
            let (selection, control_stop) = selection.data.into_parts();
            assert_eq!(selection.len(), width, "selection width must match outputs");
            assert!(
                !selection.is_empty_fiber(),
                "StructuredPartition expects one selection per input fiber"
            );
            let blank = selection.is_blank();
            if let Some(previous) = blank_mode {
                assert_eq!(
                    previous, blank,
                    "selection stream mixes blank and inline values"
                );
            } else {
                blank_mode = Some(blank);
            }
            let selected = if blank {
                let selected = table
                    .as_ref()
                    .expect("blank selections require a routing .npy file")
                    .get(row)
                    .expect("routing tensor has too few rows")
                    .to_sel_vec();
                row += 1;
                selected
            } else {
                selection.to_sel_vec()
            };
            if selection.read_from_mu() {
                self.time
                    .incr_cycles(div_ceil(selection.size_in_bytes() as u64, PMU_BW));
            }

            group_open = true;
            let mut first = true;
            let (template, final_stop) = loop {
                let elem = self
                    .input
                    .dequeue(&self.time)
                    .expect("input ended inside a selected fiber")
                    .data;
                let (value, stop) = elem.into_parts();
                if value.is_empty_fiber() {
                    assert!(
                        first && self.rank > 0 && stop >= self.rank,
                        "empty input must be a complete fiber with a stop"
                    );
                }
                first = false;
                if value.read_from_mu() {
                    self.time
                        .incr_cycles(div_ceil(value.size_in_bytes() as u64, PMU_BW));
                }
                let fiber_end = self.rank == 0 || stop >= self.rank;
                // Only the final selected fiber in each bucket closes the group.
                let local_stop = if fiber_end { self.rank } else { stop };
                for lane in &selected {
                    buckets[*lane].push(Elem::with_stop(value.clone(), local_stop));
                }
                if fiber_end {
                    break (value, stop);
                }
            };
            assert_eq!(
                final_stop - self.rank,
                control_stop,
                "data and selection stops disagree"
            );
            if control_stop == 0 {
                continue;
            }

            for (lane, bucket) in buckets.iter_mut().enumerate() {
                if let Some(last) = bucket.pop() {
                    bucket.push(Elem::ValStop(last.into_parts().0, final_stop));
                } else {
                    bucket.push(Elem::ValStop(template.empty_fiber_like(), final_stop));
                }
                self.buckets[lane]
                    .enqueue(
                        &self.time,
                        ChannelElement::new(self.time.tick(), Bucket(std::mem::take(bucket))),
                    )
                    .unwrap();
            }
            for ack in &self.drained {
                ack.dequeue(&self.time).unwrap();
            }
            group_open = false;
        }
        assert!(!group_open, "selection stream ended before a group stop");
        assert!(
            self.input.dequeue(&self.time).is_err(),
            "data has more fibers than selections"
        );
        if blank_mode == Some(true) {
            assert_eq!(row, table.unwrap().len(), "routing tensor has extra rows");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::{
        accum::{Accum, AccumConfig},
        broadcast::BroadcastContext,
        crd_dropper::CrdDropper,
        map::{UnaryMapConfig, UnaryMapToMultiHot},
        reshape::ReshapeNoPadStream,
        scan::EnumerateScan,
        static_reassemble::StaticReassemble,
        test_support::collect,
    };
    use crate::{
        functions::{accum_fn, map_fn},
        primitives::tile::Tile,
        utils::events::SimpleEvent,
    };
    use dam::utility_contexts::GeneratorContext;
    use std::sync::Arc;

    fn tile(value: u64) -> Tile<u64> {
        Tile::new(ndarray::ArcArray2::from_elem((1, 1), value), 8, false)
    }
    fn data(value: u64, stop: u32) -> Elem<Tile<u64>> {
        Elem::with_stop(tile(value), stop)
    }
    fn sel(indices: Vec<usize>, stop: u32, width: usize) -> Elem<MultiHotN> {
        Elem::with_stop(MultiHotN::from_sel_vec(indices, width, false), stop)
    }

    fn partition(
        input: Vec<Elem<Tile<u64>>>,
        control: Vec<Elem<MultiHotN>>,
        rank: u32,
        width: usize,
        path: String,
    ) -> (bool, Vec<Vec<Elem<Tile<u64>>>>) {
        let mut builder = ProgramBuilder::default();
        let (data_tx, data_rx) = builder.unbounded();
        let (sel_tx, sel_rx) = builder.unbounded();
        builder.add_child(GeneratorContext::new(move || input.into_iter(), data_tx));
        builder.add_child(GeneratorContext::new(move || control.into_iter(), sel_tx));
        let mut outputs = Vec::new();
        let mut collected = Vec::new();
        for _ in 0..width {
            let (tx, rx) = builder.bounded(1);
            outputs.push(tx);
            collected.push(collect(&mut builder, rx));
        }
        let op = StructuredPartition::new(
            &mut builder,
            data_rx,
            sel_rx,
            outputs,
            rank,
            FlatPartitionConfig {
                switch_cycles: vec![1; width],
                write_back_mu: false,
            },
            path,
        );
        builder.add_child(op);
        let run = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        let values = collected
            .iter()
            .map(|v| v.lock().unwrap().clone())
            .collect();
        (run.passed(), values)
    }

    #[test]
    fn keeps_outer_groups_and_routes_empty_selections_to_no_branch() {
        let (passed, output) = partition(
            vec![data(1, 0), data(2, 1), data(3, 0), data(4, 2), data(5, 3)],
            vec![sel(vec![0, 2], 0, 3), sel(vec![], 1, 3), sel(vec![1], 2, 3)],
            1,
            3,
            String::new(),
        );
        assert!(passed);
        let empty = Tile::<u64>::new_empty_fiber(vec![1, 1], 8);
        assert_eq!(
            output[0],
            vec![data(1, 0), data(2, 2), Elem::ValStop(empty.clone(), 3)]
        );
        assert_eq!(output[1], vec![Elem::ValStop(empty.clone(), 2), data(5, 3)]);
        assert_eq!(output[2], output[0]);
    }

    #[test]
    fn rank_zero_and_all_empty_groups() {
        let (passed, output) = partition(
            vec![data(1, 0), data(2, 1), data(3, 2)],
            vec![sel(vec![0], 0, 2), sel(vec![0], 1, 2), sel(vec![], 2, 2)],
            0,
            2,
            String::new(),
        );
        assert!(passed);
        let empty = Tile::<u64>::new_empty_fiber(vec![1, 1], 8);
        assert_eq!(
            output[0],
            vec![data(1, 0), data(2, 1), Elem::ValStop(empty.clone(), 2)]
        );
        assert_eq!(
            output[1],
            vec![Elem::ValStop(empty.clone(), 1), Elem::ValStop(empty, 2)]
        );
        let (passed, output) = partition(vec![], vec![], 0, 1, String::new());
        assert!(passed);
        assert!(output[0].is_empty());
    }

    #[test]
    fn blank_selections_use_routing_tensor_including_zero_rows() {
        use npyz::WriterBuilder;
        let path =
            std::env::temp_dir().join(format!("structured_partition_{}.npy", std::process::id()));
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[2, 2])
            .writer(std::fs::File::create(&path).unwrap())
            .begin_nd()
            .unwrap();
        writer.extend([0i64, 1, 0, 0]).unwrap();
        writer.finish().unwrap();
        let (passed, output) = partition(
            vec![data(1, 1), data(2, 2)],
            vec![
                Elem::Val(MultiHotN::new_blank(2, false)),
                Elem::ValStop(MultiHotN::new_blank(2, false), 1),
            ],
            1,
            2,
            path.to_string_lossy().into_owned(),
        );
        std::fs::remove_file(path).unwrap();
        assert!(passed);
        assert_eq!(output[1], vec![data(1, 2)]);
        assert!(output[0][0].clone().into_parts().0.is_empty_fiber());
    }

    #[test]
    fn rejects_routing_tables_with_missing_or_extra_rows() {
        use npyz::WriterBuilder;
        for rows in [1, 3] {
            let path = std::env::temp_dir().join(format!(
                "structured_partition_rows_{}_{}.npy",
                std::process::id(),
                rows
            ));
            let mut writer = npyz::WriteOptions::new()
                .default_dtype()
                .shape(&[rows, 2])
                .writer(std::fs::File::create(&path).unwrap())
                .begin_nd()
                .unwrap();
            writer.extend(vec![1i64; (rows * 2) as usize]).unwrap();
            writer.finish().unwrap();
            let (passed, _) = partition(
                vec![data(1, 1), data(2, 2)],
                vec![
                    Elem::Val(MultiHotN::new_blank(2, false)),
                    Elem::ValStop(MultiHotN::new_blank(2, false), 1),
                ],
                1,
                2,
                path.to_string_lossy().into_owned(),
            );
            std::fs::remove_file(path).unwrap();
            assert!(!passed);
        }
    }

    #[test]
    fn rejects_bad_stops_counts_widths_and_mixed_selection_modes() {
        for (input, selections) in [
            (vec![data(1, 2)], vec![sel(vec![0], 0, 2)]),
            (vec![data(1, 2), data(2, 2)], vec![sel(vec![0], 1, 2)]),
            (vec![data(1, 2)], vec![sel(vec![0], 1, 3)]),
            (vec![data(1, 1)], vec![sel(vec![0], 0, 2)]),
            (
                vec![data(1, 1), data(2, 2)],
                vec![
                    sel(vec![0], 0, 2),
                    Elem::ValStop(MultiHotN::new_blank(2, false), 1),
                ],
            ),
            (
                vec![data(1, 2)],
                vec![Elem::ValStop(MultiHotN::new_blank(2, false), 1)],
            ),
        ] {
            assert!(!partition(input, selections, 1, 2, String::new()).0);
        }
    }

    #[test]
    fn bounded_partition_reshape_reassemble_enumerate_dropper_pipeline() {
        let mut builder = ProgramBuilder::default();
        let (data_tx, data_rx) = builder.bounded(1);
        let (sel_tx, sel_rx) = builder.bounded(1);
        // Empty lane 0 and a long later bucket expose serial-drain deadlocks.
        // The next groups exercise trailing empty branches and empty parents.
        let input = vec![
            data(1, 0),
            data(2, 1),
            data(3, 0),
            data(4, 1),
            data(5, 0),
            data(6, 2),
            data(0, 2),
            data(8, 2),
            data(9, 3),
        ];
        let selections = vec![
            sel(vec![2, 3], 0, 4),
            sel(vec![2], 0, 4),
            sel(vec![2], 1, 4),
            sel(vec![1], 1, 4),
            sel(vec![], 1, 4),
            sel(vec![], 2, 4),
        ];
        builder.add_child(GeneratorContext::new(move || input.into_iter(), data_tx));
        builder.add_child(GeneratorContext::new(
            move || selections.into_iter(),
            sel_tx,
        ));
        let mut outputs = Vec::new();
        let mut lanes = Vec::new();
        for _ in 0..4 {
            let (tx, rx) = builder.bounded(1);
            let (reshape_tx, reshape_rx) = builder.bounded(1);
            outputs.push(tx);
            lanes.push(reshape_rx);
            builder.add_child(ReshapeNoPadStream::new(
                rx, reshape_tx, 2, 1, None, 3, false, 0,
            ));
        }
        let partition = StructuredPartition::new(
            &mut builder,
            data_rx,
            sel_rx,
            outputs,
            1,
            FlatPartitionConfig {
                switch_cycles: vec![1; 4],
                write_back_mu: false,
            },
            String::new(),
        );
        builder.add_child(partition);
        let (merged_tx, merged_rx) = builder.bounded(1);
        builder.add_child(StaticReassemble::<SimpleEvent, _>::new(
            lanes,
            merged_tx,
            2,
            FlatPartitionConfig {
                switch_cycles: vec![1; 4],
                write_back_mu: false,
            },
            0,
        ));
        let (drop_tx, drop_rx) = builder.bounded(1);
        let (scan_tx, scan_rx) = builder.bounded(1);
        let mut broadcast = BroadcastContext::new(merged_rx);
        broadcast.add_target(drop_tx);
        broadcast.add_target(scan_tx);
        builder.add_child(broadcast);
        let (id_tx, id_rx) = builder.bounded(1);
        builder.add_child(EnumerateScan::new(scan_rx, id_tx, 2, false));
        let (last_tx, last_rx) = builder.bounded(1);
        builder.add_child(Accum::<SimpleEvent, _, _>::new(
            id_rx,
            last_tx,
            Arc::new(|x, state, bw, wb| accum_fn::last(x, state, bw, wb, 0)),
            Arc::new(|| Tile::new_zero([1, 1], 8, false)),
            2,
            AccumConfig {
                compute_bw: 1,
                write_back_mu: false,
            },
            0,
        ));
        let (control_tx, control_rx) = builder.bounded(1);
        builder.add_child(UnaryMapToMultiHot::<SimpleEvent, _>::new(
            last_rx,
            control_tx,
            Arc::new(|x, bw, wb| map_fn::index_to_multihot(x, 4, bw, wb)),
            UnaryMapConfig {
                compute_bw: 1,
                write_back_mu: false,
            },
            0,
        ));
        let (out_tx, out_rx) = builder.bounded(1);
        let (out_sel_tx, out_sel_rx) = builder.bounded(1);
        builder.add_child(CrdDropper::new(
            drop_rx, control_rx, out_tx, out_sel_tx, 2, 1, false,
        ));
        let output = collect(&mut builder, out_rx);
        let selections = collect(&mut builder, out_sel_rx);
        let run = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        run.dump_failures();
        assert!(run.passed());
        let empty = Tile::<u64>::new_empty_fiber(vec![1, 1], 8);
        assert_eq!(
            *output.lock().unwrap(),
            vec![
                data(1, 0),
                data(2, 1),
                data(3, 0),
                data(4, 1),
                data(5, 0),
                data(6, 2),
                data(1, 0),
                data(2, 3),
                data(0, 3),
                Elem::ValStop(empty.clone(), 3),
                Elem::ValStop(empty, 4)
            ]
        );
        assert_eq!(
            *selections.lock().unwrap(),
            vec![
                sel(vec![2], 0, 4),
                sel(vec![3], 1, 4),
                sel(vec![1], 1, 4),
                Elem::ValStop(MultiHotN::new_empty_fiber(4), 1),
                Elem::ValStop(MultiHotN::new_empty_fiber(4), 2)
            ]
        );
    }
}
