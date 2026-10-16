use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, EmptyFiber, StopType};
use crate::primitives::select::MultiHotN;
use crate::utils::calculation::div_ceil;
use dam::context_tools::*;

/// Remove empty rank-r fibers and their selections, retaining outer boundaries.
/// Data is consumed before its selection, which may itself be reduced from data.
#[context_macro]
pub struct CrdDropper<A: DAMType> {
    input: Receiver<Elem<A>>,
    control: Receiver<Elem<MultiHotN>>,
    output: Sender<Elem<A>>,
    selections: Sender<Elem<MultiHotN>>,
    rank: StopType,
    switch_cycles: u64,
    write_back_mu: bool,
}

impl<A: DAMType + Bufferizable + EmptyFiber> CrdDropper<A> {
    pub fn new(
        input: Receiver<Elem<A>>,
        control: Receiver<Elem<MultiHotN>>,
        output: Sender<Elem<A>>,
        selections: Sender<Elem<MultiHotN>>,
        rank: StopType,
        switch_cycles: u64,
        write_back_mu: bool,
    ) -> Self {
        assert!(rank > 0, "CrdDropper needs a positive fiber rank");
        let ctx = Self {
            input,
            control,
            output,
            selections,
            rank,
            switch_cycles,
            write_back_mu,
            context_info: Default::default(),
        };
        ctx.input.attach_receiver(&ctx);
        ctx.control.attach_receiver(&ctx);
        ctx.output.attach_sender(&ctx);
        ctx.selections.attach_sender(&ctx);
        ctx
    }

    fn emit(&mut self, mut fiber: Vec<Elem<A>>, selection: MultiHotN, stop: StopType) {
        let last = fiber
            .pop()
            .expect("a retained fiber needs a terminal token");
        fiber.push(Elem::ValStop(last.into_parts().0, stop));
        self.time.incr_cycles(self.switch_cycles);
        if self.write_back_mu {
            self.time
                .incr_cycles(div_ceil(selection.size_in_bytes() as u64, PMU_BW));
        }
        self.selections
            .enqueue(
                &self.time,
                ChannelElement::new(
                    self.time.tick(),
                    Elem::with_stop(
                        selection.clone_with_updated_read_from_mu(self.write_back_mu),
                        stop - self.rank,
                    ),
                ),
            )
            .unwrap();
        for elem in fiber {
            let (value, stop) = elem.into_parts();
            self.time.incr_cycles(self.switch_cycles);
            if self.write_back_mu {
                self.time
                    .incr_cycles(div_ceil(value.size_in_bytes() as u64, PMU_BW));
            }
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
    }
}

impl<A: DAMType + Bufferizable + EmptyFiber> Context for CrdDropper<A> {
    fn run(&mut self) {
        let mut pending: Option<(Vec<Elem<A>>, MultiHotN)> = None;
        let mut parent_open = false;
        let mut selection_width = None;
        while let Ok(first) = self.input.dequeue(&self.time) {
            parent_open = true;
            let mut fiber = Vec::new();
            let mut elem = first.data;
            let (empty, stop) = loop {
                let (value, stop) = elem.into_parts();
                let empty = value.is_empty_fiber();
                assert!(
                    !empty || (fiber.is_empty() && stop >= self.rank),
                    "an empty marker must occupy a complete rank-r fiber"
                );
                if value.read_from_mu() {
                    self.time
                        .incr_cycles(div_ceil(value.size_in_bytes() as u64, PMU_BW));
                }
                fiber.push(Elem::with_stop(value, stop));
                if stop >= self.rank {
                    break (empty, stop);
                }
                elem = self
                    .input
                    .dequeue(&self.time)
                    .expect("input ended inside a fiber")
                    .data;
            };
            let selection = self
                .control
                .dequeue(&self.time)
                .expect("missing selection for data fiber");
            let (selection, select_stop) = selection.data.into_parts();
            assert!(
                !selection.is_empty_fiber(),
                "expected a selection for each input fiber"
            );
            assert_eq!(
                select_stop,
                stop - self.rank,
                "data and selection stops disagree"
            );
            if let Some(width) = selection_width {
                assert_eq!(width, selection.len(), "selection width changed");
            }
            selection_width = Some(selection.len());
            if selection.read_from_mu() {
                self.time
                    .incr_cycles(div_ceil(selection.size_in_bytes() as u64, PMU_BW));
            }
            let empty_template = fiber[0].clone().into_parts().0.empty_fiber_like();
            if !empty {
                if let Some((previous, previous_selection)) = pending.take() {
                    self.emit(previous, previous_selection, self.rank);
                }
                pending = Some((fiber, selection.clone()));
            }
            if stop > self.rank {
                if let Some((last, last_selection)) = pending.take() {
                    self.emit(last, last_selection, stop);
                } else {
                    self.emit(
                        vec![Elem::ValStop(empty_template, stop)],
                        selection.empty_fiber_like(),
                        stop,
                    );
                }
                parent_open = false;
            }
        }
        assert!(!parent_open, "input ended before a parent stop");
        assert!(
            self.control.dequeue(&self.time).is_err(),
            "extra selections after data ended"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        operator::test_support::collect,
        primitives::{select::SelectAdapter, tile::Tile},
    };
    use dam::{simulation::ProgramBuilder, utility_contexts::GeneratorContext};

    fn data(value: Option<i64>, stop: u32) -> Elem<Tile<i64>> {
        let tile = match value {
            Some(value) => Tile::new(ndarray::ArcArray2::from_elem((1, 1), value), 8, false),
            None => Tile::new_empty_fiber(vec![1, 1], 8),
        };
        Elem::with_stop(tile, stop)
    }
    fn selection(index: usize, stop: u32) -> Elem<MultiHotN> {
        Elem::with_stop(MultiHotN::from_sel_vec(vec![index], 4, false), stop)
    }
    fn run(
        input: Vec<Elem<Tile<i64>>>,
        control: Vec<Elem<MultiHotN>>,
    ) -> (bool, Vec<Elem<Tile<i64>>>, Vec<Elem<MultiHotN>>) {
        run_with_write_back(input, control, false)
    }

    fn run_with_write_back(
        input: Vec<Elem<Tile<i64>>>,
        control: Vec<Elem<MultiHotN>>,
        write_back_mu: bool,
    ) -> (bool, Vec<Elem<Tile<i64>>>, Vec<Elem<MultiHotN>>) {
        let mut builder = ProgramBuilder::default();
        let (input_tx, input_rx) = builder.unbounded();
        let (control_tx, control_rx) = builder.unbounded();
        let (output_tx, output_rx) = builder.bounded(1);
        let (select_tx, select_rx) = builder.bounded(1);
        builder.add_child(GeneratorContext::new(move || input.into_iter(), input_tx));
        builder.add_child(GeneratorContext::new(
            move || control.into_iter(),
            control_tx,
        ));
        builder.add_child(CrdDropper::new(
            input_rx,
            control_rx,
            output_tx,
            select_tx,
            2,
            1,
            write_back_mu,
        ));
        let output = collect(&mut builder, output_rx);
        let selection = collect(&mut builder, select_rx);
        let result = builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
        let data = output.lock().unwrap().clone();
        let selections = selection.lock().unwrap().clone();
        (result.passed(), data, selections)
    }

    #[test]
    fn write_back_applies_to_values_but_not_empty_parent_markers() {
        let (passed, output, selections) = run_with_write_back(
            vec![data(Some(0), 3), data(None, 4)],
            vec![selection(1, 1), selection(2, 2)],
            true,
        );
        assert!(passed);
        for (index, (value, selection)) in output.into_iter().zip(selections).enumerate() {
            assert_eq!(value.into_parts().0.read_from_mu(), index == 0);
            assert_eq!(selection.into_parts().0.read_from_mu(), index == 0);
        }
    }

    #[test]
    fn repairs_trailing_stops_and_preserves_original_branch_ids_and_zero_tiles() {
        let (passed, output, selections) = run(
            vec![
                data(None, 2),
                data(Some(0), 1),
                data(Some(1), 2),
                data(None, 2),
                data(None, 5),
            ],
            vec![
                selection(0, 0),
                selection(1, 0),
                selection(2, 0),
                selection(3, 3),
            ],
        );
        assert!(passed);
        assert_eq!(output, vec![data(Some(0), 1), data(Some(1), 5)]);
        assert_eq!(selections, vec![selection(1, 3)]);
    }

    #[test]
    fn consecutive_empty_parents_emit_empty_on_both_outputs() {
        let (passed, output, selections) = run(
            vec![data(None, 2), data(None, 3), data(None, 4)],
            vec![selection(0, 0), selection(1, 1), selection(0, 2)],
        );
        assert!(passed);
        assert_eq!(output, vec![data(None, 3), data(None, 4)]);
        assert_eq!(
            selections,
            vec![
                Elem::ValStop(MultiHotN::new_empty_fiber(4), 1),
                Elem::ValStop(MultiHotN::new_empty_fiber(4), 2)
            ]
        );
    }

    #[test]
    fn blank_and_zero_sized_tiles_are_retained() {
        for tile in [
            Tile::<i64>::new_blank(vec![1, 1], 8, false),
            Tile::new_empty([0, 1], 8, false),
        ] {
            let (passed, output, selections) =
                run(vec![Elem::ValStop(tile.clone(), 3)], vec![selection(2, 1)]);
            assert!(passed);
            assert_eq!(output, vec![Elem::ValStop(tile, 3)]);
            assert_eq!(selections, vec![selection(2, 1)]);
        }
        let (passed, output, selections) = run(vec![], vec![]);
        assert!(passed && output.is_empty() && selections.is_empty());
    }

    #[test]
    fn rejects_malformed_fibers_and_selection_alignment() {
        for (input, control) in [
            (vec![data(None, 0)], vec![selection(0, 0)]),
            (vec![data(None, 1)], vec![selection(0, 0)]),
            (vec![data(Some(1), 0), data(None, 3)], vec![selection(0, 1)]),
            (vec![data(Some(1), 3)], vec![selection(0, 0)]),
            (vec![data(Some(1), 3)], vec![]),
            (vec![], vec![selection(0, 1)]),
            (vec![data(Some(1), 2)], vec![selection(0, 0)]),
        ] {
            assert!(!run(input, control).0);
        }
    }
}
