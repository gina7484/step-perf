use crate::primitives::elem::{Elem, StopType};
use crate::primitives::tile::Tile;
use dam::context_tools::*;

pub(crate) fn count_range_elems<T>(
    count: usize,
    bytes_per_elem: usize,
    read_from_mu: bool,
    input_stop: Option<StopType>,
    id: u32,
) -> Vec<Elem<Tile<T>>>
where
    T: Clone + TryFrom<usize>,
{
    assert!(
        count > 0,
        "[Counter {id}] count must be positive; empty segments are not supported in this milestone"
    );
    let output_stop = input_stop.map_or(1, |level| level + 1);
    (0..count)
        .map(|index| {
            let value = T::try_from(index)
                .unwrap_or_else(|_| panic!("[Counter {id}] cannot represent index {index}"));
            let tile = Tile::new(
                ndarray::arr2(&[[value]]).into_shared(),
                bytes_per_elem,
                read_from_mu,
            );
            if index + 1 == count {
                Elem::ValStop(tile, output_stop)
            } else {
                Elem::Val(tile)
            }
        })
        .collect()
}

#[context_macro]
pub struct Counter {
    count: u64,
    out_stream: Sender<Elem<Tile<u64>>>,
    id: u32,
}

impl Counter {
    pub fn new(count: u64, out_stream: Sender<Elem<Tile<u64>>>, id: u32) -> Self {
        let ctx = Self {
            count,
            out_stream,
            id,
            context_info: Default::default(),
        };
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
}

impl Context for Counter {
    fn run(&mut self) {
        let count = usize::try_from(self.count)
            .unwrap_or_else(|_| panic!("[Counter {}] count does not fit usize", self.id));
        for elem in count_range_elems(count, 8, false, None, self.id) {
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem,
                    },
                )
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::ApproxCheckerContext;

    use super::{count_range_elems, Counter};
    use crate::primitives::elem::Elem;

    #[test]
    fn counter_emits_rank_one_range() {
        let mut builder = ProgramBuilder::default();
        let (sender, receiver) = builder.unbounded();
        builder.add_child(Counter::new(4, sender, 17));
        builder.add_child(ApproxCheckerContext::new(
            || count_range_elems::<u64>(4, 8, false, None, 17).into_iter(),
            receiver,
            |actual, expected| actual == expected,
        ));
        builder
            .initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn range_helper_raises_the_terminal_stop() {
        let values = count_range_elems::<u64>(2, 8, false, Some(3), 1);
        assert!(matches!(values[0], Elem::Val(_)));
        assert!(matches!(values[1], Elem::ValStop(_, 4)));
    }

    #[test]
    #[should_panic(expected = "count must be positive")]
    fn range_helper_rejects_zero() {
        let _ = count_range_elems::<u64>(0, 8, false, None, 2);
    }
}
