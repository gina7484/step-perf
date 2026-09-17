use crate::memory::PMU_BW;
use crate::trace::TracingSender as Sender;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::select::{MultiHotN, SelectAdapter};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::{collections::BTreeMap, marker::PhantomData};

pub struct ShuffleConfig {
    pub rank: StopType,
    pub num_buckets: usize,
    pub write_back_mu: bool,
}

#[context_macro]
pub struct Shuffle<E, A: DAMType> {
    in_stream: Receiver<Elem<A>>,
    index_stream: Receiver<Elem<MultiHotN>>,
    out_stream: Sender<Elem<A>>,
    out_index_stream: Sender<Elem<MultiHotN>>,
    config: ShuffleConfig,
    id: u32,
    _phantom: PhantomData<E>,
}

impl<E, A> Shuffle<E, A>
where
    E: LoggableEventSimple + LogEvent + Sync + Send,
    A: Bufferizable + DAMType,
{
    pub fn new(
        in_stream: Receiver<Elem<A>>,
        index_stream: Receiver<Elem<MultiHotN>>,
        out_stream: Sender<Elem<A>>,
        out_index_stream: Sender<Elem<MultiHotN>>,
        config: ShuffleConfig,
        id: u32,
    ) -> Self {
        assert!(config.rank > 0, "Shuffle rank must be positive");
        assert!(
            config.num_buckets > 0,
            "Shuffle bucket count must be positive"
        );
        let ctx = Self {
            in_stream,
            index_stream,
            out_stream,
            out_index_stream,
            config,
            id,
            _phantom: PhantomData,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.index_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx.out_index_stream.attach_sender(&ctx);
        ctx
    }

    fn flush(
        &mut self,
        storage: &[(A, StopType)],
        buckets: BTreeMap<usize, Vec<usize>>,
        group_stop: StopType,
    ) {
        assert!(
            !buckets.is_empty(),
            "Shuffle_{}: entirely empty shuffle groups are unsupported",
            self.id
        );
        let last_bucket = *buckets.keys().next_back().unwrap();
        for (bucket, references) in buckets {
            let mask = MultiHotN::from_sel_vec(vec![bucket], self.config.num_buckets, false);
            let index_elem = if bucket == last_bucket {
                Elem::ValStop(mask, group_stop - (self.config.rank - 1))
            } else {
                Elem::Val(mask)
            };
            // Send the mask before the data so consumers can select the bucket.
            self.out_index_stream
                .enqueue(
                    &self.time,
                    ChannelElement::new(self.time.tick(), index_elem),
                )
                .unwrap();
            let last_reference = references.len() - 1;
            for (position, reference) in references.into_iter().enumerate() {
                let (data, inner_stop) = &storage[reference];
                let memory_cycles = div_ceil(data.size_in_bytes() as u64, PMU_BW);
                // Every bucket reference reads the shared data, including duplicates.
                self.time.incr_cycles(memory_cycles);
                if self.config.write_back_mu {
                    self.time.incr_cycles(memory_cycles);
                }
                let stop = if position == last_reference {
                    if bucket == last_bucket {
                        group_stop
                            .checked_add(1)
                            .expect("Shuffle stop level overflow")
                    } else {
                        self.config.rank
                    }
                } else {
                    *inner_stop
                };
                let value = data.clone_with_updated_read_from_mu(self.config.write_back_mu);
                let elem = if stop == 0 {
                    Elem::Val(value)
                } else {
                    Elem::ValStop(value, stop)
                };
                self.out_stream
                    .enqueue(&self.time, ChannelElement::new(self.time.tick(), elem))
                    .unwrap();
            }
        }
    }
}

impl<E, A> Context for Shuffle<E, A>
where
    E: LoggableEventSimple + LogEvent + Sync + Send,
    A: Bufferizable + DAMType,
{
    fn run(&mut self) {
        // Tiles are stored once. Each bucket contains indices into this storage.
        let mut storage = Vec::new();
        let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut group_start = None;
        while let Ok(index_elem) = self.index_stream.dequeue(&self.time) {
            let start = *group_start.get_or_insert(self.time.tick().time());
            let (mask, index_stop) = match index_elem.data {
                Elem::Val(mask) => (mask, 0),
                Elem::ValStop(mask, stop) => (mask, stop),
            };
            assert_eq!(
                mask.len(),
                self.config.num_buckets,
                "Shuffle mask width mismatch"
            );
            if mask.read_from_mu() {
                self.time
                    .incr_cycles(div_ceil(mask.size_in_bytes() as u64, PMU_BW));
            }
            let selected = mask.to_sel_vec();
            let item_rank = self.config.rank - 1;
            let stop = loop {
                let input = self.in_stream.dequeue(&self.time).unwrap_or_else(|_| {
                    panic!(
                        "Shuffle_{}: input ended before the selected item was complete",
                        self.id
                    )
                });
                let (data, stop) = match input.data {
                    Elem::Val(data) => (data, 0),
                    Elem::ValStop(data, stop) => (data, stop),
                };
                let memory_cycles = div_ceil(data.size_in_bytes() as u64, PMU_BW);
                if data.read_from_mu() {
                    self.time.incr_cycles(memory_cycles);
                }
                if !selected.is_empty() {
                    // A dropped item needs no storage; all other items are written once.
                    self.time.incr_cycles(memory_cycles);
                    let reference = storage.len();
                    storage.push((data, stop.min(item_rank)));
                    for &bucket in &selected {
                        buckets.entry(bucket).or_default().push(reference);
                    }
                }
                if stop >= item_rank {
                    break stop;
                }
            };
            assert_eq!(
                Some(stop),
                index_stop.checked_add(item_rank),
                "Shuffle_{}: input and index stop levels do not match",
                self.id
            );
            if stop >= self.config.rank {
                self.flush(&storage, std::mem::take(&mut buckets), stop);
                storage.clear();
                group_start = None;
                dam::logging::log_event(&E::new(
                    "Shuffle".to_string(),
                    self.id,
                    start,
                    self.time.tick().time(),
                    true,
                ))
                .unwrap();
            }
        }
        assert!(
            group_start.is_none(),
            "Shuffle_{}: index ended before a group boundary",
            self.id
        );
        assert!(
            self.in_stream.dequeue(&self.time).is_err(),
            "Shuffle_{}: input outlived the index stream",
            self.id
        );
    }
}
