use std::marker::PhantomData;

use dam::context_tools::*;
use dam::logging::LogEvent;

use crate::primitives::buffer::Buffer;
use crate::primitives::elem::Bufferizable;
use crate::primitives::elem::{Elem, StopType};

use crate::utils::events::LoggableEventSimple;

/// Streamify expands each input `Buffer<T>` into a stream of tiles indexed by
/// `out_shape_tiled` with linear strides into the buffer's tile-grid.
///
/// For each lex-order position `(i_0, ..., i_{K-1})` over `out_shape_tiled`,
/// the emitted tile is the buffer entry at flat index `Σ i_j · stride[j]`.
/// Stop tokens follow the same outermost-differing-dim convention used by
/// `Buffer::to_elem_iter`: the previous position carries stop level
/// `K - changed_index - 1`, and the final position of each input buffer
/// carries stop level `K + outer_stop_lev`, where `outer_stop_lev` is any
/// stop level on the input `Elem::ValStop(buff, ...)` (0 for `Elem::Val`).
#[context_macro]
pub struct Streamify<E: LoggableEventSimple, T: Bufferizable + Clone> {
    pub stride: Vec<usize>,
    pub out_shape_tiled: Vec<usize>,
    pub in_stream: Receiver<Elem<Buffer<T>>>,
    pub out_stream: Sender<Elem<T>>,
    pub id: u32,
    _phantom: PhantomData<E>,
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
    > Streamify<E, T>
where
    Buffer<T>: DAMType,
{
    pub fn new(
        stride: Vec<usize>,
        out_shape_tiled: Vec<usize>,
        in_stream: Receiver<Elem<Buffer<T>>>,
        out_stream: Sender<Elem<T>>,
        id: u32,
    ) -> Self {
        assert_eq!(
            stride.len(),
            out_shape_tiled.len(),
            "Streamify: stride and out_shape_tiled must have same length"
        );
        assert!(
            !out_shape_tiled.is_empty(),
            "Streamify: out_shape_tiled must be non-empty"
        );
        assert!(
            out_shape_tiled.iter().all(|&d| d > 0),
            "Streamify: out_shape_tiled entries must be positive"
        );
        let ctx = Self {
            stride,
            out_shape_tiled,
            in_stream,
            out_stream,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }

    fn emit_buffer(&mut self, buff: &Buffer<T>, outer_stop_lev: StopType) {
        let k = self.out_shape_tiled.len();
        let total: usize = self.out_shape_tiled.iter().product();
        // Buffers are constructed via `ArcArray::from_shape_vec`, so their
        // backing storage is C-contiguous and `as_slice` always succeeds.
        let flat: &[T] = buff
            .as_slice()
            .expect("Streamify: buffer is not C-contiguous");

        let mut idx = vec![0usize; k];
        for pos in 0..total {
            let linear_idx: usize = idx
                .iter()
                .zip(self.stride.iter())
                .map(|(i, s)| i * s)
                .sum();
            assert!(
                linear_idx < flat.len(),
                "Streamify: linear index {} out of bounds for buffer of {} tiles \
                 (idx={:?}, stride={:?})",
                linear_idx,
                flat.len(),
                idx,
                self.stride,
            );
            let tile = flat[linear_idx].clone();

            let is_last = pos + 1 == total;
            let stop_level = if is_last {
                k as StopType + outer_stop_lev
            } else {
                let changed_index = next_changed_dim(&idx, &self.out_shape_tiled);
                (k - changed_index - 1) as StopType
            };

            let elem = if stop_level == 0 {
                Elem::Val(tile)
            } else {
                Elem::ValStop(tile, stop_level)
            };
            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick(),
                        data: elem,
                    },
                )
                .unwrap();
            self.time.incr_cycles(1);

            if !is_last {
                advance_lex(&mut idx, &self.out_shape_tiled);
            }
        }
    }
}

impl<
        E: LoggableEventSimple + LogEvent + std::marker::Sync + std::marker::Send,
        T: Bufferizable + DAMType,
    > Context for Streamify<E, T>
where
    Buffer<T>: DAMType,
{
    fn run(&mut self) {
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement {
                    time: _,
                    data: buff_elem,
                }) => {
                    let start_time = self.time.tick().time();
                    let (buff, outer_stop_lev) = match &buff_elem {
                        Elem::Val(b) => (b.clone(), 0 as StopType),
                        Elem::ValStop(b, lev) => (b.clone(), *lev),
                    };
                    self.emit_buffer(&buff, outer_stop_lev);
                    self.in_stream.dequeue(&self.time).unwrap();

                    dam::logging::log_event(&E::new(
                        "Streamify".to_string(),
                        self.id,
                        start_time,
                        self.time.tick().time(),
                        false,
                    ))
                    .unwrap();
                }
                Err(_) => return,
            }
        }
    }
}

/// Advance `idx` to the next lex-order position over `shape`.
/// Caller guarantees there is a next position.
fn advance_lex(idx: &mut [usize], shape: &[usize]) {
    let mut k = idx.len();
    while k > 0 {
        k -= 1;
        idx[k] += 1;
        if idx[k] < shape[k] {
            return;
        }
        idx[k] = 0;
    }
    panic!("advance_lex: idx already at maximum");
}

/// Find the outermost dim that will flip when advancing `idx` by one in
/// lex order over `shape`.  Smaller index = more outer.
fn next_changed_dim(idx: &[usize], shape: &[usize]) -> usize {
    let mut k = idx.len();
    while k > 0 {
        k -= 1;
        if idx[k] + 1 < shape[k] {
            return k;
        }
    }
    panic!("next_changed_dim: idx already at maximum");
}

#[cfg(test)]
mod tests {
    use crate::utils::events::{LoggableEventSimple, DUMMY_ID};
    use crate::{
        define_simple_event,
        operator::bufferize::Bufferize,
        primitives::{elem::Elem, tile::Tile},
        utils::events::SimpleEvent,
    };
    use dam::dam_macros::event_type;
    use dam::{
        simulation::{
            LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder, RunOptionsBuilder,
        },
        utility_contexts::{
            ApproxCheckerContext, CheckerContext, FunctionContext, GeneratorContext,
        },
    };
    use serde::{Deserialize, Serialize};

    define_simple_event!(BufferizeEvent);
    define_simple_event!(StreamifyEvent);
    #[test]
    fn round_trip_test_3d() {
        // Tiled stream shape: [1,3,2,2] => bufferize_rank=2 yields buffers of (2,2)
        // with outer stream (1,3).  Streamify with out_shape_tiled=(2,2,2) and
        // stride=(0,2,1) inserts a leading repeat-of-2 dim, giving output stream
        // shape (1,3,2,2,2) — equivalent to the old (repeat_factor=[2], rank=2).
        type VT = u32;

        let mut ctx = ProgramBuilder::default();
        let bufferize_rank = 2;

        let input_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
        ];

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || input_tiled_stream.into_iter(),
            snd,
        ));

        let (buff_snd, buff_rcv) = ctx.bounded(1);
        ctx.add_child(Bufferize::<BufferizeEvent, _>::new(
            rcv,
            buff_snd,
            bufferize_rank,
            DUMMY_ID,
        ));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(super::Streamify::<StreamifyEvent, _>::new(
            vec![0, 2, 1],
            vec![2, 2, 2],
            buff_rcv,
            out_snd,
            DUMMY_ID,
        ));

        let output_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 3),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 4),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || output_tiled_stream.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        let logging = true;
        if logging {
            let initialized = ctx.initialize(Default::default()).unwrap();
            let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
                dam::logging::LogFilter::AllowAll,
            ));
            let run_options = run_options.logging(LoggingOptions::Mongo(
                MongoOptionsBuilder::default()
                    .db("test_streamify".to_string())
                    .uri("mongodb://127.0.0.1:27017".to_string())
                    .build()
                    .unwrap(),
            ));
            let summary = initialized.run(run_options.build().unwrap());
            println!("{}, {:?}", summary.passed(), summary.elapsed_cycles());
        } else {
            ctx.initialize(Default::default())
                .unwrap()
                .run(Default::default());
        }
    }

    #[test]
    fn round_trip_test_0d() {
        // Tiled stream shape: [2,2,2] => bufferize_rank=2 yields buffers of (2,2)
        // with outer stream (2).  Streamify with out_shape_tiled=(2,2) and
        // stride=(2,1) walks the buffer once — equivalent to the old empty
        // repeat_factor with rank=2.
        type VT = u32;

        let mut ctx = ProgramBuilder::default();
        let bufferize_rank = 2;

        let input_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
        ];

        let (snd, rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || input_tiled_stream.into_iter(),
            snd,
        ));

        let (buff_snd, buff_rcv) = ctx.bounded(1);
        ctx.add_child(Bufferize::<SimpleEvent, _>::new(
            rcv,
            buff_snd,
            bufferize_rank,
            DUMMY_ID,
        ));

        let (out_snd, out_rcv) = ctx.unbounded();
        ctx.add_child(super::Streamify::<SimpleEvent, _>::new(
            vec![2, 1],
            vec![2, 2],
            buff_rcv,
            out_snd,
            DUMMY_ID,
        ));

        let output_tiled_stream = vec![
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 1),
            Elem::Val(Tile::<VT>::new_blank(vec![2, 2], 2, false)),
            Elem::ValStop(Tile::<VT>::new_blank(vec![2, 2], 2, false), 2),
        ];

        ctx.add_child(ApproxCheckerContext::new(
            move || output_tiled_stream.into_iter(),
            out_rcv,
            |x, y| x == y,
        ));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
