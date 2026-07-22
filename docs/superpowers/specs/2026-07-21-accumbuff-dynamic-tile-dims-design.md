# AccumBuff — dynamic `tile_row` / `tile_col` — Design

**Date:** 2026-07-21

## Problem

The `AccumBuffer` proto (`step_perf_ir/proto/ops.proto`) carries `tile_row`
and `tile_col`, the shape of each accumulator-buffer slot tile. The Python
emitter (`src/sim/__init__.py`) sets either field to `0` exactly when that
dimension of the accumulator tile's dtype is **dynamic** — its concrete value
is a runtime symbol, not a compile-time constant:

```python
accum_pb.tile_row = (
    accum_tp.buff_dtype.shape[0].as_const
    if not accum_tp.buff_dtype.shape[0].is_dynamic
    else 0
)
# ... same for tile_col / shape[1]
```

The current `AccumBuff` operator (`src/operator/accum_buff.rs`) never sees
`tile_row` / `tile_col`. The proto driver bakes them straight into the
`init_accum` closure as `Tile::new_zero([tile_row, tile_col], ...)`. When a
field is `0`, that produces a zero-sized accumulator dimension (e.g.
`[0, C]`). The buffer accumulates with `add`, and the first real input tile
`[R, C]` then fails `add`'s shape assertion
(`in1_shape_0 == in2_shape_0 || in1_shape_0 == 1 || in2_shape_0 == 1`), so the
simulation panics.

We want the operator to treat `0` as "dynamic": resolve that dimension from
the input tiles at runtime, exactly as the non-buffer `Accum` path already
handles its `DynEmpty` init (`src/proto_driver/mod.rs`, "use the size of the
first tile").

## Solution

Resolve each dynamic (`0`) tile dimension **from the first input tile of each
reduction group**, size all accumulator slots with the resolved dimensions,
and rebuild them at every group boundary. Non-dynamic dimensions keep their
proto value unchanged.

Per-group resolution (rather than resolve-once-per-run) is chosen because an
`AccumBuff` run with passthrough dimensions (stop levels `> rank`) emits one
buffer per outer iteration, and a dynamic dimension may legitimately differ
between those groups (e.g. varying per-group token counts in the MoE graph).
The overhead is negligible: the accumulator buffer is already rebuilt once per
group, so per-group resolution adds only a couple of integer reads and one
sentinel check per input, and no modeled cycles.

### Why the operator (not `add`, not the closure alone)

- `add` combines `data` with the accumulator elementwise; it cannot absorb a
  zero-sized accumulator, and changing its semantics would affect every other
  `add` call site. So the fix lives in the operator.
- The operator cannot build the zero tile itself: `Tile::new_zero` requires
  `T: num::Zero`, which the operator's `OT: DAMType` bound does not provide.
  The zero tile must therefore be produced by the `init_accum` closure (which
  is monomorphized over the concrete element type in the proto driver), while
  the *dimensions* are resolved by the operator. Hence the closure is changed
  to accept the resolved `(rows, cols)`.

### Change 1 — `src/operator/accum_buff.rs`

**Struct / `new`:**

- Change the closure type
  `init_accum: Arc<dyn Fn() -> Tile<OT> + Sync + Send>`
  → `Arc<dyn Fn(usize, usize) -> Tile<OT> + Sync + Send>`
  (arguments are the resolved `rows`, `cols`).
- Add two fields `tile_row: usize`, `tile_col: usize` (the raw proto values;
  `0` = dynamic).
- Add matching `new` parameters, ordered `..., rank, buffer_shape, tile_row,
  tile_col, config, id`.

**`run()` — lazy per-group resolution.** An empty `accumulators` vec is the
"unresolved" sentinel (`n > 0` is asserted, so a resolved buffer is never
empty). The dynamic dimensions are resolved inside the loop's existing
`peek_next`, so no separate pre-loop peek is needed:

```rust
fn run(&mut self) {
    let n: usize = self.buffer_shape.iter().product();
    assert!(n > 0, "AccumBuff: buffer_shape must describe at least one accumulator slot");

    // Slots are (re)built lazily at the start of each reduction group. An empty
    // vec means "unresolved": the next input's shape fills in any dynamic (0)
    // tile dimension. process_accum_flush empties the vec, so the next group
    // re-resolves.
    let mut accumulators: Vec<Tile<OT>> = Vec::new();
    let mut index: usize = 0;
    loop {
        match self.in_stream.peek_next(&self.time) {
            Ok(ChannelElement { time: _, data }) => {
                if accumulators.is_empty() {
                    let (in_rows, in_cols) = match &data {
                        Elem::Val(t) | Elem::ValStop(t, _) => (t.shape[0], t.shape[1]),
                    };
                    let rows = if self.tile_row == 0 { in_rows } else { self.tile_row };
                    let cols = if self.tile_col == 0 { in_cols } else { self.tile_col };
                    accumulators = (0..n).map(|_| (self.init_accum)(rows, cols)).collect();
                }
                match data {
                    Elem::Val(x) => {
                        self.process_accum(x, &mut accumulators, index);
                        index = (index + 1) % n;
                    }
                    Elem::ValStop(x, level) => {
                        if level < self.rank {
                            self.process_accum(x, &mut accumulators, index);
                            index = (index + 1) % n;
                        } else if level == self.rank {
                            let out_buffer = self.process_accum_flush(x, &mut accumulators, index);
                            index = 0;
                            self.out_stream.enqueue(&self.time, ChannelElement {
                                time: self.time.tick(), data: Elem::Val(out_buffer),
                            }).unwrap();
                        } else {
                            let out_buffer = self.process_accum_flush(x, &mut accumulators, index);
                            index = 0;
                            self.out_stream.enqueue(&self.time, ChannelElement {
                                time: self.time.tick(), data: Elem::ValStop(out_buffer, level - self.rank),
                            }).unwrap();
                        }
                    }
                }
            }
            Err(_) => return,
        }
    }
}
```

**`process_accum_flush`:** drop the in-place reinit. After writing the
reduction-completing tile into `accumulators[index]`, snapshot the slots with
`std::mem::take(accumulators)` (which leaves the vec empty → the next group
re-resolves) and build the returned `Buffer`:

```rust
let slots: Vec<Tile<OT>> = std::mem::take(accumulators);
let arr = ArcArray::from_shape_vec(self.buffer_shape.clone(), slots)
    .expect("AccumBuff: buffer_shape does not match the number of accumulator slots");
Buffer::new(arr, self.time.tick().time())
```

`process_accum` is unchanged.

### Change 2 — `src/proto_driver/mod.rs` (`OpType::AccumBuffer`, ~3749–3778)

Make the `Zero` init arm dimension-parameterized and thread the two raw fields
into `new`. `tile_row` / `tile_col` locals already exist in this block.

```rust
let init_accum: Arc<dyn Fn(usize, usize) -> Tile<f32> + Send + Sync> = match accum
    .init_func.unwrap().init_fn.unwrap()
{
    init_func::InitFn::Zero(_zero) => Arc::new(move |rows, cols| {
        Tile::new_zero([rows, cols], dtype_bytes, accum.write_back_mu)
    }),
    _ => todo!(),
};

add_child!(builder, AccumBuff::<SimpleEvent, _, _>::new(
    rcv, snd, func, init_accum, accum.rank,
    to_usize_vec(accum.buffer_shape),
    tile_row, tile_col,                 // NEW
    AccumConfig { compute_bw: accum.compute_bw as u64, write_back_mu: accum.write_back_mu },
    operation.id,
));
```

Other `init_func` variants remain `todo!()` (unchanged scope).

### Change 3 — tests (`src/operator/accum_buff.rs`)

- `run_reduction` gains `tile_row: usize`, `tile_col: usize` parameters and
  passes them to `AccumBuff::new`; its init closure becomes
  `move |_rows, _cols| zero_init(read_from_mu)`. The three existing tests call
  it with the static `(1, 2)` (matching `scalar_tile`'s `1x2` shape), so no
  dynamic resolution occurs and their behavior is identical.
- **New test — dynamic dimension sized from first input (single group):**
  `tile_row = 0`, `tile_col` static, with an init closure that honors the
  arguments (`move |rows, cols| Tile::new_zero([rows, cols], 4, read_from_mu)`)
  and non-scalar input tiles, asserting the accumulator is sized from the first
  input and the elementwise sums are correct.
- **New test — per-group varying dimension (the definitive per-group test):**
  a stream with a passthrough dimension (`rank = 2`, `buffer_shape = [1]`,
  reduced `K = 2`, passthrough `I = 2`) whose two groups carry tiles of
  **different** row counts (e.g. `[2, 2]` then `[3, 2]`) with `tile_row = 0`.
  Each emitted buffer must be sized to its own group's first input. This case
  passes under per-group resolution and would panic under resolve-once (adding
  a `[3, 2]` tile into a `[2, 2]` accumulator violates `add`'s shape
  assertion), so it pins the chosen semantics.

## Key Decisions

- **`0` means dynamic**, per dimension, independently — matching the Python
  emitter. A non-zero field is used verbatim.
- **Per-group resolution**, re-derived from the first input of each reduction
  group; negligible overhead (no extra modeled cycles; one fewer buffer
  rebuild than resolve-once).
- **Empty-vec sentinel** for "unresolved", set by `std::mem::take` in
  `process_accum_flush`; unambiguous because `n > 0`.
- **Closure takes `(rows, cols)`** so the operator, which lacks `num::Zero`,
  can defer zero-tile construction to the concrete-typed closure.
- Input tiles are always rank-2, so `shape[0]` / `shape[1]` are valid.

## Testing

`source setup.sh` (with the pyo3 env exports — `PYO3_PYTHON` and
`LD_LIBRARY_PATH` pointing at the `pytorch-step` conda env — so linking finds
`libpython`), then run the `accum_buff` tests. Confirm the three existing
tests still pass and both new tests pass.

## Out of Scope (YAGNI)

- Resolution scope other than per-group (resolve-once was considered and
  rejected).
- `init_func` variants other than `Zero` for `AccumBuffer` (still `todo!()`).
- Applying dynamic resolution to the non-buffer `Accum` path (it already has
  `DynEmpty`).
- Charging modeled cycles for rebuilding / re-zeroing the accumulator buffer.
