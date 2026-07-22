# AccumBuff Dynamic tile_row/tile_col Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AccumBuff` treat `tile_row`/`tile_col == 0` as a dynamic accumulator dimension, resolving it from the first input tile of each reduction group.

**Architecture:** The proto driver already turns the `AccumBuffer` proto's `init_func`/`tile_row`/`tile_col` into an `init_accum` closure. We (1) change that closure to accept the resolved `(rows, cols)` — the operator can't build zero tiles itself because `Tile::new_zero` needs `T: num::Zero`, absent from the operator's `OT: DAMType` bound — and pass `tile_row`/`tile_col` into the operator; then (2) rewrite the operator's `run` loop to build accumulator slots lazily at each group boundary, resolving any `0` dimension from the first input's shape.

**Tech Stack:** Rust, the `dam` simulation framework, `ndarray`, `pyo3` (build-time link to conda Python).

## Global Constraints

- **Build/test env (pyo3 links against conda Python):** every `cargo` build/test command MUST run with the conda env active and these exports, or linking fails with `cannot find -lpython3.12`:
  ```bash
  source /home/gina/miniconda3/etc/profile.d/conda.sh && conda activate pytorch-step && \
  export PYO3_PYTHON=~/miniconda3/envs/pytorch-step/bin/python && \
  export LD_LIBRARY_PATH=~/miniconda3/envs/pytorch-step/lib:$LD_LIBRARY_PATH
  ```
  Shell state does not persist between commands, so prefix each `cargo` command with this line.
- **Commit authorship:** author as the repo user (`gina7484`). No `Co-Authored-By` trailers, no "Generated with Claude Code" footer.
- **Scope:** only the `Zero` `init_func` variant is handled for `AccumBuffer` (all others remain `todo!()`). Input tiles are always rank-2, so `shape[0]`/`shape[1]` are valid.
- **`0` means dynamic**, per dimension, independently; a non-zero field is used verbatim.

---

## File Structure

- `src/operator/accum_buff.rs` — the `AccumBuff` operator: struct, `new`, `run`, `process_accum_flush`, and the `#[cfg(test)]` module. All operator and test changes live here.
- `src/proto_driver/mod.rs` — the `OpType::AccumBuffer` construction arm (~lines 3749–3779): the `init_accum` closure and the `AccumBuff::new` call.

No new files.

---

## Task 1: Plumb `tile_row`/`tile_col` into `AccumBuff` (signature refactor, behavior preserved)

Pure plumbing: change the `init_accum` closure to take `(rows, cols)`, store `tile_row`/`tile_col` on the struct, and thread both through the proto driver and the test helper. Behavior is unchanged because the operator still passes the (static) `tile_row`/`tile_col` eagerly and the existing tests use non-zero dimensions. Guarded by the existing three tests staying green.

**Files:**
- Modify: `src/operator/accum_buff.rs` (struct ~46–64, `new` ~77–102, `run` ~204, `process_accum_flush` ~176–178, test helper ~288–325 and its 3 call sites ~378–384, ~439–445, ~507)
- Modify: `src/proto_driver/mod.rs` (~3752–3779)

**Interfaces:**
- Produces: `AccumBuff::<E, T, OT>::new(in_stream, out_stream, func, init_accum, rank, buffer_shape, tile_row, tile_col, config, id)` where `init_accum: Arc<dyn Fn(usize, usize) -> Tile<OT> + Sync + Send>` and `tile_row: usize`, `tile_col: usize`. Struct gains `tile_row: usize`, `tile_col: usize` fields.

- [ ] **Step 1: Change the `init_accum` field type and add the two fields**

In `src/operator/accum_buff.rs`, in the `struct AccumBuff` definition, change the `init_accum` field type and add `tile_row`/`tile_col` after `buffer_shape`:

```rust
    init_accum: Arc<dyn Fn(usize, usize) -> Tile<OT> + Sync + Send>,
```

and after the `buffer_shape: Vec<usize>,` field, insert:

```rust
    /// Row extent of each accumulator slot tile, or `0` if the row dimension
    /// is dynamic and resolved from the first input tile of each reduction
    /// group.
    tile_row: usize,
    /// Column extent of each accumulator slot tile, or `0` if the column
    /// dimension is dynamic and resolved from the first input tile of each
    /// reduction group.
    tile_col: usize,
```

- [ ] **Step 2: Update `new` to accept and store the new parameters**

In `new`, change the `init_accum` parameter type and add `tile_row`/`tile_col` params (after `buffer_shape`), and add them to the struct literal (after `buffer_shape`):

```rust
    pub fn new(
        in_stream: Receiver<Elem<Tile<T>>>,
        out_stream: Sender<Elem<Buffer<Tile<OT>>>>,
        func: Arc<dyn Fn(&Tile<T>, &Tile<OT>, u64, bool) -> (u64, Tile<OT>) + Send + Sync>,
        init_accum: Arc<dyn Fn(usize, usize) -> Tile<OT> + Sync + Send>,
        rank: StopType,
        buffer_shape: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
        config: AccumConfig,
        id: u32,
    ) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            func,
            init_accum,
            rank,
            buffer_shape,
            tile_row,
            tile_col,
            config,
            id,
            context_info: Default::default(),
            _phantom: PhantomData,
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);
        ctx
    }
```

- [ ] **Step 3: Pass the dimensions to the two `init_accum` call sites**

In `run`, change the initial accumulator build (currently `(self.init_accum)()`):

```rust
        let mut accumulators: Vec<Tile<OT>> =
            (0..n).map(|_| (self.init_accum)(self.tile_row, self.tile_col)).collect();
```

In `process_accum_flush`, change the reinit inside `std::mem::replace` (currently `(self.init_accum)()`):

```rust
        let n = accumulators.len();
        let slots: Vec<Tile<OT>> = std::mem::replace(
            accumulators,
            (0..n)
                .map(|_| (self.init_accum)(self.tile_row, self.tile_col))
                .collect(),
        );
```

- [ ] **Step 4: Update the proto driver's closure and `new` call**

In `src/proto_driver/mod.rs`, `OpType::AccumBuffer` arm, change the `init_accum` closure type/body and add `tile_row`/`tile_col` to the `AccumBuff::new` call:

```rust
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
```

(The `tile_row`/`tile_col` locals already exist just above this block.)

- [ ] **Step 5: Update the `run_reduction` test helper**

In `src/operator/accum_buff.rs`, in the `#[cfg(test)] mod tests`, add `tile_row`/`tile_col` and an `init_accum` closure parameter to `run_reduction` (replacing the hardcoded `zero_init` closure and the now-unused `read_from_mu` param), and pass them to `new`. Parameterizing the init closure lets both the static and dynamic tests share this one runner:

```rust
    fn run_reduction(
        in_stream_data: Vec<Elem<Tile<i32>>>,
        ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>>,
        rank: u32,
        buffer_shape: Vec<usize>,
        tile_row: usize,
        tile_col: usize,
        init_accum: Arc<dyn Fn(usize, usize) -> Tile<i32> + Send + Sync>,
    ) {
        let mut ctx = ProgramBuilder::default();
        let (in_data_snd, in_data_rcv) = ctx.unbounded();
        let (out_data_snd, out_data_rcv) = ctx.unbounded();
        ctx.add_child(GeneratorContext::new(
            || in_stream_data.into_iter(),
            in_data_snd,
        ));
        ctx.add_child(AccumBuff::<SimpleEvent, _, _>::new(
            in_data_rcv,
            out_data_snd,
            Arc::new(move |tile1, tile2, comp_bw, write_back_mu| {
                accum_fn::add(tile1, tile2, comp_bw, write_back_mu, 0)
            }),
            init_accum,
            rank,
            buffer_shape,
            tile_row,
            tile_col,
            AccumConfig {
                compute_bw: 1000,
                write_back_mu: true,
            },
            0, // id
        ));
        ctx.add_child(ApproxCheckerContext::new(
            || ground_truth_data.into_iter(),
            out_data_rcv,
            tolerance_fn,
        ));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
```

- [ ] **Step 6: Update the three existing `run_reduction` call sites**

`zero_init` produces a `1x2` tile, so pass `tile_row = 1`, `tile_col = 2` (non-zero → no dynamic resolution, behavior identical) and a closure that ignores the resolved dims.

In `test_accum_buff_rank2`:
```rust
        run_reduction(
            in_stream_data,
            ground_truth_data,
            2,
            vec![j_dim],
            1,
            2,
            Arc::new(move |_rows, _cols| zero_init(read_from_mu)),
        );
```

In `test_accum_buff_rank3`:
```rust
        run_reduction(
            in_stream_data,
            ground_truth_data,
            3,
            vec![j_dim, k_dim],
            1,
            2,
            Arc::new(move |_rows, _cols| zero_init(read_from_mu)),
        );
```

In `test_accum_buff_reduce_two_dims`:
```rust
        run_reduction(
            in_stream_data,
            ground_truth_data,
            3,
            vec![j_dim],
            1,
            2,
            Arc::new(move |_rows, _cols| zero_init(read_from_mu)),
        );
```

- [ ] **Step 7: Build and run the existing tests to verify no regression**

Run:
```bash
source /home/gina/miniconda3/etc/profile.d/conda.sh && conda activate pytorch-step && \
export PYO3_PYTHON=~/miniconda3/envs/pytorch-step/bin/python && \
export LD_LIBRARY_PATH=~/miniconda3/envs/pytorch-step/lib:$LD_LIBRARY_PATH && \
cargo test accum_buff
```
Expected: compiles; `test_accum_buff_rank2`, `test_accum_buff_rank3`, `test_accum_buff_reduce_two_dims` all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/operator/accum_buff.rs src/proto_driver/mod.rs
git commit -m "AccumBuff: pass tile_row/tile_col and dims to init_accum closure"
```

---

## Task 2: Resolve dynamic dimensions per reduction group

Rewrite `run` to build accumulator slots lazily — at the first input of each reduction group — resolving any `0` dimension from that input's shape; and change `process_accum_flush` to snapshot with `std::mem::take` so the emptied buffer signals the next group to re-resolve. Driven by two new failing tests.

**Files:**
- Modify: `src/operator/accum_buff.rs` (`run`, `process_accum_flush`, and add two tests + the `filled_tile` helper)

**Interfaces:**
- Consumes: `run_reduction(in_stream_data, ground_truth_data, rank, buffer_shape, tile_row, tile_col, init_accum)` and `AccumBuff::new(...)` from Task 1.
- Produces (test helper): `filled_tile(rows: usize, cols: usize, v: i32, read_from_mu: bool) -> Tile<i32>`.

- [ ] **Step 1: Write the two failing tests (and the `filled_tile` helper)**

In `src/operator/accum_buff.rs`, inside `mod tests`, add the `filled_tile` helper and two tests. Both tests reuse the `run_reduction` helper from Task 1, passing `tile_row = 0` (dynamic) and a dimension-honoring `Tile::new_zero` init closure:

```rust
    /// A `rows x cols` tile whose every element is `v`.
    fn filled_tile(rows: usize, cols: usize, v: i32, read_from_mu: bool) -> Tile<i32> {
        Tile::new(Array2::from_elem((rows, cols), v).into(), 4, read_from_mu)
    }

    /// Dynamic row (`tile_row = 0`) resolved from the first input, single
    /// group. Reduce K=2 onto a 2-slot buffer of `2x3` tiles.
    /// `tile(k,j)` holds `k*10 + j`. slot(j) = sum_k tile(k,j).
    #[test]
    fn test_accum_buff_dynamic_row_single_group() {
        let read_from_mu = true;
        let (rows, cols) = (2usize, 3usize);
        let (k_dim, j_dim) = (2usize, 2usize);

        // Row-major [K, J], J retained (buffer_shape=[J]), K reduced, rank=2.
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        for k in 0..k_dim {
            for j in 0..j_dim {
                let tile = filled_tile(rows, cols, (k * 10 + j) as i32, read_from_mu);
                if j == j_dim - 1 {
                    if k == k_dim - 1 {
                        in_stream_data.push(Elem::ValStop(tile, 2)); // end reduced == rank
                    } else {
                        in_stream_data.push(Elem::ValStop(tile, 1)); // end retained run
                    }
                } else {
                    in_stream_data.push(Elem::Val(tile));
                }
            }
        }

        let slots: Vec<Tile<i32>> = (0..j_dim)
            .map(|j| {
                let sum: i32 = (0..k_dim).map(|k| (k * 10 + j) as i32).sum();
                filled_tile(rows, cols, sum, read_from_mu)
            })
            .collect();
        let arr = ArcArray::from_shape_vec(vec![j_dim], slots).unwrap();
        let ground_truth_data = vec![Elem::Val(Buffer::new(arr, 0))];

        run_reduction(
            in_stream_data,
            ground_truth_data,
            2,
            vec![j_dim],
            0, // tile_row dynamic
            cols,
            Arc::new(move |r, c| Tile::new_zero([r, c], 4, read_from_mu)),
        );
    }

    /// Per-group re-resolution: two passthrough groups whose tiles have
    /// DIFFERENT row counts. buffer_shape=[1], rank=2, reduced K=2,
    /// passthrough I=2, `tile_row = 0`. Group 0 tiles are `2x2`, group 1
    /// tiles are `3x2`. Each emitted buffer must be sized to its own group's
    /// first input. This PANICS under resolve-once semantics (adding a `3x2`
    /// tile into a `2x2` accumulator), so it pins per-group resolution.
    #[test]
    fn test_accum_buff_dynamic_row_per_group() {
        let read_from_mu = true;
        let cols = 2usize;
        let group_rows = [2usize, 3usize]; // I = 2 groups
        let k_dim = 2usize;

        // Row-major [I, K, retained(=1)]. Retained level 1, reduced K level 2,
        // passthrough I level 3, rank=2.
        let mut in_stream_data: Vec<Elem<Tile<i32>>> = Vec::new();
        for i in 0..group_rows.len() {
            for k in 0..k_dim {
                let tile = filled_tile(group_rows[i], cols, (i * 10 + k) as i32, read_from_mu);
                if k == k_dim - 1 {
                    if i == group_rows.len() - 1 {
                        in_stream_data.push(Elem::ValStop(tile, 3)); // end passthrough
                    } else {
                        in_stream_data.push(Elem::ValStop(tile, 2)); // end reduced == rank
                    }
                } else {
                    in_stream_data.push(Elem::ValStop(tile, 1)); // end retained run
                }
            }
        }

        // Per group: slot0 = sum_k (i*10 + k), tile sized group_rows[i] x cols.
        let mut ground_truth_data: Vec<Elem<Buffer<Tile<i32>>>> = Vec::new();
        for i in 0..group_rows.len() {
            let sum: i32 = (0..k_dim).map(|k| (i * 10 + k) as i32).sum();
            let slot = filled_tile(group_rows[i], cols, sum, read_from_mu);
            let arr = ArcArray::from_shape_vec(vec![1usize], vec![slot]).unwrap();
            let buffer = Buffer::new(arr, 0);
            if i == group_rows.len() - 1 {
                ground_truth_data.push(Elem::ValStop(buffer, 1)); // level 3 - rank 2
            } else {
                ground_truth_data.push(Elem::Val(buffer));
            }
        }

        run_reduction(
            in_stream_data,
            ground_truth_data,
            2,
            vec![1],
            0, // tile_row dynamic
            cols,
            Arc::new(move |r, c| Tile::new_zero([r, c], 4, read_from_mu)),
        );
    }
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:
```bash
source /home/gina/miniconda3/etc/profile.d/conda.sh && conda activate pytorch-step && \
export PYO3_PYTHON=~/miniconda3/envs/pytorch-step/bin/python && \
export LD_LIBRARY_PATH=~/miniconda3/envs/pytorch-step/lib:$LD_LIBRARY_PATH && \
cargo test accum_buff::tests::test_accum_buff_dynamic_row
```
Expected: both `test_accum_buff_dynamic_row_single_group` and `test_accum_buff_dynamic_row_per_group` FAIL — a panic from `functions::accum_fn::add` (the eager `Tile::new_zero([0, cols])` accumulator fails the shape assertion `in1_shape_0 == in2_shape_0 || ... == 1`).

- [ ] **Step 3: Rewrite `run` for lazy per-group resolution**

In `src/operator/accum_buff.rs`, replace the body of `fn run` with:

```rust
    fn run(&mut self) {
        let n: usize = self.buffer_shape.iter().product();
        assert!(
            n > 0,
            "AccumBuff: buffer_shape must describe at least one accumulator slot"
        );
        // Accumulator slots, (re)built lazily at the start of each reduction
        // group. An empty vec means "unresolved": the next input's shape fills
        // in any dynamic (0) tile dimension. `process_accum_flush` empties the
        // vec via `std::mem::take`, so each new group re-resolves.
        let mut accumulators: Vec<Tile<OT>> = Vec::new();
        let mut index: usize = 0;
        loop {
            match self.in_stream.peek_next(&self.time) {
                Ok(ChannelElement { time: _, data }) => {
                    if accumulators.is_empty() {
                        // Start of a reduction group: resolve dynamic tile dims
                        // from this first input and build the buffer.
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
                                // Intermediate retained-dimension boundary: keep
                                // accumulating across the reduced dimension.
                                self.process_accum(x, &mut accumulators, index);
                                index = (index + 1) % n;
                            } else if level == self.rank {
                                let out_buffer =
                                    self.process_accum_flush(x, &mut accumulators, index);
                                index = 0;
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::Val(out_buffer),
                                        },
                                    )
                                    .unwrap();
                            } else {
                                let out_buffer =
                                    self.process_accum_flush(x, &mut accumulators, index);
                                index = 0;
                                self.out_stream
                                    .enqueue(
                                        &self.time,
                                        ChannelElement {
                                            time: self.time.tick(),
                                            data: Elem::ValStop(out_buffer, level - self.rank),
                                        },
                                    )
                                    .unwrap();
                            }
                        }
                    }
                }
                Err(_) => return,
            }
        }
    }
```

- [ ] **Step 4: Change `process_accum_flush` to snapshot with `std::mem::take`**

In `process_accum_flush`, replace the final snapshot/reinit block (the `let n = accumulators.len();` + `std::mem::replace(...)` lines) with:

```rust
        // Snapshot the completed slots, leaving `accumulators` empty. The empty
        // vec signals the run loop to re-resolve tile dimensions and rebuild the
        // buffer from the first input of the next reduction group.
        let slots: Vec<Tile<OT>> = std::mem::take(accumulators);

        let arr = ArcArray::from_shape_vec(self.buffer_shape.clone(), slots)
            .expect("AccumBuff: buffer_shape does not match the number of accumulator slots");

        Buffer::new(arr, self.time.tick().time())
```

(The rest of `process_accum_flush` — cycle accounting, the `func` call writing `accumulators[index] = out_tile`, the `dequeue`, and logging — is unchanged.)

- [ ] **Step 5: Run the full `accum_buff` test suite to verify it passes**

Run:
```bash
source /home/gina/miniconda3/etc/profile.d/conda.sh && conda activate pytorch-step && \
export PYO3_PYTHON=~/miniconda3/envs/pytorch-step/bin/python && \
export LD_LIBRARY_PATH=~/miniconda3/envs/pytorch-step/lib:$LD_LIBRARY_PATH && \
cargo test accum_buff
```
Expected: all five PASS — `test_accum_buff_rank2`, `test_accum_buff_rank3`, `test_accum_buff_reduce_two_dims`, `test_accum_buff_dynamic_row_single_group`, `test_accum_buff_dynamic_row_per_group`.

- [ ] **Step 6: Commit**

```bash
git add src/operator/accum_buff.rs
git commit -m "AccumBuff: resolve dynamic tile_row/tile_col per reduction group"
```

---

## Self-Review

**1. Spec coverage:**
- "`0` means dynamic, per dimension" → Task 2 Step 3 (`if self.tile_row == 0 { in_rows } else { self.tile_row }`, same for col). ✓
- "resolved from the first input of each reduction group" → Task 2 Step 3 (lazy build on empty `accumulators`) + Step 4 (`std::mem::take` empties per group). ✓
- "closure takes `(rows, cols)`; operator lacks `num::Zero`" → Task 1 Steps 1–2, 4. ✓
- "proto driver `Zero` arm parameterized; other variants `todo!()`" → Task 1 Step 4. ✓
- "existing three tests behave identically" → Task 1 Steps 5–7 (pass `1, 2`; closure ignores args). ✓
- "new single-group dynamic test" → Task 2 `test_accum_buff_dynamic_row_single_group`. ✓
- "per-group varying test that would panic under resolve-once" → Task 2 `test_accum_buff_dynamic_row_per_group`. ✓
- "empty-vec sentinel via `std::mem::take`, unambiguous because `n > 0`" → Task 2 Steps 3–4. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"; all steps carry complete code. The `todo!()` items are real, intentionally-unchanged Rust. ✓

**3. Type consistency:** `init_accum: Fn(usize, usize) -> Tile<OT>` is used identically in the struct (T1S1), `new` (T1S2), both call sites (T1S3), the proto driver (T1S4, `Tile<f32>`), and the single `run_reduction` test helper (T1S5). `AccumBuff::new`'s argument order (`..., buffer_shape, tile_row, tile_col, config, id`) matches between the struct literal, the proto driver call, and the test helper. `run_reduction`'s new `init_accum` parameter is supplied at all five call sites — three static (via `zero_init`) and two dynamic (via `Tile::new_zero`) — and `filled_tile`'s signature matches its call sites. ✓
