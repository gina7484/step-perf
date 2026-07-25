# Playbook: debugging `graph.pb` simulation failures

How to find and fix the root cause when running a serialized `graph.pb` through the
Rust simulator panics or reports `Passed: false`. The method is: **surface every
panic → separate the root from the cascade → map the root panic to an operator →
inspect the graph.pb node that feeds it → decide whether the fix belongs in graph
generation, serialization, or the Rust sim → reproduce minimally → fix, regenerate,
verify.**

> The worked example at the bottom is a real case (an MoE dynamic-M matmul that
> serialized as static `Matmul`). Read it once end-to-end; the steps above are the
> generalization of it.

---

## TL;DR checklist

- [ ] **Reproduce** with the cargo test; watch for `Passed: false` even when the test says `ok` (dam swallows Context panics).
- [ ] **List every panic**: `... 2>&1 | grep -iE "panicked at" | sort | uniq -c`.
- [ ] **Classify** each: `assertion ... failed` / domain `panic!` = candidate **root**; `.unwrap()` on enqueue/dequeue, "stream closed", "all input streams closed" = **cascade**.
- [ ] **Map** the root panic `file:line` → operator (`src/operator`, `src/memory`) → pure function (`src/functions`).
- [ ] **Dump the graph.pb node** feeding that operator (`dump_graph.py`, below) and read its fields/dtypes.
- [ ] **Trace to source**: grep the parent repo for where that field/func is set (generation in `step_py`/`backends`, serialization in `src/sim/__init__.py`).
- [ ] **Reproduce minimally** with a function-level Rust unit test (reliable — no dam).
- [ ] **Fix → regenerate graph.pb → re-run cargo test**; success = `Passed: true` and zero `panicked at`.

---

## 0. Environment setup

Two different environments, don't mix them up:

**Running the Rust sim (cargo test)** — pyo3 links against the conda Python, so from
`step-perf/`:

```bash
source /home/gina/miniconda3/etc/profile.d/conda.sh   # if `conda activate` complains
conda activate pytorch-step
source scripts/python_path.sh          # sets PYO3_PYTHON + LD_LIBRARY_PATH
```

Without this, linking fails with `cannot find -lpython3.12`.

**Inspecting / regenerating graph.pb (Python)** — needs the package layout on
`PYTHONPATH`. `setup.sh` sets it with `$(pwd)`, so **source it from the repo root**
`/home/gina/Desktop/research/pytorch_step`, *not* from `step-perf/` (sourcing it from
`step-perf/` points `PYTHONPATH` at the Rust `src/`, and `import step_py` fails). If you
can't `cd` there, set it explicitly:

```bash
R=/home/gina/Desktop/research/pytorch_step
export PYTHONPATH="$R/parabolic/src:$R/src:$R/src/sim:$R/src/proto:$R/src/backends:$R/models:$PYTHONPATH"
```

> The recurring `libtinfo.so.6: no version information available` line is harmless noise.

---

## 1. Reproduce and surface the real panic

```bash
RUST_BACKTRACE=1 cargo test --package step_perf --lib -- \
  test::protobuf_test::test::run_graph --exact --show-output 2>&1 | tail -80
```

**Gotcha (important):** dam runs each operator in its own coroutine and swallows the
`Context` panic. The test prints `test result: ok` **and** a separate
`Passed: false, Elapsed Cycles: ...` line. **`Passed: false` is the failure signal — not
the test result.** So never trust "ok" alone here; scan the output for panics.

List *every* distinct panic and how many times each fired:

```bash
RUST_BACKTRACE=0 cargo test --package step_perf --lib -- \
  test::protobuf_test::test::run_graph --exact --show-output 2>&1 \
  | grep -iE "panicked at" | sort | uniq -c
```

Example output — nine sites, not one:

```
   8 thread '<unnamed>' panicked at src/functions/map_accum_fn.rs:23:9:
  24 thread '<unnamed>' panicked at src/memory/linear_offchip_load_ref.rs:358:18:
  24 thread '<unnamed>' panicked at src/operator/flatten.rs:74:30:
   8 thread '<unnamed>' panicked at src/operator/map.rs:115:21:
   ...
```

The counts are a hint about graph structure: here `8` = one per expert, `24` = 8 experts × a
3-deep upstream chain.

---

## 2. Separate the root cause from the cascade

In a dataflow sim, when one context panics and dies, its channels drop, so **neighbors
die too**. Most of the panic sites are usually cascade noise. Classify each:

| Panic text | Meaning | Priority |
|---|---|---|
| `assertion \`left == right\` failed` (with `left:`/`right:` values) | A real logic/shape check failed | **Root candidate — fix first** |
| domain `panic!("... don't match ...")` with a specific message | Real logic bug | **Root candidate** |
| `.unwrap()` on `enqueue`/`dequeue` (panic line is a `.unwrap()`) | Producer/consumer on a dropped channel | Cascade — usually clears itself |
| `"One stream closed earlier"` | Upstream neighbor died | Cascade |
| `"All input streams are closed or empty"` | All upstreams died | Cascade |

Strategy: **fix the root candidate(s), regenerate, re-run.** The cascade panics almost
always disappear once the trigger is gone. (In the worked example, fixing the single
assertion cleared all 9 sites.) Confirm which `.unwrap()` a line is by opening the file at
that `file:line`.

---

## 3. Map the root panic to an operator and function

Panic lines are `file:line:col`. Use `RUST_BACKTRACE=1` to see the operator → function
call chain (the swallowed panic still prints its backtrace with `--show-output`). The
codebase splits into:

| Layer | Location | Role |
|---|---|---|
| Dispatch / construction | `src/proto_driver/mod.rs` | `match` on `OpType`; builds each operator from its proto fields (incl. closures like `init_accum`) |
| Operators (dataflow contexts) | `src/operator/*.rs` | e.g. `map_accum.rs`, `map.rs`, `flatten.rs`, `partition.rs`, `reassemble.rs`, `accum.rs` |
| Memory / loaders | `src/memory/*.rs` | e.g. `linear_offchip_load_ref.rs`, `offchip_store.rs` |
| Pure compute functions | `src/functions/*.rs` | e.g. `map_accum_fn.rs` (matmul), `map_fn.rs`, `accum_fn.rs` |

Read the panicking assertion and note **which values mismatched** (`left`/`right`). Then
ask: *where does the bad operand come from?* Trace it backward — an operand a function
asserts on was usually built in `proto_driver/mod.rs` from a proto field (e.g. an
accumulator built via `Tile::new_zero([tile_row, tile_col], ...)` where `tile_row` is a
proto field). That points you at the graph.pb node.

---

## 4. Inspect the graph.pb node (the key tool)

Save this as `dump_graph.py` (anywhere) and run it after sourcing the Python env (§0):

```python
import sys
import graph_pb2                          # resolves once setup.sh is sourced
from google.protobuf import text_format

path = sys.argv[1] if len(sys.argv) > 1 else "graph.pb"
sel  = sys.argv[2] if len(sys.argv) > 2 else None   # an op id, or an op_type name

g = graph_pb2.ProgramGraph()
with open(path, "rb") as f:
    g.ParseFromString(f.read())

matched = 0
for op in g.operators:
    kind = op.WhichOneof("op_type")
    if sel is None:                                   # list mode
        print(f"[id={op.id:>4}] {kind:26} {op.name}")
    elif sel == str(op.id) or sel == kind:            # full-dump mode
        print(text_format.MessageToString(op)); print("-" * 70)
        matched += 1
if sel and not matched:
    print(f"(no op matched {sel!r} — pass an id, or a type like 'binarymap_accum')")
```

Usage:

```bash
python dump_graph.py graph.pb                 # list every op: id, type, name
python dump_graph.py graph.pb 118             # full dump of op id 118
python dump_graph.py graph.pb binarymap_accum # full dump of all ops of that type
```

`text_format.MessageToString` prints the whole node, including nested oneofs and dtypes:

```
tile_col: 512
rank: 1
func { dyn_matmul {} }
dtype_a { bf16 { row_dynamic: "u0"  col_static: 64  } }
dtype_b { bf16 { row_dynamic: "u0"  col_static: 512 } }
```

**How to read it:**

- **Dynamic vs static dims** live in the dtype: `row_dynamic: "u3"` (a symbolic size) vs
  `row_static: 64`. Symbolic names like `u0..u7` are the classic MoE per-expert token
  counts. A dtype with a `*_dynamic` dim is your prime suspect whenever a downstream
  function asserts a concrete shape.
- **proto3 omits zero-valued fields.** A field that's `0` (e.g. `tile_row: 0`) simply
  **won't appear** in the dump. If you expect `tile_row` and don't see it, it's `0` — and
  a `0` shape fed into a static-shape assertion is a common root cause.
- Schemas are in `step_perf_ir/proto/{graph,ops,func,datatype}.proto`; the generated
  Python bindings are `src/proto/*_pb2.py` (`graph_pb2`, `ops_pb2`, `func_pb2`,
  `datatype_pb2`).

If the failure is *wrong values* rather than a crash, observe the actual tiles flowing on
a channel with `FilePrinterContext` (per-instance `FilePrinterContext_{id}.log`, no STDOUT
interleaving) or `PrinterContext` — see `docs/superpowers/specs/2026-07-15-file-printer-context-design.md`.

---

## 5. Trace back to generation / serialization and decide where to fix

A `graph.pb` is produced by the Python side in two stages, both in the parent repo
(`/home/gina/Desktop/research/pytorch_step/src`):

1. **Graph generation** — kernels/backends build the op graph
   (`src/step_py/kernels/*.py`, `src/backends/**`, and the `dyn_tiling/**` test drivers).
2. **Serialization** — `src/sim/__init__.py :: serialize()` walks the graph and writes the
   proto fields. (There is a second, parallel serializer, `src/sim/hwsim_ser.py`; only
   patch it if your flow uses it.)

Grep for the offending field/func to find where it's set, e.g.:

```bash
cd /home/gina/Desktop/research/pytorch_step
grep -rn "tile_row\|BinaryMapAccum\|map_accum_fn.Matmul" --include=*.py src/ | grep -v _pb2
```

Then choose the fix location deliberately:

- **Serialization (`src/sim/__init__.py`)** — when the generator legitimately emits a
  single op for both static and dynamic cases, and the proto just needs to pick the right
  variant. Lets existing generators run unchanged. *(This is what the worked example did.)*
- **Generation (`step_py`/`backends`)** — when the graph itself is wrong at the source.
- **Rust sim (`src/proto_driver` or `src/functions`)** — when the proto is right but the
  simulator mishandles a valid case. Prefer routing to an existing handler over loosening
  an assertion (loosening hides real bugs in the cases the assertion was catching).

---

## 6. Reproduce minimally

**Function-level Rust unit test — the reliable reproducer.** Calling the pure function in
`src/functions/*.rs` directly (not through dam) means panics are reported normally, so you
get a fast, deterministic repro and a regression test. Template:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::tile::Tile;
    use ndarray::ArcArray2;

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn repro_the_bug() {
        let in1 = Tile::new_padded(ArcArray2::from_elem((14, 64), 1.0f32), 2, false, 14);
        let in2 = Tile::new_padded(ArcArray2::from_elem((64, 512), 1.0f32), 2, false, 64);
        let acc: Tile<f32> = Tile::new_zero([0, 512], 2, false);   // the graph.pb shape
        let _ = matmul(&in1, &in2, &acc, 6400, false, false);
    }
}
```

Run just it (fast, no full graph):

```bash
cargo test --package step_perf --lib -- functions::map_accum_fn::tests --show-output
```

> Reconstruct the operands to match the graph.pb node you dumped in §4 (shapes, dtypes,
> and whether a dim is `0`/dynamic). If you keep the repro as a permanent regression test,
> put it where it documents the contract; if it was only for diagnosis, revert it once the
> real fix lands so the diff stays scoped.

---

## 7. Fix, regenerate, verify

**`graph.pb` is a gitignored build artifact** (see `.gitignore`: `*.pb`, `*.npy`, ...). A
serialization/generation fix therefore has **no effect until you regenerate graph.pb.**
Regenerate via whichever Python entry point built it — it calls
`simulate(graph, ..., "graph.pb")`, which writes `graph.pb` to the **current working
directory**. Run it with cwd = `step-perf/` so it lands where the cargo test reads it, e.g.:

```bash
# from the pytorch_step repo root, with the Python env set (§0):
cd step-perf && python -c "
import sys; sys.path.insert(0, '../dyn_tiling/ported_to_curr'); sys.path.insert(0, '..')
import test_mixtral_sweep as t; t.test_gemm_dyn_tile()"
```

`simulate()` also invokes the pyo3 Rust sim (the *installed* `step_perf` module, which may
be a stale build). If you only want to re-serialize and will verify via cargo, stub it
first so the pyo3 run is skipped:

```python
import step_perf
step_perf.run_graph = lambda *a, **k: (True, 0, 0.0, 0.0)   # serialize only
```

Confirm the node changed as intended, then run the sim:

```bash
python dump_graph.py graph.pb 118     # e.g. func now shows dyn_matmul
cargo test --package step_perf --lib -- test::protobuf_test::test::run_graph --exact --show-output 2>&1 \
  | grep -iE "panicked at|Passed:"
```

**Success = `Passed: true` and zero `panicked at` lines.** Two verification lenses exist:
the **cargo test** always uses the current `step-perf/src` (source of truth for the Rust
sim); the **pyo3 `simulate()`** path uses the installed `step_perf` module, which can lag
behind uncommitted Rust changes — prefer the cargo test after touching Rust.

---

## Worked example: dynamic-M matmul serialized as static `Matmul`

**Symptom.** `run_graph` → `Passed: false`; `grep panicked at | sort | uniq -c` showed 9
sites. Eight were `assertion left == right failed` at `src/functions/map_accum_fn.rs:23`
(`left: 0`, `right: 14`); the other ~72 were `.unwrap()`/"stream closed" cascades.

**Root.** Line 23 is `matmul`'s `assert_eq!(accumulator.shape[0], in1.shape[0])`.
Backtrace: `map_accum::process_map_accum` → `matmul`. The accumulator is built in
`proto_driver/mod.rs` via `Tile::new_zero([tile_row, tile_col], ...)`. Dumping the node
(`dump_graph.py graph.pb 118`) showed `tile_col: 512`, **no `tile_row` (so it's 0)**,
`func { matmul {} }`, and `dtype_a { bf16 { row_dynamic: "u0" col_static: 64 } }` — the
input rows (M) are **dynamic** (an 8-expert MoE, `u0..u7`).

So M is dynamic, the accumulator is a `[0, 512]` placeholder, but `func=Matmul` asserts a
static `[M, N]` accumulator and panics on the `0`. `dyn_matmul` already handles a 0-row
accumulator; `matmul` does not.

**Fix location.** Generation deliberately uses one `map_accum_fn.Matmul()` for both static
and dynamic tiles, so the fix went in **serialization** (`src/sim/__init__.py`): when the
input tile's rows are dynamic, emit `DynMatmul` in the proto (Rust already routes
`DynMatmul → dyn_matmul`). No Rust change, no generator change.

```python
# src/sim/__init__.py, in serialize() for BinaryMapAccum:
in1_dtype = op.in1.stream_tp.stream_dtype
input_has_dynamic_rows = isinstance(in1_dtype, Tile) and in1_dtype.shape[0].is_dynamic
binarymapaccum_pb.func.CopyFrom(to_pb_map_accum_func(op.fn, input_has_dynamic_rows))
```

**Verify.** Regenerated graph.pb (nodes 118–125 now dump `func { dyn_matmul {} }`), re-ran
the cargo test → `Passed: true, Elapsed Cycles: 48239`, zero panics. All 8 cascade sites
cleared with the single root fix.
