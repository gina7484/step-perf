# FLOP Assumptions for `step-perf` Functions

This document records how each function under `step-perf/src/functions` assumes
its FLOP count. It covers [`map_fn.rs`](./map_fn.rs),
[`accum_fn.rs`](./accum_fn.rs), and [`map_accum_fn.rs`](./map_accum_fn.rs).

## How the model works

Every function returns `(cycles, output_tile)`. The first value is **not** raw
FLOPs — it is a **cycle count** computed as:

```
cycles = div_ceil(total_FLOPs, flop_per_cycle)
```

So the **FLOP assumption** is whatever expression sits inside `div_ceil(...)`
as the numerator. Ops that don't model compute at all just hard-code `1` (or
`0`) cycles and take no `flop_per_cycle` argument.

**Notation:** for matmul, `in1 = [M,K]`, `in2 = [K,N]` (or `[N,K]` transposed),
output `[M,N]`. For element-wise ops, `[R,C]` is the **broadcasted output**
shape `(max(r1,r2), max(c1,c2))`; for unary/constant ops it is the input shape.

## `map_fn.rs` — element-wise / map ops

| Function | FLOPs assumed | Per-element rate / rationale |
|---|---|---|
| `matmul` | `2*M*K*N` | Standard matmul: 1 mul + 1 add per MAC, K MACs per output, M*N outputs |
| `div, mul, add` | `R*C` | 1 FLOP per output element (broadcasted) |
| `add_constant` | `R*C` (input shape) | 1 FLOP per input element — ⚠️ body actually does `x * constant` |
| `sub_constant` | `R*C` (input shape) | 1 FLOP per input element |
| `mul_constant` | `R*C` (input shape) | 1 FLOP per input element |
| `silu` | `R*C*8` | **8 FLOPs/elem** (explicit assumption for `x/(1+e^-x)`) |
| `exp` | `R*C*4` | **4 FLOPs/elem** (transcendental approx) |
| `pow2` | `R*C*4` | 4 FLOPs/elem |
| `tanh` | `R*C*4` | 4 FLOPs/elem |
| `pow` | `R*C*4` | 4 FLOPs/elem |
| `rsqrt` | `R*C*4` | 4 FLOPs/elem |
| `row_wise_sum` | `R*C` (input shape) | 1 FLOP per **input** element (reduction over columns) |
| `set_offset` | **1 cycle** | metadata op — no `flop_per_cycle` |
| `mask_row` | **1 cycle** | scatter/one-hot, no compute modeled |
| `row_wise_append` | **1 cycle** | data movement |
| `cache_write_addr_gen` | **1 cycle** | address arithmetic |
| `is_equal_scalar` | **1 cycle** | scalar comparison |
| `broadcast_rows` | **1 cycle** | data movement (`_flop_per_cycle` unused) |
| `select_to_scalar` | **1 cycle** | metadata extraction |
| `multihot_to_u64` | **1 cycle** | encoding conversion |
| `u64_to_multihot` | **1 cycle** | encoding conversion |
| `to_const_int` | **1 cycle** | constant materialization |

## `accum_fn.rs` — matmul-with-accumulator

| Function | FLOPs assumed | Per-element rate / rationale |
|---|---|---|
| `matmul` | `2*M*K*N` | Same as map matmul; the accumulator add (`acc + map`) is **not** counted separately |
| `dyn_matmul` | `2*M*K*N` | Same; first tile may skip accumulation but FLOP count is unchanged |

## `map_accum_fn.rs` — accumulating map / retile ops

| Function | FLOPs assumed | Per-element rate / rationale |
|---|---|---|
| `mul` | `R*C` | 1 FLOP per output element |
| `add` | `R*C` | 1 FLOP per output element |
| `retile_col` | **0 cycles** | concat/regrouping — explicit `TODO` to add grouping cost |
| `retile_row` | **0 cycles** | concat/regrouping — explicit `TODO` |
| `signal_req_all_read` | **1 cycle** | control signal |

## Summary of the assumptions

1. **Matmul** → `2*M*K*N` (the textbook 2*MKN; accumulation add never added on top).
2. **Element-wise binary** (add/mul/div) → 1 FLOP per **broadcasted output** element.
3. **Element-wise unary/constant** (`*_constant`, `row_wise_sum`) → 1 FLOP per **input** element.
4. **Transcendentals** → fixed multiplier per element: `silu = 8`, everything else
   (`exp`, `pow2`, `tanh`, `pow`, `rsqrt`) = `4`.
5. **Metadata / data-movement / control** ops → hard-coded **1 cycle**, bypassing the FLOP model entirely.
6. **Retile (concat)** ops → **0 cycles**, flagged as a `TODO`.

### Things to double-check

- `add_constant` multiplies (`x * constant`) despite its name — the FLOP count is
  unaffected, but the semantics look like a bug.
- `silu` is the only special function costed at 8 rather than 4 FLOPs/elem,
  presumably because it decomposes into exp + add + div. If you want consistency,
  that is the one to reconcile with your intended cost model.
