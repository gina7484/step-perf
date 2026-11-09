# Resource counting

## On-chip memory

The simulator records, for each operator, the most on-chip bytes it held at
once. `run_graph` returns `{op id: bytes}` for the operators that held
anything. In pytorch_step, `run_timing_simulation` writes these to
`onchip_memory.csv`, largest first, and returns their sum as `onchip_bytes`.

Sizes come from the tiles at run time, as `bytes_per_elem` times the shape, so
dynamic and ragged tiles count at their real size. bf16 runs as `Tile<f32>` but
still counts 2 bytes per element.

### Where the code is

- **Which ops count.** `counts()` in `src/utils/onchip.rs` decides this once
  per op, at build time, from the op's proto: its kind, BinaryMap's function,
  and the declared dtype. A tile with a dynamic dimension is never 1x1, so it
  counts.
- **How many bytes.** Each counted op calls `self.onchip.observe(bytes)` in
  its `run` function, where it creates the storage. The File column below
  lists them. Two loads make the call from a helper that `run` reaches:
  `emit_blank_tile` in StridedOffChipLoad and `stream_tiles` in
  LinearOffChipLoadRef.

| Op | Counts when | Bytes observed, max kept | File |
|---|---|---|---|
| StridedOffChipLoad | always | 2 × each tile it emits | `src/memory/strided_offchip_load.rs` |
| LinearOffChipLoad | always | 2 × each tile it emits | `src/memory/linear_offchip_load.rs` |
| LinearOffChipLoadRef | always | 2 × each tile it emits | `src/memory/linear_offchip_load_ref.rs` |
| Bufferize | always | each buffer it builds | `src/operator/bufferize.rs` |
| AccumBuffer | always | each buffer it outputs | `src/operator/accum_buff.rs` |
| BinaryMap | function is Matmul | both input tiles, per call | `src/operator/map.rs` |
| Accum | output isn't a scalar or a 1x1 tile | each output tile | `src/operator/accum.rs` |
| Scan | output isn't a scalar or a 1x1 tile | each output tile | `src/operator/scan.rs` |
| RepeatStatic, RepeatRef | type isn't a Buffer, a scalar, or a 1x1 tile | each value it repeats | `src/operator/repeat.rs` |
| PromoteOuter | type isn't a Buffer, a scalar, or a 1x1 tile | each value it passes | `src/operator/promote.rs` |

Every other operator counts 0.

- A multi-hot value counts 1 byte per candidate.
- A repeat of a Buffer counts 0 because the op that built the buffer already
  counted it.

Not counted yet: Accum with Mean or Var, which runs on `AccumRowStat`, and
BinaryMap with DynMatmul.

### Adding an operator

1. Add its condition to `counts()` in `src/utils/onchip.rs`.
2. In its context, add an `onchip: OnChipCounter` field set to
   `onchip::counter()` in `new()`, and call `self.onchip.observe(bytes)` where
   it creates the storage.
3. Add a test to `src/proto_driver/onchip_tests.rs` and a row to the table.
