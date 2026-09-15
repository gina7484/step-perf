# step-perf

## Simulation traffic report

Each graph simulation reports total off-chip traffic in bytes alongside elapsed
cycles and duration, with separate read and write totals. `HBMContext` counts the
addresses it receives at runtime, so the total follows the accesses issued for
ragged and data-dependent shapes.

Each address counts as `HBMConfig.addr_offset` bytes. Repeated accesses count
again, and partially used requests count at their full request size. Loads with
`simulate_ramulator=false` bypass HBM and contribute no traffic to this report.

Python callers can read the counters from the final item in the return tuple:

```python
cycles, duration_ms, duration_s, traffic = simulate(...)
print(traffic["total_bytes"])
print(traffic["read_bytes"])
print(traffic["write_bytes"])
```

The lower-level `step_perf.run_graph(...)` returns
`(passed, cycles, duration_ms, duration_s, traffic)`. Rust `parse_proto` returns
`(passed, cycles, duration, traffic_stats)`, where `traffic_stats` is an
`Arc<HBMTrafficStats>`.

## Get Started:
To run the tests that use Ramulator:
```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 **cargo test --package step-perf --lib -- test::<test_file_name>::test::<name_of_the_test_fn> --exact --show-output
```
<br/>

Example:
```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 cargo test --package step-perf --lib -- ramulator::ramulator_context::test::ramulator_e2e_small --exact --show-output 
```
<br/>

To log data with MongoDB:
* Running MongoDB in foreground
    ```bash
    sudo mongod --config /etc/mongod.conf
    ```


## Performance model for each node

### Compute Ops
---
#### Accum


#### Map

#### Scan


### Shape Ops
---
#### Flatten, Promote, Reshape
0 cycle latency

#### Repeat
If the data arriaved on cycle `x` and is reapted `y` times, the data is enqueued from cycle `x ~ (x+y-1)`.

### Routing & Mergine Ops (Includes ops for control flow)
#### FlatPartition

#### FlatReassemble

#### EagerMerge
0 cycle latency.

#### EagerParallelize


#### Parallelize
0 cycle latency.

#### Merge
0 cycle latency.

### On-chip memory access
#### Streamify

#### StreamifyRef (or DynStreamify)

#### Bufferize

#### RetileStreamify


### Off-chip memory access
#### LinearOffChipLoad

#### LinearOffChipLoadRef

#### OffChipStore
