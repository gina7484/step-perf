# step-perf

## Get Started

To run the tests that use Ramulator:

```bash
LD_LIBRARY_PATH=step-perf/external/ramulator2_wrapper/ext/ramulator2 **cargo test --package step-perf --lib -- test::<test_file_name>::test::<name_of_the_test_fn> --exact --show-output
```

<br/>

Example:

```bash
LD_LIBRARY_PATH=step-perf/external/ramulator2_wrapper/ext/ramulator2 cargo test --package step-perf --lib -- ramulator::ramulator_context::test::ramulator_e2e_small --exact --show-output 
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

#### OffChipLoad

#### OffChipLoadRef (or DynOffChipLoad)

#### OffChipStore
