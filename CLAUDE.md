# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

Build the project:
```bash
cargo build
```

Run regular tests:
```bash
cargo test
```

Run tests that use Ramulator (requires special library path):
```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 cargo test --package step-perf --lib -- test::<test_file_name>::test::<name_of_the_test_fn> --exact --show-output
```

Example Ramulator test:
```bash
LD_LIBRARY_PATH=/home/ginasohn/step-perf/external/ramulator2_wrapper/ext/ramulator2 cargo test --package step-perf --lib -- ramulator::ramulator_context::test::ramulator_e2e_small --exact --show-output
```

## MongoDB Setup (for logging)
```bash
sudo mongod --config /etc/mongod.conf
```

## Architecture Overview

This is a Rust-based performance modeling framework built on top of the DAM (Decoupled Access/Execute Machine) architecture for simulating compute and memory operations.

### Core Modules

- **operator/**: High-level compute operators (map, accum, map_accum, repeat, partition) that implement streaming compute patterns
- **memory/**: Memory subsystem modeling including offchip load/store operations and memory events with PMU bandwidth modeling (64 bytes/cycle)
- **primitives/**: Low-level building blocks (buffer, elem, tile, select) for constructing operators
- **ramulator/**: Memory simulation integration with Ramulator2 for detailed DRAM modeling
- **functions/**: Core computational functions (map_fn, map_accum_fn) used by operators
- **utils/**: Utility functions for calculations and other common operations

### Key Dependencies

- **DAM**: Core framework for decoupled compute/memory modeling with coroutines, MongoDB logging, and DOT graph generation
- **Ramulator2**: External DRAM simulator (currently commented out in dependencies but used via wrapper)
- Data processing: ndarray, half-precision floats, CSV handling, serialization

### Data Flow

The system models streaming data through operators using `ActEntry` as the basic streaming unit between compute elements. Operators are composed using DAM's channel-based communication model.