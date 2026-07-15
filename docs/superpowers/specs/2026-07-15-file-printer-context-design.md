# FilePrinterContext — Design

**Date:** 2026-07-15

## Problem

`PrinterContext` (from the external `dam-rs` crate) prints every element it
dequeues from a channel to STDOUT. When several `PrinterContext`s run in
parallel, their output interleaves on the terminal and there is no way to tell
which line came from which context. We want a variant that writes to a
per-instance file instead, so each printer's output is isolated.

## Solution

Add a new context, `FilePrinterContext<T>`, that mirrors `PrinterContext`'s
behavior but writes each dequeued element to a file named
`FilePrinterContext_{id}.log`, where `id` is a per-instance identifier
(matching the `id: u32` convention used by the operators in `src/operator/`).

### Location

New module `src/utils/file_printer.rs`, registered with
`pub mod file_printer;` in `src/utils/mod.rs`. It is a debug-only utility
context, so it lives under `utils` rather than `operator`.

### Struct

Local context using `#[context_macro]` (from `dam::context_tools`), generic
over `T: DAMType`, with two fields: the input `chan: Receiver<T>` and
`id: u32`.

```rust
use std::fs::File;
use std::io::{BufWriter, Write};
use dam::context_tools::*;

/// A context which prints elements of a channel to a file named
/// `FilePrinterContext_{id}.log`. Useful for disambiguating output when
/// several printers run in parallel. Debugging only — can emit a lot of text.
#[context_macro]
pub struct FilePrinterContext<T: DAMType> {
    chan: Receiver<T>,
    id: u32,
}

impl<T: DAMType> Context for FilePrinterContext<T> {
    fn init(&mut self) {}

    fn run(&mut self) {
        let path = format!("FilePrinterContext_{}.log", self.id);
        let mut writer = BufWriter::new(File::create(&path).unwrap()); // truncates
        loop {
            match self.chan.dequeue(&self.time) {
                Ok(x) => writeln!(writer, "{:?}", x).unwrap(),
                Err(_) => break,
            }
            self.time.incr_cycles(1);
        }
        writer.flush().unwrap();
    }
}

impl<T: DAMType> FilePrinterContext<T> {
    pub fn new(chan: Receiver<T>, id: u32) -> Self {
        let s = Self { chan, id, context_info: Default::default() };
        s.chan.attach_receiver(&s);
        s
    }
}
```

## Key Decisions

- **File lifetime:** the file is opened once at the start of `run()` via
  `File::create` (which truncates / overwrites), wrapped in a `BufWriter`, and
  explicitly flushed when `run()` finishes. No `File` handle is stored in the
  struct, so no extra trait constraints are introduced.
- **File mode:** truncate/overwrite, so each run starts with a clean file.
- **Filename:** `FilePrinterContext_{id}.log`, created in the current working
  directory.
- **Timing:** `incr_cycles(1)` per element, identical to `PrinterContext`, so
  substituting one for the other does not change simulated latency.
- **Format:** `{:?}` on the whole dequeued `ChannelElement`, matching
  `PrinterContext`.

## Testing

A unit test in `file_printer.rs` (following the pattern in
`src/operator/map.rs`):

1. Build a small `ProgramBuilder` program with a `GeneratorContext` feeding a
   `FilePrinterContext` with a distinct `id`.
2. Run the simulation.
3. Assert `FilePrinterContext_{id}.log` exists and contains the expected lines
   (one `{:?}`-formatted `ChannelElement` per input value).
4. Clean up the file afterward.

## Out of Scope (YAGNI)

- Configurable output directory or filename beyond the `id` suffix.
- Append mode.
- Custom formatting / serialization of elements.
