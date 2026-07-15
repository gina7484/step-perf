# FilePrinterContext Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `FilePrinterContext<T>` debug context that writes each dequeued channel element to a per-instance file `FilePrinterContext_{id}.log`, so parallel printers no longer interleave on STDOUT.

**Architecture:** A local DAM context mirroring `dam`'s `PrinterContext`, defined with `#[context_macro]`, generic over `T: DAMType`, carrying `chan: Receiver<T>` and `id: u32`. It opens the output file (truncating) at the start of `run()`, writes each element with `writeln!("{:?}", ...)`, and flushes on completion. Lives under `src/utils/`.

**Tech Stack:** Rust, `dam` crate (`context_tools`), `std::fs`/`std::io`.

## Global Constraints

- Build/test env (pyo3 links against conda Python): before `cargo test`, run `source /home/gina/miniconda3/etc/profile.d/conda.sh` (if needed), `source setup.sh` from repo root `/home/gina/Desktop/research/pytorch_step`, then `conda activate pytorch-step`, `export PYO3_PYTHON=~/miniconda3/envs/pytorch-step/bin/python`, `export LD_LIBRARY_PATH=~/miniconda3/envs/pytorch-step/lib:$LD_LIBRARY_PATH`. Otherwise linking fails with `cannot find -lpython3.12`.
- Filename format is exactly `FilePrinterContext_{id}.log`, created in the current working directory, opened with truncate/overwrite.
- Timing must match `PrinterContext`: `self.time.incr_cycles(1)` per element.
- No `Co-Authored-By` / "Generated with Claude" trailers in commits.

---

### Task 1: FilePrinterContext with passing unit test

**Files:**
- Create: `src/utils/file_printer.rs`
- Modify: `src/utils/mod.rs` (add `pub mod file_printer;`)
- Test: `src/utils/file_printer.rs` (inline `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `dam::context_tools::*` (`Context`, `Receiver`, `DAMType`, `context_macro`); for the test, `dam::simulation::ProgramBuilder` and `dam::utility_contexts::GeneratorContext`.
- Produces: `pub struct FilePrinterContext<T: DAMType>` with `pub fn new(chan: Receiver<T>, id: u32) -> Self`. Writes `FilePrinterContext_{id}.log` (one `{:?}`-formatted `ChannelElement` per line).

- [ ] **Step 1: Write the failing test**

Create `src/utils/file_printer.rs` with only the test module first (the type does not exist yet, so it fails to compile = the failing state):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::GeneratorContext;
    use std::fs;
    use std::io::BufRead;

    #[test]
    fn file_printer_writes_one_line_per_element() {
        let id = 9999_u32;
        let path = format!("FilePrinterContext_{}.log", id);
        let _ = fs::remove_file(&path); // start clean

        let n = 5_i32;
        let mut ctx = ProgramBuilder::default();
        let (snd, rcv) = ctx.unbounded::<i32>();
        ctx.add_child(GeneratorContext::new(move || (0..n).into_iter(), snd));
        ctx.add_child(FilePrinterContext::new(rcv, id));
        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());

        let contents = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), n as usize, "expected one line per element");
        assert!(lines.iter().all(|l| !l.trim().is_empty()));

        fs::remove_file(&path).unwrap();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `/home/gina/Desktop/research/pytorch_step/step-perf`, after the Global Constraints env setup):
```bash
cargo test --lib utils::file_printer 2>&1 | tail -20
```
Expected: FAIL — compile error, `cannot find type FilePrinterContext in this scope` (and `file_printer` module not declared until Step 3's mod.rs edit; if the module isn't registered the test won't run at all — that also counts as the not-yet-implemented state).

- [ ] **Step 3: Write the minimal implementation**

Prepend the implementation above the test module in `src/utils/file_printer.rs`:

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
    /// Constructs a FilePrinterContext from a receiver and an id.
    pub fn new(chan: Receiver<T>, id: u32) -> Self {
        let s = Self {
            chan,
            id,
            context_info: Default::default(),
        };
        s.chan.attach_receiver(&s);
        s
    }
}
```

Register the module in `src/utils/mod.rs` by adding this line (alphabetical order, after `pub mod events;`):
```rust
pub mod file_printer;
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cargo test --lib utils::file_printer 2>&1 | tail -20
```
Expected: PASS — `test result: ok. 1 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/utils/file_printer.rs src/utils/mod.rs
git commit -m "feat: add FilePrinterContext for per-instance file output"
```

---

## Self-Review

- **Spec coverage:** location under `src/utils/` (Task 1 mod.rs edit) ✓; struct with `chan` + `id` ✓; `#[context_macro]` / `T: DAMType` ✓; open-at-run-start, truncate, BufWriter, flush ✓; filename `FilePrinterContext_{id}.log` ✓; `incr_cycles(1)` timing ✓; `{:?}` on `ChannelElement` ✓; unit test with `GeneratorContext` + file assertion + cleanup ✓. Out-of-scope items (configurable dir, append, custom formatting) correctly excluded.
- **Placeholder scan:** none — all code and commands are concrete.
- **Type consistency:** `FilePrinterContext::new(chan, id)` signature is identical in the implementation and the test; filename format string is identical in impl and test.
