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

#[cfg(test)]
mod tests {
    use super::*;
    use dam::simulation::ProgramBuilder;
    use dam::utility_contexts::GeneratorContext;
    use std::fs;

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
