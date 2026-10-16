pub mod accum;
pub mod accum_buff;
pub mod accum_row_stat;
pub mod bufferize;
pub mod counter;
pub mod map;
pub mod map_accum;
// pub mod mux_demux;
pub mod broadcast;
pub mod dynstreamify;
pub mod eager_merge;
pub mod expand;
pub mod flatmap;
pub mod flatmap_decomp;
pub mod flatten;
pub mod parallelize;
pub mod partition;
pub mod promote;
pub mod reassemble;
pub mod repeat;
pub mod reshape;
pub mod scan;
pub mod static_reassemble;
pub mod streamify;

#[cfg(test)]
pub(crate) mod test_support {
    use dam::{context_tools::*, simulation::ProgramBuilder};
    use std::sync::{Arc, Mutex};

    #[context_macro]
    struct Collect<T: DAMType> {
        input: Receiver<T>,
        values: Arc<Mutex<Vec<T>>>,
    }

    impl<T: DAMType> Context for Collect<T> {
        fn run(&mut self) {
            while let Ok(elem) = self.input.dequeue(&self.time) {
                self.values.lock().unwrap().push(elem.data);
                self.time.incr_cycles(1);
            }
        }
    }

    pub fn collect<T: DAMType + 'static>(
        builder: &mut ProgramBuilder<'_>,
        input: Receiver<T>,
    ) -> Arc<Mutex<Vec<T>>> {
        let values = Arc::new(Mutex::new(Vec::new()));
        let ctx = Collect {
            input,
            values: values.clone(),
            context_info: Default::default(),
        };
        ctx.input.attach_receiver(&ctx);
        builder.add_child(ctx);
        values
    }
}

use dam::types::StaticallySized;
use serde::Deserialize;

/// This expresses the data being streamed directly between compute units
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct ActEntry {}

impl StaticallySized for ActEntry {
    const SIZE: usize = 1;
}
