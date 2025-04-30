use dam::context_tools::*;
use dam::dam_macros::event_type;
use serde::{Deserialize, Serialize};

// Logging values
#[derive(Serialize, Deserialize, Debug)]
#[event_type]
pub struct SimpleLogData {
    pub arrive: u64,
    pub out_done: u64,
}

#[context_macro]
pub struct CustomGeneratorContext<T: Clone, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    iterator: Option<FType>,
    output: Sender<T>,
}

impl<T: DAMType, IType, FType> Context for CustomGeneratorContext<T, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    fn init(&mut self) {}

    fn run(&mut self) {
        if let Some(func) = self.iterator.take() {
            for val in (func)() {
                let current_time = self.time.tick();

                self.output
                    .enqueue(&self.time, ChannelElement::new(current_time + 1, val))
                    .unwrap();
                self.time.incr_cycles(1);

                dam::logging::log_event(&SimpleLogData {
                    arrive: self.time.tick().time() - 1,
                    out_done: self.time.tick().time(),
                })
                .unwrap();
            }
        } else {
            panic!("Iterator has already been consumed");
        }
    }
}

impl<T: DAMType, IType, FType> CustomGeneratorContext<T, IType, FType>
where
    IType: Iterator<Item = T>,
    FType: FnOnce() -> IType + Send + Sync,
{
    /// Constructs a GeneratorContext from an iterator and the output channel
    pub fn new(iterator: FType, output: Sender<T>) -> CustomGeneratorContext<T, IType, FType> {
        let gc = CustomGeneratorContext {
            iterator: Some(iterator),
            output,
            context_info: Default::default(),
        };
        gc.output.attach_sender(&gc);
        gc
    }
}

#[context_macro]
pub struct ReceiverBackpressureContext {
    in_stream: Receiver<u32>,
    out_stream: Sender<u32>,
}

impl ReceiverBackpressureContext {
    pub fn new(in_stream: Receiver<u32>, out_stream: Sender<u32>) -> Self {
        let ctx = Self {
            in_stream,
            out_stream,
            context_info: Default::default(),
        };
        ctx.in_stream.attach_receiver(&ctx);
        ctx.out_stream.attach_sender(&ctx);

        ctx
    }
}

impl Context for ReceiverBackpressureContext {
    fn run(&mut self) {
        loop {
            let data = match self.in_stream.dequeue(&self.time) {
                Ok(ChannelElement { time: t, data }) => {
                    println!("ChannelElement data: {}", data);
                    println!("ChannelElement time: {}", t.time());
                    println!("Received time: {}", self.time.tick().time());
                    data
                }
                Err(_) => return,
            };

            self.out_stream
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: self.time.tick() + 1,
                        data,
                    },
                )
                .unwrap();
            self.time.incr_cycles(3);
        }
    }
}

#[cfg(test)]
mod test_backpressure {
    use dam::{
        simulation::{
            DotConvertible, LogFilterKind, LoggingOptions, MongoOptionsBuilder, ProgramBuilder,
            RunOptionsBuilder,
        },
        utility_contexts::CheckerContext,
    };

    use super::{CustomGeneratorContext, ReceiverBackpressureContext};

    #[test]
    fn test_with_dequeue() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);
        let (out_snd, out_rcv) = ctx.bounded(2);

        ctx.add_child(CustomGeneratorContext::new(|| 0..10u32, in_snd));
        ctx.add_child(ReceiverBackpressureContext::new(in_rcv, out_snd));
        ctx.add_child(CheckerContext::new(|| 0..10u32, out_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_backpressure_log() {
        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.bounded(2);
        let (out_snd, out_rcv) = ctx.bounded(2);

        ctx.add_child(CustomGeneratorContext::new(|| 0..10u32, in_snd));
        ctx.add_child(ReceiverBackpressureContext::new(in_rcv, out_snd));
        ctx.add_child(CheckerContext::new(|| 0..10u32, out_rcv));

        let initialized = ctx.initialize(Default::default()).unwrap();

        let run_options = RunOptionsBuilder::default().log_filter(LogFilterKind::Blanket(
            // dam::logging::LogFilter::Some([SimpleLogData::NAME.to_owned()].into()),
            dam::logging::LogFilter::AllowAll,
        ));
        let run_options = run_options.logging(LoggingOptions::Mongo(
            MongoOptionsBuilder::default()
                .db("backpressure_log".to_string())
                .uri("mongodb://127.0.0.1:27017".to_string())
                .build()
                .unwrap(),
        ));
        let summary = initialized.run(run_options.build().unwrap());

        #[cfg(feature = "dot")]
        {
            println!("{}", summary.to_dot_string());
        }
        dbg!(summary.elapsed_cycles());
    }
}
