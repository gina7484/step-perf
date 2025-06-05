#[cfg(test)]
mod test {
    use dam::{
        channel::ChannelElement,
        simulation::{InitializationOptions, ProgramBuilder, RunOptions},
        utility_contexts::{FunctionContext, GeneratorContext},
    };

    #[test]
    fn simple_timing() {
        let mut parent = ProgramBuilder::default();

        let (snd, rcv) = parent.bounded(6);

        let mut snd_ctx = FunctionContext::new();
        snd.attach_sender(&snd_ctx);
        snd_ctx.set_run(move |time| {
            for i in vec![1, 2, 3, 4, 5] {
                let current_time = time.tick();
                println!("current_time: {}", current_time.time());
                snd.enqueue(time, ChannelElement::new(current_time + 1, i))
                    .unwrap();
                time.incr_cycles(1);
            }
        });
        parent.add_child(snd_ctx);

        let mut read_ctx = FunctionContext::new();
        rcv.attach_receiver(&read_ctx);
        read_ctx.set_run(move |time| loop {
            // match rcv.dequeue(time) {
            //     Ok(ChannelElement {
            //         time: channel_elem_time,
            //         data,
            //     }) => {
            //         println!(
            //             "[Dequeued at time {}] ChannelElement: time({}), data({})",
            //             time.tick().time(),
            //             channel_elem_time.time(),
            //             data
            //         );
            //     }
            //     Err(_) => {
            //         println!(
            //             "[No more elements to dequeue at time {}]",
            //             time.tick().time()
            //         );
            //         return;
            //     }
            // }
            match rcv.peek() {
                dam::channel::PeekResult::Something(ChannelElement {
                    time: peeked_time,
                    data,
                }) => {
                    println!(
                        "[Peeked at time {}] ChannelElement: time({}), data({})",
                        time.tick().time(),
                        peeked_time.time(),
                        data
                    );
                    match rcv.dequeue(time) {
                        Ok(ChannelElement {
                            time: channel_elem_time,
                            data,
                        }) => {
                            println!(
                                "[Dequeued at time {}] ChannelElement: time({}), data({})",
                                time.tick().time(),
                                channel_elem_time.time(),
                                data
                            );
                        }
                        Err(_) => {
                            return;
                        }
                    }
                }
                dam::channel::PeekResult::Nothing(_) => {}
                dam::channel::PeekResult::Closed => return,
            }
            time.incr_cycles(1);
        });

        parent.add_child(read_ctx);

        println!("Finished building");

        let executed = parent
            .initialize(InitializationOptions::default())
            .unwrap()
            .run(RunOptions::default());

        println!("Elapsed: {:?}", executed.elapsed_cycles());
    }

    #[test]
    fn dequeue_at_same_cycle() {
        let mut parent = ProgramBuilder::default();

        let (snd, rcv) = parent.bounded(6);

        let mut snd_ctx = FunctionContext::new();
        snd.attach_sender(&snd_ctx);
        snd_ctx.set_run(move |time| {
            for i in vec![1, 2, 3, 4, 5] {
                let current_time = time.tick();
                println!("current_time: {}", current_time.time());
                snd.enqueue(time, ChannelElement::new(current_time, i))
                    .unwrap();
            }
        });
        parent.add_child(snd_ctx);

        let mut read_ctx = FunctionContext::new();
        rcv.attach_receiver(&read_ctx);
        read_ctx.set_run(move |time| loop {
            match rcv.dequeue(time) {
                Ok(ChannelElement {
                    time: channel_elem_time,
                    data,
                }) => {
                    println!(
                        "[Dequeued at time {}] ChannelElement: time({}), data({})",
                        time.tick().time(),
                        channel_elem_time.time(),
                        data
                    );
                }
                Err(_) => {
                    println!(
                        "[No more elements to dequeue at time {}]",
                        time.tick().time()
                    );
                    return;
                }
            }
        });

        parent.add_child(read_ctx);

        println!("Finished building");

        let executed = parent
            .initialize(InitializationOptions::default())
            .unwrap()
            .run(RunOptions::default());

        println!("Elapsed: {:?}", executed.elapsed_cycles());
    }
}
