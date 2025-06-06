use dam::{
    channel::{Receiver, Sender},
    context::Context,
    dam_macros::context_macro,
    types::DAMType,
};

/// Since DAM channels are single-producer single-consumer, Broadcasts can be used to send from a single channel to multiple channels.

#[context_macro]
pub struct BroadcastContext<T: Clone> {
    receiver: Receiver<T>,
    targets: Vec<Sender<T>>,
}

impl<T: DAMType> Context for BroadcastContext<T> {
    fn run(&mut self) {
        loop {
            let value = self.receiver.dequeue(&self.time);
            match value {
                Ok(mut data) => {
                    for target in &self.targets {
                        target.wait_until_available(&self.time).unwrap();
                    }
                    data.time = self.time.tick();
                    for target in &self.targets {
                        target.enqueue(&self.time, data.clone()).unwrap();
                    }
                }
                Err(_) => return,
            }
        }
    }
}

impl<T: DAMType> BroadcastContext<T> {
    /// Sets up a broadcast context with an empty target list.
    pub fn new(receiver: Receiver<T>) -> Self {
        let x = Self {
            receiver,
            targets: vec![],
            context_info: Default::default(),
        };
        x.receiver.attach_receiver(&x);
        x
    }

    /// Registers a target for the broadcast
    pub fn add_target(&mut self, target: Sender<T>) {
        target.attach_sender(self);
        self.targets.push(target);
    }
}
