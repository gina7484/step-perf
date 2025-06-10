use dam::{
    channel::{Receiver, Sender},
    context::Context,
    dam_macros::context_macro,
    types::DAMType,
};

use crate::primitives::{elem::Elem, tile::Tile};

/// Since DAM channels are single-producer single-consumer, Broadcasts can be used to send from a single channel to multiple channels.

#[context_macro]
pub struct BroadcastContext<T: Clone> {
    receiver: Receiver<Elem<T>>,
    targets: Vec<Sender<Elem<T>>>,
}

impl<T: DAMType> Context for BroadcastContext<T>
where
    Elem<T>: DAMType,
{
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

impl<T: DAMType> BroadcastContext<T>
where
    Elem<T>: DAMType,
{
    /// Sets up a broadcast context with an empty target list.
    pub fn new(receiver: Receiver<Elem<T>>) -> Self {
        let x = Self {
            receiver,
            targets: vec![],
            context_info: Default::default(),
        };
        x.receiver.attach_receiver(&x);
        x
    }

    /// Registers a target for the broadcast
    pub fn add_target(&mut self, target: Sender<Elem<T>>) {
        target.attach_sender(self);
        self.targets.push(target);
    }
}
