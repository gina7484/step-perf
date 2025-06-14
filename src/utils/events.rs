// Define a trait for event types that can be logged

use dam::dam_macros::event_type;
use serde::{Deserialize, Serialize};

pub const DUMMY_ID: u32 = 0;

pub trait LoggableEventSimple {
    fn new(name: String, id: u32, start_ns: u64, end_ns: u64, is_stop: bool) -> Self;
}

#[derive(Serialize, Deserialize, Debug)]
#[event_type]
pub struct SimpleEvent {
    name: String,
    id: u32,
    start_ns: u64,
    end_ns: u64,
    is_stop: bool,
}

impl LoggableEventSimple for SimpleEvent {
    fn new(name: String, id: u32, start_ns: u64, end_ns: u64, is_stop: bool) -> Self {
        SimpleEvent {
            name,
            id,
            start_ns,
            end_ns,
            is_stop,
        }
    }
}

impl SimpleEvent {
    pub const NAME: &'static str = stringify!(SimpleEvent);
}

#[macro_export]
macro_rules! define_simple_event {
    ($event_name:ident) => {
        #[derive(Serialize, Deserialize, Debug)]
        #[event_type]
        struct $event_name {
            name: String,
            id: u32,
            start_ns: u64,
            end_ns: u64,
            is_stop: bool,
        }

        // Implement the trait for $event_name
        impl LoggableEventSimple for $event_name {
            fn new(name: String, id: u32, start_ns: u64, end_ns: u64, is_stop: bool) -> Self {
                $event_name {
                    name: stringify!($event_name).to_string(),
                    id,
                    start_ns,
                    end_ns,
                    is_stop,
                }
            }
        }

        impl $event_name {
            pub const NAME: &'static str = stringify!($event_name);
        }
    };
}
