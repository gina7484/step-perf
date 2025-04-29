use dam::dam_macros::event_type;
use serde::{Deserialize, Serialize};

// Define a trait for event types that can be logged
pub trait LoggableEvent {
    fn new(start: u64, end: u64) -> Self;
}
