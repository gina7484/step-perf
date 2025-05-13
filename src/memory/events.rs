// Define a trait for event types that can be logged

pub trait LoggableEventSimple {
    fn new(start_ns: u64, end_ns: u64, is_stop: bool) -> Self;
}

#[macro_export]
macro_rules! define_simple_event {
    ($event_name:ident) => {
        #[derive(Serialize, Deserialize, Debug)]
        #[event_type]
        struct $event_name {
            start_ns: u64,
            end_ns: u64,
            is_stop: bool,
        }

        // Implement the trait for $event_name
        impl LoggableEventSimple for $event_name {
            fn new(start_ns: u64, end_ns: u64, is_stop: bool) -> Self {
                $event_name {
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
