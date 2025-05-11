// Define a trait for event types that can be logged

pub trait LoggableEvent {
    fn new(
        outer: u32,
        m: u32,
        n: u32,
        k: u32,
        start_ns: u64,
        end_ns: u64,
        output_tile_available: bool,
        num_elems: u32,
    ) -> Self;
}
