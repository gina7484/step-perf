pub mod matmul;

use dam::types::StaticallySized;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct HBMEntry {
    outer: u32,
    m: u32,
    n: u32,
    k: u32,
    #[serde(rename = "start(ms)")]
    start_ms: f64,
    #[serde(rename = "end(ms)")]
    end_ms: f64,
    output_tile_available: bool,
    num_elems: u32,
}

impl StaticallySized for HBMEntry {
    const SIZE: usize = u32::SIZE * 5 + bool::SIZE * 2 + f64::SIZE * 2;
}

pub fn parse_csv(file_path: &str) -> Vec<HBMEntry> {
    let mut reader = csv::Reader::from_path(file_path).expect("Failed to open file");
    let mut entries = Vec::new();

    for result in reader.deserialize() {
        match result {
            Ok(entry) => entries.push(entry),
            Err(err) => eprintln!("Error parsing CSV entry: {}", err),
        }
    }

    entries
}
