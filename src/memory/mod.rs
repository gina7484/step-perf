pub mod events;
pub mod hbm_ld_st;

use dam::types::StaticallySized;
use serde::{Deserialize, Deserializer};
use std::str::FromStr;

// A custom deserializer to handle Python-style booleans
fn deserialize_python_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "True" => Ok(true),
        "False" => Ok(false),
        _ => Err(serde::de::Error::custom(format!(
            "Invalid boolean value: {}",
            s
        ))),
    }
}

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
    #[serde(deserialize_with = "deserialize_python_bool")]
    output_tile_available: bool,
    num_elems: u32,
}

impl StaticallySized for HBMEntry {
    const SIZE: usize = u32::SIZE * 5 + bool::SIZE * 2 + f64::SIZE * 2;
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct PMUEntry {
    pub outer: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub output_tile_available: bool,
    pub num_elems: u32,
}

impl StaticallySized for PMUEntry {
    const SIZE: usize = u32::SIZE * 5 + bool::SIZE;
}

impl PMUEntry {
    pub fn bw(&self) -> f64 {
        // (# of elements) / ns
        /*
        In the SN40L paper
            "These are complemented by 1040 distributed Pattern Memory Units (PMUs)
            that in aggregate provide hundreds of TBps of on-chip memory bandwidth"

        As a single SN40L chip has 1040 PMUs, we can assume that each PMU has a bandwidth of
        1000 TBps / 1040 PMUs = 1.923 TBps

        As we assume using fp16, this will be 1.923 TBps / 2 = 0.9615 * 1e12 elements / s
        = 0.9615 elements / ns
         */
        0.9615
    }
}

pub fn hbm_to_pmu(hbm_entry: &HBMEntry) -> PMUEntry {
    PMUEntry {
        outer: hbm_entry.outer,
        m: hbm_entry.m,
        n: hbm_entry.n,
        k: hbm_entry.k,
        output_tile_available: hbm_entry.output_tile_available,
        num_elems: hbm_entry.num_elems,
    }
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
