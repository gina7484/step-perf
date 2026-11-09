//! Timing-only tile loads from allocation-relative element offsets.

use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
#[cfg(test)]
use std::sync::{Arc, Mutex};

use dam::channel::PeekResult;
use dam::context_tools::*;
use dam::logging::LogEvent;

use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::tile::Tile;
use crate::ramulator::hbm_context::ParAddrs;
use crate::utils::events::LoggableEventSimple;
use crate::utils::onchip::{self, OnChipCounter};
use crate::utils::request_profile::{ProfiledReceiver, ProfiledSender};

const TILE_BUFFERS: usize = 2;

struct PendingTile {
    requests: Vec<u64>,
    issued: usize,
    outstanding: HashSet<u64>,
    stop: Option<StopType>,
    started_at: Option<u64>,
    finished_at: Option<u64>,
}

/// Read one tile per scalar base, preserving the input's stop levels.
///
/// Bases and tile strides are measured in elements of a flat allocation.
/// Output position [r, c] reads base + r * tile_stride[0] + c * tile_stride[1].
/// Strides may be zero. Every requested element must lie within the allocation;
/// partial tiles and negative strides are not supported.
///
/// HBM requests cover the aligned addr_offset-byte blocks touched by the tile.
/// Each block is requested once per tile, in first-touch order. Two tile buffers
/// overlap HBM requests, using the existing unbounded request/response channels.
/// A buffer is released when its tile is emitted in input order. Responses are
/// matched by address; a repeated address satisfies the oldest outstanding read.
/// Output payloads have no underlying values. Only dispatch and HBM completion
/// contribute latency; address arithmetic and packing have no separate cost.
#[context_macro]
pub struct StridedOffChipLoad<E: LoggableEventSimple, T: DAMType> {
    pub allocation_elements: u64,
    pub tile_shape: [usize; 2],
    pub tile_stride: [u64; 2],
    pub n_byte: usize,
    pub base_addr_byte: u64,
    pub addr_offset: u64,
    pub par_dispatch: usize,
    pub addr_snd: Sender<ParAddrs>,
    pub resp_addr_rcv: Receiver<u64>,
    pub raddr: ProfiledReceiver<Elem<Tile<u64>>>,
    pub rdata: ProfiledSender<Elem<Tile<T>>>,
    pub id: u32,
    #[cfg(test)]
    request_trace: Option<Arc<Mutex<Vec<(u64, Vec<u64>)>>>>,
    onchip: OnChipCounter,
    _phantom: PhantomData<E>,
}

impl<E: LoggableEventSimple + LogEvent + Sync + Send, T: DAMType> StridedOffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    pub fn new(
        allocation_elements: u64,
        tile_shape: [usize; 2],
        tile_stride: [u64; 2],
        n_byte: usize,
        base_addr_byte: u64,
        addr_offset: u64,
        par_dispatch: usize,
        addr_snd: Sender<ParAddrs>,
        resp_addr_rcv: Receiver<u64>,
        raddr: Receiver<Elem<Tile<u64>>>,
        rdata: Sender<Elem<Tile<T>>>,
        id: u32,
    ) -> Self {
        assert!(
            tile_shape.iter().all(|&n| n > 0),
            "tile dimensions must be positive"
        );
        assert!(n_byte > 0, "element width must be positive");
        assert!(addr_offset > 0, "HBM request width must be positive");
        assert!(par_dispatch > 0, "parallel dispatch must be positive");
        tile_shape[0]
            .checked_mul(tile_shape[1])
            .and_then(|count| count.checked_mul(n_byte))
            .expect("tile size overflow");

        let ctx = Self {
            allocation_elements,
            tile_shape,
            tile_stride,
            n_byte,
            base_addr_byte,
            addr_offset,
            par_dispatch,
            addr_snd,
            resp_addr_rcv,
            raddr: raddr.into(),
            rdata: rdata.into(),
            id,
            #[cfg(test)]
            request_trace: None,
            onchip: onchip::counter(),
            _phantom: PhantomData,
            context_info: Default::default(),
        };
        ctx.addr_snd.attach_sender(&ctx);
        ctx.resp_addr_rcv.attach_receiver(&ctx);
        ctx.raddr.attach_receiver(&ctx);
        ctx.rdata.attach_sender(&ctx);
        ctx
    }

    /// Enumerate element offsets in output row/column order, retaining repeats.
    pub fn element_offsets(&self, base: u64) -> Vec<u64> {
        let [rows, cols] = self.tile_shape;
        let mut offsets = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let row_offset = (row as u64)
                .checked_mul(self.tile_stride[0])
                .expect("element offset overflow");
            for col in 0..cols {
                let col_offset = (col as u64)
                    .checked_mul(self.tile_stride[1])
                    .expect("element offset overflow");
                let offset = base
                    .checked_add(row_offset)
                    .and_then(|offset| offset.checked_add(col_offset))
                    .expect("element offset overflow");
                assert!(
                    offset < self.allocation_elements,
                    "[StridedOffChipLoad {}] element offset {offset} outside allocation of {} elements",
                    self.id, self.allocation_elements,
                );
                offsets.push(offset);
            }
        }
        offsets
    }

    /// Cover every element's bytes, coalescing blocks within this tile only.
    pub fn hbm_requests(&self, offsets: &[u64]) -> Vec<u64> {
        let mut seen = HashSet::new();
        let mut requests = Vec::new();
        for &offset in offsets {
            let first_byte = offset
                .checked_mul(self.n_byte as u64)
                .and_then(|byte| self.base_addr_byte.checked_add(byte))
                .expect("byte address overflow");
            let last_byte = first_byte
                .checked_add(self.n_byte as u64 - 1)
                .expect("byte address overflow");
            for block in first_byte / self.addr_offset..=last_byte / self.addr_offset {
                let address = block * self.addr_offset;
                if seen.insert(address) {
                    requests.push(address);
                }
            }
        }
        requests
    }

    fn prepare_tile(&self, token: Elem<Tile<u64>>) -> PendingTile {
        let (base, stop) = decode_scalar(token);
        let offsets = self.element_offsets(base);
        PendingTile {
            requests: self.hbm_requests(&offsets),
            issued: 0,
            outstanding: HashSet::new(),
            stop,
            started_at: None,
            finished_at: None,
        }
    }

    /// Issue at most one batch per cycle, finishing one tile's dispatch first.
    fn issue_next_batch(&self, tiles: &mut VecDeque<PendingTile>) {
        let Some(tile) = tiles
            .iter_mut()
            .find(|tile| tile.issued < tile.requests.len())
        else {
            return;
        };
        let end = tile
            .issued
            .saturating_add(self.par_dispatch)
            .min(tile.requests.len());
        let chunk = &tile.requests[tile.issued..end];
        tile.started_at.get_or_insert(self.time.tick().time());
        self.addr_snd
            .enqueue(
                &self.time,
                ChannelElement {
                    time: self.time.tick() + 1,
                    data: ParAddrs::new(chunk.to_vec()),
                },
            )
            .unwrap();
        #[cfg(test)]
        if let Some(trace) = &self.request_trace {
            trace
                .lock()
                .unwrap()
                .push((self.time.tick().time(), chunk.to_vec()));
        }
        tile.outstanding.extend(chunk.iter().copied());
        tile.issued = end;
    }

    fn collect_ready_responses(&self, tiles: &mut VecDeque<PendingTile>) {
        loop {
            match self.resp_addr_rcv.peek() {
                PeekResult::Something(response) if response.time <= self.time.tick() => {}
                PeekResult::Closed => {
                    assert!(
                        tiles.iter().all(|tile| tile.finished_at.is_some()),
                        "HBM response channel closed with unfinished tiles",
                    );
                    return;
                }
                _ => return,
            }
            let address = self.resp_addr_rcv.dequeue(&self.time).unwrap().data;
            // Responses contain addresses, not tile IDs. Consume exactly one
            // outstanding read, oldest first when two tiles share a block.
            let tile = tiles
                .iter_mut()
                .find(|tile| tile.outstanding.contains(&address))
                .expect("HBM response has no outstanding request");
            tile.outstanding.remove(&address);
            if tile.issued == tile.requests.len() && tile.outstanding.is_empty() {
                tile.finished_at = Some(self.time.tick().time());
            }
        }
    }

    fn emit_completed_tiles(&self, tiles: &mut VecDeque<PendingTile>) {
        while tiles.front().is_some_and(|tile| tile.finished_at.is_some()) {
            let tile = tiles.pop_front().unwrap();
            let start = tile.started_at.unwrap();
            let finish = tile.finished_at.unwrap();
            let bytes = (tile.requests.len() as u64)
                .checked_mul(self.addr_offset)
                .expect("read traffic overflow");
            self.rdata.record_memory(start, finish, bytes, false);
            dam::logging::log_event(&E::new(
                "StridedOffChipLoad".into(),
                self.id,
                start,
                finish,
                tile.stop.is_some(),
            ))
            .unwrap();
            self.emit_blank_tile(tile.stop);
        }
    }

    fn emit_blank_tile(&self, stop: Option<StopType>) {
        let tile = Tile::new_blank(self.tile_shape.to_vec(), self.n_byte, true);
        // Double buffering keeps two tiles on chip.
        self.onchip.observe(2 * tile.size_in_bytes());
        let data = match stop {
            Some(level) => Elem::ValStop(tile, level),
            None => Elem::Val(tile),
        };
        self.rdata
            .enqueue(
                &self.time,
                ChannelElement {
                    time: self.time.tick(),
                    data,
                },
            )
            .unwrap();
    }
}

fn decode_scalar(token: Elem<Tile<u64>>) -> (u64, Option<StopType>) {
    let (tile, stop) = match token {
        Elem::Val(tile) => (tile, None),
        Elem::ValStop(tile, level) => (tile, Some(level)),
    };
    assert_eq!(tile.shape, [1, 1], "tile base must be a scalar [1,1] tile");
    let values = tile
        .underlying
        .as_ref()
        .expect("tile base must have a concrete value");
    (values[[0, 0]], stop)
}

impl<E: LoggableEventSimple + LogEvent + Sync + Send, T: DAMType> Context
    for StridedOffChipLoad<E, T>
where
    Elem<Tile<T>>: DAMType,
{
    fn run(&mut self) {
        let mut tiles = VecDeque::new();
        let mut input_closed = false;
        loop {
            self.collect_ready_responses(&mut tiles);
            self.emit_completed_tiles(&mut tiles);

            // Poll the base input so a delayed next base cannot hold up replies
            // or completed outputs. Each accepted base occupies one tile buffer.
            if !input_closed && tiles.len() < TILE_BUFFERS {
                match self.raddr.peek() {
                    PeekResult::Something(token) if token.time <= self.time.tick() => {
                        let token = self.raddr.dequeue(&self.time).unwrap();
                        tiles.push_back(self.prepare_tile(token.data));
                    }
                    PeekResult::Closed => input_closed = true,
                    _ => {}
                }
            }
            self.issue_next_batch(&mut tiles);

            if input_closed && tiles.is_empty() {
                break;
            }
            self.time.incr_cycles(1);
        }
    }
}

#[cfg(test)]
mod tests;
