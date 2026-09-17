//! Optional ATen boundary summaries. No event history or database is retained.
//!
//! Build-time thread-local state binds observers to existing channel endpoints.
//! Runtime observers own independent atomic counters; they never lock a global
//! collector or change simulated time. The run owner exports their summaries.
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed},
    Arc,
};

use crate::primitives::elem::Elem;
use dam::channel::adapters::RecvAdapter;
use dam::channel::{ChannelElement, ChannelID, DequeueError, EnqueueError, Receiver, Sender};
use dam::context::Context;
use dam::structures::TimeManager;
use dam::types::DAMType;
use serde::{Deserialize, Serialize};

const UNSET: u64 = u64::MAX;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Layout {
    pub groups: Vec<Vec<u32>>,
    /// A stop at this level or above advances the batch coordinate. Zero
    /// advances on every token. None denotes a single shared request group.
    pub advance_stop: Option<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PortSpec {
    pub op_id: u32,
    pub direction: String,
    pub source_id: u32,
    pub stream_idx: Option<u32>,
    pub aten: Vec<String>,
    pub layout: Layout,
    #[serde(default)]
    pub memory: bool,
}

#[derive(Debug, Deserialize)]
pub struct ProfileSpec {
    pub version: u32,
    pub graph_fingerprint: String,
    pub ports: Vec<PortSpec>,
}

#[derive(Default)]
struct Summary {
    first: AtomicU64,
    last: AtomicU64,
    tokens: AtomicU64,
    read_bytes: AtomicU64,
    write_bytes: AtomicU64,
    memory_start: AtomicU64,
    memory_finish: AtomicU64,
}
impl Summary {
    fn new() -> Self {
        Self {
            first: AtomicU64::new(UNSET),
            memory_start: AtomicU64::new(UNSET),
            ..Self::default()
        }
    }
    fn observe(&self, cycle: u64) {
        self.first.fetch_min(cycle, Relaxed);
        self.last.fetch_max(cycle, Relaxed);
        self.tokens.fetch_add(1, Relaxed);
    }
}

pub struct Observer {
    spec: PortSpec,
    groups: Vec<Summary>,
    coordinate: AtomicUsize,
    peeked: AtomicBool,
    last_group: AtomicUsize,
    bound: AtomicBool,
}
impl Observer {
    fn new(spec: PortSpec) -> Self {
        assert!(
            !spec.layout.groups.is_empty(),
            "profile request groups cannot be empty"
        );
        let groups = spec.layout.groups.iter().map(|_| Summary::new()).collect();
        Self {
            spec,
            groups,
            coordinate: AtomicUsize::new(0),
            peeked: AtomicBool::new(false),
            last_group: AtomicUsize::new(0),
            bound: AtomicBool::new(false),
        }
    }
    fn current(&self) -> usize {
        self.coordinate.load(Relaxed) % self.groups.len()
    }
    fn advance(&self, stop: Option<u32>) {
        if let Some(level) = self.spec.layout.advance_stop {
            if level == 0 || stop.is_some_and(|s| s >= level) {
                self.coordinate.fetch_add(1, Relaxed);
            }
        }
    }
    fn input(&self, cycle: u64, stop: Option<u32>, consumed: bool) {
        let group = self.current();
        if !self.peeked.swap(true, Relaxed) {
            self.groups[group].observe(cycle);
        }
        if consumed {
            self.last_group.store(group, Relaxed);
            self.advance(stop);
            self.peeked.store(false, Relaxed);
        }
    }
    fn output(&self, cycle: u64, stop: Option<u32>) {
        let group = self.current();
        self.groups[group].observe(cycle);
        self.last_group.store(group, Relaxed);
        self.advance(stop);
    }
    fn memory(&self, group: usize, start: u64, finish: u64, bytes: u64, write: bool) {
        let record = &self.groups[group];
        record.memory_start.fetch_min(start, Relaxed);
        record.memory_finish.fetch_max(finish, Relaxed);
        if write {
            record.write_bytes.fetch_add(bytes, Relaxed);
        } else {
            record.read_bytes.fetch_add(bytes, Relaxed);
        }
    }
}

#[derive(Serialize)]
pub struct PortResult {
    pub spec: PortSpec,
    pub bound: bool,
    pub groups: Vec<GroupResult>,
}
#[derive(Serialize)]
pub struct GroupResult {
    pub requests: Vec<u32>,
    pub first_cycle: Option<u64>,
    pub last_cycle: Option<u64>,
    pub tokens: u64,
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub memory_start: Option<u64>,
    pub memory_finish: Option<u64>,
}

pub struct Session {
    observers: Vec<Arc<Observer>>,
    fingerprint: String,
}
struct BuildState {
    session: Arc<Session>,
    op_id: Option<u32>,
    endpoints: HashMap<(ChannelID, bool), Arc<Observer>>,
}
thread_local! { static BUILD: RefCell<Option<BuildState>> = const { RefCell::new(None) }; }

/// FNV-1a is a stale-sidecar check, not a security digest.
pub fn fingerprint(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    format!("{hash:016x}")
}
impl Session {
    pub fn begin(spec_path: &str, graph_bytes: &[u8]) -> Result<Arc<Self>, String> {
        let bytes = std::fs::read(spec_path).map_err(|e| e.to_string())?;
        let spec: ProfileSpec = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if spec.version != 1 {
            return Err("unsupported request profile version".into());
        }
        if spec.graph_fingerprint != fingerprint(graph_bytes) {
            return Err("request profile does not match graph.pb".into());
        }
        for port in &spec.ports {
            if port.layout.groups.is_empty() || port.layout.groups.iter().any(Vec::is_empty) {
                return Err("empty request group in profile specification".into());
            }
            if port.layout.advance_stop.is_none() && port.layout.groups.len() != 1 {
                return Err("a shared profile port must have one request group".into());
            }
            if port.direction != "input" && port.direction != "output" {
                return Err("invalid profile port direction".into());
            }
        }
        let session = Arc::new(Self {
            fingerprint: spec.graph_fingerprint,
            observers: spec
                .ports
                .into_iter()
                .map(|s| Arc::new(Observer::new(s)))
                .collect(),
        });
        BUILD.with(|b| {
            assert!(b.borrow().is_none(), "nested profile construction");
            *b.borrow_mut() = Some(BuildState {
                session: session.clone(),
                op_id: None,
                endpoints: HashMap::new(),
            });
        });
        Ok(session)
    }
    pub fn write(&self, path: &str, cycles: u64, passed: bool) -> Result<(), String> {
        let mut results = Vec::new();
        for observer in &self.observers {
            let groups = observer
                .groups
                .iter()
                .zip(&observer.spec.layout.groups)
                .map(|(g, requests)| {
                    let first = g.first.load(Relaxed);
                    let memory_start = g.memory_start.load(Relaxed);
                    GroupResult {
                        requests: requests.clone(),
                        first_cycle: (first != UNSET).then_some(first),
                        last_cycle: (first != UNSET).then_some(g.last.load(Relaxed)),
                        tokens: g.tokens.load(Relaxed),
                        read_bytes: g.read_bytes.load(Relaxed),
                        write_bytes: g.write_bytes.load(Relaxed),
                        memory_start: (memory_start != UNSET).then_some(memory_start),
                        memory_finish: (memory_start != UNSET)
                            .then_some(g.memory_finish.load(Relaxed)),
                    }
                })
                .collect();
            results.push(PortResult {
                spec: observer.spec.clone(),
                bound: observer.bound.load(Relaxed),
                groups,
            });
        }
        let report = serde_json::json!({ "version": 1, "graph_fingerprint": self.fingerprint, "cycles": cycles, "passed": passed, "ports": results });
        let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        serde_json::to_writer(std::io::BufWriter::new(file), &report).map_err(|e| e.to_string())
    }
}
/// Scope construction separately from execution. Operator-owned handles keep
/// observers alive after this thread-local registry is cleared.
pub fn end_build() {
    BUILD.with(|b| *b.borrow_mut() = None);
}
pub fn set_current_op(id: u32) {
    BUILD.with(|b| {
        if let Some(s) = b.borrow_mut().as_mut() {
            s.op_id = Some(id);
        }
    });
}
pub fn register_endpoint(
    channel: ChannelID,
    output: bool,
    source_id: u32,
    stream_idx: Option<u32>,
) {
    BUILD.with(|b| {
        if let Some(s) = b.borrow_mut().as_mut() {
            for observer in &s.session.observers {
                let p = &observer.spec;
                if Some(p.op_id) == s.op_id
                    && (p.direction == "output") == output
                    && p.source_id == source_id
                    && p.stream_idx == stream_idx
                {
                    s.endpoints.insert((channel, output), observer.clone());
                }
            }
        }
    });
}
fn bind(channel: ChannelID, output: bool) -> Option<Arc<Observer>> {
    BUILD
        .with(|b| {
            b.borrow()
                .as_ref()
                .and_then(|s| s.endpoints.get(&(channel, output)).cloned())
        })
        .inspect(|o| {
            o.bound.store(true, Relaxed);
        })
}

pub trait ProfileToken {
    fn profile_stop(&self) -> Option<u32>;
}
impl<T> ProfileToken for Elem<T> {
    fn profile_stop(&self) -> Option<u32> {
        match self {
            Elem::Val(_) => None,
            Elem::ValStop(_, s) => Some(*s),
        }
    }
}

pub struct ProfiledReceiver<T: Clone> {
    inner: Receiver<T>,
    observer: Option<Arc<Observer>>,
}
impl<T: DAMType> From<Receiver<T>> for ProfiledReceiver<T> {
    fn from(inner: Receiver<T>) -> Self {
        let observer = bind(inner.id(), false);
        Self { inner, observer }
    }
}
impl<T: Clone> Deref for ProfiledReceiver<T> {
    type Target = Receiver<T>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T: DAMType + ProfileToken> ProfiledReceiver<T> {
    pub fn peek_next(&self, manager: &TimeManager) -> Result<ChannelElement<T>, DequeueError> {
        let value = self.inner.peek_next(manager)?;
        if let Some(o) = &self.observer {
            o.input(
                manager.tick().time().max(value.time.time()),
                value.data.profile_stop(),
                false,
            );
        }
        Ok(value)
    }
    pub fn dequeue(&self, manager: &TimeManager) -> Result<ChannelElement<T>, DequeueError> {
        let value = self.inner.dequeue(manager)?;
        if let Some(o) = &self.observer {
            o.input(
                manager.tick().time().max(value.time.time()),
                value.data.profile_stop(),
                true,
            );
        }
        Ok(value)
    }
    pub fn record_memory_current(&self, start: u64, finish: u64, bytes: u64, write: bool) {
        if let Some(o) = &self.observer {
            o.memory(o.current(), start, finish, bytes, write);
        }
    }
}
impl<T: DAMType + ProfileToken> RecvAdapter<T> for ProfiledReceiver<T> {
    fn attach_receiver(&self, ctx: &dyn Context) {
        self.inner.attach_receiver(ctx);
    }
    fn peek(&self) -> dam::channel::PeekResult<T> {
        self.inner.peek()
    }
    fn peek_next(&self, manager: &TimeManager) -> Result<ChannelElement<T>, DequeueError> {
        self.peek_next(manager)
    }
    fn dequeue(&self, manager: &TimeManager) -> Result<ChannelElement<T>, DequeueError> {
        self.dequeue(manager)
    }
}

pub struct ProfiledSender<T: Clone> {
    inner: Sender<T>,
    observer: Option<Arc<Observer>>,
}
impl<T: DAMType> From<Sender<T>> for ProfiledSender<T> {
    fn from(inner: Sender<T>) -> Self {
        let observer = bind(inner.id(), true);
        Self { inner, observer }
    }
}
impl<T: Clone> Deref for ProfiledSender<T> {
    type Target = Sender<T>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T: DAMType + ProfileToken> ProfiledSender<T> {
    pub fn enqueue(
        &self,
        manager: &TimeManager,
        value: ChannelElement<T>,
    ) -> Result<(), EnqueueError> {
        let stop = value.data.profile_stop();
        let ready = value.time.time();
        self.inner.enqueue(manager, value)?;
        if let Some(o) = &self.observer {
            o.output(manager.tick().time().max(ready), stop);
        }
        Ok(())
    }
    pub fn record_memory(&self, start: u64, finish: u64, bytes: u64, write: bool) {
        if let Some(o) = &self.observer {
            o.memory(o.current(), start, finish, bytes, write);
        }
    }
}

/// Clear construction state even if graph construction unwinds.
pub struct BuildGuard;
impl Drop for BuildGuard {
    fn drop(&mut self) {
        end_build();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observer(groups: Vec<Vec<u32>>, advance_stop: Option<u32>) -> Observer {
        Observer::new(PortSpec {
            op_id: 7,
            direction: "input".into(),
            source_id: 1,
            stream_idx: None,
            aten: vec!["attention".into()],
            layout: Layout {
                groups,
                advance_stop,
            },
            memory: true,
        })
    }

    #[test]
    fn peek_records_start_before_compute_without_counting_twice() {
        let o = observer(vec![vec![0], vec![1]], Some(2));
        o.input(10, None, false);
        o.input(20, None, false);
        o.input(40, None, true);
        o.input(50, Some(2), false);
        o.input(80, Some(2), true);
        o.input(90, Some(3), true);
        assert_eq!(o.groups[0].first.load(Relaxed), 10);
        assert_eq!(o.groups[0].last.load(Relaxed), 50);
        assert_eq!(o.groups[0].tokens.load(Relaxed), 2);
        assert_eq!(o.groups[1].first.load(Relaxed), 90);
        assert_eq!(o.coordinate.load(Relaxed), 2);
    }

    #[test]
    fn ragged_rows_advance_on_stop_rank_not_fixed_tile_count() {
        let o = observer(vec![vec![0], vec![1], vec![2]], Some(2));
        for (time, stop) in [
            (5, None),
            (8, Some(1)),
            (10, Some(2)),
            (30, Some(2)),
            (40, None),
            (42, None),
            (45, Some(3)),
        ] {
            o.memory(o.current(), time - 1, time, 64, false);
            o.output(time, stop);
        }
        assert_eq!(
            o.groups
                .iter()
                .map(|g| g.tokens.load(Relaxed))
                .collect::<Vec<_>>(),
            vec![3, 1, 3]
        );
        assert_eq!(
            o.groups
                .iter()
                .map(|g| g.read_bytes.load(Relaxed))
                .collect::<Vec<_>>(),
            vec![192, 64, 192]
        );
    }

    #[test]
    fn shared_tile_traffic_is_counted_once() {
        let o = observer(vec![vec![0, 1, 2, 3]], None);
        o.memory(o.current(), 0, 10, 4096, false);
        o.output(12, Some(3));
        o.output(24, Some(3));
        assert_eq!(o.coordinate.load(Relaxed), 0);
        assert_eq!(o.groups[0].read_bytes.load(Relaxed), 4096);
        assert_eq!(o.groups[0].first.load(Relaxed), 12);
        assert_eq!(o.groups[0].last.load(Relaxed), 24);
    }

    #[test]
    fn graph_fingerprint_is_stable_and_changes_with_bytes() {
        assert_eq!(fingerprint(b""), "cbf29ce484222325");
        assert_eq!(fingerprint(b"hello"), "a430d84680aabd0b");
        assert_ne!(fingerprint(b"graph1"), fingerprint(b"graph2"));
    }
}
