//! Optional debug dump of the graph built by `build_from_proto`.
//!
//! When enabled (via the `dump_prefix` argument threaded down from
//! `run_graph` -> `parse_proto` -> `build_from_proto`), this records, for every
//! node added to the `ProgramBuilder` via `add_child`, its type name + proto op
//! id + DAM identifier, together with the channel IDs (and their entry kind and
//! datatype) of the senders & receivers wired into it.
//!
//! This is meant to help debug `ProgramBuilder::initialize` failures, which
//! report bare `ChannelID`s (`DisconnectedSender`/`DisconnectedReceiver`) with
//! no context about which node owns them.
//!
//! Capture is gated behind a thread-local; when inactive every hook is a cheap
//! no-op. `build_from_proto` runs single-threaded (the simulation only goes
//! multi-threaded later, in `run`), so a thread-local is safe here.

use std::cell::RefCell;

use dam::channel::ChannelID;
use dam::macro_support::Identifiable;

/// Direction of a channel handle relative to the node that holds it.
#[derive(Clone, Copy)]
pub enum Dir {
    /// The node holds the receiver -- this is one of the node's inputs.
    In,
    /// The node holds the sender -- this is one of the node's outputs.
    Out,
}

/// Which `ChannelMapEntry` variant the channel belongs to. In
/// `get_sender`/`get_receiver` this is fully determined by whether a stream
/// index was supplied (`Some` => `Broadcast`, `None` => `Single`); a mismatch
/// already `panic!`s in those functions.
#[derive(Clone, Copy)]
pub enum EntryKind {
    Single,
    Broadcast,
}

impl EntryKind {
    fn from_idx(idx: Option<u32>) -> Self {
        if idx.is_some() {
            EntryKind::Broadcast
        } else {
            EntryKind::Single
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            EntryKind::Single => "Single",
            EntryKind::Broadcast => "Broadcast",
        }
    }
}

struct ChannelRec {
    dir: Dir,
    /// `ChannelID`'s `Display`, e.g. `Channel(7)`.
    id: String,
    kind: EntryKind,
    dtype: &'static str,
}

struct NodeRec {
    /// `<TypeName>_<proto_op_id>`, matching the SVG visualization labels.
    label: String,
    /// The DAM `Identifier` of the node.
    dam_id: usize,
    channels: Vec<ChannelRec>,
}

#[derive(Default)]
struct GraphDump {
    current_op: Option<u32>,
    /// Channels captured for the node currently being built; flushed into the
    /// node's record when it hits `add_child` (see [`record_node`]).
    pending: Vec<ChannelRec>,
    nodes: Vec<NodeRec>,
}

thread_local! {
    static DUMP: RefCell<Option<GraphDump>> = const { RefCell::new(None) };
}

/// Start capturing. Resets any previous state on this thread.
pub fn begin() {
    DUMP.with(|d| *d.borrow_mut() = Some(GraphDump::default()));
}

/// Stop capturing and discard state.
pub fn end() {
    DUMP.with(|d| *d.borrow_mut() = None);
}

/// Set the proto op id of the operation currently being built, used to label
/// nodes as `<TypeName>_<op_id>`. No-op unless capturing is active.
pub fn set_current_op(op_id: u32) {
    DUMP.with(|d| {
        if let Some(g) = d.borrow_mut().as_mut() {
            g.current_op = Some(op_id);
        }
    });
}

/// Clear the current op id, for nodes added outside the per-op loop (e.g. the
/// final `HBMContext`). No-op unless capturing is active.
pub fn clear_current_op() {
    DUMP.with(|d| {
        if let Some(g) = d.borrow_mut().as_mut() {
            g.current_op = None;
        }
    });
}

/// Record a channel handed to the node currently being built. No-op unless
/// capturing is active.
pub fn capture_channel(id: ChannelID, dir: Dir, idx: Option<u32>, dtype: &'static str) {
    DUMP.with(|d| {
        if let Some(g) = d.borrow_mut().as_mut() {
            g.pending.push(ChannelRec {
                dir,
                id: format!("{}", id),
                kind: EntryKind::from_idx(idx),
                dtype,
            });
        }
    });
}

/// Record a node added via `add_child`, attaching every channel captured since
/// the previous node. No-op unless capturing is active.
pub fn record_node<C: Identifiable + ?Sized>(node: &C) {
    DUMP.with(|d| {
        if let Some(g) = d.borrow_mut().as_mut() {
            let label = match g.current_op {
                Some(op) => format!("{}_{}", node.name(), op),
                None => node.name(),
            };
            let channels = std::mem::take(&mut g.pending);
            g.nodes.push(NodeRec {
                label,
                dam_id: node.id().id,
                channels,
            });
        }
    });
}

/// Strip module-path segments from a `std::any::type_name` string, keeping the
/// type names and generic structure: `step_perf::primitives::tile::Tile<f32>`
/// becomes `Tile<f32>`, `..::Buffer<..::Tile<f32>>` becomes `Buffer<Tile<f32>>`.
/// A segment is "module path" if it is an identifier immediately followed by
/// `::`.
fn short_type(full: &str) -> String {
    let mut out = String::with_capacity(full.len());
    let bytes = full.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '_' || c.is_ascii_alphanumeric() {
            let start = i;
            while i < bytes.len() {
                let cc = bytes[i] as char;
                if cc == '_' || cc.is_ascii_alphanumeric() {
                    i += 1;
                } else {
                    break;
                }
            }
            // Drop the identifier (and the `::`) if it's a module-path segment.
            if full[i..].starts_with("::") {
                i += 2;
            } else {
                out.push_str(&full[start..i]);
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// Render the captured nodes into the contents of the "nodes" dump file.
pub fn render_nodes() -> String {
    DUMP.with(|d| {
        let borrowed = d.borrow();
        let Some(g) = borrowed.as_ref() else {
            return String::new();
        };
        let mut out = String::new();
        out.push_str("=== Nodes added via add_child (insertion order) ===\n\n");
        for (seq, node) in g.nodes.iter().enumerate() {
            out.push_str(&format!(
                "#{:<4} {:<28} dam_id={}\n",
                seq, node.label, node.dam_id
            ));
            for c in node.channels.iter().filter(|c| matches!(c.dir, Dir::In)) {
                out.push_str(&format!(
                    "       in : {}  [{}, {}]\n",
                    c.id,
                    c.kind.as_str(),
                    short_type(c.dtype)
                ));
            }
            for c in node.channels.iter().filter(|c| matches!(c.dir, Dir::Out)) {
                out.push_str(&format!(
                    "       out: {}  [{}, {}]\n",
                    c.id,
                    c.kind.as_str(),
                    short_type(c.dtype)
                ));
            }
            out.push('\n');
        }
        out
    })
}
