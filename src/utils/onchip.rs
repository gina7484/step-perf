//! On-chip memory per operator: the most bytes each counted op held at once.
//!
//! `build_from_proto` opens a table with [`begin`], names the operator it is
//! building with [`set_current_op`], and closes the table with [`end`]. A
//! context takes an [`OnChipCounter`] with [`counter`] in its `new()` and
//! reports sizes with [`OnChipCounter::observe`]. Whether an operator counts is
//! decided once from its proto by [`counts`], so contexts never see the proto.
//! The rules are listed in `docs/resource_counting.md`.
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

use crate::primitives::buffer::Buffer;
use crate::primitives::elem::Bufferizable;
use crate::primitives::select::{IndexN, MultiHotN};
use crate::primitives::tile::Tile;
use crate::proto_driver::proto_headers::graph_proto::{
    self as pb, data_type::Type, elemto_elem_func::ElemElemFn, operation::OpType, DataType,
    Operation,
};

/// Keeps the largest size its operator reported. The counter of an operator
/// that doesn't count, or of a context built outside a build, ignores sizes.
#[derive(Clone, Default)]
pub struct OnChipCounter(Option<Arc<AtomicU64>>);

impl OnChipCounter {
    pub fn observe(&self, bytes: usize) {
        if let Some(max) = &self.0 {
            max.fetch_max(bytes as u64, Relaxed);
        }
    }
}

/// Every counter handed out during one build, with its operator id.
#[derive(Default)]
pub struct OnChipTable(Vec<(u32, Arc<AtomicU64>)>);

impl OnChipTable {
    /// Bytes per operator that held anything, read after the run finishes.
    /// An operator built from several counting contexts reports their sum.
    pub fn bytes_by_op(&self) -> HashMap<u32, u64> {
        let mut bytes = HashMap::new();
        for (id, max) in &self.0 {
            let held = max.load(Relaxed);
            if held > 0 {
                *bytes.entry(*id).or_insert(0) += held;
            }
        }
        bytes
    }
}

#[derive(Default)]
struct Build {
    /// The operator being built, and whether it counts.
    current: Option<(u32, bool)>,
    table: OnChipTable,
}

thread_local! {
    static BUILD: RefCell<Option<Build>> = const { RefCell::new(None) };
}

/// Start an empty table for this thread's build, dropping any earlier one.
pub fn begin() {
    BUILD.with(|build| *build.borrow_mut() = Some(Build::default()));
}

/// Name the operator whose contexts are constructed next.
pub fn set_current_op(op: &Operation) {
    BUILD.with(|build| {
        if let Some(build) = build.borrow_mut().as_mut() {
            build.current = Some((op.id, counts(op)));
        }
    });
}

/// A counter for the operator being built, entered in the table if it counts.
pub fn counter() -> OnChipCounter {
    BUILD.with(|build| match build.borrow_mut().as_mut() {
        Some(Build {
            current: Some((id, true)),
            table,
        }) => {
            let max = Arc::new(AtomicU64::new(0));
            table.0.push((*id, Arc::clone(&max)));
            OnChipCounter(Some(max))
        }
        _ => OnChipCounter::default(),
    })
}

/// Close this thread's table and return it.
pub fn end() -> OnChipTable {
    BUILD.with(|build| {
        build
            .borrow_mut()
            .take()
            .map(|build| build.table)
            .unwrap_or_default()
    })
}

/// Whether `op` counts toward on-chip memory, by its kind and declared types.
pub fn counts(op: &Operation) -> bool {
    match op.op_type.as_ref() {
        Some(
            OpType::StridedOffChipLoad(_)
            | OpType::LinearOffChipLoad(_)
            | OpType::LinearOffChipLoadRef(_)
            | OpType::Bufferize(_)
            | OpType::AccumBuffer(_),
        ) => true,
        Some(OpType::Binarymap(map)) => matches!(
            map.func
                .as_ref()
                .and_then(|func| func.elem_elem_fn.as_ref()),
            Some(ElemElemFn::Matmul(_))
        ),
        Some(OpType::Accum(accum)) => !is_scalar(accum.dtype_b.as_ref()),
        // Scan protos without dtype_b keep their state in the input type.
        Some(OpType::Scan(scan)) => !is_scalar(scan.dtype_b.as_ref().or(scan.dtype_a.as_ref())),
        Some(OpType::RepeatStatic(repeat)) => holds_value(repeat.dtype.as_ref()),
        Some(OpType::RepeatRef(repeat)) => holds_value(repeat.dtype.as_ref()),
        Some(OpType::PromoteOuter(promote)) => holds_value(promote.dtype.as_ref()),
        _ => false,
    }
}

/// A repeated or promoted value takes storage unless it is a scalar, or a
/// buffer reference whose storage its producer already counted.
fn holds_value(dtype: Option<&DataType>) -> bool {
    let buffer = matches!(dtype.and_then(|d| d.r#type.as_ref()), Some(Type::Buffer(_)));
    !buffer && !is_scalar(dtype)
}

/// A scalar type, or a tile type declared 1x1.
fn is_scalar(dtype: Option<&DataType>) -> bool {
    macro_rules! one_by_one {
        ($tile:expr, $module:ident) => {
            matches!(
                ($tile.row_size.as_ref(), $tile.col_size.as_ref()),
                (
                    Some(pb::$module::RowSize::RowStatic(1)),
                    Some(pb::$module::ColSize::ColStatic(1))
                )
            )
        };
    }
    match dtype.and_then(|d| d.r#type.as_ref()) {
        Some(Type::ScalarU64(_) | Type::ScalarI64(_) | Type::ScalarBool(_)) => true,
        Some(Type::F16(tile)) => one_by_one!(tile, f16),
        Some(Type::F32(tile)) => one_by_one!(tile, f32),
        Some(Type::Bf16(tile)) => one_by_one!(tile, bf16),
        Some(Type::U64(tile)) => one_by_one!(tile, u64),
        Some(Type::I64(tile)) => one_by_one!(tile, i64),
        Some(Type::Bool(tile)) => one_by_one!(tile, bool),
        _ => false,
    }
}

/// Bytes a value occupies, for contexts generic over what they carry.
pub trait OnChipBytes {
    fn onchip_bytes(&self) -> usize;
}

impl<T> OnChipBytes for Tile<T> {
    fn onchip_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

impl<T: Bufferizable> OnChipBytes for Buffer<T> {
    fn onchip_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

impl OnChipBytes for MultiHotN {
    fn onchip_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

impl OnChipBytes for IndexN {
    fn onchip_bytes(&self) -> usize {
        self.size_in_bytes()
    }
}

// Plain numbers stream through these contexts in their unit tests.
macro_rules! scalar_bytes {
    ($($scalar:ty),*) => {
        $(impl OnChipBytes for $scalar {
            fn onchip_bytes(&self) -> usize {
                std::mem::size_of::<$scalar>()
            }
        })*
    };
}
scalar_bytes!(bool, i32, i64, u32, u64, usize, f32, f64);
