use dam::types::DAMType;

pub type StopType = u32;

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum Elem<ValType> {
    Val(ValType),
    ValStop(ValType, StopType),
}

impl<V> Elem<V> {
    pub fn into_parts(self) -> (V, StopType) {
        match self {
            Self::Val(value) => (value, 0),
            Self::ValStop(value, stop) => {
                assert!(stop > 0, "use Val for a token without a stop");
                (value, stop)
            }
        }
    }

    pub fn with_stop(value: V, stop: StopType) -> Self {
        if stop == 0 {
            Self::Val(value)
        } else {
            Self::ValStop(value, stop)
        }
    }
}

/// A structural empty fiber retains its payload type and nominal metadata.
/// The enclosing operator determines its rank from the stream and stop token.
pub trait EmptyFiber {
    fn is_empty_fiber(&self) -> bool;
    fn empty_fiber_like(&self) -> Self;
}

/// Default to a default stop type, usually S0.
impl<ValType: Default> Default for Elem<ValType> {
    fn default() -> Self {
        Elem::Val(ValType::default()).into()
    }
}

impl<VT: DAMType> DAMType for Elem<VT> {
    fn dam_size(&self) -> usize {
        match self {
            Elem::Val(v) => v.dam_size(),
            Elem::ValStop(v, s) => v.dam_size() + s.dam_size(),
        }
    }
}

pub trait Bufferizable {
    fn size_in_bytes(&self) -> usize;
    fn read_from_mu(&self) -> bool;
    fn clone_with_updated_read_from_mu(&self, read_from_mu: bool) -> Self;
}
