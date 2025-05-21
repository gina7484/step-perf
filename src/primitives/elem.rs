use dam::types::DAMType;

pub type StopType = u32;

#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub enum Elem<ValType> {
    Val(ValType),
    ValStop(ValType, StopType),
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
            Elem::ValStop(_v, s) => s.dam_size(),
        }
    }
}
