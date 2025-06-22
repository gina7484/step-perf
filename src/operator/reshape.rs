use crate::memory::PMU_BW;
use crate::primitives::elem::{Bufferizable, Elem, StopType};
use crate::primitives::{select::SelectAdapter, tile::Tile};
use crate::utils::calculation::div_ceil;
use crate::utils::events::LoggableEventSimple;
use dam::{context_tools::*, logging::LogEvent};
use std::marker::PhantomData;

#[context_macro]
pub struct Reshape<E, A: DAMType> {
    in_stream: Receiver<Elem<A>>,
    out_stream: Sender<Elem<A>>,
    
}