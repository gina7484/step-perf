use std::{collections::HashMap, fmt, marker::PhantomData};

use crate::primitives::{buffer::Buffer, elem::Elem, select::MultiHotN, tile::Tile};
use dam::{
    channel::{Receiver, Sender},
    simulation::ProgramBuilder,
    types::DAMType,
};
use derive_more::Constructor;

pub enum ChanType<T: DAMType> {
    Receiver(Receiver<Elem<T>>),
    Sender(Sender<Elem<T>>),
}

impl<T: DAMType> fmt::Debug for ChanType<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChanType::Receiver(_) => write!(f, "Rcv"),
            ChanType::Sender(_) => write!(f, "Snd"),
        }
    }
}

#[derive(Debug)]
pub enum ChannelMapEntry<T: DAMType> {
    Single(ChanType<T>),
    Broadcast(HashMap<u32, ChanType<T>>),
}

pub fn inspect_sender<T: DAMType>(snd: &Sender<T>, target_id: u32) {
    let target_id = format!("Channel({})", target_id);
    let curr_id = format!("{}", snd.id());
    if target_id == curr_id {
        panic!("{} sender found here", target_id);
    }
}

pub fn inspect_receiver<T: DAMType>(
    rcv: &Receiver<T>,
    target_id: u32,
    id: u32,
    idx: Option<u32>,
    location: &str,
) {
    let target_id = format!("Channel({})", target_id);
    let curr_id = format!("{}", rcv.id());
    if target_id == curr_id {
        panic!(
            "{} receiver found {} (op id({}), stream idx({:?}))",
            target_id, location, id, idx
        );
    }
}

const DEFAULT_CHAN_SIZE: usize = 1024;

#[derive(Default, Constructor)]
pub struct ChannelMap<'a, T: DAMType> {
    pub map: Option<HashMap<u32, ChannelMapEntry<T>>>,
    // HashMap between the node id & channel(s) coming out from the node
    // When a channel gets created, either the sender or the receiver side will be
    // consumed, so the map will contain the unconnected side.
    // If the node outputs multiple streams, the ChannelMapEntry will be a map of
    // the unconnected side of each channel.
    _marker: PhantomData<&'a ()>,
}

impl<'a, T: DAMType> ChannelMap<'a, T>
where
    T: 'a,
{
    pub fn inspect(&self) {
        match self.map.as_ref() {
            Some(map) => {
                for (idx, chanmapentr) in map.iter() {
                    match chanmapentr {
                        ChannelMapEntry::Single(chan_type) => match chan_type {
                            ChanType::Receiver(receiver) => {
                                println!("{}: rcv", *idx)
                            }
                            ChanType::Sender(sender) => {
                                println!("{}: snd", *idx)
                            }
                        },
                        ChannelMapEntry::Broadcast(hash_map) => {
                            println!("{}: Broadcast", *idx)
                        }
                    }
                }
            }
            None => println!("None"),
        }
    }
    pub fn get_receiver(
        &mut self,
        id: u32,
        idx: Option<u32>,
        builder: &mut ProgramBuilder<'a>,
        capacity: Option<usize>,
    ) -> Receiver<Elem<T>> {
        // if id == 272 {
        //     println!("get_sender: {:?}", idx);
        //     println!("{:?}", self.map.as_ref().unwrap().get(&id));
        // }
        match &mut self.map {
            Some(chan_map) => match idx {
                Some(stream_idx) => match chan_map.get_mut(&id) {
                    // Broadcast
                    Some(ChannelMapEntry::Broadcast(x)) => match x.remove(&stream_idx) {
                        Some(ChanType::Receiver(rcv)) => rcv,
                        None => {
                            match capacity {
                                Some(cap) => {
                                    let (snd, rcv) = builder.bounded::<Elem<T>>(cap);
                                    //inspect_sender(&snd, 24);
                                    x.insert(stream_idx, ChanType::Sender(snd));
                                    rcv
                                }
                                None => {
                                    // Default capacity
                                    let (snd, rcv) = builder.bounded::<Elem<T>>(DEFAULT_CHAN_SIZE);
                                    x.insert(stream_idx, ChanType::Sender(snd));
                                    rcv
                                }
                            }
                        }
                        _ => panic!("Check whether your id or ChannelMap is correct"),
                    },
                    None => {
                        match capacity {
                            Some(cap) => {
                                let (snd, rcv) = builder.bounded::<Elem<T>>(cap);
                                //inspect_sender(&snd, 24);
                                let mut broadcast_map = HashMap::new();
                                broadcast_map.insert(stream_idx, ChanType::Sender(snd));
                                chan_map.insert(id, ChannelMapEntry::Broadcast(broadcast_map));
                                rcv
                            }
                            None => {
                                // Default capacity
                                let (snd, rcv) = builder.bounded::<Elem<T>>(DEFAULT_CHAN_SIZE);
                                //inspect_sender(&snd, 24);
                                let mut broadcast_map = HashMap::new();
                                broadcast_map.insert(stream_idx, ChanType::Sender(snd));
                                chan_map.insert(id, ChannelMapEntry::Broadcast(broadcast_map));
                                rcv
                            }
                        }
                    }
                    _ => panic!("Check whether your id or ChannelMap is correct"),
                },
                None => match chan_map.remove(&id) {
                    // Single
                    Some(ChannelMapEntry::Single(ChanType::Receiver(x))) => {
                        // inspect_receiver(&x, 141, id, idx, "L122");
                        x
                    }
                    None => {
                        match capacity {
                            Some(cap) => {
                                let (snd, rcv) = builder.bounded::<Elem<T>>(cap);
                                // inspect_sender(&snd, 24);
                                chan_map.insert(id, ChannelMapEntry::Single(ChanType::Sender(snd)));
                                rcv
                            }
                            None => {
                                // Default capacity
                                let (snd, rcv) = builder.bounded::<Elem<T>>(DEFAULT_CHAN_SIZE);
                                // inspect_sender(&snd, 24);
                                chan_map.insert(id, ChannelMapEntry::Single(ChanType::Sender(snd)));
                                rcv
                            }
                        }
                    }
                    _ => panic!("Check whether your id or ChannelMap is correct"),
                },
            },
            None => {
                self.instantiate();
                self.get_receiver(id, idx, builder, capacity)
            }
        }
    }

    pub fn get_sender(
        &mut self,
        id: u32,
        idx: Option<u32>,
        builder: &mut ProgramBuilder<'a>,
        capacity: Option<usize>,
    ) -> Sender<Elem<T>> {
        match &mut self.map {
            Some(chan_map) => match idx {
                Some(stream_idx) => match chan_map.get_mut(&id) {
                    // Broadcast
                    Some(ChannelMapEntry::Broadcast(x)) => match x.remove(&stream_idx) {
                        Some(ChanType::Sender(snd)) => snd,
                        None => {
                            match capacity {
                                Some(cap) => {
                                    let (snd, rcv) = builder.bounded::<Elem<T>>(cap);
                                    // inspect_receiver(&rcv, 229, id, idx, "203");
                                    x.insert(stream_idx, ChanType::Receiver(rcv));
                                    snd
                                }
                                None => {
                                    // Default capacity
                                    let (snd, rcv) = builder.bounded::<Elem<T>>(DEFAULT_CHAN_SIZE);
                                    // inspect_receiver(&rcv, 229, id, idx, "210");
                                    x.insert(stream_idx, ChanType::Receiver(rcv));
                                    snd
                                }
                            }
                        }
                        _ => panic!("Check whether your id or ChannelMap is correct"),
                    },
                    None => {
                        match capacity {
                            Some(cap) => {
                                let (snd, rcv) = builder.bounded::<Elem<T>>(cap);
                                // inspect_receiver(&rcv, 229, id, idx, "L222");
                                let mut broadcast_map = HashMap::new();
                                broadcast_map.insert(stream_idx, ChanType::Receiver(rcv));
                                chan_map.insert(id, ChannelMapEntry::Broadcast(broadcast_map));
                                snd
                            }
                            None => {
                                // Default capacity
                                let (snd, rcv) = builder.bounded::<Elem<T>>(DEFAULT_CHAN_SIZE);
                                // inspect_receiver(&rcv, 229, id, idx, "L231");
                                let mut broadcast_map = HashMap::new();
                                broadcast_map.insert(stream_idx, ChanType::Receiver(rcv));
                                chan_map.insert(id, ChannelMapEntry::Broadcast(broadcast_map));
                                snd
                            }
                        }
                    }
                    _ => panic!("Check whether your id or ChannelMap is correct"),
                },
                None => match chan_map.remove(&id) {
                    // Single
                    Some(ChannelMapEntry::Single(ChanType::Sender(x))) => x,
                    None => {
                        match capacity {
                            Some(cap) => {
                                let (snd, rcv) = builder.bounded::<Elem<T>>(cap);
                                // inspect_receiver(&rcv, 229, id, idx, "L248");
                                chan_map
                                    .insert(id, ChannelMapEntry::Single(ChanType::Receiver(rcv)));
                                snd
                            }
                            None => {
                                // Default capacity
                                let (snd, rcv) = builder.bounded::<Elem<T>>(DEFAULT_CHAN_SIZE);
                                // inspect_receiver(&rcv, 229, id, idx, "L256");
                                chan_map
                                    .insert(id, ChannelMapEntry::Single(ChanType::Receiver(rcv)));
                                snd
                            }
                        }
                    }
                    _ => panic!("Check whether your id or ChannelMap is correct"),
                },
            },
            None => {
                self.instantiate();
                self.get_sender(id, idx, builder, capacity)
            }
        }
    }

    pub fn instantiate(&mut self) {
        self.map = Some(HashMap::new());
    }
}

#[derive(Default)]
pub struct ChannelMapCollection<'a> {
    pub dummy: ChannelMap<'a, ()>,
    // data types
    pub tile_f32: ChannelMap<'a, Tile<f32>>,
    pub tile_u64: ChannelMap<'a, Tile<u64>>,
    // buffered data types
    pub buff_tile_f32: ChannelMap<'a, Buffer<Tile<f32>>>,
    // select
    pub multihot: ChannelMap<'a, MultiHotN>,
    pub u64: ChannelMap<'a, u64>,
    pub bool: ChannelMap<'a, bool>,
}
