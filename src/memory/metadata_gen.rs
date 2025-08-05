use crate::primitives::{elem::Elem, tile::Tile};
use dam::{context_tools::*, types::DAMType};

use itertools::enumerate;
use ndarray::{Array2, IntoDimension, IxDyn, IxDynImpl};

#[context_macro]
pub struct MetadataGen<T: Clone> {
    pub underlying: ndarray::ArcArray<T, IxDyn>,
    pub snd: Sender<Elem<Tile<u64>>>,
    pub id: u32,
}

impl<T: npyz::Deserialize + Clone + TryInto<u64> + TryFrom<u64> + Send + Sync> MetadataGen<T> {
    pub fn new(npy_path: String, snd: Sender<Elem<Tile<u64>>>, id: u32) -> Self {
        let mut file = std::fs::File::open(npy_path).unwrap();

        // Read the data and shape of the `.npy` file
        let file_data = npyz::NpyFile::new(&mut file).unwrap();
        let shape_vec = file_data
            .shape()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<usize>>();

        let shape: ndarray::Dim<IxDynImpl> = shape_vec.into_dimension();

        let vec_data: Vec<T> = file_data.into_vec().unwrap();
        let underlying = ndarray::ArcArray::from_shape_vec(shape, vec_data).unwrap();

        let ctx = Self {
            underlying,
            snd,
            id,
            context_info: Default::default(),
        };

        ctx.snd.attach_sender(&ctx);

        ctx
    }

    fn get_elem_array(&self) -> Vec<Elem<Tile<u64>>> {
        let mut result = Vec::new();
        let shape = self.underlying.shape();

        // Handle 1D arrays
        if shape.len() == 1 {
            for (i, val) in self.underlying.iter().enumerate() {
                let val_u64 = val
                    .clone()
                    .try_into()
                    .unwrap_or_else(|_| panic!("Error converting T into u64"));
                if i == shape[0] - 1 {
                    result.push(Elem::ValStop(
                        Tile::new(
                            Array2::from_shape_vec((1, 1), vec![val_u64])
                                .unwrap()
                                .to_shared(),
                            8,
                            false,
                        ),
                        1,
                    ));
                } else {
                    result.push(Elem::Val(Tile::new(
                        Array2::from_shape_vec((1, 1), vec![val_u64])
                            .unwrap()
                            .to_shared(),
                        8,
                        false,
                    )));
                }
            }
            return result;
        }

        // Handle 2D and higher dimensional arrays
        let total_elements = self.underlying.len();
        let elements_per_row = shape[1..].iter().product::<usize>();
        let num_rows = shape[0];

        for (i, val) in self.underlying.iter().enumerate() {
            // Convert flat index to multi-dimensional indices
            let mut remaining = i;
            let mut multi_index = vec![0; shape.len()];

            // Calculate multi-dimensional indices
            for dim in (0..shape.len()).rev() {
                multi_index[dim] = remaining % shape[dim];
                remaining /= shape[dim];
            }

            // Determine the highest-dimensional stop token needed
            let mut highest_stop_token: Option<u32> = None;
            let mut all_inner_dims_at_end = true;

            // Check from innermost to outermost
            for dim in (0..shape.len()).rev() {
                // If all inner dimensions are at their end, check this dimension
                if all_inner_dims_at_end {
                    let is_dim_size_one = shape[dim] == 1;
                    let is_last_elem = multi_index[dim] == shape[dim] - 1;

                    // If at end or dim size is 1, update the highest stop token
                    if is_last_elem || is_dim_size_one {
                        highest_stop_token = Some((shape.len() - dim) as u32);
                    }

                    // Update tracking for outer dimensions
                    // Only continue checking outer dimensions if this one is at its last element
                    all_inner_dims_at_end = is_last_elem;
                }
            }

            let val_u64 = val
                .clone()
                .try_into()
                .unwrap_or_else(|_| panic!("Error converting T into u64"));
            if let Some(stop_type) = highest_stop_token {
                result.push(Elem::ValStop(
                    Tile::new(
                        Array2::from_shape_vec((1, 1), vec![val_u64])
                            .unwrap()
                            .to_shared(),
                        8,
                        false,
                    ),
                    stop_type,
                ));
            } else {
                result.push(Elem::Val(Tile::new(
                    Array2::from_shape_vec((1, 1), vec![val_u64])
                        .unwrap()
                        .to_shared(),
                    8,
                    false,
                )));
            }
        }

        result
    }
}

impl<T: npyz::Deserialize + Clone + TryInto<u64> + TryFrom<u64> + Send + Sync> Context
    for MetadataGen<T>
{
    fn run(&mut self) {
        let elems = self.get_elem_array();
        let start_time = self.time.tick();
        for (idx, elem) in enumerate(elems) {
            self.snd
                .enqueue(
                    &self.time,
                    ChannelElement {
                        time: start_time + idx as u64,
                        data: elem,
                    },
                )
                .unwrap();
        }
    }
}

#[cfg(test)]
mod test {
    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{ApproxCheckerContext, GeneratorContext, PrinterContext},
    };

    use crate::primitives::{elem::Elem, tile::Tile};

    use super::MetadataGen;

    #[test]
    fn test_3d() {
        // shape: [2, 3, 2]
        // cargo test --package step_perf --lib -- memory::metadata_gen::test::test_3d --exact --show-output

        /*
        import torch
        import numpy as np
        a = torch.tensor([[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]],dtype=torch.uint64)
        print(a.shape)
        np.save("medatagen_3d.npy",a.detach().numpy())
         */
        let npy_path = "medatagen_3d.npy";

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded::<Elem<Tile<u64>>>();

        ctx.add_child(MetadataGen::<u64>::new(npy_path.to_owned(), in_snd, 0));

        ctx.add_child(PrinterContext::new(in_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }

    #[test]
    fn test_3d_i64() {
        // shape: [4]
        // cargo test --package step_perf --lib -- memory::metadata_gen::test::test_3d_i64 --exact --show-output

        /*
        import torch
        import numpy as np
        a = torch.arange(4)
        print(a.dtype)
        print(a.shape)
        np.save("medatagen_3d_i64.npy",a.detach().numpy())
         */
        let npy_path = "medatagen_3d_i64.npy";

        let mut ctx = ProgramBuilder::default();
        let (in_snd, in_rcv) = ctx.unbounded::<Elem<Tile<u64>>>();

        ctx.add_child(MetadataGen::<u64>::new(npy_path.to_owned(), in_snd, 0));

        ctx.add_child(PrinterContext::new(in_rcv));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
