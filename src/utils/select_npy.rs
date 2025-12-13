use crate::primitives::{elem::Elem, select::MultiHotN};

/// Reads a multi-hot tensor from an .npy file and returns a Vec of Elem<MultiHotN> instances
/// that encode the tensor structure using stop markers.
///
/// For a tensor of shape [4,6,8]:
/// - Every 6th element gets a stop marker (end of dimension 1)
/// - Every 24th element gets a higher stop marker (end of dimension 2)
/// - The stop type corresponds to the highest dimension that ends at that position
pub fn read_multihot_elem_from_npy<T>(
    file_path: &str,
) -> Result<Vec<Elem<MultiHotN>>, Box<dyn std::error::Error>>
where
    T: npyz::Deserialize + num_traits::Zero,
{
    let mut file = std::fs::File::open(file_path)?;
    let npy_file = npyz::NpyFile::new(&mut file)?;

    let shape_vec = npy_file
        .shape()
        .iter()
        .map(|x| *x as usize)
        .collect::<Vec<usize>>();

    // Validate that we have at least a 1D tensor
    if shape_vec.is_empty() {
        return Err("Expected at least 1D tensor".into());
    }

    let vector_length = shape_vec[shape_vec.len() - 1]; // Length of each multi-hot vector

    // Read the data directly using the generic type
    let vec_data: Vec<T> = npy_file.into_vec()?;

    // Convert to boolean data using Zero trait
    let bool_data: Vec<bool> = vec_data.iter().map(|x| !x.is_zero()).collect();

    // Convert the flat boolean data into MultiHotN vectors
    let multihot_vectors: Vec<MultiHotN> = bool_data
        .chunks(vector_length)
        .map(|chunk| MultiHotN::new(chunk.to_vec(), false))
        .collect();

    // Calculate the number of elements in each dimension (excluding the last dimension)
    let elem_shape: Vec<usize> = if shape_vec.len() > 1 {
        shape_vec[..shape_vec.len() - 1].to_vec()
    } else {
        vec![1] // For 1D case
    };

    // Wrap each MultiHotN in Elem with appropriate stop markers
    let result: Vec<Elem<MultiHotN>> = multihot_vectors
        .into_iter()
        .enumerate()
        .map(|(i, multihot)| {
            let stop_level = get_stop_level(i, &elem_shape);
            match stop_level {
                0 => Elem::Val(multihot),
                level => Elem::ValStop(multihot, level as u32),
            }
        })
        .collect();

    Ok(result)
}

/// Helper function to determine the stop level for a given position in the flattened tensor
/// Returns 0 for no stop, or the highest dimension level that ends at this position
fn get_stop_level(flat_index: usize, shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 0;
    }

    let mut max_stop_level = 0;

    // For each dimension, calculate the period (how often it ends)
    // and check if the current position is at the end of that dimension
    for dim_level in 1..=shape.len() {
        // Calculate how many elements before this dimension repeats
        let period: usize = shape[shape.len() - dim_level..].iter().product();

        // Check if we're at the end of this dimension
        if (flat_index + 1) % period == 0 {
            max_stop_level = dim_level;
        }
    }

    max_stop_level
}

/// Returns an iterator of Elem<MultiHotN> with structure encoding
pub fn read_multihot_elem_from_npy_iter<T>(
    file_path: &str,
) -> Result<impl Iterator<Item = Elem<MultiHotN>>, Box<dyn std::error::Error>>
where
    T: npyz::Deserialize + num_traits::Zero,
{
    let vectors = read_multihot_elem_from_npy::<T>(file_path)?;
    Ok(vectors.into_iter())
}

#[cfg(test)]
mod tests {
    use super::*;

    use dam::{
        simulation::ProgramBuilder,
        utility_contexts::{BroadcastContext, ConsumerContext, GeneratorContext, PrinterContext},
    };

    use crate::{
        primitives::{elem::Elem, select::MultiHotN},
        proto_driver::proto_headers::graph_proto::Broadcast,
    };

    #[test]
    fn test_conversion() {
        let file_path = "./step-perf/select.npy";

        let mut ctx = ProgramBuilder::default();
        let (sel_snd, sel_rcv) = ctx.unbounded::<Elem<MultiHotN>>();

        ctx.add_child(GeneratorContext::new(
            || read_multihot_elem_from_npy_iter::<i64>(file_path).unwrap(),
            sel_snd,
        ));

        let (sel_snd1, sel_rcv1) = ctx.unbounded();
        let (sel_snd2, sel_rcv2) = ctx.unbounded();

        let mut brd = BroadcastContext::new(sel_rcv);
        brd.add_target(sel_snd1);
        brd.add_target(sel_snd2);
        ctx.add_child(brd);

        ctx.add_child(PrinterContext::new(sel_rcv1));
        ctx.add_child(ConsumerContext::new(sel_rcv2));

        ctx.initialize(Default::default())
            .unwrap()
            .run(Default::default());
    }
}
