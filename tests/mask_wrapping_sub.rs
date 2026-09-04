use ndarray::Array2;
use step_perf::{functions::map_fn, primitives::tile::Tile};

#[test]
fn whole_padding_tile_has_zero_valid_columns() {
    let a = Tile::new(Array2::from_elem((1,1), 1u64).to_shared(), 8, false);
    let b = Tile::new(Array2::from_elem((1,1), 16u64).to_shared(), 8, false);
    let (_, count) = map_fn::sub_u64_wrapping(&a, &b, 1, false);
    assert_eq!(count.underlying.as_ref().unwrap()[[0,0]], 1u64.wrapping_sub(16));
    let (_, mask) = map_fn::mask_col::<u64,f32>(&count, false, 1, 16, false);
    assert!(mask.underlying.unwrap().iter().all(|x| *x == 0.0));
}
