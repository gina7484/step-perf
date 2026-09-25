use step_perf::functions::map_fn::{log, softplus};
use step_perf::primitives::tile::Tile;

#[test]
fn preserve_padding_and_timing_only_metadata() {
    let blank = Tile::<f32>::new_blank_padded(vec![3, 4], 4, false, 2);
    for (cycles, output) in [log(&blank, 8, true), softplus(&blank, 16, true)] {
        assert_eq!(cycles, 6);
        assert_eq!(output.shape, vec![3, 4]);
        assert_eq!(output.offset, 2);
        assert_eq!(output.bytes_per_elem, 4);
        assert!(output.read_from_mu);
        assert!(output.underlying.is_none());
    }
    let values = ndarray::arr2(&[[1.0_f32, 2.0], [0.5, 4.0]]);
    let padded = Tile::new_padded(values.to_shared(), 4, true, 1);
    for output in [log(&padded, 8, false).1, softplus(&padded, 8, false).1] {
        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(output.offset, 1);
        assert!(!output.read_from_mu);
        assert!(output.underlying.is_some());
    }
}

#[test]
fn natural_log_and_stable_softplus() {
    let values = ndarray::arr2(&[[1.0_f32, std::f32::consts::E, 0.0, -1.0]]);
    let result = log(&Tile::new(values.to_shared(), 4, false), 16, false).1;
    let arr = result.underlying.unwrap();
    assert_eq!(arr[[0, 0]], 0.0);
    assert!((arr[[0, 1]] - 1.0).abs() < 1e-6);
    assert_eq!(arr[[0, 2]], f32::NEG_INFINITY);
    assert!(arr[[0, 3]].is_nan());

    let values = ndarray::arr2(&[[0.0_f32, -50.0, 100.0, f32::NEG_INFINITY, f32::INFINITY, f32::NAN]]);
    let result = softplus(&Tile::new(values.to_shared(), 4, false), 16, false).1;
    let arr = result.underlying.unwrap();
    assert!((arr[[0, 0]] - std::f32::consts::LN_2).abs() < 1e-7);
    assert!((arr[[0, 1]] / 1.92874985e-22 - 1.0).abs() < 1e-6);
    assert_eq!(arr[[0, 2]], 100.0);
    assert_eq!(arr[[0, 3]], 0.0);
    assert_eq!(arr[[0, 4]], f32::INFINITY);
    assert!(arr[[0, 5]].is_nan());
}
