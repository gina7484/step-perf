use step_perf::memory::tile_base_addr;

#[test]
fn adjacent_heads_share_rows_not_tile_sized_contiguous_blocks() {
    // [token, 2 heads * 32 columns], 16-token tiles, FP32.
    assert_eq!(tile_base_addr(0, 2, 16, 32, 4), 0);
    assert_eq!(tile_base_addr(1, 2, 16, 32, 4), 128);
    assert_eq!(tile_base_addr(2, 2, 16, 32, 4), 4096);
    assert_eq!(tile_base_addr(3, 2, 16, 32, 4), 4224);
    assert_eq!(tile_base_addr(3, 1, 16, 32, 4), 6144);
}
