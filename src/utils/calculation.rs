/// Performs ceiling division of two u64 values
pub fn div_ceil(a: u64, b: u64) -> u64 {
    // Handle division by zero
    assert!(b != 0, "Division by zero");

    // If division is exact, return the result
    if a % b == 0 {
        return a / b;
    }

    // Otherwise, add 1 to the integer division result
    a / b + 1
}
