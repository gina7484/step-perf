pub fn to_usize_vec<T: TryInto<usize>>(vec: Vec<T>) -> Vec<usize>
where
    <T as TryInto<usize>>::Error: std::fmt::Debug,
{
    vec.into_iter()
        .map(|x| x.try_into().expect("Conversion to usize failed"))
        .collect()
}

pub fn to_u64_vec<T: TryInto<u64>>(vec: Vec<T>) -> Vec<u64>
where
    <T as TryInto<u64>>::Error: std::fmt::Debug,
{
    vec.into_iter()
        .map(|x| x.try_into().expect("Conversion to usize failed"))
        .collect()
}
