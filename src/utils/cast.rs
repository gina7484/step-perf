pub fn to_usize_vec<T: TryInto<usize>>(vec: Vec<T>) -> Vec<usize>
where
    <T as TryInto<usize>>::Error: std::fmt::Debug,
{
    vec.into_iter()
        .map(|x| x.try_into().expect("Conversion to usize failed"))
        .collect()
}
