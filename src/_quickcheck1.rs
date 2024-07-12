impl<T, const N: usize> quickcheck1::Arbitrary for crate::CopyVec<T, N>
where
    T: Copy + 'static + quickcheck1::Arbitrary,
{
    fn arbitrary(g: &mut quickcheck1::Gen) -> Self {
        let mut this = Self::new();
        for _ in 0..usize::arbitrary(g) {
            if this.push_within_capacity(T::arbitrary(g)).is_err() {
                break;
            }
        }
        this
    }
}
