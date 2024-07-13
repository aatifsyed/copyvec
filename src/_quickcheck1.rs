use alloc::{boxed::Box, vec::Vec};

use quickcheck1::Arbitrary;

use crate::CopyVec;

impl<T, const N: usize> quickcheck1::Arbitrary for crate::CopyVec<T, N>
where
    T: Copy + 'static + quickcheck1::Arbitrary,
{
    fn arbitrary(g: &mut quickcheck1::Gen) -> Self {
        vec2copyvec(Arbitrary::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(
            (*self)
                .into_iter()
                .collect::<Vec<T>>()
                .shrink()
                .map(vec2copyvec),
        )
    }
}

fn vec2copyvec<T: Copy, const N: usize>(vec: Vec<T>) -> CopyVec<T, N> {
    let mut this = CopyVec::new();
    let _ignore_too_many = this.try_extend(vec);
    this
}
