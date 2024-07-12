use core::{fmt, marker::PhantomData};

use serde1::{de::Error as _, ser::SerializeSeq as _, Deserialize, Serialize};

use crate::CopyVec;

impl<'de, T, const N: usize> Deserialize<'de> for CopyVec<T, N>
where
    T: Copy + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde1::Deserializer<'de>,
    {
        struct Visitor<T, const N: usize>(PhantomData<fn() -> [T; N]>);
        impl<'de, T, const N: usize> serde1::de::Visitor<'de> for Visitor<T, N>
        where
            T: Copy + Deserialize<'de>,
        {
            type Value = CopyVec<T, N>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_fmt(format_args!("a sequence of at most {} elements", N))
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde1::de::SeqAccess<'de>,
            {
                let mut this = CopyVec::new();
                while let Some(it) = seq.next_element()? {
                    this.try_push(it).map_err(A::Error::custom)?
                }
                Ok(this)
            }
        }

        deserializer.deserialize_seq(Visitor(PhantomData))
    }
}

impl<T, const N: usize> Serialize for CopyVec<T, N>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde1::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len()))?;
        for it in self {
            seq.serialize_element(it)?
        }
        seq.end()
    }
}
