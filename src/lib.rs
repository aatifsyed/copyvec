//! A stack-allocated sequence that mirror's [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)'s API,
//! but:
//! - Implements [`Copy`] (and can only hold [`Copy`] types).
//! - Does not grow.
//! - Is `#[no_std]`/no-`alloc` compatible.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use core::{
    array,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    fmt,
    hash::Hash,
    iter::{FusedIterator, Take},
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
    slice::{self, SliceIndex},
};

#[cfg(feature = "std")]
use std::io;

/// A contiguous growable array type, with a fixed, stack-alllocated capacity.
#[derive(Copy)]
pub struct CopyVec<T, const N: usize> {
    occupied: usize,
    inner: [MaybeUninit<T>; N],
}

impl<T, const N: usize> CopyVec<T, N> {
    /// Constructs a new, empty `CopyVec<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::CopyVec;
    /// let mut vec: CopyVec<i32, 10> = CopyVec::new();
    /// ```
    pub const fn new() -> Self {
        Self {
            occupied: 0,
            inner: [const { MaybeUninit::uninit() }; N],
        }
    }

    // pub fn with_capacity(capacity: usize) -> Vec<T>
    // pub fn try_with_capacity(capacity: usize) -> Result<Vec<T>, TryReserveError>
    // pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize ) -> Vec<T>
    // pub const fn new_in(alloc: A) -> Vec<T, A>
    // pub fn with_capacity_in(capacity: usize, alloc: A) -> Vec<T, A>
    // pub fn try_with_capacity_in(capacity: usize, alloc: A) -> Result<Vec<T, A>, TryReserveError>
    // pub unsafe fn from_raw_parts_in(ptr: *mut T, length: usize, capacity: usize, alloc: A) -> Vec<T, A>
    // pub fn into_raw_parts(self) -> (*mut T, usize, usize)
    // pub fn into_raw_parts_with_alloc(self) -> (*mut T, usize, usize, A)

    /// Returns the total number of elements the vector can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::CopyVec;
    /// let mut vec: CopyVec<i32, 10> = CopyVec::new();
    /// vec.push(42);
    /// assert_eq!(vec.capacity(), 10);
    /// ```
    pub const fn capacity(&self) -> usize {
        N
    }

    // pub fn reserve(&mut self, additional: usize)
    // pub fn reserve_exact(&mut self, additional: usize)
    // pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>
    // pub fn try_reserve_exact( &mut self, additional: usize) -> Result<(), TryReserveError>
    // pub fn shrink_to_fit(&mut self)
    // pub fn shrink_to(&mut self, min_capacity: usize)

    /// Converts the vector into [`Box<[T]>`](Box).
    #[cfg(feature = "alloc")]
    pub fn into_boxed_slice(self) -> Box<[T]>
    where
        T: Clone, // TODO(0xaatif): https://github.com/rust-lang/rust/issues/63291
                  //                remove this bound
    {
        self.as_slice().into()
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater or equal to the vector’s current length, this has no effect.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.occupied = len;
            for i in len..self.len() {
                unsafe { self.inner[i].assume_init_drop() };
            }
        }
    }
    const fn slice_parts(&self) -> (*const T, usize) {
        (self.inner.as_ptr().cast(), self.occupied)
    }
    pub const fn as_slice(&self) -> &[T] {
        let (data, len) = self.slice_parts();
        unsafe { slice::from_raw_parts(data, len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let (data, len) = self.slice_parts();
        unsafe { slice::from_raw_parts_mut(data.cast_mut(), len) }
    }
    // pub fn as_ptr(&self) -> *const T
    // pub fn as_mut_ptr(&mut self) -> *mut T
    // pub fn allocator(&self) -> &A
    // pub unsafe fn set_len(&mut self, new_len: usize)
    // pub fn swap_remove(&mut self, index: usize) -> T
    // pub fn insert(&mut self, index: usize, element: T)
    // pub fn remove(&mut self, index: usize) -> T

    // pub fn retain<F>(&mut self, f: F)
    // where
    //     F: FnMut(&T) -> bool,

    // pub fn retain_mut<F>(&mut self, f: F)
    // where
    //     F: FnMut(&mut T) -> bool,

    // pub fn dedup_by_key<F, K>(&mut self, key: F)
    // where
    //     F: FnMut(&mut T) -> K,
    //     K: PartialEq,

    // pub fn dedup_by<F>(&mut self, same_bucket: F)
    // where
    //     F: FnMut(&mut T, &mut T) -> bool,

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    /// - If the underlying storage has been exhausted.
    ///   Use [`Self::push_within_capacity`] to handle the error instead.
    pub fn push(&mut self, value: T)
    where
        T: Copy,
    {
        match self.push_within_capacity(value) {
            Ok(()) => {}
            Err(_) => panic!("exceeded capacity of vector"),
        }
    }

    pub fn push_within_capacity(&mut self, value: T) -> Result<(), T>
    where
        T: Copy,
    {
        match self.len() > self.capacity() {
            true => Err(value),
            false => {
                self.inner[self.len()].write(value);
                self.occupied += 1;
                Ok(())
            }
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        match self.occupied == 0 {
            true => None,
            false => {
                self.occupied -= 1;
                Some(unsafe { self.inner[self.len()].assume_init_read() })
            }
        }
    }

    // pub fn pop_if<F>(&mut self, f: F) -> Option<T>
    // where
    //     F: FnOnce(&mut T) -> bool,

    // pub fn append(&mut self, other: &mut Vec<T, A>)

    // pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, A> ⓘ
    // where
    //     R: RangeBounds<usize>,

    pub fn clear(&mut self) {
        self.truncate(0)
    }
    pub const fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub const fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
    // pub fn split_off(&mut self, at: usize) -> Vec<T, A>

    // pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    // where
    //     F: FnMut() -> T,

    // pub fn leak<'a>(self) -> &'a mut [T]
    // pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>]
    // pub fn split_at_spare_mut(&mut self) -> (&mut [T], &mut [MaybeUninit<T>])
    // pub fn resize(&mut self, new_len: usize, value: T) where T: Clone
    // pub fn extend_from_slice(&mut self, other: &[T]) where T: Clone

    // pub fn extend_from_within<R>(&mut self, src: R)
    // where
    //     R: RangeBounds<usize>,

    // pub fn into_flattened(self) -> Vec<T, A> (for CopyVec<[T; M], N>)
    // pub fn dedup(&mut self) where T: PartialEq
    // pub fn splice<R, I>(&mut self, range: R, replace_with: I) -> Splice<'_, <I as IntoIterator>::IntoIter, A>

    // pub fn extract_if<F>(&mut self, filter: F) -> ExtractIf<'_, T, F, A> ⓘ
    // where
    //     F: FnMut(&mut T) -> bool,
}

/// Methods that don't mirror [`std::vec::Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)'s API.
impl<T, const N: usize> CopyVec<T, N> {
    /// Create a full [`CopyVec`] from an array.
    pub const fn from_array(array: [T; N]) -> Self
    where
        T: Copy,
    {
        let mut inner = [const { MaybeUninit::uninit() }; N];
        let mut ix = N;
        while let Some(nix) = ix.checked_sub(1) {
            ix = nix;
            inner[ix] = MaybeUninit::new(array[ix])
        }
        Self { occupied: N, inner }
    }
    /// Like [`Self::push_within_capacity`], but returns a [`std::error::Error`].
    pub fn try_push(&mut self, value: T) -> Result<(), Error>
    where
        T: Copy,
    {
        self.push_within_capacity(value).map_err(|_| Error {
            capacity: self.capacity(),
            excess: None,
        })
    }
    /// Fallible verson of [`FromIterator`], since `N` is expected to be small.
    pub fn try_from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self, Error>
    where
        T: Copy,
    {
        let mut this = Self::new();
        this.try_extend(iter)?;
        Ok(this)
    }
    /// Fallible version of [`Extend`], since `N` is expected to be small.
    pub fn try_extend<I: IntoIterator<Item = T>>(&mut self, iter: I) -> Result<(), Error>
    where
        T: Copy,
    {
        let mut excess = 0;
        for it in iter {
            match self.push_within_capacity(it) {
                Ok(()) => {}
                Err(_) => excess += 1,
            }
        }
        match excess == 0 {
            true => Ok(()),
            false => Err(Error {
                capacity: self.capacity(),
                excess: Some(excess),
            }),
        }
    }
}

pub struct Error {
    capacity: usize,
    /// [`None`] if returned from [`CopyVec::try_push`].
    excess: Option<usize>,
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Error")
            .field(&DebugWithDisplay(self))
            .finish()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self { capacity, excess } = self;
        match excess {
            Some(excess) => f.write_fmt(format_args!(
                "exceed fixed capacity of {} by {} elements",
                capacity, excess
            )),
            None => f.write_fmt(format_args!("exceeded fixed capacity of {}", capacity)),
        }
    }
}

struct DebugWithDisplay<T>(T);
impl<T> fmt::Debug for DebugWithDisplay<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        T::fmt(&self.0, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl<T, const N: usize> AsMut<[T]> for CopyVec<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
impl<T, const N: usize> AsMut<CopyVec<T, N>> for CopyVec<T, N> {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<T, const N: usize> AsRef<[T]> for CopyVec<T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T, const N: usize> AsRef<CopyVec<T, N>> for CopyVec<T, N> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T, const N: usize> Borrow<[T]> for CopyVec<T, N> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T, const N: usize> BorrowMut<[T]> for CopyVec<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize> Clone for CopyVec<T, N>
where
    T: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, const N: usize> fmt::Debug for CopyVec<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl<T, const N: usize> Default for CopyVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Deref for CopyVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<T, const N: usize> DerefMut for CopyVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

// impl<'a, T, A> Extend<&'a T> for Vec<T, A>
// where
//     T: Copy + 'a,
//     A: Allocator,

// impl<T, A> Extend<T> for Vec<T, A>
// where
//     A: Allocator,

// impl<T> From<&[T]> for Vec<T>
// where
//     T: Clone,

// impl<T, const N: usize> From<&[T; N]> for Vec<T>
// where
//     T: Clone,

// impl<'a, T> From<&'a Vec<T>> for Cow<'a, [T]>
// where
//     T: Clone,

// impl<T> From<&mut [T]> for Vec<T>
// where
//     T: Clone,

// impl<T, const N: usize> From<&mut [T; N]> for Vec<T>
// where
//     T: Clone,

// impl From<&str> for Vec<u8>

// impl<T, const N: usize> From<[T; N]> for Vec<T>

// impl<T, A> From<BinaryHeap<T, A>> for Vec<T, A>
// where
//     A: Allocator,

// impl<T, A> From<Box<[T], A>> for Vec<T, A>
// where
//     A: Allocator,

// impl From<CString> for Vec<u8>

// impl<'a, T> From<Cow<'a, [T]>> for Vec<T>
// where
//     [T]: ToOwned<Owned = Vec<T>>,

// impl From<String> for Vec<u8>

// impl From<Vec<NonZero<u8>>> for CString

// impl<'a, T> From<Vec<T>> for Cow<'a, [T]>
// where
//     T: Clone,

// impl<T, A> From<Vec<T, A>> for Arc<[T], A>
// where
//     A: Allocator + Clone,

// impl<T, A> From<Vec<T, A>> for BinaryHeap<T, A>
// where
//     T: Ord,
//     A: Allocator,

// impl<T, A> From<Vec<T, A>> for Box<[T], A>
// where
//     A: Allocator,

// impl<T, A> From<Vec<T, A>> for Rc<[T], A>
// where
//     A: Allocator,

// impl<T, A> From<Vec<T, A>> for VecDeque<T, A>
// where
//     A: Allocator,

// impl<T, A> From<VecDeque<T, A>> for Vec<T, A>
// where
//     A: Allocator,

// impl<T> FromIterator<T> for Vec<T>

impl<T, const N: usize> Hash for CopyVec<T, N>
where
    T: Hash,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T, I, const N: usize> Index<I> for CopyVec<T, N>
where
    I: SliceIndex<[T]>,
{
    type Output = <I as SliceIndex<[T]>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T, I, const N: usize> IndexMut<I> for CopyVec<T, N>
where
    I: SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a CopyVec<T, N> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut CopyVec<T, N> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

impl<T, const N: usize> IntoIterator for CopyVec<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;
    fn into_iter(self) -> Self::IntoIter {
        let Self { occupied, inner } = self;
        IntoIter {
            inner: inner.into_iter().take(occupied),
        }
    }
}

/// Consuming iterator for a [`CopyVec`].
pub struct IntoIter<T, const N: usize> {
    inner: Take<array::IntoIter<MaybeUninit<T>, N>>,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|it| unsafe { it.assume_init() })
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth(n).map(|it| unsafe { it.assume_init() })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.next_back().map(|it| unsafe { it.assume_init() })
    }
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.nth_back(n).map(|it| unsafe { it.assume_init() })
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}
impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> Ord for CopyVec<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

// impl<T, U, A> PartialEq<&[U]> for Vec<T, A>
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A, const N: usize> PartialEq<&[U; N]> for Vec<T, A>
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A> PartialEq<&mut [U]> for Vec<T, A>
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A> PartialEq<[U]> for Vec<T, A>
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A, const N: usize> PartialEq<[U; N]> for Vec<T, A>
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A> PartialEq<Vec<U, A>> for &[T]
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A> PartialEq<Vec<U, A>> for &mut [T]
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A> PartialEq<Vec<U, A>> for [T]
// where
//     A: Allocator,
//     T: PartialEq<U>,

// impl<T, U, A> PartialEq<Vec<U, A>> for Cow<'_, [T]>
// where
//     A: Allocator,
//     T: PartialEq<U> + Clone,

// impl<T, U, A> PartialEq<Vec<U, A>> for VecDeque<T, A>
// where
//     A: Allocator,
//     T: PartialEq<U>,

impl<T, U, const N: usize, const M: usize> PartialEq<CopyVec<U, M>> for CopyVec<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &CopyVec<U, M>) -> bool {
        self.as_slice().eq(other.as_slice())
    }
}

impl<T, const N: usize, const M: usize> PartialOrd<CopyVec<T, M>> for CopyVec<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &CopyVec<T, M>) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

// impl<T, const N: usize> TryFrom<Vec<T>> for Box<[T; N]>

// impl<T, A, const N: usize> TryFrom<Vec<T, A>> for [T; N]
// where
//     A: Allocator,

#[cfg(feature = "std")]
impl<const N: usize> io::Write for CopyVec<u8, N> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut written = 0;
        for it in buf {
            match self.try_push(*it) {
                Ok(()) => written += 1,
                Err(_) => return Ok(written),
            }
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self.try_extend(buf.iter().copied()) {
            Ok(()) => Ok(()),
            Err(e) => Err(io::Error::new(io::ErrorKind::WriteZero, e)),
        }
    }
}

// impl<T, A> DerefPure for Vec<T, A>
// where
//     A: Allocator,

impl<T, const N: usize> Eq for CopyVec<T, N> where T: Eq {}

#[cfg(all(test, feature = "std"))]
mod tests {
    use fmt::Debug;
    use quickcheck::Arbitrary;

    use super::*;

    #[derive(Debug, Clone)]
    enum Op<T> {
        Push(T),
        Pop,
        Truncate(usize),
        Clear,
    }

    impl<T> Arbitrary for Op<T>
    where
        T: Clone + Arbitrary,
    {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let options = [
                Op::Push(T::arbitrary(g)),
                Op::Pop,
                Op::Truncate(usize::arbitrary(g)),
                Op::Clear,
            ];
            g.choose(&options).unwrap().clone()
        }
    }

    fn check_invariants<const N: usize, T: PartialEq + Debug>(
        ours: &mut CopyVec<T, N>,
        theirs: &mut Vec<T>,
    ) {
        assert!(ours.iter().eq(theirs.iter()));
        assert!(ours.iter_mut().eq(theirs.iter_mut()));
        assert_eq!(ours.capacity(), theirs.capacity());
        assert_eq!(ours.len(), theirs.len());
        assert_eq!(ours.as_slice(), theirs.as_slice());
    }

    fn do_test<const N: usize, T: PartialEq + Copy + Debug>(ops: Vec<Op<T>>) {
        let mut ours = CopyVec::<T, N>::new();
        let mut theirs = Vec::new();
        theirs.reserve_exact(N);
        check_invariants(&mut ours, &mut theirs);
        for op in ops {
            match op {
                Op::Push(it) => {
                    assert_eq!(
                        ours.push_within_capacity(it),
                        push_within_capacity(&mut theirs, it)
                    );
                }
                Op::Pop => {
                    assert_eq!(ours.pop(), theirs.pop())
                }
                Op::Truncate(u) => {
                    ours.truncate(u);
                    theirs.truncate(u)
                }
                Op::Clear => {
                    ours.clear();
                    theirs.clear();
                }
            }
            check_invariants(&mut ours, &mut theirs)
        }
    }

    fn push_within_capacity<T>(v: &mut Vec<T>, value: T) -> Result<(), T> {
        match v.spare_capacity_mut().is_empty() {
            true => Err(value),
            false => {
                v.push(value);
                Ok(())
            }
        }
    }

    quickcheck::quickcheck! {
        fn quickcheck_10_u8(ops: Vec<Op<u8>>) -> () {
            do_test::<10, _>(ops)
        }
    }
}
