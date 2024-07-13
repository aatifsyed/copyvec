//! A stack-allocated sequence that mirror's [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html)'s API,
//! but:
//! - Implements [`Copy`] (and can only hold [`Copy`] types).
//! - Does not grow.
//! - Is `#[no_std]`/no-`alloc` compatible.
//!
//! ```
//! # use copyvec::{CopyVec, copyvec};
//!
//! // const-friendly
//! const VEC: CopyVec<&str, 10> = CopyVec::new();
//!
//! // easy initialising
//! let mut vec = copyvec!["a", "b", "c"; + 2];
//!                                    // ^~ with excess capacity
//!
//! // use the API's you know
//! vec.push("d");
//!
//! // including iteration
//! for it in &mut vec {
//!     if *it == "a" {
//!         *it = "A"
//!     }
//! }
//!
//! assert_eq!(vec, ["A", "b", "c", "d"]);
//! vec.retain(|it| *it == "b" || *it == "c")
//! assert_eq!(vec.remove(0), "b");
//! ```
//!
//! # Other features
//! - [`serde`](https://docs.rs/serde/)
//! - [`quickcheck`](https://docs.rs/quickcheck/)
//!
//!
//! If you like this crate, you may also enjoy [`stackstack`](https://docs.rs/stackstack)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::{borrow::Cow, boxed::Box};

#[cfg(all(feature = "quickcheck1", feature = "alloc"))]
mod _quickcheck1;
#[cfg(feature = "serde1")]
mod _serde1;

use core::{
    array,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    fmt,
    hash::Hash,
    iter::{self, FusedIterator, Take},
    mem::{self, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr,
    slice::{self, SliceIndex},
};

#[cfg(feature = "std")]
use std::io;

/// Create a [`CopyVec`] filled with the given arguments.
///
/// The syntax is similar to [`vec!`](std::vec!)'s,
/// but additional capacity may be specified with `+ <extra>`.
///
/// ```
/// # use copyvec::copyvec;
/// let mut exact = copyvec!["a", "b", "c"];
/// assert_eq!(exact.capacity(), 3);
///
/// let with_spare = copyvec!["a", "b", "c"; + 5];
/// assert_eq!(with_spare.capacity(), 8);
///
/// let exact = copyvec!["a"; 3];
/// assert_eq!(exact, ["a", "a", "a"]);
///
/// let with_spare = copyvec!["a"; 3; + 5];
/// assert_eq!(with_spare, ["a", "a", "a"]);
/// ```
///
/// It may also be used in `const` expressions:
/// ```
/// # use copyvec::copyvec;
/// const _: () = {
///     copyvec!["a", "b", "c"];
///     copyvec!["a", "b", "c"; + 5];
///     copyvec!["a"; 3];
///     copyvec!["a"; 3; + 5];
/// };
/// ```
#[macro_export]
macro_rules! copyvec {
    [$($el:expr),* $(,)?; + $extra:expr] => {
        $crate::__private::from_slice::<_, {
            [
                $($crate::__private::stringify!($el)),*
            ].len() + $extra
        }>(&[
            $($el),*
        ])
    };
    [$($el:expr),* $(,)?] => {
        $crate::CopyVec::from_array([$($el,)*])
    };
    [$fill:expr; $len:expr; + $extra:expr] => {
        $crate::__private::from_slice::<_, { $len + $extra }>(
            &[$fill; $len]
        )
    };
    [$fill:expr; $len:expr] => {
        $crate::CopyVec::from_array([$fill; $len])
    };
}

#[doc(hidden)]
pub mod __private {
    use super::*;
    pub use core::stringify;

    pub const fn from_slice<T, const N: usize>(slice: &[T]) -> CopyVec<T, N>
    where
        T: Copy,
    {
        if slice.len() > N {
            panic!("initializer length is greater than backing storage length")
        }
        let mut buf = [const { MaybeUninit::uninit() }; N];
        let mut ix = slice.len();
        while let Some(nix) = ix.checked_sub(1) {
            ix = nix;
            buf[ix] = MaybeUninit::new(slice[ix]);
        }
        CopyVec {
            occupied: slice.len(),
            buf,
        }
    }
}

/// A contiguous growable array type, with a fixed, stack-alllocated capacity.
#[derive(Copy)]
pub struct CopyVec<T, const N: usize> {
    occupied: usize,
    buf: [MaybeUninit<T>; N],
}

impl<T, const N: usize> CopyVec<T, N> {
    /// Constructs a new, empty `CopyVec<T>`, with space for `N` elements.
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
            buf: [const { MaybeUninit::uninit() }; N],
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
    /// This is always equal to `N` for non-zero sized types.
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
        match T::IS_ZST {
            true => usize::MAX,
            false => N,
        }
    }

    // pub fn reserve(&mut self, additional: usize)
    // pub fn reserve_exact(&mut self, additional: usize)
    // pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError>
    // pub fn try_reserve_exact( &mut self, additional: usize) -> Result<(), TryReserveError>
    // pub fn shrink_to_fit(&mut self)
    // pub fn shrink_to(&mut self, min_capacity: usize)

    /// Converts the vector into [`Box<[T]>`][Box].
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let v = copyvec![1, 2, 3];
    ///
    /// let slice = v.into_boxed_slice();
    /// assert_eq!(&*slice, &[1, 2, 3])
    /// ```
    #[cfg(feature = "alloc")]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        let mut v = alloc::vec::Vec::<T>::with_capacity(self.len());
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), v.as_mut_ptr(), self.len());
            v.set_len(self.len());
            self.set_len(0);
        }
        v.into_boxed_slice()
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater or equal to the vector’s current length, this has no effect.
    ///
    // The [`drain`](Self::drain) method can emulate truncate, but causes the excess elements to be returned instead of dropped.
    //
    /// # Examples
    ///
    /// Truncating a five element vector to two elements:
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3, 4, 5];
    /// vec.truncate(2);
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// No truncation occurs when `len` is greater than the vector's current
    /// length:
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3];
    /// vec.truncate(8);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    ///
    /// Truncating when `len == 0` is equivalent to calling the [`clear`](Self::clear)
    /// method.
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3];
    /// vec.truncate(0);
    /// assert_eq!(vec, []);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        if len > self.occupied {
            return;
        }
        let remaining_len = self.occupied - len;
        unsafe {
            let s = ptr::slice_from_raw_parts_mut(self.as_mut_ptr().add(len), remaining_len);
            self.occupied = len;
            ptr::drop_in_place(s);
        }
    }
    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// use std::io::{self, Write};
    /// let buffer = copyvec![1, 2, 3, 5, 8];
    /// io::sink().write(buffer.as_slice()).unwrap();
    /// ```
    pub const fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// use std::io::{self, Read};
    /// let mut buffer = copyvec![0; 3];
    /// io::repeat(0b101).read_exact(buffer.as_mut_slice()).unwrap();
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
    /// See [`std::vec::Vec::as_ptr`].
    pub const fn as_ptr(&self) -> *const T {
        self.buf.as_ptr().cast()
    }
    /// See [`std::vec::Vec::as_mut_ptr`].
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.as_mut_ptr().cast()
    }
    // pub fn allocator(&self) -> &A

    /// See [`std::vec::Vec::set_len`].
    ///
    /// # Safety
    /// - `new_len` must be less than or equal to [`capacity()`](Self::capacity).
    /// - The elements at `old_len..new_len` must be initialized.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        self.occupied = new_len
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering of the remaining elements, but is *O*(1).
    /// If you need to preserve the element order, use [`remove`](Self::remove) instead.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut v = copyvec!["foo", "bar", "baz", "qux"];
    ///
    /// assert_eq!(v.swap_remove(1), "bar");
    /// assert_eq!(v, ["foo", "qux", "baz"]);
    ///
    /// assert_eq!(v.swap_remove(0), "foo");
    /// assert_eq!(v, ["baz", "qux"]);
    /// ```
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        if index >= len {
            panic!("swap_remove index (is {index}) should be < len (is {len})")
        }
        unsafe {
            let value = ptr::read(self.as_ptr().add(index));
            let base_ptr = self.as_mut_ptr();
            ptr::copy(base_ptr.add(len - 1), base_ptr.add(index), 1);
            self.set_len(len - 1);
            value
        }
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`, or the capacity is exceeded.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3; + 2];
    /// vec.insert(1, 4);
    /// assert_eq!(vec, [1, 4, 2, 3]);
    /// vec.insert(4, 5);
    /// assert_eq!(vec, [1, 4, 2, 3, 5]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*([`len`](Self::len)) time. All items after the insertion index must be
    /// shifted to the right. In the worst case, all elements are shifted when
    /// the insertion index is 0.
    pub fn insert(&mut self, index: usize, element: T)
    where
        T: Copy,
    {
        let len = self.len();
        if index > len {
            panic!("insertion index (is {index}) should be <= len (is {len})",)
        }
        self.push(element);
        self.as_mut_slice()[index..].rotate_right(1)
    }
    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of *O*(*n*). If you don't need the order of elements
    /// to be preserved, use [`swap_remove`](Self::swap_remove) instead.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut v = copyvec![1, 2, 3];
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v, [1, 3]);
    /// ```
    pub fn remove(&mut self, index: usize) -> T {
        let len = self.len();
        if index >= len {
            panic!("removal index (is {index}) should be < len (is {len})");
        }
        unsafe {
            let ret;
            {
                let ptr = self.as_mut_ptr().add(index);
                ret = ptr::read(ptr);
                ptr::copy(ptr.add(1), ptr, len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4];
    /// vec.retain(|&x| x % 2 == 0);
    /// assert_eq!(vec, [2, 4]);
    /// ```
    ///
    /// Because the elements are visited exactly once in the original order,
    /// external state may be used to decide which elements to keep.
    ///
    /// ```
    /// let mut vec = vec![1, 2, 3, 4, 5];
    /// let keep = [false, true, true, false, true];
    /// let mut iter = keep.iter();
    /// vec.retain(|_| *iter.next().unwrap());
    /// assert_eq!(vec, [2, 3, 5]);
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.retain_mut(|it| f(it))
    }

    /// Retains only the elements specified by the predicate, passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e` such that `f(&mut e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3, 4];
    /// vec.retain_mut(|x| if *x <= 3 {
    ///     *x += 1;
    ///     true
    /// } else {
    ///     false
    /// });
    /// assert_eq!(vec, [2, 3, 4]);
    /// ```
    pub fn retain_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let mut retain = [true; N];
        let retain = &mut retain[..self.len()];
        for (it, retain) in iter::zip(self.iter_mut(), &mut *retain) {
            *retain = f(it);
        }
        let mut ct = 0;
        for (ix, retain) in retain.iter().enumerate() {
            if !retain {
                self.remove(ix - ct);
                ct += 1;
            }
        }
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same
    /// key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![10, 20, 21, 30, 20];
    ///
    /// vec.dedup_by_key(|i| *i / 10);
    ///
    /// assert_eq!(vec, [10, 20, 30, 20]);
    /// ```
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.dedup_by(|l, r| key(l) == key(r))
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality
    /// relation.
    ///
    /// The `same_bucket` function is passed references to two elements from the vector and
    /// must determine if the elements compare equal. The elements are passed in opposite order
    /// from their order in the slice, so if `same_bucket(a, b)` returns `true`, `a` is removed.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec!["foo", "bar", "Bar", "baz", "bar"];
    ///
    /// vec.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    ///
    /// assert_eq!(vec, ["foo", "bar", "baz", "bar"]);
    /// ```
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        let mut buf = [const { MaybeUninit::<T>::uninit() }; N];
        let mut occupied = 0;
        let mut iter = self.iter_mut();

        let Some(mut bucket) = iter.next() else {
            return;
        };

        for next in iter {
            match same_bucket(next, bucket) {
                true => continue,
                false => {
                    unsafe { ptr::swap_nonoverlapping(buf[occupied].as_mut_ptr(), bucket, 1) };
                    bucket = next;
                    occupied += 1;
                }
            }
        }
        unsafe { ptr::swap_nonoverlapping(buf[occupied].as_mut_ptr(), bucket, 1) };
        occupied += 1;
        *self = Self { occupied, buf }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    /// - If the underlying storage has been exhausted.
    ///   Use [`Self::push_within_capacity`] or [`Self::try_push`]
    ///   to handle the error instead,
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2; + 1];
    /// vec.push(3);
    /// assert_eq!(vec, [1, 2, 3]);
    /// ```
    pub fn push(&mut self, value: T)
    where
        T: Copy,
    {
        match self.push_within_capacity(value) {
            Ok(()) => {}
            Err(_) => panic!("exceeded fixed capacity {} of vector", self.capacity()),
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an error is returned
    /// with the element.
    ///
    /// See also [`Self::try_push`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::CopyVec;
    /// let mut vec = CopyVec::<_, 1>::new();
    /// vec.push_within_capacity('a').unwrap();
    /// vec.push_within_capacity('b').unwrap_err();
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) time.
    pub fn push_within_capacity(&mut self, value: T) -> Result<(), T>
    where
        T: Copy,
    {
        match self.len() >= self.capacity() {
            true => Err(value),
            false => {
                if !T::IS_ZST {
                    self.buf[self.len()].write(value);
                }
                self.occupied += 1;
                Ok(())
            }
        }
    }
    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3];
    /// assert_eq!(vec.pop(), Some(3));
    /// assert_eq!(vec, [1, 2]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) time.
    pub fn pop(&mut self) -> Option<T> {
        match self.occupied == 0 {
            true => None,
            false => {
                self.occupied -= 1;
                Some(unsafe { ptr::read(self.as_ptr().add(self.len())) })
            }
        }
    }

    /// Removes and returns the last element in a vector if the predicate
    /// returns `true`, or [`None`] if the predicate returns false or the vector
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 3, 4];
    /// let pred = |x: &mut i32| *x % 2 == 0;
    ///
    /// assert_eq!(vec.pop_if(pred), Some(4));
    /// assert_eq!(vec, [1, 2, 3]);
    /// assert_eq!(vec.pop_if(pred), None);
    /// ```
    pub fn pop_if<F>(&mut self, f: F) -> Option<T>
    where
        F: FnOnce(&mut T) -> bool,
    {
        let last = self.last_mut()?;
        if f(last) {
            self.pop()
        } else {
            None
        }
    }

    // pub fn append(&mut self, other: &mut Vec<T, A>)

    // pub fn drain<R>(&mut self, range: R) -> Drain<'_, T, A> ⓘ
    // where
    //     R: RangeBounds<usize>,

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut v = copyvec![1, 2, 3];
    ///
    /// v.clear();
    ///
    /// assert!(v.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.truncate(0)
    }
    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let a = copyvec![1, 2, 3];
    /// assert_eq!(a.len(), 3);
    /// ```
    pub const fn len(&self) -> usize {
        self.occupied
    }
    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::CopyVec;
    /// let mut v = CopyVec::<_, 1>::new();
    /// assert!(v.is_empty());
    ///
    /// v.push(1);
    /// assert!(!v.is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
    // pub fn split_off(&mut self, at: usize) -> Vec<T, A>

    // pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    // where
    //     F: FnMut() -> T,

    // pub fn leak<'a>(self) -> &'a mut [T]

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`](Self::set_len) method.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::CopyVec;
    /// // Vector is big enough for 10 elements.
    /// let mut v = CopyVec::<_, 10>::new();
    ///
    /// // Fill in the first 3 elements.
    /// let uninit = v.spare_capacity_mut();
    /// uninit[0].write(0);
    /// uninit[1].write(1);
    /// uninit[2].write(2);
    ///
    /// // Mark the first 3 elements of the vector as being initialized.
    /// unsafe {
    ///     v.set_len(3);
    /// }
    ///
    /// assert_eq!(&v, &[0, 1, 2]);
    /// ```
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.len()).cast(),
                self.capacity() - self.len(),
            )
        }
    }

    // pub fn split_at_spare_mut(&mut self) -> (&mut [T], &mut [MaybeUninit<T>])
    // pub fn resize(&mut self, new_len: usize, value: T) where T: Clone
    // pub fn extend_from_slice(&mut self, other: &[T]) where T: Clone

    // pub fn extend_from_within<R>(&mut self, src: R)
    // where
    //     R: RangeBounds<usize>,

    // pub fn into_flattened(self) -> Vec<T, A> (for CopyVec<[T; M], N>)

    /// Removes consecutive repeated elements in the vector according to the
    /// [`PartialEq`] trait implementation.
    ///
    /// If the vector is sorted, this removes all duplicates.
    ///
    /// # Examples
    ///
    /// ```
    /// # use copyvec::copyvec;
    /// let mut vec = copyvec![1, 2, 2, 3, 2];
    ///
    /// vec.dedup();
    ///
    /// assert_eq!(vec, [1, 2, 3, 2]);
    /// ```
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|l, r| l == r)
    }
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
        let mut buf = [const { MaybeUninit::uninit() }; N];
        let mut ix = N;
        while let Some(nix) = ix.checked_sub(1) {
            ix = nix;
            buf[ix] = MaybeUninit::new(array[ix])
        }
        Self { occupied: N, buf }
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

/// Error returned from [`CopyVec::try_push`], [`CopyVec::try_extend`] or [`CopyVec::try_from_iter`].
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
        let Self {
            occupied,
            buf: inner,
        } = self;
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

impl<T, U, const N: usize> PartialEq<&[U]> for CopyVec<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<&[U; M]> for CopyVec<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&[U; M]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, U, const N: usize> PartialEq<&mut [U]> for CopyVec<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &&mut [U]) -> bool {
        self.as_slice() == *other
    }
}

impl<T, U, const N: usize> PartialEq<[U]> for CopyVec<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice() == other
    }
}

impl<T, U, const N: usize, const M: usize> PartialEq<[U; M]> for CopyVec<T, N>
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &[U; M]) -> bool {
        self.as_slice() == other
    }
}

impl<T, U, const N: usize> PartialEq<CopyVec<U, N>> for &[T]
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &CopyVec<U, N>) -> bool {
        *self == other.as_slice()
    }
}

impl<T, U, const N: usize> PartialEq<CopyVec<U, N>> for &mut [T]
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &CopyVec<U, N>) -> bool {
        *self == other.as_slice()
    }
}

impl<T, U, const N: usize> PartialEq<CopyVec<U, N>> for [T]
where
    T: PartialEq<U>,
{
    fn eq(&self, other: &CopyVec<U, N>) -> bool {
        self == other.as_slice()
    }
}

#[cfg(feature = "alloc")]
impl<T, U, const N: usize> PartialEq<CopyVec<U, N>> for Cow<'_, [T]>
where
    T: PartialEq<U> + Clone,
{
    fn eq(&self, other: &CopyVec<U, N>) -> bool {
        *self == other.as_slice()
    }
}

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

trait Ext: Sized {
    const IS_ZST: bool = mem::size_of::<Self>() == 0;
}
impl<T> Ext for T {}

#[cfg(all(test, feature = "std", feature = "quickcheck"))]
mod tests {
    use fmt::Debug;
    use quickcheck::Arbitrary;
    use quickcheck1 as quickcheck;

    use super::*;

    #[derive(Debug, Clone)]
    enum Op<T> {
        Push(T),
        Pop,
        Truncate(usize),
        Clear,
        Insert(usize, T),
        SwapRemove(usize),
        Remove(usize),
        Dedup,
        RetainLt(T),
        RetainGt(T),
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
                Op::Insert(usize::arbitrary(g), T::arbitrary(g)),
                Op::SwapRemove(usize::arbitrary(g)),
                Op::Remove(usize::arbitrary(g)),
                Op::Dedup,
                Op::RetainLt(T::arbitrary(g)),
                Op::RetainGt(T::arbitrary(g)),
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

    fn do_test<const N: usize, T: PartialEq + Copy + Debug + PartialOrd>(ops: Vec<Op<T>>) {
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
                Op::Insert(ix, it) => {
                    if ix <= theirs.len() && !theirs.spare_capacity_mut().is_empty() {
                        ours.insert(ix, it);
                        theirs.insert(ix, it);
                    }
                }
                Op::SwapRemove(ix) => {
                    if ix < theirs.len() {
                        assert_eq!(ours.swap_remove(ix), theirs.swap_remove(ix))
                    }
                }
                Op::Remove(ix) => {
                    if ix < theirs.len() {
                        assert_eq!(ours.remove(ix), theirs.remove(ix))
                    }
                }
                Op::Dedup => {
                    ours.dedup();
                    theirs.dedup();
                }
                Op::RetainLt(r) => {
                    ours.retain(|t| *t < r);
                    theirs.retain(|t| *t < r);
                }
                Op::RetainGt(r) => {
                    ours.retain(|t| *t > r);
                    theirs.retain(|t| *t > r);
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
        fn quickcheck_0_u8(ops: Vec<Op<u8>>) -> () {
            do_test::<10, _>(ops)
        }
        fn quickcheck_0_unit(ops: Vec<Op<()>>) -> () {
            do_test::<10, _>(ops)
        }
        fn quickcheck_10_u8(ops: Vec<Op<u8>>) -> () {
            do_test::<10, _>(ops)
        }
        fn quickcheck_10_unit(ops: Vec<Op<()>>) -> () {
            do_test::<10, _>(ops)
        }
    }
}
