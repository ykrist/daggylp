#![feature(generic_associated_types)]
#![feature(option_result_unwrap_unchecked)]
#![allow(warnings)]
mod graph;

pub use graph::{Var, Constraint, Graph, SolveStatus, Weight, InfKind};

pub mod mrs;
#[cfg(any(test, feature = "viz"))]
pub mod viz;
mod iis;
mod error;
mod model_states;

#[cfg(test)]
pub mod test;

mod scc;
pub mod edge_storage;

pub use error::*;
use std::iter::FusedIterator;

pub(crate) fn set_with_capacity<K>(n: usize) -> fnv::FnvHashSet<K> {
  fnv::FnvHashSet::with_capacity_and_hasher(n, Default::default())
}

pub(crate) fn map_with_capacity<K, V>(n: usize) -> fnv::FnvHashMap<K, V> {
  fnv::FnvHashMap::with_capacity_and_hasher(n, Default::default())
}

#[derive(Clone, Debug)]
pub(crate) struct ArrayIntoIter<T, const N: usize> {
  array: [T; N],
  pos: usize,
}

impl<T: Copy, const N: usize> ArrayIntoIter<T, N> {
  pub fn new(array: [T; N]) -> Self {
    ArrayIntoIter { array, pos: 0 }
  }
}

impl<T: Copy, const N: usize> Iterator for ArrayIntoIter<T, N> {
  type Item = T;

  fn next(&mut self) -> Option<T> {
    match self.array.get(self.pos).copied() {
      some @ Some(_) => {
        self.pos += 1;
        some
      }
      None => None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let rem = self.array.len() - self.pos;
    (rem, Some(rem))
  }

  fn nth(&mut self, n: usize) -> Option<T> {
    self.pos += n;
    self.next()
  }
}

impl<T: Copy, const N: usize> ExactSizeIterator for ArrayIntoIter<T, N> {}
impl<T: Copy, const N: usize> FusedIterator for ArrayIntoIter<T, N> {}


// #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
// struct UsizeNiche<const N: usize>(usize);

// struct Index(Option<u32>);

// impl<const N: usize> UsizeNiche<N> {
//   fn assert_not_niche(value: usize) {
//     assert!(value != N, "value is equal to niche value ({})", N)
//   }
//
//   const fn new_none() -> Self {
//     UsizeNiche(N)
//   }
//
//   const unsafe fn new_unchecked(value: usize) -> Self {
//     UsizeNiche(value)
//   }
//
//
//   fn new_some(value: usize) -> Self {
//     Self::assert_not_niche(value);
//     UsizeNiche(value)
//   }
//
//   fn assert_some(&self) {
//
//   }
//
//   fn is_none(&self) -> bool {
//     self.0 == N
//   }
//
//   fn is_some(&self) -> bool {
//     self.0 != N
//   }
//
//   unsafe fn value_unchecked(&self) -> usize {
//     self.0
//   }
//
//   fn to_option(&self) -> Option<usize> {
//     if self.is_none() { None }
//     else { Some(self.0) }
//   }
//
//   fn value(&self) -> usize {
//     Self::assert_not_niche(self.0);
//     self.0
//   }
//
//   unsafe fn map_inplace_unchecked(&mut self, f: impl FnOnce(usize) -> usize) {
//     self.0 = f(self.0);
//   }
//
//   fn map_inplace(&mut self, f: impl FnOnce(usize) -> usize) {
//     if self.is_some() {
//       let v = f(self.0);
//       if v == N {
//         panic!("cannot map to niche value ({})", N);
//       }
//       self.0 = v;
//     }
//   }
//
//
//   fn map(mut self, f: impl FnOnce(usize) -> usize) -> Self {
//     self.map_inplace(f);
//     self
//   }
// }
//
// type IndexNiche = UsizeNiche<{usize::MAX}>;
//

#[cfg(all(test, feature = "test-helpers"))]
mod test_helpers {
  use crate::test;
  #[test]
  fn mark_failed_as_regression() -> anyhow::Result<()> {
      test::mark_failed_as_regression()
  }
}

