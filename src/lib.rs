#![feature(generic_associated_types)]
#![feature(option_result_unwrap_unchecked)]
#![feature(array_windows)]
#![allow(warnings)]
mod graph;

pub use graph::{Constraint, Graph, InfKind, SolveStatus, Var, Weight};

mod iis;
pub mod mrs;
#[cfg(any(test, feature = "viz"))]
pub mod viz;
pub use iis::Iis;

mod error;
mod model_states;
pub(crate) use model_states::*;

#[cfg(test)]
pub mod test;

pub mod edge_storage;
mod scc;

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
      None => None,
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

#[cfg(all(test, feature = "test-helpers"))]
mod test_helpers {
  use crate::test::GraphData;
  use crate::viz::GraphViz;
  use crate::{test, Graph, SolveStatus::*};
  use anyhow::Context;

  #[test]
  fn mark_failed_as_regression() -> anyhow::Result<()> {
    test::mark_failed_as_regression()
  }
}
