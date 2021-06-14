mod graph;
mod path_iis;
mod cycles;
mod mrs;
mod viz;
pub mod test_utils;
mod error;

pub use error::*;

pub(crate) fn set_with_capacity<K>(n: usize) -> fnv::FnvHashSet<K> {
  fnv::FnvHashSet::with_capacity_and_hasher(n, Default::default())
}

pub(crate) fn map_with_capacity<K, V>(n: usize) -> fnv::FnvHashMap<K, V> {
  fnv::FnvHashMap::with_capacity_and_hasher(n, Default::default())
}

