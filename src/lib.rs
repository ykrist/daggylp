mod graph;
mod path_iis;
mod cycles;
mod mrs;
mod viz;


pub(crate) fn set_with_capacity<K>(n: usize) -> fnv::FnvHashSet<K> {
  fnv::FnvHashSet::with_capacity_and_hasher(n, Default::default())
}
