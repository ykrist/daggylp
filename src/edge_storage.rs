use crate::graph::{NodeIdx, Edge};
use std::borrow::Cow;
use fnv::{FnvHashSet, FnvHashMap};
use std::iter;
use std::cmp::max;
use std::slice::Iter as SliceIter;

mod private {
  pub trait Sealed {}
}

use private::Sealed;
use std::ops::Deref;
use std::hint::unreachable_unchecked;
use std::mem::MaybeUninit;

pub trait EdgeDir: Sealed + Sized {
  fn new_neighbours_iter<'a, E: Neighbours<Self>>(lookup: &'a E, node: usize) -> E::Neigh<'a> {
    lookup.neighbours(node)
  }

  fn traverse_edge(e: &Edge) -> usize;

  fn is_forwards() -> bool;
}

pub enum ForwardDir {}

impl Sealed for ForwardDir {}

impl EdgeDir for ForwardDir {
  fn traverse_edge(e: &Edge) -> usize { e.to }

  fn is_forwards() -> bool { true }
}

pub enum BackwardDir {}

impl Sealed for BackwardDir {}

impl EdgeDir for BackwardDir {
  fn traverse_edge(e: &Edge) -> usize { e.from }

  fn is_forwards() -> bool { false }
}

type EdgesToNodes<I> = iter::Map<I, fn(&Edge) -> usize>;

pub trait Neighbours<D: EdgeDir> {
  type Neigh<'a>: Iterator<Item= & 'a Edge>;

  fn neighbours(&self, node: usize) -> Self::Neigh<'_>;

  fn neighbour_nodes(&self, node: usize) -> EdgesToNodes<Self::Neigh<'_>> {
    self.neighbours(node).map(D::traverse_edge)
  }
}


pub trait BuildEdgeStorage {
  type Finish;
  fn new(num_nodes: usize) -> Self;

  fn size_hint(&mut self, num_edges: usize) {}

  fn add_edge(&mut self, e: Edge);

  fn finish(self) -> Self::Finish;
}


pub trait EdgeLookup: Clone + Neighbours<ForwardDir> + Neighbours<BackwardDir> {
  type Builder: BuildEdgeStorage<Finish=Self>;
  type AllEdges<'a>: Iterator<Item= & 'a Edge>;
  /// Execute any queued removals.
  fn remove_update<F: Fn(&Edge) -> bool>(&mut self, should_remove: Option<F>);

  /// Remove a single edge.  If [`EdgeLookup::mark_for_removal`] is called, [`EdgeLookup::remove_update`]
  /// is guaranteed  to be called *before* the lookup is used for queries.  This allows the lookup to
  /// remove edges lazily, e.g. via an internal queue.
  fn mark_for_removal(&mut self, from: usize, to: usize);

  /// Remove a set of edges.  The default implementation calls [`EdgeLookup::mark_for_removal`] for every
  /// element in the set. If [`EdgeLookup::mark_for_removal_batch`] is called, [`EdgeLookup::remove_update`]
  /// is guaranteed  to be called *before* the lookup is used for queries.  This allows the lookup to
  /// remove edges lazily, e.g. via an internal queue.
  fn mark_for_removal_batch(&mut self, edges: Cow<FnvHashSet<(usize, usize)>>) {
    for &(from, to) in edges.iter() {
      self.mark_for_removal(from, to)
    }
  }

  fn add_new_component(&mut self, num_nodes: usize, edges: FnvHashMap<(usize, usize), Edge>);

  fn find_edge(&self, from: usize, to: usize) -> &Edge;

  fn all_edges(&self) -> Self::AllEdges<'_>;

  fn successors(&self, node: usize) -> <Self as Neighbours<ForwardDir>>::Neigh<'_> {
    <Self as Neighbours<ForwardDir>>::neighbours(self, node)
  }

  fn successor_nodes(&self, node: usize) -> EdgesToNodes<<Self as Neighbours<ForwardDir>>::Neigh<'_>> {
    <Self as Neighbours<ForwardDir>>::neighbour_nodes(self, node)
  }

  fn predecessors(&self, node: usize) -> <Self as Neighbours<BackwardDir>>::Neigh<'_> {
    <Self as Neighbours<BackwardDir>>::neighbours(self, node)
  }

  fn predecessor_nodes(&self, node: usize) -> EdgesToNodes<<Self as Neighbours<BackwardDir>>::Neigh<'_>> {
    <Self as Neighbours<BackwardDir>>::neighbour_nodes(self, node)
  }

  fn neighbours<D: EdgeDir>(&self, node: usize) -> <Self as Neighbours<D>>::Neigh<'_>
    where Self: Neighbours<D>
  {
    D::new_neighbours_iter(self, node)
  }

  fn neighbour_nodes<D: EdgeDir>(&self, node: usize) -> EdgesToNodes<<Self as Neighbours<D>>::Neigh<'_>>
    where Self: Neighbours<D>
  {
    D::new_neighbours_iter(self, node).map(D::traverse_edge)
  }
}

/// Edges are storage in an array of arrays (two heap indirections in general)
/// This type is parameterised over inner array type, which can be `Vec`,
/// `smallvec::SmallVec` or `arrayvec::ArrayVec`.
#[derive(Default, Debug, Clone)]
pub struct AdjacencyList<V> {
  edges_from: Vec<V>,
  edges_to: Vec<V>,
  edge_removal_queue: FnvHashSet<(usize, usize)>,
}


impl<V: EdgeStorage<Edge>> Neighbours<ForwardDir> for AdjacencyList<V> {
  type Neigh<'a> = std::slice::Iter<'a, Edge>;

  fn neighbours(&self, node: usize) -> Self::Neigh<'_> {
    (&*self.edges_from[node]).iter()
  }
}

impl<V: EdgeStorage<Edge>> Neighbours<BackwardDir> for AdjacencyList<V> {
  type Neigh<'a> = std::slice::Iter<'a, Edge>;

  fn neighbours(&self, node: usize) -> Self::Neigh<'_> {
    self.edges_to[node].iter()
  }
}

impl<V: EdgeStorage<Edge>> BuildEdgeStorage for AdjacencyList<V> {
  type Finish = Self;

  fn new(num_nodes: usize) -> Self {
    AdjacencyList {
      edges_from: vec![V::default(); num_nodes],
      edges_to: vec![V::default(); num_nodes],
      edge_removal_queue: FnvHashSet::default(),
    }
  }

  fn finish(self) -> Self { self }

  fn add_edge(&mut self, e: Edge) {
    self.edges_to[e.to].push(e);
    self.edges_from[e.from].push(e);
  }
}

type CsrAllEdges<'a, V> = iter::FlatMap<
  std::slice::Iter<'a, V>,
  &'a [Edge],
  fn(&'a V) -> &'a [Edge]
>;

impl<V: EdgeStorage<Edge>> EdgeLookup for AdjacencyList<V> {
  type Builder = Self;
  type AllEdges<'a> = CsrAllEdges<'a, V>;

  fn all_edges(&self) -> Self::AllEdges<'_> {
    self.edges_from.iter().flat_map(Deref::deref)
  }

  fn remove_update<F: Fn(&Edge) -> bool>(&mut self, should_remove: Option<F>) {
    // TODO might be better to check which start/end nodes are in edge_remove_queue
    let edges_from = &mut self.edges_from;
    let edges_to = &mut self.edges_to;
    let edge_remove_queue = &self.edge_removal_queue;

    if let Some(mut should_remove) = should_remove {
      for edge_lookup in &mut [edges_from, edges_to] {
        for edgelist in edge_lookup.iter_mut() {
          edgelist.retain(|e| !(
            should_remove(e) || edge_remove_queue.contains(&(e.from, e.to))
          ))
        }
      }
    } else {
      for edge_lookup in &mut [edges_from, edges_to] {
        for edgelist in edge_lookup.iter_mut() {
          edgelist.retain(|e| !edge_remove_queue.contains(&(e.from, e.to)))
        }
      }
    }
    self.edge_removal_queue.clear();
  }

  fn mark_for_removal_batch(&mut self, edges: Cow<FnvHashSet<(usize, usize)>>) {
    if self.edge_removal_queue.capacity() == 0 {
      match edges {
        Cow::Owned(edges) => {
          self.edge_removal_queue = edges;
        }
        Cow::Borrowed(edges) => {
          self.edge_removal_queue = edges.clone();
        }
      }
    } else {
      self.edge_removal_queue.extend(edges.iter())
    }
  }

  fn mark_for_removal(&mut self, from: usize, to: usize) {
    self.edge_removal_queue.insert((from, to));
  }

  fn add_new_component(&mut self, num_new_nodes: usize, new_edges: FnvHashMap<(usize, usize), Edge>) {
    for edge_lookup in &mut [&mut self.edges_from, &mut self.edges_to] {
      edge_lookup.extend(std::iter::repeat_with(V::default).take(num_new_nodes))
    }

    for ((from, to), e) in new_edges {
      self.edges_from[from].push(e);
      self.edges_to[to].push(e);
    }
  }


  fn find_edge(&self, from: usize, to: usize) -> &Edge {
    for e in self.edges_from[from].iter() {
      if to == e.to {
        return e;
      }
    }
    unreachable!("no edge from {} to {}", from, to)
  }
}

pub trait EdgeStorage<T>: Deref<Target=[T]> + Clone + Default + 'static
{
  fn push(&mut self, item: T);

  fn retain(&mut self, f: impl FnMut(&T) -> bool);
}

impl<T: 'static + Clone> EdgeStorage<T> for Vec<T> {
  fn push(&mut self, item: T) {
    Vec::push(self, item);
  }

  fn retain(&mut self, f: impl FnMut(&T) -> bool) {
    Vec::retain(self, f)
  }
}

#[cfg(feature = "smallvec")]
mod smallvec_support {
  use super::*;
  use smallvec::SmallVec;

  impl<T: 'static + Clone, const N: usize> EdgeStorage<T> for SmallVec<[T; N]> {
    fn push(&mut self, item: T) {
      SmallVec::push(self, item);
    }

    fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
      SmallVec::retain(self, move |e: &mut T| f(e))
    }
  }
}

#[cfg(feature = "arrayvec")]
mod arrayvec_support {
  use super::*;
  use arrayvec::ArrayVec;

  impl<T: 'static + Clone, const N: usize> EdgeStorage<T> for ArrayVec<T, N> {
    fn push(&mut self, item: T) {
      ArrayVec::push(self, item);
    }

    fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
      ArrayVec::retain(self, move |e: &mut T| f(e))
    }
  }
}


mod csr_alist {
  use super::*;

  #[derive(Debug, Clone)]
  pub struct CsrAdjacencyListBuilder {
    num_nodes: usize,
    edges: Vec<Option<Edge>>,
  }

  /// Safety: `entries` cannot contain `None` entries.
  unsafe fn sort_and_build_start_len_vec(dest: &mut Vec<(usize, usize)>, entries: &mut [Option<Edge>], num_new_nodes: usize, idx_node: impl Fn(&Edge) -> usize) {
    entries.sort_unstable_by_key(|entry| idx_node(entry.as_ref().unwrap_unchecked()));
    let mut sorted_entries =
      entries.iter().map(|e| idx_node(e.as_ref().unwrap_unchecked()));

    let last_node = num_new_nodes - 1;
    let mut prev_node = dest.len();
    let mut start = 0;
    let mut len = 0;
    for node in sorted_entries {
      len += 1;
      if node != prev_node {
        debug_assert!(node > prev_node);
        let num_skipped = node - prev_node - 1;
        dest.extend(iter::repeat((start, 0)).take(num_skipped));
        dest.push((start, len));
        start += len;
        len = 0;
      }
      prev_node = node;
    }

    debug_assert!(last_node >= prev_node);
    dest.push((start, len));
    debug_assert_eq!(start + len, entries.len());
    dest.extend(iter::repeat((start, 0)).take(last_node - prev_node));
  }

  impl BuildEdgeStorage for CsrAdjacencyListBuilder {
    type Finish = CsrAdjacencyList;

    fn new(num_nodes: usize) -> Self {
      // assume a relatively sparse graph, with not many lonely nodes
      CsrAdjacencyListBuilder { num_nodes, edges: Vec::with_capacity(2 * num_nodes) }
    }

    fn add_edge(&mut self, e: Edge) {
      self.edges.push(Some(e));
    }

    fn finish(self) -> CsrAdjacencyList {
      let CsrAdjacencyListBuilder { mut edges, num_nodes } = self;

      let mut edges_from = edges.clone();
      let mut edges_from_start_and_len = Vec::with_capacity(num_nodes);
      // Safety: we never add None entries to `edges` in the builder
      unsafe { sort_and_build_start_len_vec(&mut edges_from_start_and_len, &mut edges_from, num_nodes, |e| e.from) };

      let mut edges_to = edges;
      let mut edges_to_start_and_len = Vec::with_capacity(num_nodes);
      // Safety: we never add None entries to `edges` in the builder
      unsafe { sort_and_build_start_len_vec(&mut edges_to_start_and_len, &mut edges_to, num_nodes, |e| e.to) };

      CsrAdjacencyList {
        edges_to,
        edges_to_start_and_len,
        edges_from,
        edges_from_start_and_len,
      }
    }
  }

  #[derive(Debug, Clone)]
  pub struct CsrAdjacencyList {
    edges_to: Vec<Option<Edge>>,
    edges_to_start_and_len: Vec<(usize, usize)>,
    edges_from: Vec<Option<Edge>>,
    edges_from_start_and_len: Vec<(usize, usize)>,
  }


  #[inline(always)]
  fn conditionally_remove_entry(entry: &mut Option<Edge>, remove: impl Fn(&Edge) -> bool) {
    if entry.is_some() {
      let e = unsafe { entry.as_ref().unwrap_unchecked() };
      if remove(e) {
        *entry = None;
      }
    }
  }


  impl Neighbours<ForwardDir> for CsrAdjacencyList {
    type Neigh<'a> = iter::FilterMap<SliceIter<'a, Option<Edge>>, fn(&Option<Edge>) -> Option<&Edge>>;

    fn neighbours(&self, node: NodeIdx) -> Self::Neigh<'_> {
      let (start, len) = self.edges_from_start_and_len[node as usize];
      self.edges_from[start..start + len].iter().filter_map(Option::as_ref)
    }
  }

  impl Neighbours<BackwardDir> for CsrAdjacencyList {
    type Neigh<'a> = iter::FilterMap<SliceIter<'a, Option<Edge>>, fn(&Option<Edge>) -> Option<&Edge>>;

    fn neighbours(&self, node: NodeIdx) -> Self::Neigh<'_> {
      let (start, len) = self.edges_to_start_and_len[node as usize];
      self.edges_to[start..start + len].iter().filter_map(Option::as_ref)
    }
  }

  impl EdgeLookup for CsrAdjacencyList {
    type Builder = CsrAdjacencyListBuilder;
    type AllEdges<'a> = iter::FilterMap<SliceIter<'a, Option<Edge>>, fn(&Option<Edge>) -> Option<&Edge>>;

    fn remove_update<F: Fn(&Edge) -> bool>(&mut self, should_remove: Option<F>) {
      if let Some(should_remove) = should_remove {
        let edges_from = &mut self.edges_from;
        let edges_to = &mut self.edges_to;
        for edgelist in &mut [edges_from, edges_to] {
          for entry in edgelist.iter_mut() {
            conditionally_remove_entry(entry, &should_remove);
          }
        }
      }
    }


    fn mark_for_removal(&mut self, from: usize, to: usize) {
      let (start, len) = self.edges_to_start_and_len[to];
      for entry in self.edges_to[start..start + len].iter_mut() {
        conditionally_remove_entry(entry, |e| e.from == e.from);
      }

      let (start, len) = self.edges_from_start_and_len[from];
      for entry in self.edges_from[start..start + len].iter_mut() {
        conditionally_remove_entry(entry, |e| e.to == e.to);
      }
    }

    fn add_new_component(&mut self, num_new_nodes: usize, new_edges: FnvHashMap<(usize, usize), Edge>) {
      let component_start = self.edges_from.len();
      debug_assert_eq!(component_start, self.edges_to.len());
      self.edges_from.extend(new_edges.into_values().map(Some));
      self.edges_from_start_and_len.reserve(num_new_nodes);
      // Safety: the entries we just added from `start` onwards are all `Some(..)`.
      unsafe {
        sort_and_build_start_len_vec(&mut self.edges_from_start_and_len,
                                     &mut self.edges_from[component_start..],
                                     num_new_nodes,
                                     |e| e.from)
      }

      self.edges_to.extend_from_slice(&self.edges_from[component_start..]);
      self.edges_to_start_and_len.reserve(num_new_nodes);
      // Safety: the entries we just added from `start` onwards are all `Some(..)`.
      unsafe {
        sort_and_build_start_len_vec(&mut self.edges_to_start_and_len,
                                     &mut self.edges_to[component_start..],
                                     num_new_nodes,
                                     |e| e.to)
      }
      debug_assert_eq!(self.edges_to.len(), self.edges_from.len());
      debug_assert_eq!(self.edges_to_start_and_len.len(), self.edges_from_start_and_len.len());
    }

    fn all_edges(&self) -> Self::AllEdges<'_> {
      self.edges_to.iter().filter_map(Option::as_ref)
    }

    fn find_edge(&self, from: usize, to: usize) -> &Edge {
      let (start, len) = self.edges_from_start_and_len[from];
      for e in &self.edges_from[start..start + len] {
        match e {
          Some(e) if e.to == to => return e,
          _ => {}
        }
      }
      unreachable!()
    }
  }
}
