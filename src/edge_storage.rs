use crate::graph::Edge;
use std::borrow::Cow;
use fnv::{FnvHashSet, FnvHashMap};
use std::iter;
use std::cmp::max;

mod private {
  pub trait Sealed {}
}

use private::Sealed;

pub trait EdgeDir : Sealed + Sized {
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
  type Neigh<'a>: Iterator<Item=&'a Edge>;

  fn neighbours(&self, node: usize) -> Self::Neigh<'_>;

  fn neighbour_nodes(&self, node: usize) -> EdgesToNodes<Self::Neigh<'_>>{
    self.neighbours(node).map(D::traverse_edge)
  }
}


pub trait EdgeLookupBuilder {
  type Finish;
  fn new(num_nodes: usize) -> Self;

  fn size_hint(&mut self, num_edges: usize) {}

  fn add_edge(&mut self, e: Edge);

  fn finish(self) -> Self::Finish;
}



pub trait EdgeLookup: Clone + Neighbours<ForwardDir> + Neighbours<BackwardDir> {
  type Builder: EdgeLookupBuilder<Finish=Self>;

  /// Execute any queued removals.
  fn remove_update<F: FnMut(&Edge) -> bool>(&mut self, should_remove: Option<F>);

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

  fn successors(&self, node: usize) -> <Self as Neighbours<ForwardDir>>::Neigh<'_> {
    <Self as Neighbours<ForwardDir>>::neighbours(self, node)
  }

  fn successor_nodes(&self, node: usize) -> EdgesToNodes<<Self as Neighbours<ForwardDir>>::Neigh<'_>>  {
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


#[derive(Default, Debug, Clone)]
pub struct CsrEdgeStorage {
  edges_from: Vec<Vec<Edge>>,
  edges_to: Vec<Vec<Edge>>,
  edge_removal_queue: FnvHashSet<(usize, usize)>,
}


impl Neighbours<ForwardDir> for CsrEdgeStorage {
  type Neigh<'a> = std::slice::Iter<'a, Edge>;

  fn neighbours(&self, node: usize) -> Self::Neigh<'_> {
    self.edges_from[node].iter()
  }
}

impl Neighbours<BackwardDir> for CsrEdgeStorage {
  type Neigh<'a> = std::slice::Iter<'a, Edge>;

  fn neighbours(&self, node: usize) -> Self::Neigh<'_> {
    self.edges_to[node].iter()
  }
}

impl EdgeLookupBuilder for CsrEdgeStorage {
  type Finish = Self;

  fn new(num_nodes: usize) -> Self {
    CsrEdgeStorage {
      edges_from: vec![Vec::new(); num_nodes],
      edges_to: vec![Vec::new(); num_nodes],
      edge_removal_queue: FnvHashSet::default(),
    }
  }

  fn finish(self) -> Self { self }

  fn add_edge(&mut self, e: Edge) {
    self.edges_to[e.to].push(e);
    self.edges_from[e.from].push(e);
  }
}


impl EdgeLookup for CsrEdgeStorage {
  type Builder = Self;


  fn remove_update<F: FnMut(&Edge) -> bool>(&mut self, should_remove: Option<F>) {
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
      edge_lookup.extend(std::iter::repeat_with(Vec::new).take(num_new_nodes))
    }

    for ((from, to), e) in new_edges {
      self.edges_from[from].push(e);
      self.edges_to[to].push(e);
    }
  }


  fn find_edge(&self, from: usize, to: usize) -> &Edge {
    for e in &self.edges_from[from] {
      if to == e.to {
        return e;
      }
    }
    unreachable!("no edge from {} to {}", from, to)
  }
}