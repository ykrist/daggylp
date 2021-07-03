use std::sync::atomic::{AtomicU32, Ordering};
use fnv::{FnvHashSet, FnvHashMap};
use std::hash::Hash;
use crate::graph::EdgeKind::{SccOut, SccIn, SccToScc};
use crate::set_with_capacity;
use crate::iis::Iis;
use crate::scc::SccInfo;

use std::cmp::min;
use std::option::Option::Some;
use std::iter::{ExactSizeIterator, Map};
use crate::graph::ModelState::Unsolved;
use std::borrow::Cow;
use crate::test_utils::NodeData;
use crate::edge_storage::{AdjacencyList, EdgeDir, BuildEdgeStorage};
pub use crate::edge_storage::{EdgeLookup};

pub type Weight = i64;
pub(crate) type NodeIdx = usize;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum InfKind {
  Cycle,
  Path,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum SolveStatus {
  Infeasible(InfKind),
  Optimal,
}

#[derive(Debug, Clone)]
pub(crate) enum ModelState {
  Unsolved,
  InfCycle { sccs: Vec<FnvHashSet<NodeIdx>>, first_inf_scc: usize },
  InfPath(Vec<NodeIdx>),
  Optimal,
  Mrs,
  Dirty { rebuild_sccs: bool },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EdgeKind {
  Regular,
  SccIn(NodeIdx),
  SccOut(NodeIdx),
  SccToScc { from: NodeIdx, to: NodeIdx },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edge {
  pub(crate) from: NodeIdx,
  pub(crate) to: NodeIdx,
  pub(crate) weight: Weight,
  /// used for MRS calculation
  pub(crate) kind: EdgeKind,
}


#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Constraint {
  Ub(Var),
  Lb(Var),
  Edge(Var, Var),
}

#[derive(Hash, Copy, Clone, Debug, Eq, PartialEq)]
pub struct Var {
  pub(crate) graph_id: u32,
  pub(crate) node: NodeIdx,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NodeKind {
  /// Regular node
  Var,
  /// Strongly-connected component Member, contains scc index
  SccMember(NodeIdx),
  /// Artificial SCC node, contains scc index
  Scc(NodeIdx),
}

impl NodeKind {
  // Should this node kind be ignored during forward-labelling
  pub(crate) fn ignored_fwd_label(&self) -> bool {
    self.is_scc_member()
  }

  // Should this node kind be ignored during back-labelling of an acyclic IIS?
  pub(crate) fn ignored_iis_back_label(&self) -> bool {
    self.is_scc()
  }

  pub(crate) fn is_scc_member(&self) -> bool { matches!(self, NodeKind::SccMember(_)) }

  pub(crate) fn is_scc(&self) -> bool { matches!(self, NodeKind::Scc(_)) }
}

#[derive(Debug, Clone)]
pub struct Node {
  pub(crate) x: Weight,
  pub(crate) obj: Weight,
  pub(crate) lb: Weight,
  pub(crate) ub: Weight,
  pub(crate) active_pred: Option<NodeIdx>,
  pub(crate) kind: NodeKind,
}

impl Node {
  fn new(lb: Weight, ub: Weight, obj: Weight, kind: NodeKind) -> Self {
    Node {
      lb,
      ub,
      obj,
      kind,
      active_pred: None,
      x: lb,
    }
  }
}


pub(crate) trait GraphId {
  fn graph_id(&self) -> u32;

  fn var_from_node_id(&self, node: NodeIdx) -> Var {
    Var { node, graph_id: self.graph_id() }
  }
}

macro_rules! impl_graphid {
  ([$($x:tt)*] $($y:tt)*) => {
    impl <$($x)*> GraphId for $($y)* {
      fn graph_id(&self) -> u32 { self.id }
    }
  };

  ($t:path) => {
    impl GraphId for $t {
      fn graph_id(&self) -> u32 { self.id }
    }
  };
}

pub struct Graph<E = AdjacencyList<Vec<Edge>>> {
  id: u32,
  pub(crate) nodes: Vec<Node>,
  pub(crate) sccs: Vec<SccInfo>,
  pub(crate) edges: E,
  pub(crate) state: ModelState,
  first_scc_node: usize,
  // FIXME make sure this is kept up-to-date
}


pub struct GraphNodesBuilder<E> {
  _edges: std::marker::PhantomData<E>,
  id: u32,
  nodes: Vec<Node>,
}

impl_graphid!([E] GraphNodesBuilder<E>);


impl<E: EdgeLookup> GraphNodesBuilder<E> {
  pub fn add_var(&mut self, obj: Weight, lb: Weight, ub: Weight) -> Var {
    assert!(obj >= 0);
    let n = Node::new(lb, ub, obj, NodeKind::Var);
    let v = self.var_from_node_id(self.nodes.len());
    self.nodes.push(n);
    v
  }

  pub fn finish_nodes(self) -> GraphEdgesBuilder<E> {
    let edges = E::Builder::new(self.nodes.len());
    GraphEdgesBuilder {
      id: self.id,
      nodes: self.nodes,
      edges,
    }
  }
}

pub struct GraphEdgesBuilder<E: EdgeLookup> {
  id: u32,
  nodes: Vec<Node>,
  edges: E::Builder,
}

impl_graphid!([E: EdgeLookup] GraphEdgesBuilder<E>);


impl<E: EdgeLookup> GraphEdgesBuilder<E> {
  pub fn num_edges_hint(&mut self, num_edges: usize) {
    self.edges.size_hint(num_edges)
  }

  pub fn add_constr(&mut self, lhs: Var, d: Weight, rhs: Var) {
    assert!(d >= 0, "Constant cannot be negative");
    assert!(lhs != rhs, "Variables must be different");
    let e = Edge {
      from: lhs.node,
      to: rhs.node,
      weight: d,
      kind: EdgeKind::Regular,
    };
    self.edges.add_edge(e);
  }

  pub fn finish(self) -> Graph<E> {
    let first_scc_node = self.nodes.len();
    Graph {
      id: self.id,
      nodes: self.nodes,
      sccs: Vec::new(),
      first_scc_node,
      edges: self.edges.finish(),
      state: Unsolved,
    }
  }
}
// once it's been finalised, not more adding of edges/nodes.  Only modification allowed is to remove
// an edge.


impl_graphid!([E] Graph<E>);

impl<E: EdgeLookup> Graph<E> {
  pub fn new() -> GraphNodesBuilder<E> {
    static NEXT_ID: AtomicU32 = AtomicU32::new(0);
    GraphNodesBuilder {
      _edges: std::marker::PhantomData,
      id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
      nodes: Vec::new(),
    }
  }

  pub(crate) fn reset_nodes(&mut self) {
    for n in self.nodes.iter_mut() {
      n.x = n.lb;
      n.active_pred = None;
    }
  }


  fn update(&mut self) {
    if let ModelState::Dirty { rebuild_sccs } = &self.state {
      let should_remove = if *rebuild_sccs {
        Some(|edge: &Edge| matches!(&edge.kind,
            EdgeKind::SccIn(_)
            | EdgeKind::SccOut(_)
            | EdgeKind::SccToScc { .. }
        ))
      } else {
        None
      };
      self.edges.remove_update(should_remove);
      if *rebuild_sccs {
        self.nodes.truncate(self.first_scc_node);
        for n in self.nodes.iter_mut() {
          debug_assert!(matches!(&n.kind, NodeKind::Var | NodeKind::SccMember(_)));
          n.kind = NodeKind::Var;
        }
      }
      self.state = Unsolved;
    }
  }


  pub fn remove_iis(&mut self, iis: &Iis) {
    self.mark_dirty_batch(iis.edges.iter().copied());
    self.edges.mark_for_removal_batch(Cow::Borrowed(&iis.edges))
  }

  pub fn remove_iis_owned(&mut self, iis: Iis) {
    self.mark_dirty_batch(iis.edges.iter().copied());
    self.edges.mark_for_removal_batch(Cow::Owned(iis.edges));
  }

  pub(crate) fn remove_edge(&mut self, from: NodeIdx, to: NodeIdx) {
    self.mark_dirty(from, to);
    self.edges.mark_for_removal(from, to);
  }

  fn mark_dirty_batch(&mut self, edges: impl Iterator<Item=(NodeIdx, NodeIdx)>) {
    for (from, to) in edges {
      if self.mark_dirty(from, to) {
        break;
      }
    }
  }

  fn mark_dirty(&mut self, from: NodeIdx, to: NodeIdx) -> bool {
    if let ModelState::Dirty { rebuild_sccs } = &self.state {
      if *rebuild_sccs {
        return true;
      }
    }

    let rebuild_sccs = {
      let from_node = &self.nodes[from];
      matches!(&from_node.kind, &NodeKind::SccMember(_)) &&
        &from_node.kind == &self.nodes[to].kind
    };
    self.state = ModelState::Dirty { rebuild_sccs };
    rebuild_sccs
  }

  pub fn solve(&mut self) -> SolveStatus {
    self.update();
    if let Some(state) = self.forward_label() {
      self.state = state;
    } else {
      let sccs = self.find_sccs();
      let inf_idx = sccs.iter().enumerate()
        .find(|(_, scc)| !self.scc_is_feasible(scc))
        .map(|(idx, _)| idx);

      self.state = if let Some(inf_idx) = inf_idx {
        ModelState::InfCycle { sccs, first_inf_scc: inf_idx }
      } else {
        self.condense(sccs);
        self.forward_label().expect("second forward labelling should not find cycles")
      };
    }
    match &self.state {
      ModelState::InfCycle { .. } => SolveStatus::Infeasible(InfKind::Cycle),
      ModelState::InfPath(_) => SolveStatus::Infeasible(InfKind::Path),
      ModelState::Optimal => SolveStatus::Optimal,
      ModelState::Unsolved | ModelState::Mrs | ModelState::Dirty { .. } => unreachable!(),
    }
  }

  pub(crate) fn edge_to_constraint(&self, e: &Edge) -> Constraint {
    Constraint::Edge(self.var_from_node_id(e.from), self.var_from_node_id(e.to))
  }


  fn forward_label<'b>(&'b mut self) -> Option<ModelState> {
    let mut queue = vec![];
    let mut violated_ubs = vec![];
    let mut num_unvisited_pred = vec![0u64; self.nodes.len()];

    // Preprocessing - find source nodes and count number of edges
    for (i, node) in self.nodes.iter().enumerate() {
      if node.kind.is_scc_member() {
        continue;
      }
      self.edges.successors(i).for_each(|e: &Edge| {});
      for j in self.edges.successor_nodes(i) {
        if !self.nodes[j].kind.is_scc_member() {
          num_unvisited_pred[j] += 1;
        }
      }
    }

    for (i, node) in self.nodes.iter_mut().enumerate() {
      node.active_pred = None;
      node.x = node.lb;

      if node.kind.is_scc_member() {
        continue;
      }
      if num_unvisited_pred[i] == 0 {
        queue.push(i);
      }
    }


    // Solve - traverse DAG in topologically-sorted order
    // let nodes = &self.nodes;


    while let Some(i) = queue.pop() {
      let node = &self.nodes[i];
      let x = node.x;
      for e in self.edges.successors(i) {
        let nxt_node = &mut self.nodes[e.to];
        if nxt_node.kind.is_scc_member() {
          continue;
        }
        let nxt_x = x + e.weight;
        if nxt_node.x < nxt_x {
          nxt_node.x = nxt_x;
          nxt_node.active_pred = Some(i);
          if nxt_x > nxt_node.ub {
            violated_ubs.push(e.to);
          }
        }
        num_unvisited_pred[e.to] -= 1;
        if num_unvisited_pred[e.to] == 0 {
          queue.push(e.to)
        }
      }
    }

    // Post-processing: check infeasibility kinds
    if num_unvisited_pred.iter().any(|&cnt| cnt != 0) {
      None
    } else if !violated_ubs.is_empty() {
      Some(ModelState::InfPath(violated_ubs))
    } else {
      // FIXME: move to separate post-processing stage and do lazily
      for scc in &self.sccs {
        for &n in &scc.nodes {
          self.nodes[n].x = self.nodes[scc.scc_node].x;
        }
      }


      Some(ModelState::Optimal)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_utils::*;
  use crate::viz::*;
  use test_case::test_case;
  use SolveStatus::*;
  use InfKind::*;

  // #[test_case("simple-f" => Optimal)]
  // #[test_case("simple-cycle-cei" => Infeasible(Cycle))]
  // #[test_case("simple-cycle-f" => Optimal)]
  // #[test_case("complex-scc-cei" => Infeasible(Cycle))]
  // #[test_case("multiple-sccs-f" => Optimal)]
  // #[test_case("k8-f" => Optimal)]
  // #[test_case("k8-cei" => Infeasible(Cycle))]
  // #[test_case("k8-cbi" => Infeasible(Cycle))]
  // fn solve(input_name: &str) -> SolveStatus {
  //   let mut g = GraphSpec::load_from_file(test_input(input_name)).unwrap().build();
  //   let status = g.solve();
  //   // if draw {
  //   g.viz().save_svg(test_output(&format!("solve-{}.svg", input_name)));
  //   status
  // }
}
