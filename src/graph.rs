use std::sync::atomic::{AtomicU32, Ordering};
use fnv::{FnvHashSet, FnvHashMap};
use std::hash::Hash;
use crate::graph::EdgeKind::{SccOut, SccIn, SccToScc};
use crate::set_with_capacity;
use crate::iis::Iis;
use crate::scc::SccInfo;

use std::cmp::min;
use std::option::Option::Some;
use std::iter::ExactSizeIterator;


pub type Weight = i64;

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
  InfCycle { sccs: Vec<FnvHashSet<usize>>, first_inf_scc: usize },
  InfPath(Vec<usize>),
  Optimal,
  Mrs
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EdgeKind {
  Regular,
  SccIn(usize),
  SccOut(usize),
  SccToScc{ from: usize, to: usize }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edge {
  pub(crate) from: usize,
  pub(crate) to: usize,
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
  graph_id: u32,
  node: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum NodeKind {
  /// Regular node
  Var,
  /// Strongly-connected component Member, contains scc index
  SccMember(usize),
  /// Artificial SCC node, contains scc index
  Scc(usize),
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

  pub(crate) fn is_scc_member(&self) -> bool { matches!(self, NodeKind::SccMember(_))}

  pub(crate) fn is_scc(&self) -> bool { matches!(self, NodeKind::Scc(_))}
}

#[derive(Debug, Clone)]
pub struct Node {
  pub(crate) x: Weight,
  pub(crate) obj: Weight,
  pub(crate) lb: Weight,
  pub(crate) ub: Weight,
  pub(crate) active_pred: Option<usize>,
  pub(crate) kind: NodeKind,
}


pub struct Graph {
  pub(crate) id: u32,
  pub(crate) nodes: Vec<Node>,
  pub(crate) sccs: Vec<SccInfo>,
  pub(crate) num_active_nodes: usize,
  pub(crate) edges_from: Vec<Vec<Edge>>,
  pub(crate) edges_to: Vec<Vec<Edge>>,
  pub(crate) source_nodes: Vec<usize>,
  pub(crate) parameters: Parameters,
  pub(crate) state: ModelState,
}

#[derive(Copy, Clone, Debug)]
pub struct Parameters {
  pub(crate) minimal_cyclic_iis: bool,
  pub(crate) minimal_acyclic_iis: bool,
  pub(crate) size_hint_vars: usize,
  pub(crate) size_hint_constrs: usize,
}

impl Default for Parameters {
  fn default() -> Self {
    Parameters {
      minimal_cyclic_iis: false,
      minimal_acyclic_iis: true,
      size_hint_vars: 0,
      size_hint_constrs: 0,
    }
  }
}

impl Parameters {
  fn build() -> ParamBuilder { ParamBuilder { params: Default::default() } }
}

pub struct ParamBuilder {
  params: Parameters,
}

impl ParamBuilder {
  pub fn min_cyclic_iis(mut self, val: bool) -> Self {
    self.params.minimal_cyclic_iis = val;
    self
  }

  pub fn min_acyclic_iis(mut self, val: bool) -> Self {
    self.params.minimal_acyclic_iis = val;
    self
  }

  pub fn size_hint(mut self, n_vars: usize, n_constrs: usize) -> Self {
    self.params.size_hint_vars = n_vars;
    self.params.size_hint_constrs = n_constrs;
    self
  }

  pub fn finish(self) -> Parameters { self.params }
}

pub struct GraphBuilder {}
// once it's been finalised, not more adding of edges/nodes.  Only modification allowed is to remove
// an edge.

pub(crate) enum ForwardDir {}
pub(crate) enum BackwardDir {}


pub(crate) trait EdgeDir {
  fn new_neigh_iter<'a>(graph: &'a Graph, node: usize) -> NeighboursIter<'a, Self>;

  fn next_neigh(edges: &mut std::slice::Iter<Edge>) -> Option<usize>;

  fn is_forwards() -> bool;
}

pub(crate) struct NeighboursIter<'a, D: ?Sized> {
  dir: std::marker::PhantomData<D>,
  edges: std::slice::Iter<'a, Edge>,
}

impl EdgeDir for ForwardDir {
  fn new_neigh_iter<'a>(graph: &'a Graph, node: usize) ->  NeighboursIter<'a, Self> {
    NeighboursIter {
      edges: graph.edges_from[node].iter(),
      dir: std::marker::PhantomData
    }
  }

  fn next_neigh(edges: &mut std::slice::Iter<Edge>) -> Option<usize> {
    edges.next().map(|e| e.to)
  }

  fn is_forwards() -> bool { true }
}

impl EdgeDir for BackwardDir {
  fn new_neigh_iter<'a>(graph: &'a Graph, node: usize) ->  NeighboursIter<'a, Self> {
    NeighboursIter {
      edges: graph.edges_to[node].iter(),
      dir: std::marker::PhantomData
    }
  }

  fn next_neigh(edges: &mut std::slice::Iter<Edge>) -> Option<usize> {
    edges.next().map(|e| e.from)
  }

  fn is_forwards() -> bool { false }
}

impl<D: EdgeDir> Iterator for NeighboursIter<'_, D> {
  type Item = usize;
  fn next(&mut self) -> Option<Self::Item> {
    D::next_neigh(&mut self.edges)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.edges.size_hint()
  }
}

impl<D: EdgeDir> ExactSizeIterator for NeighboursIter<'_, D>{}

impl Graph {
  pub fn new_with_params(params: Parameters) -> Self {
    static NEXT_ID: AtomicU32 = AtomicU32::new(0);
    Graph {
      id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
      nodes: Vec::with_capacity(params.size_hint_vars),
      sccs: Vec::new(),
      num_active_nodes: 0,
      edges_from: Vec::with_capacity(params.size_hint_constrs),
      edges_to: Vec::with_capacity(params.size_hint_constrs),
      source_nodes: Vec::new(),
      parameters: params,
      state: ModelState::Unsolved,
    }
  }


  pub fn new() -> Self {
    Self::new_with_params(Parameters::default())
  }

  pub(crate) fn reset_nodes(&mut self) {
    for n in self.nodes.iter_mut() {
      n.x = n.lb;
      n.active_pred = None;
    }
  }

  pub(crate) fn add_node(&mut self, obj: Weight, lb: Weight, ub: Weight, kind: NodeKind) -> usize {
    let id = self.nodes.len();
    self.nodes.push(Node { lb, ub, x: lb, obj, active_pred: None, kind });
    self.edges_to.push(Vec::new());
    self.edges_from.push(Vec::new());
    id
  }

  pub(crate) fn var_from_node_id(&self, node: usize) -> Var {
    Var { graph_id: self.id, node }
  }

  pub(crate) fn add_var(&mut self, obj: Weight, lb: Weight, ub: Weight) -> Var {
    assert!(obj >= 0);
    let n = self.add_node(obj, lb, ub, NodeKind::Var);
    self.var_from_node_id(n)
  }

  pub fn add_constr(&mut self, lhs: Var, d: Weight, rhs: Var) {
    assert!(d >= 0);
    assert!(lhs != rhs, "Variables must be different");
    let e = Edge {
      from: lhs.node,
      to: rhs.node,
      weight: d,
      kind: EdgeKind::Regular,
    };
    self.add_edge(e);
  }

  pub(crate) fn remove_edge(&mut self, e: Edge) {
    self.edges_to[e.to].retain(|e| e.from != e.to);
    self.edges_from[e.from].retain(|e| e.to != e.from);
  }

  pub(crate) fn add_edge(&mut self, e: Edge) {
    self.edges_to[e.to].push(e);
    self.edges_from[e.from].push(e);
  }

  pub fn solve(&mut self) -> SolveStatus {
    // FIXME check state?
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
      ModelState::Unsolved | ModelState::Mrs => unreachable!(),
    }
  }

  pub(crate) fn edge_to_constraint(&self, e: &Edge) -> Constraint {
    Constraint::Edge(self.var_from_node_id(e.from), self.var_from_node_id(e.to))
  }

  pub(crate) fn neighbours<D: EdgeDir>(&self, n: usize) -> NeighboursIter<'_, D> {
    D::new_neigh_iter(self, n)
  }

  pub(crate) fn successors(&self, n: usize) -> NeighboursIter<'_, ForwardDir> {
    ForwardDir::new_neigh_iter(self, n)
  }

  pub(crate) fn predecessors(&self, n: usize) -> NeighboursIter<'_, BackwardDir> {
    BackwardDir::new_neigh_iter(self, n)
  }

  fn forward_label(&mut self) -> Option<ModelState> {
    let mut queue = vec![];
    let mut violated_ubs = vec![];
    let mut num_unvisited_pred = vec![0u64; self.nodes.len()];

    // Preprocessing - find source nodes and count number of edges
    for (n, node) in self.nodes.iter().enumerate() {
      if node.kind.is_scc_member() {
        continue
      }
      for e in &self.edges_from[n] {
        if !self.nodes[e.to].kind.is_scc_member() {
          num_unvisited_pred[e.to] += 1;
        }
      }
    }

    for (n, node) in self.nodes.iter_mut().enumerate() {
      node.active_pred = None;
      node.x = node.lb;

      if node.kind.is_scc_member() {
        continue
      }
      if num_unvisited_pred[n] == 0 {
        queue.push(n);
      }
    }


    // Solve - traverse DAG in topologically-sorted order
    while let Some(i) = queue.pop() {
      let node = &self.nodes[i];
      let x = node.x;
      for e in &self.edges_from[i] {
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

  pub(crate) fn find_edge(&self, from: usize, to: usize) -> &Edge {
    for e in &self.edges_from[from] {
      if &e.to == &to {
        return e
      }
    }
    unreachable!()
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

  #[test_case("simple-f" => Optimal)]
  #[test_case("simple-cycle-cei" => Infeasible(Cycle))]
  #[test_case("simple-cycle-f" => Optimal)]
  #[test_case("complex-scc-cei" => Infeasible(Cycle))]
  #[test_case("multiple-sccs-f" => Optimal)]
  #[test_case("k8-f" => Optimal)]
  #[test_case("k8-cei" => Infeasible(Cycle))]
  #[test_case("k8-cbi" => Infeasible(Cycle))]
  fn solve(input_name: &str) -> SolveStatus {
    let mut g = GraphSpec::load_from_file(test_input(input_name)).unwrap().build();
    let status = g.solve();
    // if draw {
    g.viz().save_svg(test_output(&format!("solve-{}.svg", input_name)));
    status
  }

  #[test]
  fn foo() {

  }
}
