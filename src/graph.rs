use std::sync::atomic::{AtomicU32, Ordering};
use fnv::{FnvHashSet, FnvHashMap};
use std::hash::Hash;
use crate::graph::EdgeKind::{SccOut, SccIn, SccToScc};
use crate::set_with_capacity;
use crate::iis::Iis;

use std::cmp::min;
use std::option::Option::Some;
use std::iter::ExactSizeIterator;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mrs {
  pub(crate) vars: Vec<Var>,
  pub(crate) constrs: Vec<Constraint>,
}

impl Mrs {
  pub(crate) fn new_with_root(root: Var) -> Self {
    let vars = vec![root];
    let constrs = vec![Constraint::Lb(root)];
    Mrs { vars, constrs }
  }

  pub(crate) fn new_empty() -> Self {
    Mrs { vars: Vec::default(), constrs: Vec::default() }
  }

  pub(crate) fn add_var(&mut self, var: Var) {
    debug_assert!(!self.vars.contains(&var));
    self.vars.push(var);
  }

  pub(crate) fn add_constraint(&mut self, c: Constraint) {
    debug_assert!(!self.constrs.contains(&c));
    self.constrs.push(c);
  }
}


pub type Weight = u64;

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

#[derive(Debug)]
pub struct SccInfo {
  pub(crate) nodes: FnvHashSet<usize>,
  pub(crate) lb_node: usize,
  pub(crate) ub_node: usize,
  pub(crate) scc_node: usize,
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
    assert!(lhs != rhs);
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
      dbg!(&sccs);
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

  /// Find all Strongly-Connected Components with 2 or more nodes
  pub(crate) fn find_sccs(&mut self) -> Vec<FnvHashSet<usize>> {
    debug_assert_eq!(self.sccs.len(), 0);
    const UNDEF: usize = usize::MAX;

    #[derive(Copy, Clone)]
    struct NodeAttr {
      lowlink: usize,
      index: usize,
      onstack: bool,
    }


    fn tarjan(sccs: &mut Vec<FnvHashSet<usize>>, edges_from: &[Vec<Edge>], stack: &mut Vec<usize>, attr: &mut [NodeAttr], v: usize, next_idx: &mut usize) {
      let node_attr = &mut attr[v];
      node_attr.index = *next_idx;
      node_attr.lowlink = *next_idx;
      node_attr.onstack = true;
      stack.push(v);
      *next_idx += 1;

      for w in edges_from[v].iter().map(|e| e.to) {
        let w_attr = &attr[w];
        if w_attr.index == UNDEF {
          tarjan(sccs, edges_from, stack, attr, w, next_idx);
          let w_lowlink = attr[w].lowlink;
          let v_attr = &mut attr[v];
          v_attr.lowlink = min(v_attr.lowlink, w_lowlink);
        } else if w_attr.onstack {
          let w_index = w_attr.index;
          let v_attr = &mut attr[v];
          v_attr.lowlink = min(v_attr.lowlink, w_index);
        }
      }

      let v_attr = &mut attr[v];
      if v_attr.lowlink == v_attr.index {
        let w = stack.pop().unwrap();
        attr[w].onstack = false;

        if w != v { // ignore trivial SCCs of size 1
          let mut scc = set_with_capacity(16);
          scc.insert(w);
          loop {
            let w = stack.pop().unwrap();
            attr[w].onstack = false;
            scc.insert(w);
            if w == v {
              break;
            }
          }
          sccs.push(scc);
        }
      }
    }

    let mut attr = vec![NodeAttr{ lowlink: UNDEF, index: UNDEF, onstack: false }; self.nodes.len()];
    let mut next_idx = 0;
    let mut stack = Vec::with_capacity(32);
    let mut sccs = Vec::new();

    for n in 0..self.nodes.len() {
      if attr[n].index == UNDEF {
        tarjan(&mut sccs, &self.edges_from, &mut stack, &mut attr, n, &mut next_idx);
      }
    }
    dbg!(&sccs);
    sccs
  }

  pub(crate) fn scc_is_feasible(&self, scc: &FnvHashSet<usize>) -> bool {
    for &n in scc {
      for e in &self.edges_from[n] {
        if e.weight > 0 && scc.contains(&e.to) {
          return false;
        }
      }
    }

    self.find_scc_bound_infeas(scc.iter().copied()).is_none()
  }

  pub(crate) fn condense(&mut self, sccs: Vec<FnvHashSet<usize>>) {
    // Add new SCC nodes
    for scc in sccs {
      let (lb_node, lb) = scc.iter().map(|&n| (n, self.nodes[n].lb))
        .max_by_key(|pair| pair.1).unwrap();
      let (ub_node, ub) = scc.iter().map(|&n| (n, self.nodes[n].ub))
        .min_by_key(|pair| pair.1).unwrap();

      let scc_idx = self.sccs.len();
      let scc_n = self.nodes.len();
      let scc_node = Node {
        x: lb,
        ub,
        lb,
        obj: 0,
        kind: NodeKind::Scc(scc_idx),
        active_pred: None,
      };
      self.nodes.push(scc_node);

      for &n in &scc {
        dbg!(n, scc_idx);
        self.nodes[n].kind = NodeKind::SccMember(scc_idx);
      }

      self.sccs.push(SccInfo {
        nodes: scc,
        scc_node: scc_n,
        lb_node,
        ub_node,
      });
    }

    eprintln!("{:?}", &self.nodes);
    eprintln!("{:?}", &self.sccs);

    // Add new edges in and out of the SCC
    let mut new_edges = FnvHashMap::<(usize, usize), Edge>::default();
    let mut add_edge = |new_edges: &mut FnvHashMap<(usize, usize), Edge>, edge: Edge| {
      new_edges.entry((edge.from, edge.to))
        .and_modify(|e| if e.weight < edge.weight { *e = edge })
        .or_insert(edge);
    };

    for scc in &self.sccs {
      for &n in &scc.nodes {
        for e in &self.edges_to[n] {
          let mut e = *e;
          if scc.nodes.contains(&e.from) { continue }

          match self.nodes[e.from].kind {
            NodeKind::Var => {
              e.kind = SccIn(e.to);
            },
            NodeKind::SccMember(k) => {
              e.kind = SccToScc { from: e.from, to: e.to };
              e.from = self.sccs[k].scc_node;
            },
            NodeKind::Scc(_) => unreachable!(),

          };
          e.to = scc.scc_node;
          add_edge(&mut new_edges, e);
        }

        for e in &self.edges_from[n] {
          let mut e = *e;
          if scc.nodes.contains(&e.to) { continue}
          match self.nodes[e.to].kind {
            NodeKind::Var => {
              e.kind = SccOut(e.from)
            },
            NodeKind::SccMember(k) => {
              e.kind = SccToScc { from: e.from, to: e.to };
              e.to = self.sccs[k].scc_node;
            },
            NodeKind::Scc(_) => unreachable!(),
          };
          e.from = scc.scc_node;
          add_edge(&mut new_edges, e);
        }
      }
    }

    for edge_lookup in &mut [&mut self.edges_from, &mut self.edges_to] {
      edge_lookup.extend(std::iter::repeat_with(Vec::new).take(self.sccs.len()))
    }

    for ((from, to), e) in new_edges {
      self.edges_from[from].push(e);
      self.edges_to[to].push(e);
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
    let mut g = GraphGen::load_from_file(test_input(input_name)).build();
    let status = g.solve();
    // if draw {
    g.viz().save_svg(test_output(&format!("solve-{}.svg", input_name)));
    status
  }

  #[test]
  fn foo() {

  }
}
