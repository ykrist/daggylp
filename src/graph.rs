use std::sync::atomic::{AtomicU32, Ordering};
use fnv::{FnvHashSet, FnvHashMap};
use std::hash::Hash;
use crate::graph::EdgeKind::{SccOut, SccIn};
use crate::set_with_capacity;


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


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Iis {
  pub(crate) constrs: FnvHashSet<Constraint>,
}

impl Iis {
  /// First variable should **NOT** be the same as the last
  pub(crate) fn from_cycle(vars: impl IntoIterator<Item=Var>) -> Self {
    let mut vars = vars.into_iter();
    let first : Var = vars.next().unwrap();
    let mut constrs = set_with_capacity(vars.size_hint().0 + 1);
    let mut i = first;
    for j in vars {
      constrs.insert(Constraint::Edge(i, j));
      i = j;
    }
    constrs.insert(Constraint::Edge(i, first));
    Iis{ constrs }
  }

  pub(crate) fn add_constraint(&mut self, c: Constraint) {
    let unique = self.constrs.insert(c);
    debug_assert!(unique)
  }

  pub fn size(&self) -> usize { self.constrs.len() }
}

pub type Weight = u64;

#[derive(Debug, Copy, Clone)]
pub enum InfKind {
  Cycle,
  Path,
}

#[derive(Debug, Copy, Clone)]
pub enum SolveStatus {
  Infeasible(InfKind),
  Optimal,
}

#[derive(Debug, Clone)]
pub(crate) enum ModelState {
  Init,
  Cycles,
  CycleInfeasible { sccs: Vec<FnvHashSet<usize>>, first_inf_scc: usize },
  InfPath(Vec<usize>),
  Optimal,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EdgeKind {
  Regular,
  SccIn(usize),
  SccOut(usize),
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
  fn ignored_fwd_label(&self) -> bool {
    matches!(self, NodeKind::SccMember(_))
  }

  // Should this node kind be ignored during back-labelling of an acyclic IIS?
  fn ignored_iis_back_label(&self) -> bool {
    matches!(self, NodeKind::Scc(_))
  }
}

pub struct Node {
  pub(crate) x: Weight,
  pub(crate) obj: Weight,
  pub(crate) lb: Weight,
  pub(crate) ub: Weight,
  pub(crate) active_pred: Option<usize>,
  pub(crate) kind: NodeKind,
}

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
      state: ModelState::Init,
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
    self.state = self.forward_label();
    match &self.state {
      ModelState::Cycles => {
        let sccs = self.find_sccs();
        let inf_idx = sccs.iter().enumerate()
          .find(|(_, scc)| !self.scc_is_feasible(scc))
          .map(|(idx, _)| idx);

        if let Some(inf_idx) = inf_idx {
          self.state = ModelState::CycleInfeasible { sccs, first_inf_scc: inf_idx };
          SolveStatus::Infeasible(InfKind::Cycle)
        } else {
          self.condense(sccs);
          self.solve()
        }
      }
      ModelState::CycleInfeasible { .. } => SolveStatus::Infeasible(InfKind::Cycle),
      ModelState::InfPath(_) => SolveStatus::Infeasible(InfKind::Path),
      ModelState::Optimal => SolveStatus::Optimal,
      ModelState::Init => unreachable!(),
    }

  }

  pub(crate) fn edge_to_constraint(&self, e: &Edge) -> Constraint {
    Constraint::Edge(self.var_from_node_id(e.from), self.var_from_node_id(e.to))
  }

  fn forward_label(&mut self) -> ModelState {
    let mut num_active_nodes = 0;
    let mut num_visited_nodes = 0;
    let mut stack = vec![];
    let mut violated_ubs = vec![];
    let mut num_visited_pred = vec![0; self.nodes.len()];
    // Preprocessing - find source nodes and count number of active nodes
    for (n, node) in self.nodes.iter().enumerate() {
      if self.edges_to[n].is_empty() {
        stack.push(n);
      }
      if !node.kind.ignored_fwd_label() {
        num_active_nodes += 1;
      }
    }
    let num_active_nodes = num_active_nodes;

    // Solve - traverse DAG in topologically-sorted order
    while let Some(i) = stack.pop() {
      let node = &self.nodes[i];
      num_visited_nodes += 1;

      let x = node.x;
      for e in &self.edges_from[i] {
        let nxt_node = &mut self.nodes[e.to];
        if nxt_node.kind.ignored_fwd_label() {
          continue; // skip this one
        }
        let nxt_x = x + e.weight;
        if nxt_node.x < nxt_x {
          nxt_node.x = nxt_x;
          nxt_node.active_pred = Some(i);
          if nxt_x > nxt_node.ub {
            violated_ubs.push(e.to);
          }
        }
        num_visited_pred[e.to] += 1;
        if num_visited_pred[e.to] == self.edges_to[e.to].len() {
          stack.push(e.to)
        }
      }
    }

    // Post-processing: check infeasibility kinds
    if num_visited_nodes < num_active_nodes {
      ModelState::Cycles
    } else if !violated_ubs.is_empty() {
      ModelState::InfPath(violated_ubs)
    } else {
      ModelState::Optimal
    }
  }

  /// Find all Strongly-Connected Components with 2 or more nodes
  pub(crate) fn find_sccs(&mut self) -> Vec<FnvHashSet<usize>> {
    debug_assert_eq!(self.sccs.len(), 0);
    todo!()
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
    for scc in sccs {
      let (lb_node, lb) = scc.iter().map(|&n| (n, self.nodes[n].lb))
        .max_by_key(|pair| pair.1).unwrap();
      let (ub_node, ub) = scc.iter().map(|&n| (n, self.nodes[n].ub))
        .min_by_key(|pair| pair.1).unwrap();

      let scc_n = self.sccs.len();
      let scc_node = Node {
        x: lb,
        ub,
        lb,
        obj: 0,
        kind: NodeKind::Scc(scc_n),
        active_pred: None,
      };
      self.nodes.push(scc_node);

      // edges into the SCC
      let mut biggest_in = FnvHashMap::<usize, Edge>::default();
      let mut biggest_out = FnvHashMap::<usize, Edge>::default();
      for e in self.edges_from.iter().flat_map(|edges| edges.iter()) {
        if scc.contains(&e.to) && scc.contains(&e.from) {
          if !biggest_in.contains_key(&e.from) || biggest_in[&e.from].weight < e.weight {
            biggest_in.insert(e.from, *e);
          }
        }

        if scc.contains(&e.from) && scc.contains(&e.to) {
          if !biggest_out.contains_key(&e.from) || biggest_out[&e.from].weight < e.weight {
            biggest_out.insert(e.from, *e);
          }
        }
      }

      self.edges_from.push(biggest_out.into_iter()
        .map(|(_, mut e)| {
          e.kind = SccOut(e.from);
          e.from = scc_n;
          e
        })
        .collect());

      for (_, mut e) in biggest_in.into_iter() {
        e.kind = SccIn(e.to);
        e.to = scc_n;
        self.edges_from[e.from].push(e);
      }

      for &n in &scc {
        self.nodes[n].kind = NodeKind::SccMember(scc_n);
      }

      self.sccs.push(SccInfo {
        nodes: scc,
        scc_node: scc_n,
        lb_node,
        ub_node,
      })
    }
  }

  pub fn compute_iis(&mut self) -> Iis {
    match &self.state {
      ModelState::InfPath(violated_ubs) => {
        self.compute_path_iis(violated_ubs)
      },
      ModelState::CycleInfeasible { sccs, first_inf_scc } => {
        self.compute_cyclic_iis(&sccs[*first_inf_scc..])
      }
      ModelState::Optimal => panic!("cannot compute IIS on feasible model"),
      ModelState::Init => panic!("need to call solve() first"),
      ModelState::Cycles => unreachable!()
    }
  }

  pub(crate) fn find_edge(&self, from: usize, to: usize) -> &Edge {
    let _ = (from, to);
    todo!()
  }
}
