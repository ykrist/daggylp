use crate::set_with_capacity;
use crate::graph::*;
use fnv::FnvHashSet;
use crate::model_states::{ModelState, ModelAction};
use crate::graph::SolveStatus::Infeasible;

mod cycles;
mod path;


#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Iis {
  graph_id: u32,
  kind: InfKind,
  /// If the IIS contains bounds, then (lb_node, ub_node)
  bounds: Option<(usize, usize)>,
  /// Path/cycle.  If kind is `InfKind::Cycle`, first location should be the same as the last location.
  edge_nodes: Vec<usize>,
}


impl GraphId for Iis {
  fn graph_id(&self) -> u32 { self.graph_id }
}

impl Iis {
  /// Construct a pure cycle or cycle-bound IIS
  ///
  /// First node should **NOT** be the same as the last
  /// Takes an iterator of nodes `a , b , c , ... ,y, z` and adds the edges
  ///
  ///  `(a -> b), (b -> c) ,...,(y -> z), (z -> a)`
  pub(crate) fn from_cycle<E>(graph: &Graph<E>, bounds: Option<(usize, usize)>, nodes: impl IntoIterator<Item=usize>) -> Self {
    let mut nodes = nodes.into_iter();
    let mut edge_nodes = Vec::with_capacity(nodes.size_hint().0 + 1);
    let first: usize = nodes.next().unwrap();
    edge_nodes.push(first);
    edge_nodes.extend(nodes);
    edge_nodes.push(first);
    Iis { kind: InfKind::Cycle, edge_nodes, bounds, graph_id: graph.graph_id() }
  }

  /// Construct a pure cycle IIS
  ///
  /// First node should **NOT** be the same as the last
  /// Takes an iterator of nodes `a , b , c , ... ,y, z` and adds the edges
  ///
  ///  `(a <- b), (b <- c) ,...,(y <- z), (z <- a)`
  pub(crate) fn from_backwards_cycle<E>(graph: &Graph<E>,
                                        // bounds: Option<(usize, usize)>,
                                        nodes: impl IntoIterator<Item=usize>) -> Self {
    let mut nodes = nodes.into_iter();
    let mut edge_nodes = Vec::with_capacity(nodes.size_hint().0 + 1);
    let first: usize = nodes.next().unwrap();
    edge_nodes.push(first);
    edge_nodes.extend(nodes);
    edge_nodes.push(first);
    edge_nodes.reverse();
    Iis { kind: InfKind::Cycle, edge_nodes, bounds: None, graph_id: graph.graph_id() }
  }

  /// Construct a Cycle-Bound IIS
  ///
  /// Takes two iterators.
  /// `backwards_path` is iterator of nodes `a , b , c , ... ,y , z` and representing the edges
  /// `(a <- b), (b <- c) ,..., (y <- z)`
  /// where `a` is the UB violated node and `z` is the LB violated node.
  ///
  /// `forwards_path` is iterator of nodes `a , b , c , ... ,y , z` and representing the edges
  /// `(a -> b), (b -> c) ,..., (y -> z)`
  /// where `a` is the UB violated node and `z` is the LB violated node.
  ///
  pub(crate) fn from_cycle_path_pair<E>(graph: &Graph<E>,
                                        backwards_path: impl IntoIterator<Item=usize>,
                                        forwards_path: impl IntoIterator<Item=usize>) -> Self {
    let mut backwards_path = backwards_path.into_iter();
    let mut forwards_path = forwards_path.into_iter();

    let mut edge_nodes = Vec::with_capacity(backwards_path.size_hint().0 + forwards_path.size_hint().0 - 1);

    edge_nodes.extend(backwards_path);
    // edge_nodes = a <- ... <- z
    let ub_node = *edge_nodes.first().unwrap();
    let lb_node = *edge_nodes.last().unwrap();
    edge_nodes.reverse();
    // edge_nodes = z -> .... -> a
    edge_nodes.extend(forwards_path.skip(1)); // need to skip duplicate of `a`
    // edge_nodes = z -> .... -> a -> ... -> z

    Iis { kind: InfKind::Cycle, edge_nodes, bounds: Some((lb_node, ub_node)), graph_id: graph.graph_id() }
  }


  /// Construct a path IIS
  ///
  /// Takes an iterator of nodes `a , b , c , ... ,y , z` and adds the edges
  ///  `(a <- b), (b <- c) ,..., (y <- z)`
  ///
  /// Where `a` is the UB node and `z` is the LB node.
  pub(crate) fn from_backwards_path<E>(graph: &Graph<E>, nodes: impl IntoIterator<Item=usize>) -> Self {
    let mut edge_nodes: Vec<_> = nodes.into_iter().collect();
    // edge_nodes = a <- ... <- z
    let ub_node = *edge_nodes.first().unwrap();
    let lb_node = *edge_nodes.last().unwrap();
    edge_nodes.reverse();
    // edge_nodes = z -> .... -> a
    Iis { kind: InfKind::Path, edge_nodes, bounds: Some((lb_node, ub_node)), graph_id: graph.graph_id() }
  }

  pub fn len(&self) -> usize {
    self.edge_nodes.len() - 1 + if self.bounds.is_some() { 2 } else { 0 }
  }

  pub fn bounds(&self) -> Option<(Var, Var)> {
    self.bounds.map(|(lb, ub)| {
      (self.var_from_node_id(lb), self.var_from_node_id(ub))
    })
  }

  pub fn iter_bounds<'a>(&'a self) -> impl Iterator<Item=Constraint> + 'a {
    let bounds = self.bounds.map(|(lb, ub)| [
      Constraint::Lb(self.var_from_node_id(lb)),
      Constraint::Ub(self.var_from_node_id(ub))
    ]);
    bounds.into_iter().flat_map(crate::ArrayIntoIter::new)
  }

  pub(crate) fn iter_edges<'a>(&'a self) -> impl Iterator<Item=(usize, usize)> + 'a {
    self.edge_nodes.windows(2).map(|pair| (pair[0], pair[1]))
  }

  pub fn iter_edge_constraints<'a>(&'a self) -> impl Iterator<Item=(Var, Var)> + 'a {
    self.iter_edges()
      .map(move |(i, j)| (self.var_from_node_id(i), self.var_from_node_id(j))
      )
  }

  pub fn iter_edge_vars<'a>(&'a self) -> impl Iterator<Item=Var> + 'a {
    let nodes = match self.kind {
      InfKind::Path => self.edge_nodes.as_slice(),
      InfKind::Cycle => &self.edge_nodes[..self.edge_nodes.len()-1],
    };
    nodes.iter().copied().map(move |n| self.var_from_node_id(n))
  }

  pub fn iter_constraints<'a>(&'a self) -> impl Iterator<Item=Constraint> + 'a {
    self.iter_bounds()
      .chain(
      self.iter_edge_constraints().map(|(vi, vj)| Constraint::Edge(vi, vj))
      )
  }


}

impl<E: EdgeLookup> Graph<E> {
  pub fn compute_iis(&mut self, minimal: bool) -> Iis {
    self.check_allowed_action(ModelAction::ComputeIis).unwrap();
    let iis = match &self.state {
      ModelState::InfPath(violated_ubs) => {
        self.compute_path_iis(violated_ubs)
      }
      ModelState::InfCycle { sccs, first_inf_scc } => {
        self.compute_cyclic_iis(minimal, &sccs[*first_inf_scc..])
      }
      _ => unreachable!()
    };

    #[cfg(feature = "viz-extra")] {
      self.viz_data.clear_highlighted();
      self.viz_data.highlighted_edges.extend(
        iis.iter_edge_constraints().map(|(vi, vj)| (vi.node, vj.node))
      );
      self.viz_data.highlighted_nodes.extend(
        iis.iter_bounds().map(|c| match c {
          Constraint::Ub(v) | Constraint::Lb(v) => v.node,
          _ => unreachable!()
        }));
    }
    iis
  }
}