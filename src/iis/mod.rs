use crate::set_with_capacity;
use crate::graph::*;
use fnv::FnvHashSet;
use crate::model_states::ModelAction;

mod cycles;
mod path;


#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct Iis {
  graph_id: u32,
  /// If the IIS contains bounds, then (lb_node, ub_node)
  pub(crate) bounds: Option<(usize, usize)>,
  /// Edge constraints
  pub(crate) edges: FnvHashSet<(usize, usize)>,
}

impl GraphId for Iis {
  fn graph_id(&self) -> u32 { self.graph_id }
}

impl Iis {
  pub(crate) fn clear(&mut self) {
    self.edges.clear();
    self.bounds = None;
  }

  pub(crate) fn with_capacity<E>(graph: &Graph<E>, n_edges: usize) -> Self {
    Iis { edges: set_with_capacity(n_edges), bounds: None, graph_id: graph.graph_id() }
  }

  /// First variable should **NOT** be the same as the last
  /// Takes an iterator of nodes `a , b , c , ... ,y, z` and adds the edges
  ///
  ///  `(a -> b), (b -> c) ,...,(y -> z), (z -> a)`
  pub(crate) fn from_cycle<E>(graph: &Graph<E>, nodes: impl IntoIterator<Item=usize>) -> Self {
    let mut vars = nodes.into_iter();
    let mut edges = set_with_capacity(vars.size_hint().0);
    let first: usize = vars.next().unwrap();
    let mut i = first;
    for j in vars {
      edges.insert((i, j));
      i = j;
    }
    edges.insert((i, first));
    Iis { edges, bounds: None, graph_id: graph.graph_id() }
  }

  /// First variable should **NOT** be the same as the last
  /// Takes an iterator of nodes `a , b , c , ... ,y,  z` and adds the edges
  ///
  ///  `(a <- b), (b <- c) ,...,(y <- z), (z <- a)`
  pub(crate) fn from_cycle_backwards<E>(graph: &Graph<E>, nodes: impl IntoIterator<Item=usize>) -> Self {
    let mut vars = nodes.into_iter();
    let mut edges = set_with_capacity(vars.size_hint().0);
    let last: usize = vars.next().unwrap();
    let mut j = last;
    for i in vars {
      edges.insert((i, j));
      j = i;
    }
    edges.insert((last, j));
    Iis { edges, bounds: None, graph_id: graph.graph_id() }
  }


  /// Takes an iterator of nodes `a , b , c , ... ,y , z` and adds the edges
  ///  `(a -> b), (b -> c) ,..., (y -> z)`
  ///
  /// If `bounds` is `true`, add the lower bound of `a` and upper bound of `z` end to the IIS.
  pub(crate) fn add_forwards_path(&mut self, nodes: impl IntoIterator<Item=usize>, bounds: bool) {
    let mut vars = nodes.into_iter();
    self.edges.reserve(vars.size_hint().0);
    let mut i: usize = vars.next().unwrap();
    let first = i;
    for j in vars {
      let is_new = self.edges.insert((i, j));
      debug_assert!(is_new);
      i = j;
    }
    if bounds {
      self.add_bounds(first, i);
    }
  }

  /// Takes an iterator of nodes `a , b , c , ... ,y , z` and adds the edges
  ///  `(a <- b), (b <- c) ,..., (y <- z)`
  ///
  /// If `bounds` is `true`, add the lower bound of `z` and upper bound of `a` end to the IIS.
  pub(crate) fn add_backwards_path(&mut self, nodes: impl IntoIterator<Item=usize>, bounds: bool) {
    let mut vars = nodes.into_iter();
    self.edges.reserve(vars.size_hint().0);
    let mut j: usize = vars.next().unwrap();
    let last = j;
    for i in vars {
      let is_new = self.edges.insert((i, j));
      debug_assert!(is_new);
      j = i;
    }
    if bounds {
      self.add_bounds(j, last);
    }
  }

  pub(crate) fn add_bounds(&mut self, lb_node: usize, ub_node: usize) {
    debug_assert_eq!(self.bounds, None);
    self.bounds = Some((lb_node, ub_node))
  }

  pub fn len(&self) -> usize {
    self.edges.len() + if self.bounds.is_some() { 2 } else { 0 }
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

  pub fn iter_edge_constraints<'a>(&'a self) -> impl Iterator<Item=(Var, Var)> + 'a {
    self.edges.iter()
      .map(move |&(i, j)| (self.var_from_node_id(i), self.var_from_node_id(j))
      )
  }

  pub fn iter_constraints<'a>(&'a self) -> impl Iterator<Item=Constraint> + 'a {
    self.iter_bounds().chain(self.iter_edge_constraints().map(|(vi, vj)| Constraint::Edge(vi, vj)))
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