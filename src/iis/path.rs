use crate::graph::*;
use super::*;
use crate::{map_with_capacity};
use fnv::FnvHashMap;

fn update_min<T: Ord, U>(dst: &mut Option<(T, U)>, src: (T, U)) -> bool {
  if let Some(v) = dst {
    if &v.0 <= &src.0 {
      return false;
    }
  }
  *dst = Some(src);
  true
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct DpState {
  node: usize,
  deadline: Weight,
}


#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum DpValueAction {
  /// Before the recursive call to compute value of `state[t+1]`, `state[t]` gets a
  /// cache entry with a `CycleSentinel` entry, so we can detect cycles.  Once the value `state[t+1]`
  /// has been computed, `state[t]` becomes either `Pruned` or `Step`.
  CycleSentinel,
  /// This state cannot possible lead to an optimal path
  Pruned,
  /// A terminal (base-case) state with associated value
  Terminate(u32),
  /// The value of this state, and the next state in the optimal trajectory.
  Step(u32, DpState),
}

use DpValueAction::*;
use crate::edge_storage::EdgeList;

//      val            i     edge weight (i, j)
///                          j    deadline[j]
type DpCache = FnvHashMap<DpState, DpValueAction>;

struct DpShortestPath<'a> {
  dp_cache: &'a DpCache,
  path_len: u32,
  state: Option<DpState>,
}

impl<'a> DpShortestPath<'a> {
  fn new(cache: &'a DpCache, initial_state: DpState, len: u32) -> Self {
    DpShortestPath {
      dp_cache: cache,
      path_len: len,
      state: Some(initial_state),
    }
  }
}

impl Iterator for DpShortestPath<'_> {
  type Item = usize;

  fn next(&mut self) -> Option<Self::Item> {
    self.state.take()
      .map(|state| {
        let (path_len, next_state) = match self.dp_cache[&state] {
          Terminate(l) => (l, None),
          Step(l, s) => (l, Some(s)),
          Pruned | CycleSentinel => unreachable!()
        };
        self.state = next_state;
        self.path_len = path_len;
        state.node
      })
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let n = self.path_len as usize + 1;
    (n, Some(n))
  }
}

impl<E: EdgeLookup> Graph<E> {
  pub(crate) fn compute_path_iis(&self, violated_ubs: &[usize]) -> Iis {
    let mut cache = map_with_capacity(64);
    let mut shortest_path_initial_state = None;

    for &n in violated_ubs {
      let node = &self.nodes[n];
      debug_assert!(!matches!(&node.kind, NodeKind::SccMember(_)));
      let (n, node) = if let &NodeKind::Scc(k) = &node.kind {
        let n = self.sccs[k].ub_node;
        (n, &self.nodes[n])
      } else {
        (n, node)
      };
      let initial_state = DpState { node: n, deadline: node.ub };
      match self.dp(&mut cache, initial_state) {
        Pruned => continue,
        Terminate(l) => {
          debug_assert_eq!(l, 0);
          update_min(&mut shortest_path_initial_state, (0, initial_state));
          break;
        }
        Step(l, _) => {
          update_min(&mut shortest_path_initial_state, (l, initial_state));
        }
        CycleSentinel => unreachable!(),
      }
    }

    let (n_edges, initial_state) = shortest_path_initial_state.unwrap();
    let shortest_path = DpShortestPath::new(&cache, initial_state, n_edges);
    let n_edges = n_edges as usize;
    let mut iis = Iis::from_backwards_path(self, shortest_path);
    debug_assert_eq!(iis.len(), n_edges + 2);
    iis
  }

  fn dp(&self, cache: &mut DpCache, state: DpState) -> DpValueAction {
    if let Some(val_action) = cache.get(&state) {
      return *val_action;
    }

    let n = &self.nodes[state.node];
    let return_val = if state.deadline < n.lb {
      Terminate(0)
    } else if state.deadline > n.ub {
      Pruned
    } else {
      cache.insert(state, DpValueAction::CycleSentinel);

      let mut best_val_action = None;

      for e in self.edges.predecessors(state.node)
        .filter(|e| matches!(&e.kind, EdgeKind::Regular))
      {
        let new_state = DpState { node: e.from, deadline: state.deadline - e.weight };
        let mut val = match self.dp(cache, new_state) {
          Pruned => continue,
          CycleSentinel => continue,
          Terminate(val) => val,
          Step(val, _) => val,
        };

        val += 1; // add cost from the edge we just traversed
        update_min(&mut best_val_action, (val, new_state));
      }

      match best_val_action {
        None => DpValueAction::Pruned,
        Some((val, new_state)) => DpValueAction::Step(val, new_state)
      }
    };

    cache.insert(state, return_val);
    return_val
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[macro_use]
  use crate::*;
  use crate::test::*;
  use proptest::prelude::*;
  use crate::test::strategy::{node, any_bounds_nodes, MAX_EDGE_WEIGHT, default_nodes};

  fn path_iis(path: Vec<usize>) -> Iis {
    let bounds = Some((*path.first().unwrap(), *path.last().unwrap()));
    Iis { kind: InfKind::Path, bounds, edge_nodes: path, graph_id: u32::MAX }
  }

  fn graph_with_single_path_iis() -> impl SharableStrategy<Value=GraphData> {
    strategy::connected_acyclic_graph(default_nodes(10..=100), Just(0))
      .prop_map(|mut g| {
        g.nodes.first_mut().unwrap().lb = 1;
        g.nodes.last_mut().unwrap().ub = 0;
        g
      })
  }

  fn iis_graph() -> impl SharableStrategy<Value=GraphData> {
    strategy::connected_acyclic_graph(any_bounds_nodes(10..=100), 0..=MAX_EDGE_WEIGHT)
  }

  #[graph_proptest]
  #[input(graph_with_single_path_iis())]
  fn find_and_remove(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Infeasible(InfKind::Path));
    let iis = g.compute_iis(true);
    let iis_size = iis.len();
    g.remove_iis(&iis);
    let status = g.solve();
    match status {
      SolveStatus::Infeasible(InfKind::Path) => {
        let new_iis = g.compute_iis(true);
        prop_assert!(iis_size <= new_iis.len(), "second IIS should be smaller or the same size")
      }
      SolveStatus::Infeasible(InfKind::Cycle) => {
        test_case_bail!("input graph should be acyclic")
      }
      SolveStatus::Optimal => {}
    }
    Ok(())
  }

  #[graph_proptest]
  #[input(graph_with_single_path_iis())]
  fn single_path(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Infeasible(InfKind::Path));
    let iis = g.compute_iis(true);
    prop_assert!(iis.len() - 2 < g.nodes.len());
    Ok(())
  }

  #[graph_proptest]
  #[input(iis_graph())]
  fn multiple_paths(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Infeasible(InfKind::Path) | SolveStatus::Optimal);
    prop_assume!(matches!(status, SolveStatus::Infeasible(InfKind::Path)));
    let iis = g.compute_iis(true);
    prop_assert!(iis.edge_nodes.len() < g.nodes.len());
    prop_assert_matches!(iis.kind, InfKind::Path);
    let mut edge_sum: Weight = 0;
    for (i, j) in iis.iter_edges() {
      edge_sum += g.edges.find_edge(i, j).weight;
    }

    prop_assert!(iis.bounds.is_some(), "Bounds in IIS");
    let (lb_node, ub_node) = iis.bounds.unwrap();
    let lb = g.nodes[lb_node].lb;
    let ub = g.nodes[ub_node].ub;
    prop_assert!(lb + edge_sum > ub);
    Ok(())
  }



  #[graph_test]
  #[input("multiple-sccs-0.pi", path_iis(vec![0, 3, 4, 5]))]
  #[input("multiple-sccs-1.pi", path_iis(vec![0, 1, 2, 6, 4, 5]))]
  fn edge_cases(g: &mut Graph, true_iis: Iis) -> GraphTestResult {
    match g.solve() {
      SolveStatus::Infeasible(InfKind::Path) => {
        let iis = g.compute_iis(true);
        graph_testcase_assert_eq!(true_iis.bounds, iis.bounds);
        graph_testcase_assert_eq!(true_iis.edge_nodes, iis.edge_nodes);
      }
      status => {
        Err(anyhow::anyhow!("should be path infeasible, was {:?}", status))?
      }
    };
    Ok(())
  }
}