mod enumeration;
mod shortest_path;

use crate::graph::*;
use fnv::{FnvHashSet, FnvHashMap};
use crate::{Result, set_with_capacity, Error, map_with_capacity};
use std::iter::once;
use std::collections::VecDeque;
use crate::iis::Iis;

pub use shortest_path::ShortestPathAlg;


enum CyclicInfKind {
  Pure,
  Unknown,
  Bounds { lb: usize, ub: usize },
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Prune {
  // No pruning
  AllDest,
  // Find the best destination and prune others (stop)
  BestDest,
  // Find the best dest less than
  BestDestLessThan(u32),
  // All dest less than
  AllDestLessThan(u32),
}

impl Prune {
  fn all_dests(&self) -> bool {
    use Prune::*;
    matches!(self, AllDest | AllDestLessThan(..))
  }

  fn bound(&self) -> Option<u32> {
    use Prune::*;
    match self {
      BestDestLessThan(bnd) | AllDestLessThan(bnd) => Some(*bnd),
      AllDest | BestDest => None
    }
  }

  fn update_bound(&mut self, f: impl FnOnce(u32) -> u32) {
    use Prune::*;
    match self {
      BestDestLessThan(bnd) | AllDestLessThan(bnd) => {
        *bnd = f(*bnd);
      }
      AllDest | BestDest => {}
    }
  }
}

trait FindCyclicIis<A> {
  fn find_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis;

  fn find_smallest_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis;
}

impl Graph {
  /// Returns a two node-bound pairs in an SCC, (n1, lb), (n2, ub) such that ub < lb, if such a pair exists.
  pub(crate) fn find_scc_bound_infeas(&self, scc: impl Iterator<Item=usize>) -> Option<((usize, Weight), (usize, Weight))> {
    let mut nodes = scc.map(|n| (n, &self.nodes[n]));

    let (n, first_node) = nodes.next().expect("expected non-empty iterator");
    let mut min_ub_node = n;
    let mut min_ub = first_node.ub;
    let mut max_lb_node = n;
    let mut max_lb = first_node.lb;

    for (n, node) in nodes {
      if max_lb < node.lb {
        max_lb = node.lb;
        max_lb_node = n;
      }
      if min_ub > node.ub {
        min_ub = node.ub;
        min_ub_node = n;
      }
      if min_ub < max_lb {
        return Some(((max_lb_node, max_lb), (min_ub_node, min_ub)));
      }
    }

    None
  }


  pub(crate) fn compute_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    // use shortest_path::ShortestPath;
    if self.parameters.minimal_cyclic_iis {
      <Self as FindCyclicIis<ShortestPathAlg>>::find_cyclic_iis(self, sccs)
    } else {
      <Self as FindCyclicIis<ShortestPathAlg>>::find_smallest_cyclic_iis(self, sccs)
    }
  }
}

pub struct SccHandle<'a> {
  scc: &'a SccInfo
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_utils::*;
  use test_case::test_case;
  use crate::viz::GraphViz;

  fn num_iis_in_complete_graph_with_single_nonzero_edge(n: u8) -> u128 {
    let mut total = 0u128;
    for k in 2..=n {
      let mut prod = 1;
      for i in (n - k + 1)..=(n - 2) { // may be empty if k == 2
        prod *= i as u128;
      }
      total += prod;
    }
    total
  }

  fn num_cycles_in_complete_graph(n: u8) -> u128 {
    let mut c = 0;
    for k in 2..=n {
      let mut q: u128 = (n - k + 1) as u128;
      for x in (n - k + 2)..=n {
        q *= (x as u128);
      }
      c += q / (k as u128);
    }
    c
  }

  fn complete_graph_with_single_nonzero_edge(n: u8) -> GraphSpec {
    let mut g = GraphSpec::new(
      n as usize,
      IdenticalNodes { lb: 0, ub: 10, obj: 0 },
      AllEdges(0),
    );
    g.edges.insert((0, 1), 1);
    g
  }

  fn complete_graph_sccs(sizes: &[u8]) -> GraphSpec {
    let mut sizes = sizes.iter().copied();
    let n = sizes.next().unwrap();

    let mut g = complete_graph_with_single_nonzero_edge(n);
    for n in sizes {
      let f = complete_graph_with_single_nonzero_edge(n);
      g.join(&f, AllEdges(1), NoEdges());
    }
    g
  }


  fn solve_and_count_cycles_iis(g: &mut Graph, n_iis: u128, n_cycles: u128) {
    g.solve();
    let mut iis_iter = g.all_cyclic_iis().unwrap();

    let mut iis_cnt = 0;
    let mut cycle_cnt = 0;

    for iis in iis_iter.clone() {
      iis_cnt += 1;
    }

    for cycle in iis_iter.cycle_iter {
      cycle_cnt += 1;
    }

    assert_eq!(iis_cnt, n_iis);
    assert_eq!(cycle_cnt, n_cycles);
  }

  #[test_case(2)]
  #[test_case(3)]
  #[test_case(4)]
  #[test_case(5)]
  #[test_case(6)]
  #[test_case(7)]
  #[test_case(8)]
  fn count_complete(n: u8) {
    let mut g = complete_graph_with_single_nonzero_edge(n).build();
    solve_and_count_cycles_iis(&mut g,
                               num_iis_in_complete_graph_with_single_nonzero_edge(n),
                               num_cycles_in_complete_graph(n));
  }


  #[test_case(& [4, 4])]
  #[test_case(& [8, 7])]
  #[test_case(& [4, 3, 8, 6])]
  #[test_case(& [4, 6, 3, 7, 7, 6])]
  fn count_multi_scc(component_sizes: &[u8]) {
    let mut g = complete_graph_sccs(component_sizes).build();
    let (mut n_iis, mut n_cycles) = (0, 0);

    for &n in component_sizes {
      n_iis += num_iis_in_complete_graph_with_single_nonzero_edge(n);
      n_cycles += num_cycles_in_complete_graph(n);
    }
    solve_and_count_cycles_iis(&mut g, n_iis, n_cycles);
  }


  fn cycle_graph(n: u8) -> GraphSpec {
    let n = n as usize;
    GraphSpec::new(
      n,
      IdenticalNodes { lb: 0, ub: Weight::MAX, obj: 0 },
      CycleEdges { n, weight: 1 },
    )
  }

  #[test_case(2)]
  #[test_case(3)]
  #[test_case(4)]
  #[test_case(5)]
  #[test_case(6)]
  #[test_case(7)]
  #[test_case(8)]
  fn count_iis_cycle_graph(n: u8) {
    let mut g = cycle_graph(n).build();
    solve_and_count_cycles_iis(&mut g, 1, 1);
  }

  /// Build a graph of the form:
  ///
  ///        0 ---> 1
  ///        /\      |
  ///        |      \/
  ///        4 <--- 2
  ///        |      /\
  ///        \/      |
  ///        5 ---> 6
  ///        /\      |
  ///        |      \/
  ///        8 <--- 7
  ///        |      /\
  ///        \/      |
  ///        ...    ...
  ///
  /// Will have `( k + 1 ) * 2` nodes.  All edges have zero weight, except 0 -> 1.
  /// This means there is always one IIS and exactly k cycles.
  fn k_cycle_graph(k: u8) -> GraphSpec {
    let mut g = cycle_graph(4);
    g.edges.values_mut().for_each(|w| *w = 0);
    g.edges.insert((0, 1), 1);
    let node_data = *g.nodes.first().unwrap();

    let mut a = 2;
    let mut b = 3;

    for j in 1..k {
      let c = g.add_node(node_data);
      let d = g.add_node(node_data);
      g.edges.insert((b, c), 0);
      g.edges.insert((c, d), 0);
      g.edges.insert((d, a), 0);
      a = c;
      b = d;
    }
    g
  }

  #[test_case(10)]
  #[test_case(12)]
  #[test_case(16)]
  #[test_case(100)]
  #[test_case(3)]
  #[test_case(4)]
  #[test_case(55)]
  fn count_k_cycle_graph(k: u8) {
    let mut g = k_cycle_graph(k).build();
    solve_and_count_cycles_iis(&mut g, 1, k as u128);
  }
}
