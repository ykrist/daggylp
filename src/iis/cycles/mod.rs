use crate::graph::*;
use fnv::{FnvHashSet, FnvHashMap};
use crate::{set_with_capacity, Error, map_with_capacity};
use std::iter::once;
use std::collections::VecDeque;
use crate::iis::Iis;
use crate::scc::{BoundInfeas, SccInfo};

mod enumeration;
mod shortest_path;
pub use shortest_path::ShortestPathAlg;


enum CyclicInfKind {
  Pure,
  Unknown,
  Bounds(BoundInfeas),
}



trait FindCyclicIis<A> {
  fn find_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis;

  fn find_smallest_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis;
}

impl Graph {
  pub(crate) fn compute_cyclic_iis(&self, minimal: bool, sccs: &[FnvHashSet<usize>]) -> Iis {
    // use shortest_path::ShortestPath;
    if minimal {
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
  use crate::test::*;
  use crate::viz::GraphViz;
  use proptest::prelude::*;


//
//   #[test_case(2)]
//   #[test_case(3)]
//   #[test_case(4)]
//   #[test_case(5)]
//   #[test_case(6)]
//   #[test_case(7)]
//   #[test_case(8)]
//   fn count_complete(n: u8) {
//     let mut g = complete_graph_with_single_nonzero_edge(n).build();
//     solve_and_count_cycles_iis(&mut g,
//                                num_iis_in_complete_graph_with_single_nonzero_edge(n),
//                                num_cycles_in_complete_graph(n));
//   }
//
//
//   #[test_case(& [4, 4])]
//   #[test_case(& [8, 7])]
//   #[test_case(& [4, 3, 8, 6])]
//   #[test_case(& [4, 6, 3, 7, 7, 6])]
//   fn count_multi_scc(component_sizes: &[u8]) {
//     let mut g = complete_graph_sccs(component_sizes).build();
//     let (mut n_iis, mut n_cycles) = (0, 0);
//
//     for &n in component_sizes {
//       n_iis += num_iis_in_complete_graph_with_single_nonzero_edge(n);
//       n_cycles += num_cycles_in_complete_graph(n);
//     }
//     solve_and_count_cycles_iis(&mut g, n_iis, n_cycles);
//   }
//
//
//   fn cycle_graph(n: u8) -> GraphSpec {
//     let n = n as usize;
//     GraphSpec::new(
//       n,
//       IdenticalNodes { lb: 0, ub: Weight::MAX, obj: 0 },
//       CycleEdges { n, weight: 1 },
//     )
//   }
//
//   #[test_case(2)]
//   #[test_case(3)]
//   #[test_case(4)]
//   #[test_case(5)]
//   #[test_case(6)]
//   #[test_case(7)]
//   #[test_case(8)]
//   fn count_iis_cycle_graph(n: u8) {
//     let mut g = cycle_graph(n).build();
//     solve_and_count_cycles_iis(&mut g, 1, 1);
//   }
//
//   /// Build a graph of the form:
//   ///
//   ///        0 ---> 1
//   ///        /\      |
//   ///        |      \/
//   ///        4 <--- 2
//   ///        |      /\
//   ///        \/      |
//   ///        5 ---> 6
//   ///        /\      |
//   ///        |      \/
//   ///        8 <--- 7
//   ///        |      /\
//   ///        \/      |
//   ///        ...    ...
//   ///
//   /// Will have `( k + 1 ) * 2` nodes.  All edges have zero weight, except 0 -> 1.
//   /// This means there is always one IIS and exactly k cycles.
//   fn k_cycle_graph(k: u8) -> GraphSpec {
//     let mut g = cycle_graph(4);
//     g.edges.values_mut().for_each(|w| *w = 0);
//     g.edges.insert((0, 1), 1);
//     let node_data = *g.nodes.first().unwrap();
//
//     let mut a = 2;
//     let mut b = 3;
//
//     for j in 1..k {
//       let c = g.add_node(node_data);
//       let d = g.add_node(node_data);
//       g.edges.insert((b, c), 0);
//       g.edges.insert((c, d), 0);
//       g.edges.insert((d, a), 0);
//       a = c;
//       b = d;
//     }
//     g
//   }
//
//   #[test_case(10)]
//   #[test_case(12)]
//   #[test_case(16)]
//   #[test_case(100)]
//   #[test_case(3)]
//   #[test_case(4)]
//   #[test_case(55)]
//   fn count_k_cycle_graph(k: u8) {
//     let mut g = k_cycle_graph(k).build();
//     solve_and_count_cycles_iis(&mut g, 1, k as u128);
//   }
}
