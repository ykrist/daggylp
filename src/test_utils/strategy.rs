use proptest::prelude::*;
use super::generators::*;
use crate::graph::Weight;
use fnv::FnvHashMap;
use std::collections::HashMap;
use proptest::arbitrary::arbitrary;

trait Connectivity {
  fn connected(&mut self, from: usize, to: usize) -> bool;

  fn set_num_nodes(&mut self, _: usize) {}
}

struct AllEdges();

impl Connectivity for AllEdges {
  fn connected(&mut self, _: usize, _: usize) -> bool { true }
}

struct Cycle(usize);

impl Connectivity for Cycle {
  fn set_num_nodes(&mut self, n: usize) { self.0 = n }

  fn connected(&mut self, from: usize, to: usize) -> bool {
    (from + 1) % self.0  == to
  }
}

/// Assume nodes are arranged like this:
/// ```text
///           0      rank = 0
///          1 2     rank = 1
///         3 4 5    rank = 2
///        6 7 8 9   rank = 3
///       10 ...
/// ```
/// For every sub-triangle of nodes in the big triangle above,
/// ```text
///       i
///      j k
/// ```
/// `Triangular` will add edges in three ways:
///  - left-to-right: eg `j -> k`
///  - upward: eg `k -> i`
///  - downward: eg `i -> j`
struct Triangular;

impl Connectivity for Triangular {
  fn connected(&mut self, from: usize, to: usize) -> bool {
    fn node_rank(t: usize) -> usize {
      let r = ((9 + 8 * t) as f64).sqrt() * 0.5 - 1.5;
      (r - 1e-10).ceil() as usize
    }

    let from_rank = node_rank(from);
    let to_rank = node_rank(to);

    (from_rank == to_rank && from + 1 == to) // left to right edges
      || (from_rank == to_rank + 1 && from == to + from_rank + 1) // upward edges
      || (from_rank + 1 == to_rank && from + to_rank == to) // downward edges
  }
}


#[derive(Debug)]
struct EdgeData {
  from: usize,
  to: usize,
  weight: Weight,
}

type EdgeMap = FnvHashMap<(usize, usize), Weight>;

fn assign_edge_weights(mut edges: EdgeMap, weights: Vec<Weight>) -> EdgeMap {
  for (weight, w) in edges.values_mut().zip(weights) {
    *weight = w;
  }
  edges
}
//
// prop_compose! {
//   fn edge_map()
// }

fn node(bounds: impl Strategy<Value=(Weight, Weight)>, obj: impl Strategy<Value=Weight>) -> impl Strategy<Value=NodeData> {
  (obj, bounds).prop_map(|(obj, (lb, ub))| NodeData { lb, ub, obj })
}

fn graph(
  size: usize,
  mut conn: impl Connectivity,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec>
{
  assert!(size > 0);
  conn.set_num_nodes(size*(size-1));
  let nodes = vec![nodes; size];
  let edges: Vec<_> = (0..size).flat_map(|i| (0..size).map(move |j| (i, j)))
    .filter(move |&(i, j)| conn.connected(i, j))
    .collect();
  let edge_weights = vec![edge_weights; edges.len()];

  (nodes, Just(edges), edge_weights).prop_map(|(nodes, edges, edge_weights)| {
    GraphSpec {
      nodes,
      edges: edges.into_iter().zip(edge_weights).collect()
    }
  })


  //
  // prop::collection::vec(node, size)
  //   .prop_flat_map(move |nodes| {
  //     let n = nodes.len();
  //     conn.set_max_possible_edges(n*(n-1));
  //
  //
  //     let edge_weights = vec![edge_weights; edges.len()];
  //
  //     (Just(nodes), Just(edges), edge_weights)
  //     // let edges = prop::collection::vec(edge_weights, edges.len())
  //     //   .prop_map(move |weights| {
  //     //     let edges: EdgeMap = edges.iter().copied().zip(weights).collect();
  //     //     GraphSpec { edges, nodes: nodes.clone() }
  //     //   });
  //   })
  //   .prop_map(|(nodes, edges, edge_weights)| {
  //     GraphSpec {
  //       nodes,
  //       edges: edges.into_iter().zip(edge_weights).collect()
  //     }
  //   })
}

const MAX_EDGE_WEIGHT: Weight = Weight::MAX / 2;
//
// // fn edge(min_node: usize, max_node: usize) -> impl Strategy<Value=EdgeData> {
// //   (min_node..=max_node, min_node..=max_node, 0..=MAX_EDGE_WEIGHT)
// //     .prop_filter_map(|(i, j, w)| {
// //       if i == j { Some(EdgeData { from: i, to: j, weight: w }) } else { None }
// //     })
// // }
//
// fn edge_with_weight(min_node: usize, max_node: usize, weight: impl Strategy<Value=Weight>) -> impl Strategy<Value=EdgeData> {
//   (min_node..=max_node, min_node..=max_node, weight)
//     .prop_filter_map(|(i, j, w)| {
//       if i == j { Some(EdgeData { from: i, to: j, weight: w }) } else { None }
//     })
// }
//
// //
// // fn scc_bounds(max_lb: Weight, min_ub: Weight) -> impl Strategy<Value=NodeData> {
// //   (..=max_lb, min_ub..)
// //     .prop_map(|lb, ub|)
// // }
//
fn complete_graph_zero_edges(nodes: impl Strategy<Value=NodeData> + Clone) -> impl Strategy<Value=GraphSpec> {
  (1..=8usize).prop_flat_map(move |size| graph(size, AllEdges(), nodes.clone(), Just(0)))
}

fn complete_graph_nonzero_edges(nodes: impl Strategy<Value=NodeData> + Clone) -> impl Strategy<Value=GraphSpec> {
  (1..=8usize).prop_flat_map(move |size| graph(size, AllEdges(), nodes.clone(),
                                          (-MAX_EDGE_WEIGHT..=MAX_EDGE_WEIGHT).prop_filter("nonzero edge", |w| w != &0)))
}

//
// // fn complete_graph_nonzero_edges(nodes: impl Strategy<Value=NodeData>) -> impl Strategy<Value=GraphSpec> {
// //   let nodes = prop::collection::vec(nodes, 0..8);
// //   let edges = Map::
// //
// //     .prop_map(|nodes| GraphSpec::with_node_data(nodes, AllEdges(0)))
// // }
