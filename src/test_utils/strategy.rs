use proptest::prelude::*;
use super::generators::*;
use crate::graph::Weight;
use fnv::FnvHashMap;
use std::collections::HashMap;
use proptest::arbitrary::arbitrary;
use proptest::test_runner::TestRunner;
use crate::test_utils::SccGraphConn;
use crate::viz::GraphViz;
use std::fmt::Debug;


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

pub fn node(bounds: impl Strategy<Value=(Weight, Weight)> + Clone, obj: impl Strategy<Value=Weight> + Clone) -> impl Strategy<Value=NodeData> + Clone {
  (obj, bounds).prop_map(|(obj, (lb, ub))| NodeData { lb, ub, obj })
}

pub fn default_nodes() -> impl Strategy<Value=NodeData> + Clone {
  Just(NodeData{ lb: 0, ub: 1, obj: 0 })
}

pub fn graph(
  size: usize,
  mut conn: impl Connectivity,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec>
{
  dbg!(size);
  assert!(size > 0);
  conn.set_size(size, size * (size - 1));
  let nodes = vec![nodes; size];
  let edges: Vec<_> = (0..size).flat_map(|i| (0..size).map(move |j| (i, j)))
    .filter(move |&(i, j)| i != j && conn.connected(i, j))
    .collect();
  let edge_weights = vec![edge_weights; edges.len()];

  (nodes, Just(edges), edge_weights).prop_map(|(nodes, edges, edge_weights)|
    GraphSpec {
      nodes,
      edges: edges.into_iter().zip(edge_weights).collect(),
    }
  )
}

pub const MAX_EDGE_WEIGHT: Weight = Weight::MAX / 2;
pub const MAX_WEIGHT: Weight = MAX_EDGE_WEIGHT;
pub const MIN_WEIGHT: Weight = -MAX_WEIGHT;


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
pub fn complete_graph_zero_edges(nodes: impl Strategy<Value=NodeData> + Clone) -> impl Strategy<Value=GraphSpec> {
  (2..=8usize).prop_flat_map(move |size| graph(size, AllEdges(), nodes.clone(), Just(0)))
}

pub fn complete_graph_nonzero_edges(nodes: impl Strategy<Value=NodeData> + Clone) -> impl Strategy<Value=GraphSpec> {
  (2..=8usize).prop_flat_map(move |size| graph(size, AllEdges(), nodes.clone(),
                                               (-MAX_EDGE_WEIGHT..=MAX_EDGE_WEIGHT).prop_filter("nonzero edge", |w| w != &0)))
}

pub fn scc_graph_conn() -> impl Strategy<Value=SccGraphConn> {
  prop_oneof![
    Just(SccGraphConn::Cycle(Cycle::new())),
    Just(SccGraphConn::Tri(Triangular())),
    Just(SccGraphConn::Complete(AllEdges())),
  ]
}

pub fn scc_graph(
  size: usize,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec> {
  (scc_graph_conn()).prop_flat_map(move |conn| {
    graph(size, conn, nodes.clone(), edge_weights.clone())
  })
}

pub fn feasible_scc(size: usize, scc_bounds: (Weight, Weight)) -> impl Strategy<Value=GraphSpec> {
  let (lb, ub) = scc_bounds;
  scc_graph(
    size,
    node((MIN_WEIGHT..=lb, ub..=MAX_WEIGHT), Just(0)),
    Just(0),
  )
}

pub fn cycle_bound_infeasible_scc(size: usize) -> impl Strategy<Value=GraphSpec> {
  scc_graph(
    size,
    node((MIN_WEIGHT..=MAX_WEIGHT, MIN_WEIGHT..=MAX_WEIGHT), Just(0)),
    Just(0),
  ).prop_map(|mut graph| {
    let min_ub_node = (0..graph.nodes.len()).min_by_key(|&n| graph.nodes[n].ub).unwrap();
    let max_lb_node = (0..graph.nodes.len()).min_by_key(|&n| graph.nodes[n].lb).unwrap();

    let lb = graph.nodes[max_lb_node].lb;
    let ub = graph.nodes[min_ub_node].ub;
    if lb <= ub {
      graph.nodes[max_lb_node].lb = ub;
      graph.nodes[min_ub_node].ub = lb;
    }
    graph
  })
}

pub fn cycle_edge_infeasible_scc(size: usize) -> impl Strategy<Value=GraphSpec> {
  scc_graph(
    size,
    node((MIN_WEIGHT..=0, 0..=MAX_WEIGHT), Just(0)),
    MIN_WEIGHT..=MAX_WEIGHT,
  ).prop_map(|mut graph| {
    let mut last_edge = None;
    for (_, w) in graph.edges.iter_mut() {
      if w != &0 {
        return graph;
      }
      last_edge = Some(w);
    }
    *last_edge.expect("at least one edge") = 0;
    graph
  })
}

pub fn set_arbitrary_edge_to_one(graph: impl Strategy<Value=GraphSpec>) -> impl Strategy<Value=GraphSpec> {
  (graph, any::<prop::sample::Selector>())
    .prop_map(|(mut graph, selector)| {
      let e = *selector.select(graph.edges.keys());
      graph.edges.insert(e, 1);
      graph
    })
}

// fn cycle_chain_graph_zero_edges(subgraph_sizes: Vec<usize>, nodes: impl Strategy<Value=NodeData> + Clone) ->

pub fn sample_strategy<T>(s: impl Strategy<Value=T>) -> T {
  use proptest::strategy::ValueTree;
  s.new_tree(&mut TestRunner::default()).unwrap().current()
}

//
// #[cfg(test)]
// mod tests {
//   use super::*;
//   use crate::viz::GraphViz;
// }