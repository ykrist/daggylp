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
use std::cmp::min;
use crate::test_utils::SccGraphConn::Sq;

#[derive(Debug, Copy, Clone)]
pub enum SccKind {
  Feasible,
  InfEdge,
  InfBound,
}

impl SccKind {
  pub fn feasible(&self) -> bool { matches!(self, SccKind::Feasible) }

  pub fn any() -> impl Strategy<Value=Self> {
    prop_oneof![
      Just(SccKind::Feasible),
      Just(SccKind::InfEdge),
      Just(SccKind::InfBound),
    ]
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

pub fn node(bounds: impl Strategy<Value=(Weight, Weight)> + Clone, obj: impl Strategy<Value=Weight> + Clone) -> impl Strategy<Value=NodeData> + Clone {
  (obj, bounds).prop_map(|(obj, (lb, ub))| NodeData { lb, ub, obj })
}

pub fn default_nodes() -> impl Strategy<Value=NodeData> + Clone {
  Just(NodeData { lb: 0, ub: 1, obj: 0 })
}

pub fn graph_with_conn(
  size: usize,
  mut conn: impl Connectivity,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec>
{
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

fn graph_with_edgelist(
  size: usize,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_list: Vec<(usize, usize)>,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec> {
  let n_edges = edge_list.len();
  (vec![nodes; size], Just(edge_list), vec![edge_weights; n_edges])
    .prop_map(|(nodes, edge_list, weights)| {
      let edges: FnvHashMap<_, _> = edge_list.into_iter().zip(weights).collect();
      GraphSpec::with_node_and_edge_data(nodes, edges)
    })
}

pub fn acyclic_graph(
  size: usize,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec>
{
  assert!(size > 0);
  vec![any::<bool>(); size * (size - 1) / 2]
    .prop_map(move |sparsity_pattern: Vec<bool>| {
      (0..size).flat_map(|i| (i + 1..size).map(move |j| (i, j)))
        .zip(sparsity_pattern)
        .filter_map(|(edge, is_present)| if is_present { Some(edge) } else { None })
        .collect::<Vec<_>>()
    })
    .prop_flat_map(move |edge_list: Vec<(usize, usize)>|
      graph_with_edgelist(size, nodes.clone(), edge_list, edge_weights.clone())
    )
}

pub fn connected_acyclic_graph(
  size: usize,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec>
{
  assert!(size > 0);
  vec![any::<bool>(); size * (size - 1) / 2]
    .prop_map(move |sparsity_pattern: Vec<bool>| {
      (0..size).flat_map(|i| (i + 1..size).map(move |j| (i, j)))
        .zip(sparsity_pattern)
        .filter_map(|((i,j), is_present)| {
          if i + 1== j || is_present  { Some((i,j)) } else { None }
        })
        .collect::<Vec<_>>()
    })
    .prop_flat_map(move |edge_list: Vec<(usize, usize)>|
      graph_with_edgelist(size, nodes.clone(), edge_list, edge_weights.clone())
    )
}

pub fn graph(
  size: usize,
  nodes: impl Strategy<Value=NodeData> + Clone,
  edge_weights: impl Strategy<Value=Weight> + Clone,
) -> impl Strategy<Value=GraphSpec>
{
  assert!(size > 0);
  vec![any::<bool>(); size * (size - 1)]
    .prop_map(move |sparsity_pattern: Vec<bool>| {
      (0..size).flat_map(|i| (0..size).map(move |j| (i, j))).filter(|(i, j)| i != j)
        .zip(sparsity_pattern)
        .filter_map(|(edge, is_present)| if is_present { Some(edge) } else { None })
        .collect::<Vec<_>>()
    })
    .prop_flat_map(move |edge_list: Vec<(usize, usize)>|
      graph_with_edgelist(size, nodes.clone(), edge_list, edge_weights.clone())
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
  (2..=8usize).prop_flat_map(move |size| graph_with_conn(size, AllEdges(), nodes.clone(), Just(0)))
}

pub fn complete_graph_nonzero_edges(nodes: impl Strategy<Value=NodeData> + Clone) -> impl Strategy<Value=GraphSpec> {
  (2..=8usize).prop_flat_map(move |size| graph_with_conn(size, AllEdges(), nodes.clone(),
                                                         (-MAX_EDGE_WEIGHT..=MAX_EDGE_WEIGHT).prop_filter("nonzero edge", |w| w != &0)))
}

pub fn scc_graph_conn(size: usize) -> impl Strategy<Value=SccGraphConn> {
  let tri = if Triangular::strongly_connected(size) { 1 } else { 0 };
  let sq = if Square::strongly_connected(size) { 1 } else { 0 };
  let complete = if size < 9 { 1 } else { 0 };
  prop_oneof![
    3 => Just(SccGraphConn::Cycle(Cycle::new())),
    tri => Just(SccGraphConn::Tri(Triangular())),
    sq => Just(SccGraphConn::Sq(Square::new())),
    complete => Just(SccGraphConn::Complete(AllEdges())),
  ]
}

pub fn scc_graph(
  size: usize,
  feas: SccKind,
) -> impl Strategy<Value=GraphSpec> {
  let nodes = match feas {
    SccKind::Feasible | SccKind::InfEdge =>
      node((MIN_WEIGHT..=0, 0..=MAX_WEIGHT), Just(0)),
    SccKind::InfBound =>
      node((MIN_WEIGHT..=MAX_WEIGHT, MIN_WEIGHT..=MAX_WEIGHT), Just(0)),
  };
  let edge_weights = match feas {
    SccKind::Feasible | SccKind::InfBound =>
      0..=0,
    SccKind::InfEdge =>
      0..=MAX_EDGE_WEIGHT,
  };

  scc_graph_conn(size)
    .prop_flat_map(move |conn| {
      println!("conn = {:?}", conn);
      graph_with_conn(size, conn, nodes.clone(), edge_weights.clone())
    })
    .prop_map(move |mut g| {
      match feas {
        SccKind::InfEdge => {
          if g.edges.values().all(|w| w == &0) {
            *g.edges.values_mut().next().unwrap() = 1;
          }
        }
        SccKind::InfBound => {
          let max_lb = g.nodes.iter().map(|n| n.lb).max().unwrap();
          let min_ub = g.nodes.iter().map(|n| n.ub).min().unwrap();
          if max_lb <= min_ub {
            g.nodes.first_mut().unwrap().lb = min_ub + 1;
          }
        }
        SccKind::Feasible => {}
      }
      g
    })
}

//
// pub fn cycle_bound_infeasible_scc(size: usize) -> impl Strategy<Value=GraphSpec> {
//   scc_graph(
//     size,
//     node((MIN_WEIGHT..=MAX_WEIGHT, MIN_WEIGHT..=MAX_WEIGHT), Just(0)),
//     Just(0),
//   ).prop_map(|mut graph| {
//     let min_ub_node = (0..graph.nodes.len()).min_by_key(|&n| graph.nodes[n].ub).unwrap();
//     let max_lb_node = (0..graph.nodes.len()).min_by_key(|&n| graph.nodes[n].lb).unwrap();
//
//     let lb = graph.nodes[max_lb_node].lb;
//     let ub = graph.nodes[min_ub_node].ub;
//     if lb <= ub {
//       graph.nodes[max_lb_node].lb = ub;
//       graph.nodes[min_ub_node].ub = lb;
//     }
//     graph
//   })
// }
//
// pub fn cycle_edge_infeasible_scc(size: usize) -> impl Strategy<Value=GraphSpec> {
//   scc_graph(
//     size,
//     node((MIN_WEIGHT..=0, 0..=MAX_WEIGHT), Just(0)),
//     MIN_WEIGHT..=MAX_WEIGHT,
//   ).prop_map(|mut graph| {
//     let mut last_edge = None;
//     for (_, w) in graph.edges.iter_mut() {
//       if w != &0 {
//         return graph;
//       }
//       last_edge = Some(w);
//     }
//     *last_edge.expect("at least one edge") = 0;
//     graph
//   })
// }


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