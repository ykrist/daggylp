// use proptest::prelude::*;
use proptest as prop;
use proptest::{prop_oneof, prelude::{Just, any}};
use super::data::*;
use crate::graph::Weight;
use fnv::FnvHashMap;
use std::collections::HashMap;
use proptest::arbitrary::arbitrary;
use proptest::test_runner::TestRunner;
use crate::test::SccGraphConn;
use std::fmt::Debug;
use std::cmp::min;
use crate::test::SccGraphConn::Sq;
use prop::strategy::Strategy as ProptestStrategy;
use proptest::strategy::ValueTree;
use proptest::sample::SizeRange;

pub trait SharableStrategy: ProptestStrategy<Value=<Self as SharableStrategy>::Value, Tree=<Self as SharableStrategy>::Tree> + Clone + Send + Sync + 'static {
  type Value: Send + Sync + 'static;
  type Tree: Debug + Clone + ValueTree<Value=<Self as SharableStrategy>::Value>;
}

impl<V, T, S> SharableStrategy for S
  where
    V: Send + Sync + 'static,
    T: Debug + Clone + ValueTree<Value=V>,
    S: ProptestStrategy<Value=V, Tree=T> + Clone + Send + Sync + 'static
{
  type Value = V;
  type Tree = T;
}


#[derive(Debug, Copy, Clone)]
pub enum SccKind {
  Feasible,
  InfEdge,
  InfBound,
}

impl SccKind {
  pub fn feasible(&self) -> bool { matches!(self, SccKind::Feasible) }

  pub fn any() -> impl SharableStrategy<Value=Self> {
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


pub fn node(lb: Weight, ub: Weight, obj: Weight) -> NodeData {
  NodeData { lb, ub, obj }
}

pub fn nodes(lb: impl SharableStrategy<Value=Weight>,
             ub: impl SharableStrategy<Value=Weight>,
             obj: impl SharableStrategy<Value=Weight>) -> impl SharableStrategy<Value=NodeData> {
  (lb, ub, obj).prop_map(|(mut lb,mut ub, obj)| {
    if lb > ub { std::mem::swap(&mut lb, &mut ub); }
    NodeData { lb, ub, obj }
  })
}

pub fn default_nodes(size: impl Into<SizeRange>) -> impl SharableStrategy<Value=Vec<NodeData>> {
  prop::collection::vec(
    Just(NodeData { lb: 0, ub: 1, obj: 0 }),
    size,
  )
}

pub fn any_nodes(size: impl Into<SizeRange>) -> impl SharableStrategy<Value=Vec<NodeData>> {
  prop::collection::vec(
    nodes(MIN_WEIGHT..=MAX_WEIGHT, MIN_WEIGHT..=MAX_WEIGHT, 0..=MAX_WEIGHT),
    size,
  )
}
pub fn any_bounds_nodes(size: impl Into<SizeRange>) -> impl SharableStrategy<Value=Vec<NodeData>> {
  prop::collection::vec(
    nodes(MIN_WEIGHT..=MAX_WEIGHT, MIN_WEIGHT..=MAX_WEIGHT, Just(0)),
    size,
  )
}

pub fn any_edge_weight() -> impl SharableStrategy<Value=Weight> { 0..MAX_EDGE_WEIGHT }

fn edge_iter(n: usize) -> impl Iterator<Item=(usize, usize)> {
  (0..n).flat_map(move |i| (0..n).map(move |j| (i, j))).filter(|(i, j)| i != j)
}

fn acyclic_edge_iter(n: usize) -> impl Iterator<Item=(usize, usize)> {
  (0..n).flat_map(move |i| (i + 1..n).map(move |j| (i, j)))
}

pub fn graph_with_conn(
  nodes: impl SharableStrategy<Value=Vec<NodeData>>,
  mut conn: impl Connectivity + Clone + Send + Sync + 'static,
  edge_weights: impl SharableStrategy<Value=Weight>,
) -> impl SharableStrategy<Value=GraphData>
{
  nodes.prop_ind_flat_map2(move |nodes: Vec<NodeData>| {
    let size = nodes.len();
    let mut conn = conn.clone();
    conn.set_size(size, size * (size - 1));
    let edges: Vec<_> = edge_iter(size)
      .filter(|&(i, j)| conn.connected(i, j))
      .collect();
    let weights = vec![edge_weights.clone(); edges.len()];
    (Just(edges), weights)
  }).prop_map(|(nodes, (edges, edge_weights))|
    GraphData {
      nodes,
      edges: edges.into_iter().zip(edge_weights).collect(),
    }
  )
}

fn graph_with_edgelist(
  size: usize,
  nodes: impl SharableStrategy<Value=NodeData> + Clone,
  edge_list: Vec<(usize, usize)>,
  edge_weights: impl SharableStrategy<Value=Weight> + Clone,
) -> impl SharableStrategy<Value=GraphData> {
  let n_edges = edge_list.len();
  (vec![nodes; size], Just(edge_list), vec![edge_weights; n_edges])
    .prop_map(|(nodes, edge_list, weights)| {
      let edges: FnvHashMap<_, _> = edge_list.into_iter().zip(weights).collect();
      GraphData::with_node_and_edge_data(nodes, edges)
    })
}

pub fn acyclic_graph(
  nodes: impl SharableStrategy<Value=Vec<NodeData>>,
  edge_weights: impl SharableStrategy<Value=Weight>,
) -> impl SharableStrategy<Value=GraphData>
{
  nodes
    .prop_ind_flat_map2(move |nodes: Vec<NodeData>| {
      let size = nodes.len();
      vec![prop::option::weighted(0.2, edge_weights.clone()); (size * (size - 1)) / 2]
    })
    .prop_map(|(nodes, edges)| {
      let edges = edges.into_iter()
        .zip(acyclic_edge_iter(nodes.len()))
        .filter_map(|(w, e)| w.map(|w| (e, w)))
        .collect();
      GraphData { edges, nodes }
    })
}

pub fn connected_acyclic_graph(
  nodes: impl SharableStrategy<Value=Vec<NodeData>>,
  edge_weights: impl SharableStrategy<Value=Weight>,
) -> impl SharableStrategy<Value=GraphData>
{
  nodes
    .prop_ind_flat_map2(move |nodes: Vec<NodeData>| {
      let size = nodes.len();
      vec![prop::option::weighted(0.2, edge_weights.clone()); (size * (size - 1)) / 2]
    })
    .prop_map(|(nodes, edges)| {
      let edges = edges.into_iter()
        .zip(acyclic_edge_iter(nodes.len()))
        .filter_map(|(mut w, (i, j))| {
          if i + 1 == j {
            w = w.or(Some(0));
          }
          w.map(|w| ((i, j), w))
        })
        .collect();
      GraphData { edges, nodes }
    })
}

pub fn graph(
  nodes: impl SharableStrategy<Value=Vec<NodeData>>,
  edge_weights: impl SharableStrategy<Value=Weight> + Clone,
) -> impl SharableStrategy<Value=GraphData>
{
  nodes
    .prop_ind_flat_map2(move |nodes: Vec<NodeData>| {
      let size = nodes.len();
      vec![prop::option::weighted(0.2, edge_weights.clone()); (size * (size - 1))]
    })
    .prop_map(|(nodes, edges)| {
      let edges = edges.into_iter()
        .zip(edge_iter(nodes.len()))
        .filter_map(|(mut w, (i, j))|
          w.map(|w| ((i, j), w))
        )
        .collect();
      GraphData { edges, nodes }
    })
}

pub const MAX_EDGE_WEIGHT: Weight = Weight::MAX / 1_000_000;
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
pub fn complete_graph_zero_edges(nodes: impl SharableStrategy<Value=Vec<NodeData>>) -> impl SharableStrategy<Value=GraphData> {
  graph_with_conn(nodes, AllEdges(), Just(0))
}

pub fn complete_graph_nonzero_edges(nodes: impl SharableStrategy<Value=Vec<NodeData>>) -> impl SharableStrategy<Value=GraphData> {
  graph_with_conn(nodes, AllEdges(), 1..=MAX_EDGE_WEIGHT)
}

pub fn scc_graph_conn(size: usize) -> impl SharableStrategy<Value=SccGraphConn> {
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
  size: impl Into<SizeRange>,
  feas: SccKind,
) -> impl SharableStrategy<Value=GraphData> {
  let nodes = match feas {
    SccKind::Feasible | SccKind::InfEdge =>
      nodes(MIN_WEIGHT..=0, 0..=MAX_WEIGHT, Just(0)),
    SccKind::InfBound =>
      nodes(MIN_WEIGHT..=MAX_WEIGHT, MIN_WEIGHT..=MAX_WEIGHT, Just(0)),
  };
  let nodes = prop::collection::vec(nodes, size);
  let edge_weights = match feas {
    SccKind::Feasible | SccKind::InfBound =>
      0..=0,
    SccKind::InfEdge =>
      0..=MAX_EDGE_WEIGHT,
  };

  nodes
    .prop_ind_flat_map2(|nodes| scc_graph_conn(nodes.len()))
    .prop_flat_map(move |(nodes, conn)|
      graph_with_conn(Just(nodes), conn, edge_weights.clone())
    )
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


pub fn set_arbitrary_edge_to_one(graph: impl SharableStrategy<Value=GraphData>) -> impl SharableStrategy<Value=GraphData> {
  graph.prop_ind_flat_map(|graph| {
    let n_edges = graph.edges.len();
    (Just(graph), 0usize..n_edges)
  })
    .prop_map(|(mut graph, i): (GraphData, _)| {
      *graph.edges.values_mut().nth(i).unwrap() = 1;
      graph
    })
}

// fn cycle_chain_graph_zero_edges(subgraph_sizes: Vec<usize>, nodes: impl Strategy<Value=NodeData> + Clone) ->

pub fn sample_strategy<T>(s: impl SharableStrategy<Value=T>) -> T {
  use proptest::strategy::ValueTree;
  s.new_tree(&mut TestRunner::default()).unwrap().current()
}
