use std::path::Path;
use fnv::FnvHashMap;
use crate::graph::*;
use proptest::option::of;
use std::fmt;
use std::io::Write;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct NodeData {
  pub ub: Weight,
  pub lb: Weight,
  pub obj: Weight,
}

impl fmt::Debug for NodeData {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    f.write_fmt(format_args!("Nd([{},{}], {})", self.lb, self.ub, self.obj))
  }
}

#[derive(Debug, Clone)]
pub struct GraphSpec {
  pub nodes: Vec<NodeData>,
  pub edges: FnvHashMap<(usize, usize), Weight>,
}

impl GraphSpec {
  pub fn build(&self) -> Graph {
    let mut g = Graph::new();

    let vars: Vec<_> = self.nodes
      .iter()
      .map(|data| g.add_var(data.obj, data.lb, data.ub))
      .collect();

    for (&(i, j), &d) in &self.edges {
      g.add_constr(vars[i], d, vars[j]);
    }

    g
  }

  pub fn add_node(&mut self, data: NodeData) -> usize {
    let n = self.nodes.len();
    self.nodes.push(data);
    n
  }

  pub fn with_node_data(nodes: Vec<NodeData>, mut conn: impl Connectivity, mut edge_weights: impl EdgeWeights) -> Self {
    let num_nodes = nodes.len();
    let num_possible_edges = num_nodes * (num_nodes - 1);
    conn.set_size(num_nodes, num_possible_edges);
    edge_weights.set_size(num_nodes, num_possible_edges);

    let edges = (0..nodes.len())
      .flat_map(|i| (0..nodes.len()).map(move |j| (i, j)))
      .filter(move |&(i, j)| i != j && conn.connected(i, j))
      .map(move |(i, j)| ((i, j), edge_weights.weight(i, j)))
      .collect();

    GraphSpec { nodes, edges }
  }

  pub fn new(size: usize, mut nodes: impl NodeSpec, mut conn: impl Connectivity, mut edge_weights: impl EdgeWeights) -> Self {
    let nodes: Vec<_> = (0..size)
      .map(|i| {
        NodeData {
          obj: nodes.obj(i).unwrap_or(0),
          lb: nodes.lb(i).unwrap_or(0),
          ub: nodes.ub(i).unwrap_or(Weight::MAX),
        }
      })
      .collect();

    Self::with_node_data(nodes, conn, edge_weights)
  }

  pub fn save_to_file(&self, path: impl AsRef<Path>) {
    let mut writer = std::io::BufWriter::new(
      std::fs::File::create(path).unwrap());

    for n in &self.nodes {
      writeln!(&mut writer, "{} {} {}", n.lb, n.ub, n.obj).unwrap();
    }
    writeln!(&mut writer, "edges").unwrap();
    for (&(i, j), &w) in &self.edges {
      writeln!(&mut writer, "{} {} {}", i, j, w).unwrap();
    }
  }

  pub fn load_from_file(path: impl AsRef<Path>) -> Self {
    let contents = std::fs::read_to_string(path).unwrap();
    let mut nodes = Vec::new();
    let mut edges = FnvHashMap::default();

    let mut parse_edges = false;

    for line in contents.lines() {
      let line = line.trim();
      match line {
        "edges" => {
          parse_edges = true;
        }
        line => {
          let mut tok = line.split_whitespace();
          if parse_edges {
            let i: usize = tok.next().unwrap().parse().unwrap();
            let j: usize = tok.next().unwrap().parse().unwrap();
            let d: Weight = tok.next().unwrap().parse().unwrap();
            edges.insert((i, j), d);
          } else {
            let lb: Weight = tok.next().unwrap().parse().unwrap();
            let ub: Weight = tok.next().unwrap().parse().unwrap();
            let obj: Weight = tok.next().unwrap().parse().unwrap();
            nodes.push(NodeData { lb, ub, obj })
          }
        }
      }
    }
    GraphSpec { nodes, edges }
  }

  // pub fn join(&mut self, other: &GraphSpec, mut conn_to_other: impl EdgeSpec, mut conn_from_other: impl EdgeSpec) {
  //   let node_offset = self.nodes.len();
  //   for (&(i, j), &e) in &other.edges {
  //     self.edges.insert((i + node_offset, j + node_offset), e);
  //   }
  //
  //   let num_possible_edges =  self.nodes.len()*other.nodes.len();
  //   conn_to_other.num_possible_edges(num_possible_edges);
  //   conn_from_other.num_possible_edges(num_possible_edges);
  //
  //   for i in 0..self.nodes.len() {
  //     for j in 0..other.nodes.len() {
  //       if let Some(w) = conn_to_other.weight(i, j) {
  //         self.edges.insert((i, j + node_offset), w);
  //       }
  //       if let Some(w) = conn_from_other.weight(j, i) {
  //         self.edges.insert((j + node_offset, i), w);
  //       }
  //     }
  //   }
  //   self.nodes.extend_from_slice(&other.nodes);
  // }

  pub fn from_components(subgraphs: Vec<GraphSpec>, conn: Vec<(usize, usize, Box<dyn Connectivity>, Box<dyn EdgeWeights>)>) -> Self {
    let subgraph_size: Vec<_> = subgraphs.iter().map(|g| g.nodes.len()).collect();
    let offsets: Vec<_> = {
      let mut offset = &mut 0;

      subgraph_size.iter().map(|&s| {
        let o = *offset;
        *offset += s;
        o
      }).collect()
    };

    let mut subgraphs = subgraphs.into_iter();

    let mut g = subgraphs.next().expect("at least one graph");
    g.nodes.reserve(subgraph_size[1..].iter().copied().sum());

    for (&offset, h) in subgraph_size[1..].iter().zip(subgraphs) {
      g.nodes.extend_from_slice(&h.nodes);
      g.edges.extend(h.edges.into_iter().map(|((i, j), w)| ((i + offset, j + offset), w)));
    }

    for (sg_i, sg_j, mut conn, mut weights) in conn {
      let ij_nodes = subgraph_size[sg_i] + subgraph_size[sg_j];
      let ij_edges = subgraph_size[sg_i] * subgraph_size[sg_j];
      conn.set_size(ij_nodes, ij_edges);
      weights.set_size(ij_nodes, ij_edges);

      for i in 0..subgraph_size[sg_i] {
        let ni = i + offsets[sg_i];
        for j in 0..subgraph_size[sg_j] {
          if conn.connected(i, j) {
            let nj = j + offsets[sg_i];
            g.edges.insert((ni, nj), weights.weight(i, j));
          }
        }
      }
    }
    g
  }
}


pub trait Connectivity {
  fn connected(&mut self, from: usize, to: usize) -> bool;

  fn set_size(&mut self, _nodes: usize, _edges: usize) {}

  fn mask<F: FnMut(usize, usize) -> bool>(self, mask: F) -> MaskEdges<Self, F> where Self: Sized {
    MaskEdges {
      orig: self,
      mask,
    }
  }
}

#[derive(Debug, Copy, Clone)]
pub struct AllEdges();

impl Connectivity for AllEdges {
  fn connected(&mut self, _: usize, _: usize) -> bool { true }
}

#[derive(Debug, Copy, Clone)]
pub struct Cycle(usize);

impl Cycle {
  pub fn new() -> Self { Cycle(0) }
}

impl Connectivity for Cycle {
  fn set_size(&mut self, nodes: usize, _: usize) { self.0 = nodes }

  fn connected(&mut self, from: usize, to: usize) -> bool {
    (from + 1) % self.0 == to
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
#[derive(Debug, Copy, Clone)]
pub struct Triangular();

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

#[derive(Debug, Copy, Clone)]
pub enum SccGraphConn {
  Tri(Triangular),
  Complete(AllEdges),
  Cycle(Cycle),
}

impl Connectivity for SccGraphConn {
  fn set_size(&mut self, nodes: usize, edges: usize) {
    match self {
      SccGraphConn::Cycle(c) => c.set_size(nodes, edges),
      _ => {}
    }
  }

  fn connected(&mut self, from: usize, to: usize) -> bool {
    match self {
      SccGraphConn::Cycle(c) => c.connected(from, to),
      SccGraphConn::Tri(c) => c.connected(from, to),
      SccGraphConn::Complete(c) => c.connected(from, to),
    }
  }
}

pub struct MaskEdges<E, F> {
  orig: E,
  mask: F,
}

impl<E: Connectivity, F: FnMut(usize, usize) -> bool> Connectivity for MaskEdges<E, F> {
  fn set_size(&mut self, nodes: usize, edges: usize) {
    self.orig.set_size(nodes, edges)
  }

  fn connected(&mut self, from: usize, to: usize) -> bool {
    (self.mask)(from, to) && self.orig.connected(from, to)
  }
}


pub trait EdgeWeights {
  fn set_size(&mut self, _nodes: usize, _edges: usize) {}

  fn weight(&mut self, from: usize, to: usize) -> Weight;

  fn map<F: FnMut(usize, usize, Weight) -> Weight>(self, map: F) -> MapWeights<Self, F> where Self: Sized {
    MapWeights { orig: self, map }
  }

  fn mask<F: FnMut(usize, usize) -> bool>(self, mask: F) -> MaskEdges<Self, F> where Self: Sized {
    MaskEdges { orig: self, mask }
  }
}

pub trait NodeSpec {
  fn lb(&mut self, _: usize) -> Option<Weight> { None }

  fn ub(&mut self, _: usize) -> Option<Weight> { None }

  fn obj(&mut self, _: usize) -> Option<Weight> { None }

  fn combine<N: NodeSpec>(self, other: N) -> CombinedNodeSpec<Self, N> where Self: Sized {
    CombinedNodeSpec { first: self, second: other }
  }

  fn set_num_nodes(&mut self, _: usize) {}
}

pub struct MapWeights<E, F> {
  orig: E,
  map: F,
}

impl<E: EdgeWeights, F: FnMut(usize, usize, Weight) -> Weight> EdgeWeights for MapWeights<E, F> {
  fn set_size(&mut self, nodes: usize, edges: usize) {
    self.orig.set_size(nodes, edges)
  }

  fn weight(&mut self, from: usize, to: usize) -> Weight {
    (self.map)(from, to, self.orig.weight(from, to))
  }
}


pub struct NoEdges();

impl Connectivity for NoEdges {
  fn connected(&mut self, _: usize, _: usize) -> bool { false }
}

pub struct CombinedNodeSpec<A, B> {
  first: A,
  second: B,
}


macro_rules! first_then_second {
  ($method:ident) => {
      fn $method(&mut self, i: usize) -> Option<Weight> {
        self.first.$method(i).or_else(|| self.second.$method(i))
      }
  };

  ($($method:ident),+) => {
    $( first_then_second!{ $method } )*
  };

}

impl<A: NodeSpec, B: NodeSpec> NodeSpec for CombinedNodeSpec<A, B> {
  first_then_second! { ub, lb, obj }
}

pub struct DefaultNodeSpec();

impl NodeSpec for DefaultNodeSpec {}

pub struct IdenticalNodes {
  pub lb: Weight,
  pub ub: Weight,
  pub obj: Weight,
}

impl NodeSpec for IdenticalNodes {
  fn lb(&mut self, _: usize) -> Option<Weight> { Some(self.lb) }
  fn ub(&mut self, _: usize) -> Option<Weight> { Some(self.ub) }
  fn obj(&mut self, _: usize) -> Option<Weight> { Some(self.obj) }
}
