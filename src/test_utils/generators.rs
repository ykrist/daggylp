use std::path::Path;
use fnv::FnvHashMap;
use crate::graph::*;
use std::io::Write;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeData {
  pub ub: Weight,
  pub lb: Weight,
  pub obj: Weight,
}

#[derive(Debug, Clone)]
pub struct GraphGen {
  pub nodes: Vec<NodeData>,
  pub edges: FnvHashMap<(usize, usize), Weight>,
}

impl GraphGen {
  pub fn build(self) -> Graph {
    let mut g = Graph::new();

    let vars: Vec<_> = self.nodes
      .into_iter()
      .map(|data| g.add_var(data.obj, data.lb, data.ub))
      .collect();

    for ((i, j), d) in self.edges {
      g.add_constr(vars[i], d, vars[j]);
    }

    g
  }

  pub fn add_node(&mut self, data: NodeData) -> usize {
    let n = self.nodes.len();
    self.nodes.push(data);
    n
  }

  pub fn new(size: usize, mut nodes: impl NodeSpec, mut edges: impl EdgeSpec) -> Self {
    let nodes = (0..size)
      .map(|i| {
        NodeData {
          obj: nodes.obj(i).unwrap_or(0),
          lb: nodes.lb(i).unwrap_or(0),
          ub: nodes.ub(i).unwrap_or(Weight::MAX),
        }
      })
      .collect();

    let edges = (0..size)
      .flat_map(|i| (0..size).map(move |j| (i, j)))
      .filter_map(move |(i, j)| edges.weight(i, j).map(|w| ((i, j), w)))
      .collect();

    GraphGen { nodes, edges }
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
    GraphGen { nodes, edges }
  }

  pub fn join(&mut self, other: &GraphGen, mut conn_to_other: impl EdgeSpec, mut conn_from_other: impl EdgeSpec) {
    let node_offset = self.nodes.len();
    for (&(i, j), &e) in &other.edges {
      self.edges.insert((i + node_offset, j + node_offset), e);
    }
    for i in 0..self.nodes.len() {
      for j in 0..other.nodes.len() {
        if let Some(w) = conn_to_other.weight(i, j) {
          self.edges.insert((i, j + node_offset), w);
        }
        if let Some(w) = conn_from_other.weight(j, i) {
          self.edges.insert((j + node_offset, i), w);
        }
      }
    }
    self.nodes.extend_from_slice(&other.nodes);
  }
}


pub trait EdgeSpec {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight>;

  fn map_weights<F: FnMut(usize, usize, Weight) -> Weight>(self, map: F) -> MapWeights<Self, F> where Self: Sized {
    MapWeights { orig: self, map }
  }

  fn mask<F: FnMut(usize, usize) -> bool>(self, mask: F) -> MaskEdges<Self, F> where Self: Sized {
    MaskEdges { orig: self, mask }
  }
}

impl<'a, T: EdgeSpec> EdgeSpec for &'a mut T {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> { self.weight(from, to) }
}

pub struct MapWeights<E, F> {
  orig: E,
  map: F,
}

impl<E: EdgeSpec, F: FnMut(usize, usize, Weight) -> Weight> EdgeSpec for MapWeights<E, F> {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> {
    self.orig.weight(from, to).map(|w| (self.map)(from, to, w))
  }
}


pub struct MaskEdges<E, F> {
  orig: E,
  mask: F,
}

impl<E: EdgeSpec, F: FnMut(usize, usize) -> bool> EdgeSpec for MaskEdges<E, F> {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> {
    if (self.mask)(from, to) {
      self.orig.weight(from, to)
    } else {
      None
    }
  }
}

pub struct NoEdges();

impl EdgeSpec for NoEdges {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> { None }
}

pub struct AllEdges(pub Weight);

impl EdgeSpec for AllEdges {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> {
    if from == to { None } else { Some(self.0) }
  }
}

pub struct CycleEdges {
  pub n: usize,
  pub weight: Weight
}

impl EdgeSpec for CycleEdges {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> {
    if (from + 1) % self.n == to {
      Some(self.weight)
    } else {
      None
    }
  }
}

pub struct ArbitraryKEdges {
  pub k: usize,
  pub weight: Weight
}

impl EdgeSpec for ArbitraryKEdges {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> {
    if self.k > 0 {
      self.k -= 1;
      Some(self.weight)
    } else {
      None
    }
  }
}

pub struct SingleEdge {
  pub from: usize,
  pub to: usize,
  pub weight: Weight,
}

impl EdgeSpec for SingleEdge {
  fn weight(&mut self, from: usize, to: usize) -> Option<Weight> {
    if (from, to) == (self.from, self.to) {
      Some(self.weight)
    } else {
      None
    }
  }
}

pub trait NodeSpec {
  fn lb(&mut self, i: usize) -> Option<Weight> { None }

  fn ub(&mut self, i: usize) -> Option<Weight> { None }

  fn obj(&mut self, i: usize) -> Option<Weight> { None }

  fn combine<N: NodeSpec>(self, other: N) -> CombinedNodeSpec<Self, N> where Self: Sized {
    CombinedNodeSpec { first: self, second: other }
  }
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
