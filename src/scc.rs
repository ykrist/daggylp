use std::cmp::min;
use fnv::{FnvHashMap, FnvHashSet};
use crate::graph::*;
use crate::set_with_capacity;


#[derive(Debug)]
pub struct SccInfo {
  pub nodes: FnvHashSet<usize>,
  pub lb_node: usize,
  pub ub_node: usize,
  pub scc_node: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundInfeas {
  pub ub_node: usize,
  pub ub: Weight,
  pub lb_node: usize,
  pub lb: Weight,
}

impl BoundInfeas {
  #[inline]
  fn new(n: usize, node: &Node) -> Self {
    BoundInfeas {
      ub: node.ub,
      lb: node.lb,
      lb_node: n,
      ub_node: n,
    }
  }

  #[inline]
  fn update(&mut self, n: usize, node: &Node) {
    if self.lb < node.lb {
      self.lb = node.lb;
      self.lb_node = n;
    }
    if self.ub > node.ub {
      self.ub = node.ub;
      self.ub_node = n;
    }
  }
}


#[derive(Copy, Clone)]
struct NodeAttr {
  lowlink: usize,
  index: usize,
  onstack: bool,
}
const UNDEF: usize = usize::MAX;

impl<E: EdgeLookup> Graph<E> {
  fn tarjan(&self, sccs: &mut Vec<FnvHashSet<usize>>, stack: &mut Vec<usize>, attr: &mut [NodeAttr], v: usize, next_idx: &mut usize) {
    let node_attr = &mut attr[v];
    node_attr.index = *next_idx;
    node_attr.lowlink = *next_idx;
    node_attr.onstack = true;
    stack.push(v);
    *next_idx += 1;

    for w in self.edges.successor_nodes(v) {
      let w_attr = &attr[w];
      if w_attr.index == UNDEF {
        self.tarjan(sccs, stack, attr, w, next_idx);
        let w_lowlink = attr[w].lowlink;
        let v_attr = &mut attr[v];
        v_attr.lowlink = min(v_attr.lowlink, w_lowlink);
      } else if w_attr.onstack {
        let w_index = w_attr.index;
        let v_attr = &mut attr[v];
        v_attr.lowlink = min(v_attr.lowlink, w_index);
      }
    }

    let v_attr = &mut attr[v];
    if v_attr.lowlink == v_attr.index {
      let w = stack.pop().unwrap();
      attr[w].onstack = false;

      if w != v { // ignore trivial SCCs of size 1
        let mut scc = set_with_capacity(16);
        scc.insert(w);
        loop {
          let w = stack.pop().unwrap();
          attr[w].onstack = false;
          scc.insert(w);
          if w == v {
            break;
          }
        }
        sccs.push(scc);
      }
    }
  }

  /// Find all Strongly-Connected Components with 2 or more nodes
  pub(crate) fn find_sccs(&mut self) -> Vec<FnvHashSet<usize>> {
    debug_assert_eq!(self.sccs.len(), 0);


    let mut attr = vec![NodeAttr{ lowlink: UNDEF, index: UNDEF, onstack: false }; self.nodes.len()];
    let mut next_idx = 0;
    let mut stack = Vec::with_capacity(32);
    let mut sccs = Vec::new();

    for n in 0..self.nodes.len() {
      if attr[n].index == UNDEF {
        self.tarjan(&mut sccs, &mut stack, &mut attr, n, &mut next_idx);
      }
    }
    sccs
  }

  /// Returns a bound infeasibility if one exists.  If `extrema = true`, finds the greatest violation (min UB, max LB)
  /// Returns None if SCC is feasible.
  pub(crate) fn find_scc_bound_infeas(&self, scc: impl IntoIterator<Item=usize>, extrema: bool) -> Option<BoundInfeas> {
    let mut nodes = scc.into_iter().map(|n| (n, &self.nodes[n]));
    let (n, first_node) = nodes.next().expect("expected non-empty iterator");
    let mut bi = BoundInfeas::new(n, first_node);

    if extrema {
      nodes.for_each(|(n, node)| bi.update(n, node));
    } else {
      for (n, node) in nodes {
        if bi.ub < bi.lb { return Some(bi) }
        bi.update(n, node);
      }
    }

    if bi.ub < bi.lb {
      Some(bi)
    } else {
      None
    }
  }


  // TODO: make this return the scc kind
  pub(crate) fn scc_is_feasible(&self, scc: &FnvHashSet<usize>) -> bool {
    for &n in scc {
      for e in self.edges.successors(n) {
        if e.weight > 0 && scc.contains(&e.to) {
          return false;
        }
      }
    }

    self.find_scc_bound_infeas(scc.iter().copied(), false).is_none()
  }

  pub(crate) fn condense(&mut self, sccs: Vec<FnvHashSet<usize>>) {
    use EdgeKind::*;
    // Add new SCC nodes
    for scc in sccs {
      let (lb_node, lb) = scc.iter().map(|&n| (n, self.nodes[n].lb))
        .max_by_key(|pair| pair.1).unwrap();
      let (ub_node, ub) = scc.iter().map(|&n| (n, self.nodes[n].ub))
        .min_by_key(|pair| pair.1).unwrap();

      let scc_idx = self.sccs.len();
      let scc_n = self.nodes.len();
      let scc_node = Node {
        x: lb,
        ub,
        lb,
        obj: 0,
        kind: NodeKind::Scc(scc_idx),
        active_pred: None,
      };
      self.nodes.push(scc_node);

      for &n in &scc {
        self.nodes[n].kind = NodeKind::SccMember(scc_idx);
      }

      self.sccs.push(SccInfo {
        nodes: scc,
        scc_node: scc_n,
        lb_node,
        ub_node,
      });
    }

    // Add new edges in and out of the SCC
    let mut new_edges = FnvHashMap::<(usize, usize), Edge>::default();
    let mut add_edge = |new_edges: &mut FnvHashMap<(usize, usize), Edge>, edge: Edge| {
      new_edges.entry((edge.from, edge.to))
        .and_modify(|e| if e.weight < edge.weight { *e = edge })
        .or_insert(edge);
    };

    for scc in &self.sccs {
      for &n in &scc.nodes {
        for e in self.edges.predecessors(n) {
          let mut e = *e;
          if scc.nodes.contains(&e.from) { continue }

          match self.nodes[e.from].kind {
            NodeKind::Var => {
              e.kind = SccIn(e.to);
            },
            NodeKind::SccMember(k) => {
              e.kind = SccToScc { from: e.from, to: e.to };
              e.from = self.sccs[k].scc_node;
            },
            NodeKind::Scc(_) => unreachable!(),

          };
          e.to = scc.scc_node;
          add_edge(&mut new_edges, e);
        }

        for e in self.edges.successors(n) {
          let mut e = *e;
          if scc.nodes.contains(&e.to) { continue}
          match self.nodes[e.to].kind {
            NodeKind::Var => {
              e.kind = SccOut(e.from)
            },
            NodeKind::SccMember(k) => {
              e.kind = SccToScc { from: e.from, to: e.to };
              e.to = self.sccs[k].scc_node;
            },
            NodeKind::Scc(_) => unreachable!(),
          };
          e.from = scc.scc_node;
          add_edge(&mut new_edges, e);
        }
      }
    }
    self.edges.add_new_component(self.sccs.len(), new_edges);
  }
}

#[cfg(test)]
mod tests {
  #[macro_use]
  use crate::*;
  use crate::test_utils::*;
  use crate::test_utils::strategy::*;
  use proptest::prelude::*;
  use serde::{Serialize, Deserialize};
  use crate::graph::Graph;
  use proptest::test_runner::TestCaseResult;
  use crate::viz::LayoutAlgo;

  #[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
  struct SccSizeOrder(Vec<usize>);

  impl SccSizeOrder {
    fn sorted(&self) -> Vec<usize> {
      let mut v = self.0.clone();
      v.sort();
      v
    }
  }

  fn multi_scc_graph() -> impl SharableStrategy<Value=(GraphSpec, SccSizeOrder)> {
    let scc = (4..=8usize).prop_flat_map(|s| scc_graph(s, SccKind::Feasible));
    prop::collection::vec(scc, 1..=50)
      .prop_map(|sccs| {
        let sizes = SccSizeOrder(sccs.iter().map(|g| g.nodes.len()).collect());
        let conn = (1..sccs.len()).map(|child_scc| {
          let parent_scc = (child_scc - 1) / 2;
          let conn : Box<dyn Connectivity> = Box::new(AllEdges());
          let weights : Box<dyn EdgeWeights> = Box::new(AllSame(1));
          (child_scc, parent_scc, conn, weights)
        }).collect();
        let sz : Vec<_> = sccs.iter().map(|s| s.nodes.len()).collect();
        (GraphSpec::from_components(sccs, conn), sizes)
      })
  }

  struct Tests;
  impl Tests {
    fn scc_sizes(g: &mut Graph, actual_sizes: SccSizeOrder) -> TestCaseResult {
      let sccs = g.find_sccs();
      let sizes = SccSizeOrder(sccs.iter().map(|l| l.len()).collect());
      prop_assert_eq!(sizes.sorted(), actual_sizes.sorted());
      Ok(())
    }
  }

  // graph_test_dbg!(Tests; scc_sizes _);

  graph_tests!{
    Tests;
    multi_scc_graph() => scc_sizes [layout=LayoutAlgo::Fdp];
  }
}