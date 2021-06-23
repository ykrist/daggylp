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

impl Graph {

  /// Find all Strongly-Connected Components with 2 or more nodes
  pub(crate) fn find_sccs(&mut self) -> Vec<FnvHashSet<usize>> {
    debug_assert_eq!(self.sccs.len(), 0);
    const UNDEF: usize = usize::MAX;

    #[derive(Copy, Clone)]
    struct NodeAttr {
      lowlink: usize,
      index: usize,
      onstack: bool,
    }


    fn tarjan(sccs: &mut Vec<FnvHashSet<usize>>, edges_from: &[Vec<Edge>], stack: &mut Vec<usize>, attr: &mut [NodeAttr], v: usize, next_idx: &mut usize) {
      let node_attr = &mut attr[v];
      node_attr.index = *next_idx;
      node_attr.lowlink = *next_idx;
      node_attr.onstack = true;
      stack.push(v);
      *next_idx += 1;

      for w in edges_from[v].iter().map(|e| e.to) {
        let w_attr = &attr[w];
        if w_attr.index == UNDEF {
          tarjan(sccs, edges_from, stack, attr, w, next_idx);
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

    let mut attr = vec![NodeAttr{ lowlink: UNDEF, index: UNDEF, onstack: false }; self.nodes.len()];
    let mut next_idx = 0;
    let mut stack = Vec::with_capacity(32);
    let mut sccs = Vec::new();

    for n in 0..self.nodes.len() {
      if attr[n].index == UNDEF {
        tarjan(&mut sccs, &self.edges_from, &mut stack, &mut attr, n, &mut next_idx);
      }
    }
    sccs
  }

  pub(crate) fn scc_is_feasible(&self, scc: &FnvHashSet<usize>) -> bool {
    for &n in scc {
      for e in &self.edges_from[n] {
        if e.weight > 0 && scc.contains(&e.to) {
          return false;
        }
      }
    }

    self.find_scc_bound_infeas(scc.iter().copied()).is_none()
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
        for e in &self.edges_to[n] {
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

        for e in &self.edges_from[n] {
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

    for edge_lookup in &mut [&mut self.edges_from, &mut self.edges_to] {
      edge_lookup.extend(std::iter::repeat_with(Vec::new).take(self.sccs.len()))
    }

    for ((from, to), e) in new_edges {
      self.edges_from[from].push(e);
      self.edges_to[to].push(e);
    }
  }

}