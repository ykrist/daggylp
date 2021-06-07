use super::graph::*;
use fnv::FnvHashSet;
use crate::set_with_capacity;

impl Graph {
  /// return the shortest path along edges in the SCC.  Will only find paths strictly shorter than `prune`.
  pub(crate) fn shortest_path_scc(&self, scc: &FnvHashSet<usize>, start: usize, end: usize, prune: Option<u32>) -> Option<Vec<usize>> {
    let prune = prune.unwrap_or(u32::MAX);
    todo!()
  }

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
      if max_lb < min_ub {
        return Some(((max_lb_node, max_lb), (min_ub_node, min_ub)));
      }
    }

    None
  }



  /// Try to find an IIS which consists only of edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  /// Returns `None` if no such IIS exists in the SCC.
  fn find_pure_cyclic_iis(&self, scc: &FnvHashSet<usize>, smallest: bool, prune: Option<u32>) -> Option<Iis> {
    let mut smallest_iis = None; // TODO optimisation: can re-use this allocation
    let mut search_max_iis_size = prune;

    for &n in scc {
      for e in &self.edges_from[n] {
        if e.weight > 0 && scc.contains(&e.to) {
          if let Some(p) = self.shortest_path_scc(scc, e.to, e.from, search_max_iis_size) {
            let iis_size = p.len();
            let mut constrs = set_with_capacity(iis_size);
            constrs.insert(Constraint::Edge(self.var_from_node_id(e.from), self.var_from_node_id(e.to)));
            let mut vars = p.into_iter().map(|n| self.var_from_node_id(n));
            let mut vi = vars.next().unwrap();
            for vj in vars {
              constrs.insert(Constraint::Edge(vi, vj));
              vi = vj;
            }
            let iis = Iis{ constrs};
            if !smallest {
              return Some(iis)
            }
            smallest_iis = Some(iis);
            search_max_iis_size = Some(iis_size as u32);
          }
        }
      }
    }
    smallest_iis
  }

  fn cycle_iis_from_path_pair(&self, forward: &[usize], backward: &[usize]) -> Iis {
    Iis::from_cycle(forward.iter()
      .chain(&backward[1..backward.len()-1])
      .map(|&n| self.var_from_node_id(n)))
  }

  /// Try to find an IIS which consists an Upper bound, Lower bound and edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  fn find_bound_cycle_iis(&self, scc: &FnvHashSet<usize>) -> Option<Iis> {
    if let Some(((max_lb_node, _), (min_ub_node, _))) = self.find_scc_bound_infeas(scc.iter().copied()) {
      let p1 = self.shortest_path_scc(scc,  max_lb_node, min_ub_node, None).unwrap();
      let p2 = self.shortest_path_scc(scc,  min_ub_node, max_lb_node, None).unwrap();
      let mut iis = self.cycle_iis_from_path_pair(&p1, &p2);
      iis.add_constraint(Constraint::Lb(self.var_from_node_id(max_lb_node)));
      iis.add_constraint(Constraint::Ub(self.var_from_node_id(min_ub_node)));
      Some(iis)
    } else {
      None
    }
  }

  /// Try to find the *smallest* IIS which consists an Upper bound, Lower bound and edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  /// If `prune` is `Some(k)`, only looks for IIS strictly smaller than `k`
  /// Returns `None` if no such IIS exists in the SCC.
  fn find_min_bound_cycle_iis(&self, scc: &FnvHashSet<usize>, prune: Option<u32>) -> Option<Iis> {
    let mut smallest_iis = None; // TODO optimisation: can re-use this allocation
    let mut search_max_cycle_edge_count = prune.map(|n| n-2); // will always contain two bounds

    for &n1 in scc {
      for &n2 in scc {
        if self.nodes[n1].ub < self.nodes[n2].lb {
          let mut budget = search_max_cycle_edge_count;
          if let Some(p1) = self.shortest_path_scc(scc,  n1, n2, budget) {
            budget = budget.map(|n| n + 1 - p1.len() as u32);
            if let Some(p2) = self.shortest_path_scc(scc,  n2, n1, budget) {
              let mut iis = self.cycle_iis_from_path_pair(&p1, &p2);
              iis.add_constraint(Constraint::Ub(self.var_from_node_id(n1)));
              iis.add_constraint(Constraint::Lb(self.var_from_node_id(n2)));
              smallest_iis = Some(iis);
              search_max_cycle_edge_count = Some((p1.len() + p2.len() - 2) as u32);
              if let Some(sz) = search_max_cycle_edge_count {
                debug_assert!(sz >= 2);
                if sz == 2 {
                  return smallest_iis // smallest ever cyclic IIS has two edges
                }
              }
            }
          }
        }
      }
    }
    smallest_iis
  }

  pub(crate) fn compute_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    if self.parameters.minimal_cyclic_iis {
      let mut smallest_iis_size = None;
      let mut best_iis = None;

      for scc in sccs { // start from first SCC index found in ModelState
        if let Some(iis) = self.find_pure_cyclic_iis(scc, true, smallest_iis_size) {
          smallest_iis_size = Some(iis.size() as u32);
          best_iis = Some(iis);
        }
        if let Some(iis) = self.find_min_bound_cycle_iis(scc, smallest_iis_size) {
          smallest_iis_size = Some(iis.size() as u32);
          best_iis = Some(iis);
        }
      }

      best_iis.unwrap()
    } else {
      for scc in sccs { // start from first SCC index found in ModelState
        let iis = self.find_pure_cyclic_iis(scc, false, None)
          .or_else(|| self.find_bound_cycle_iis(scc));
        if let Some(iis) = iis {
          return iis
        }
      }
      unreachable!();
    }
  }
}