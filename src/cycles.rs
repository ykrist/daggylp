use super::graph::*;
use fnv::FnvHashSet;
use crate::{Result, set_with_capacity, Error};

enum CyclicInfKind {
  Pure,
  Unknown,
  Bounds{ lb: usize, ub: usize}
}

impl Graph {
  fn cycle_edges<'a>(&'a self, nodes: &'a [usize]) -> CyclesEdgeIter<'a> {
    CyclesEdgeIter::new(self, nodes)
  }

  fn cycle_infeasible(&self, nodes: &[usize]) -> Option<CyclicInfKind> {
    if self.cycle_edges(nodes).any(|e| e.weight != 0) {
      return Some(CyclicInfKind::Pure)
    }
    if let Some(((lb, _), (ub, _))) = self.find_scc_bound_infeas(nodes.iter().copied()) {
      return Some(CyclicInfKind::Bounds {lb, ub })
    }
    return None
  }


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
      if min_ub < max_lb {
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
            let iis = Iis { constrs };
            if !smallest {
              return Some(iis);
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
      .chain(&backward[1..backward.len() - 1])
      .map(|&n| self.var_from_node_id(n)))
  }

  /// Try to find an IIS which consists an Upper bound, Lower bound and edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  fn find_bound_cycle_iis(&self, scc: &FnvHashSet<usize>) -> Option<Iis> {
    if let Some(((max_lb_node, _), (min_ub_node, _))) = self.find_scc_bound_infeas(scc.iter().copied()) {
      let p1 = self.shortest_path_scc(scc, max_lb_node, min_ub_node, None).unwrap();
      let p2 = self.shortest_path_scc(scc, min_ub_node, max_lb_node, None).unwrap();
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
    let mut search_max_cycle_edge_count = prune.map(|n| n - 2); // will always contain two bounds

    for &n1 in scc {
      for &n2 in scc {
        if self.nodes[n1].ub < self.nodes[n2].lb {
          let mut budget = search_max_cycle_edge_count;
          if let Some(p1) = self.shortest_path_scc(scc, n1, n2, budget) {
            budget = budget.map(|n| n + 1 - p1.len() as u32);
            if let Some(p2) = self.shortest_path_scc(scc, n2, n1, budget) {
              let mut iis = self.cycle_iis_from_path_pair(&p1, &p2);
              iis.add_constraint(Constraint::Ub(self.var_from_node_id(n1)));
              iis.add_constraint(Constraint::Lb(self.var_from_node_id(n2)));
              smallest_iis = Some(iis);
              search_max_cycle_edge_count = Some((p1.len() + p2.len() - 2) as u32);
              if let Some(sz) = search_max_cycle_edge_count {
                debug_assert!(sz >= 2);
                if sz == 2 {
                  return smallest_iis; // smallest ever cyclic IIS has two edges
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
          smallest_iis_size = Some(iis.len() as u32);
          best_iis = Some(iis);
        }
        if let Some(iis) = self.find_min_bound_cycle_iis(scc, smallest_iis_size) {
          smallest_iis_size = Some(iis.len() as u32);
          best_iis = Some(iis);
        }
      }

      best_iis.unwrap()
    } else {
      for scc in sccs { // start from first SCC index found in ModelState
        let iis = self.find_pure_cyclic_iis(scc, false, None)
          .or_else(|| self.find_bound_cycle_iis(scc));
        if let Some(iis) = iis {
          return iis;
        }
      }
      unreachable!();
    }
  }

  pub fn all_cyclic_iis(&self) -> Result<CyclicIisIter> {
    // fixme check for infeasibility

    match &self.state {
      ModelState::Optimal | ModelState::Mrs => todo!(),
      ModelState::Unsolved => todo!(),
      ModelState::InfPath(_) => todo!(),
      ModelState::InfCycle { sccs, first_inf_scc } => {
        Ok(CyclicIisIter::new(self, &sccs[*first_inf_scc..], false))
      }
    }
  }
}

pub struct SccHandle<'a> {
  scc: &'a SccInfo
}

#[derive(Copy, Clone, Debug)]
pub struct StackVariables<'a> {
  edges_from_v: &'a [Edge],
  edge_idx: usize,
  cycle_found: bool,
  v: usize,
}

type BlockList = Vec<FnvHashSet<usize>>;

#[derive(Clone)]
pub struct CycleIter<'a> {
  one_cycle_per_scc: bool,
  local_variables: Vec<StackVariables<'a>>,
  marked: Vec<bool>,
  reached: Vec<bool>,
  blist: BlockList,
  current_path_pos: Vec<usize>,
  current_path: Vec<usize>,
  graph: &'a Graph,
  sccs: &'a [FnvHashSet<usize>],
  empty: bool,
}

impl<'a> CycleIter<'a> {
  fn no_cycle(&mut self, x: usize, y: usize) {
    self.blist[y].insert(x);
  }

  fn unmark(&mut self, y: usize) {
    self.marked[y] = false;
    for x in self.blist[y].clone() { // TODO can we avoid the clone here?
      if self.marked[x] {
        self.unmark(x);
      }
    }
    self.blist[y].clear();
  }

  pub fn new(graph: &'a Graph, sccs: &'a [FnvHashSet<usize>], one_cycle_per_scc: bool) -> Self {
    let mut iter = CycleIter {
      one_cycle_per_scc,
      local_variables: Default::default(),
      marked: vec![false; graph.nodes.len()],
      reached: vec![false; graph.nodes.len()],
      blist: vec![Default::default(); graph.nodes.len()],
      current_path_pos: vec![usize::MAX; graph.nodes.len()],
      current_path: Vec::with_capacity(64),
      graph,
      sccs,
      empty: false,
    };
    iter.init_scc();
    iter
  }

  fn init_scc(&mut self) {
    let root = *self.sccs[0].iter().max_by_key(|&&n| self.graph.edges_to[n].len()).unwrap();
    self.current_path.clear();
    self.local_variables.clear();
    self.pre_loop(root);
  }

  fn next_scc(&mut self) {
    self.sccs = &self.sccs[1..];
    if !self.sccs.is_empty() {
      self.init_scc();
    }
  }

  /// Push to the stack, and do the stuff that happens before the main loop
  fn pre_loop(&mut self, v: usize) {
    self.local_variables.push(StackVariables { cycle_found: false, v, edges_from_v: &self.graph.edges_from[v], edge_idx: 0 });
    self.marked[v] = true;
    debug_assert_eq!(self.current_path_pos[v], usize::MAX);
    self.current_path_pos[v] = self.current_path.len();
    self.current_path.push(v);
  }

  /// step through the neighbours loop until we find cycle or finish the loop
  fn neighbours_loop(&mut self) -> Option<Vec<usize>> {
    let sp = self.local_variables.len() - 1;
    let v = self.local_variables[sp].v;

    while let Some(e) = self.local_variables[sp].edges_from_v.get(self.local_variables[sp].edge_idx) {
      self.local_variables[sp].edge_idx += 1;
      let w = e.to;
      if !self.sccs[0].contains(&w) && !self.blist[v].contains(&w) {
        continue;
      }
      // TODO: check if in blocklist?
      if !self.marked[w] {
        // Recursive call begins
        self.pre_loop(w); // stack push
        if let Some(cycle) = self.neighbours_loop() { // recurse - callee will run post-loop
          self.local_variables[sp].cycle_found = true; // don't pop stack - inner call might emit more
          return Some(cycle);
        } else {
          // inner loop has finished
          // self.post_loop(); // stack pop
        }
        // recursive call end - no cycles found
        self.no_cycle(v, w);
      } else if !self.reached[w] {
        // have found a cycle
        self.local_variables[sp].cycle_found = true;
        let start = self.current_path_pos[w];
        debug_assert_ne!(start, usize::MAX);
        // println!("sss {}, {:?}", w, self.current_path);
        return Some(self.current_path[start..].to_vec());
      } else {
        self.no_cycle(v, w);
      }
    }

    None
  }

  /// Pop from the stack, and do the stuff that happens after the main loop
  fn post_loop(&mut self) {
    // cleanup
    let StackVariables { v, cycle_found, .. } = self.local_variables.pop().unwrap();

    let vv = self.current_path.pop().unwrap();
    debug_assert_eq!(vv, v);
    debug_assert_ne!(self.current_path_pos[v], usize::MAX);
    self.current_path_pos[v] = usize::MAX;

    if cycle_found {
      self.unmark(v);
    }
    self.reached[v] = true;
  }
}


impl Iterator for CycleIter<'_> {
  type Item = Vec<usize>;

  fn next(&mut self) -> Option<Self::Item> {
    while !self.sccs.is_empty() {
      let cycle = self.neighbours_loop();
      if cycle.is_some() {
        if self.one_cycle_per_scc {
          self.next_scc()
        }
        return cycle
      } else if self.local_variables.len() > 1 {
        self.post_loop() // pop the stack and continue the outer neighbours_loop
      } else {
        self.next_scc()
      }
    }
    None
  }
}

#[derive(Clone)]
pub struct CyclicIisIter<'a> {
  one_iis_per_scc: bool,
  cycle_iter: CycleIter<'a>,
  graph: &'a Graph,
}

impl<'a> CyclicIisIter<'a> {
  fn new(graph: &'a Graph, sccs: &'a [FnvHashSet<usize>], one_iis_per_scc: bool) -> Self {
    CyclicIisIter {
      cycle_iter: CycleIter::new(graph, sccs, false),
      one_iis_per_scc,
      graph,
    }
  }
}

impl Iterator for CyclicIisIter<'_> {
  type Item = Iis;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some(cycle) = self.cycle_iter.next() {
      match self.graph.cycle_infeasible(&cycle) {
        None => continue,
        Some(kind) => {
          println!("{:?}", &cycle);
          let mut iis = Iis::from_cycle(cycle.iter().map(|n| self.graph.var_from_node_id(*n)));
          match kind {
            CyclicInfKind::Pure => {},
            CyclicInfKind::Bounds { lb, ub } => {
              iis.add_constraint(Constraint::Ub(self.graph.var_from_node_id(ub)));
              iis.add_constraint(Constraint::Lb(self.graph.var_from_node_id(lb)));
            }
            CyclicInfKind::Unknown => unreachable!(),
          }
          if self.one_iis_per_scc {
            self.cycle_iter.next_scc()
          }
          return Some(iis);
        }
      }
    }
    None
  }
}

struct CyclesEdgeIter<'a> {
  nodes: std::slice::Iter<'a, usize>,
  len: usize,
  first: usize,
  prev: usize,
  graph: &'a Graph,
}

impl<'a> CyclesEdgeIter<'a> {
  fn new(graph: &'a Graph, cycle: &'a [usize]) -> Self {
    debug_assert!(cycle.len() > 1);
    let mut nodes = cycle.iter();
    let first = *nodes.next().unwrap();
    let prev = first;
    CyclesEdgeIter {
      nodes,
      first,
      prev,
      graph,
      len: cycle.len(),
    }
  }
}

impl Iterator for CyclesEdgeIter<'_> {
  type Item = Edge;

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(&n) = self.nodes.next() {
      let e = *self.graph.find_edge(self.prev, n);
      self.prev = n;
      self.len -= 1;
      Some(e)
    } else if self.len > 0 {
      let e = *self.graph.find_edge(self.prev, self.first);
      self.len -= 1;
      Some(e)
    } else {
      None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.len, Some(self.len))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_utils::*;
  use test_case::test_case;
  use crate::viz::GraphViz;

  fn num_iis_in_complete_graph_with_single_nonzero_edge(n: u8) -> u128 {
    let mut total = 0u128;
    for k in 2..=n {
      let mut prod = 1;
      for i in (n - k + 1)..=(n-2) { // may be empty if k == 2
        prod *= i as u128;
      }
      total += prod;
    }
    total
  }

  fn num_cycles_in_complete_graph(n: u8) -> u128 {
    let mut c = 0;
    for k in 2..=n {
      let mut q: u128 = (n - k + 1) as u128;
      for x in (n - k + 2)..=n {
        q *= (x as u128);
      }
      c += q / (k as u128);
    }
    c
  }

  fn complete_graph_with_single_nonzero_edge(n: u8) -> GraphGen {
    let mut g = GraphGen::new(
      n as usize,
      IdenticalNodes{ lb: 0, ub: 10, obj: 0 },
      AllEdges(0)
    );
    g.edges.insert((0,1), 1);
    g
  }

  fn complete_graph_sccs(sizes: &[u8]) -> GraphGen {
    let mut sizes = sizes.iter().copied();
    let n = sizes.next().unwrap();

    let mut g = complete_graph_with_single_nonzero_edge(n);
    for n in sizes {
      let f = complete_graph_with_single_nonzero_edge(n);
      g.join(&f, AllEdges(1), NoEdges());
    }
    g
  }


  fn solve_and_count_cycles_iis(g: &mut Graph, n_iis: u128, n_cycles: u128) {
    g.solve();
    let mut iis_iter = g.all_cyclic_iis().unwrap();

    let mut iis_cnt = 0;
    let mut cycle_cnt = 0;

    for iis in iis_iter.clone() {
      iis_cnt += 1;
    }

    for cycle in iis_iter.cycle_iter {
      cycle_cnt += 1;
    }

    assert_eq!(iis_cnt, n_iis);
    assert_eq!(cycle_cnt, n_cycles);
  }

  #[test_case(2)]
  #[test_case(3)]
  #[test_case(4)]
  #[test_case(5)]
  #[test_case(6)]
  #[test_case(7)]
  #[test_case(8)]
  fn count_complete(n: u8) {
    let mut g = complete_graph_with_single_nonzero_edge(n).build();
    solve_and_count_cycles_iis(&mut g,
                               num_iis_in_complete_graph_with_single_nonzero_edge(n),
                               num_cycles_in_complete_graph(n));
  }


  #[test_case(&[4,4])]
  #[test_case(&[8,7])]
  #[test_case(&[4,3,8,6])]
  #[test_case(&[4,6,3,7,7,6])]
  fn count_multi_scc(component_sizes: &[u8]) {
    let mut g = complete_graph_sccs(component_sizes).build();
    let (mut n_iis, mut n_cycles) = (0,0);

    for &n in component_sizes {
      n_iis += num_iis_in_complete_graph_with_single_nonzero_edge(n);
      n_cycles += num_cycles_in_complete_graph(n);
    }
    solve_and_count_cycles_iis(&mut g, n_iis, n_cycles);
  }


  fn cycle_graph(n: u8) -> GraphGen {
    let n = n as usize;
    GraphGen::new(
      n,
      IdenticalNodes{ lb: 0, ub: Weight::MAX, obj: 0 },
      CycleEdges { n, weight: 1 }
    )
  }

  #[test_case(2)]
  #[test_case(3)]
  #[test_case(4)]
  #[test_case(5)]
  #[test_case(6)]
  #[test_case(7)]
  #[test_case(8)]
  fn count_iis_cycle_graph(n: u8) {
    let mut g = cycle_graph(n).build();
    solve_and_count_cycles_iis(&mut g, 1, 1);
  }

  /// Build a graph of the form:
  ///
  ///        0 ---> 1
  ///        /\      |
  ///        |      \/
  ///        4 <--- 2
  ///        |      /\
  ///        \/      |
  ///        5 ---> 6
  ///        /\      |
  ///        |      \/
  ///        8 <--- 7
  ///        |      /\
  ///        \/      |
  ///        ...    ...
  ///
  /// Will have `( k + 1 ) * 2` nodes.  All edges have zero weight, except 0 -> 1.
  /// This means there is always one IIS and exactly k cycles.
  fn k_cycle_graph(k: u8) -> GraphGen {
    let mut g = cycle_graph(4);
    g.edges.values_mut().for_each(|w| *w = 0);
    g.edges.insert((0, 1), 1);
    let node_data = *g.nodes.first().unwrap();

    let mut a = 2;
    let mut b = 3;

    for j in 1..k {
      let c = g.add_node(node_data);
      let d = g.add_node(node_data);
      g.edges.insert((b,c), 0);
      g.edges.insert((c,d), 0);
      g.edges.insert((d,a), 0);
      a = c;
      b = d;
    }
    g
  }

  #[test_case(10)]
  #[test_case(12)]
  #[test_case(16)]
  #[test_case(100)]
  #[test_case(3)]
  #[test_case(4)]
  #[test_case(55)]
  fn count_k_cycle_graph(k: u8) {
    let mut g = k_cycle_graph(k).build();
    solve_and_count_cycles_iis(&mut g, 1, k as u128);
  }
}
