use super::*;
use crate::iis::Iis;

pub enum Enumeration {}

impl FindCyclicIis<Enumeration> for Graph {
  fn find_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    self.iter_cyclic_iis(sccs).next().unwrap()
  }

  fn find_smallest_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    self.iter_cyclic_iis(sccs).min_by_key(|iis| iis.len()).unwrap()
  }
}


#[derive(Copy, Clone, Debug)]
struct StackVariables<'a> {
  edges_from_v: &'a [Edge],
  edge_idx: usize,
  cycle_found: bool,
  v: usize,
}

type BlockList = Vec<FnvHashSet<usize>>;

/// An iterator version of the simple-cycle enumeration algorithm
/// by Loizou and Thanisch (1982)
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
        return cycle;
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
            CyclicInfKind::Pure => {}
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

impl Graph {
  fn cycle_edges<'a>(&'a self, nodes: &'a [usize]) -> CyclesEdgeIter<'a> {
    CyclesEdgeIter::new(self, nodes)
  }

  fn cycle_infeasible(&self, nodes: &[usize]) -> Option<CyclicInfKind> {
    if self.cycle_edges(nodes).any(|e| e.weight != 0) {
      return Some(CyclicInfKind::Pure);
    }
    if let Some(((lb, _), (ub, _))) = self.find_scc_bound_infeas(nodes.iter().copied()) {
      return Some(CyclicInfKind::Bounds { lb, ub });
    }
    return None;
  }

  fn iter_cycles<'a>(&'a self, sccs: &'a [FnvHashSet<usize>]) -> CycleIter<'a> {
    CycleIter::new(self, sccs, false)
  }

  fn iter_cyclic_iis<'a>(&'a self, sccs: &'a [FnvHashSet<usize>]) -> CyclicIisIter<'a> {
    CyclicIisIter::new(self, sccs, false)
  }
}