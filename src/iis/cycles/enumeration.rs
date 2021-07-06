use super::*;
use crate::iis::Iis;
use crate::test_utils::{GraphSpec};
use crate::viz::{GraphViz, LayoutAlgo};
use proptest::prelude::{TestCaseError, Strategy};
use proptest::test_runner::{TestCaseResult, TestError};
use crate::edge_storage::{ForwardDir, Neighbours};

pub enum Enumeration {}

impl FindCyclicIis<Enumeration> for Graph {
  fn find_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    self.iter_cyclic_iis(sccs.iter()).next().unwrap()
  }

  fn find_smallest_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    self.iter_cyclic_iis(sccs.iter()).min_by_key(|iis| iis.len()).unwrap()
  }
}


#[derive(Copy, Clone, Debug)]
struct StackVariables<I> {
  edges_from_v: I,
  cycle_found: bool,
  v: usize,
}

type BlockList = Vec<FnvHashSet<usize>>;

/// An iterator version of the simple-cycle enumeration algorithm
/// by Loizou and Thanisch (1982)
// #[derive(Clone)] // FIXME: add back in?
pub struct CycleIter<'a, I, E: EdgeLookup> {
  one_cycle_per_scc: bool,
  local_variables: Vec<StackVariables<<E as Neighbours<ForwardDir>>::Neigh<'a>>>,
  marked: Vec<bool>,
  reached: Vec<bool>,
  blocked_pred: BlockList,
  blocked_succ: BlockList,
  current_path_pos: Vec<usize>,
  current_path: Vec<usize>,
  graph: &'a Graph<E>,
  sccs: I,
  current_scc: Option<&'a FnvHashSet<usize>>,
  empty: bool,
}

impl<'a, I, E> CycleIter<'a, I, E>
  where
    I: 'a + Iterator<Item=&'a FnvHashSet<usize>>,
    E: EdgeLookup,
{
  fn no_cycle(&mut self, x: usize, y: usize) {
    // println!("block({} -> {}), {:?}", x, y, &self.current_path);
    self.blocked_pred[y].insert(x);
    self.blocked_succ[x].insert(y);
  }

  fn unmark(&mut self, y: usize) {
    self.marked[y] = false;
    // println!("unmark({}), {:?}", y, &self.current_path);
    for x in self.blocked_pred[y].clone() { // TODO can we avoid the clone here?
      let removed = self.blocked_succ[x].remove(&y);
      debug_assert!(removed);
      if self.marked[x] {
        self.unmark(x);
      }
    }
    self.blocked_pred[y].clear();
  }

  pub fn new(graph: &'a Graph<E>, mut sccs: I, one_cycle_per_scc: bool) -> Self {
    let mut iter = CycleIter {
      one_cycle_per_scc,
      local_variables: Default::default(),
      marked: vec![false; graph.nodes.len()],
      reached: vec![false; graph.nodes.len()],
      blocked_pred: vec![Default::default(); graph.nodes.len()],
      blocked_succ: vec![Default::default(); graph.nodes.len()],
      current_path_pos: vec![usize::MAX; graph.nodes.len()],
      current_path: Vec::with_capacity(64),
      graph,
      current_scc: None,
      sccs,
      empty: false,
    };
    iter.next_scc();
    iter
  }

  fn next_scc(&mut self) {
    self.current_scc = self.sccs.next();
    self.current_path.clear();
    self.local_variables.clear();

    if let Some(scc) = self.current_scc {
      let root = *scc.iter().next().unwrap();
      // let root = *scc.iter().max_by_key(|&&n| self.graph.edges_to[n].len()).unwrap(); // FIXME is it correct to just take any node?
      self.pre_loop(root);
    }
  }

  /// Push to the stack, and do the stuff that happens before the main loop
  fn pre_loop(&mut self, v: usize) {
    self.local_variables.push(StackVariables { cycle_found: false, v, edges_from_v: self.graph.edges.successors(v) });
    self.marked[v] = true;
    debug_assert_eq!(self.current_path_pos[v], usize::MAX);
    self.current_path_pos[v] = self.current_path.len();
    self.current_path.push(v);
    // println!("mark({}), {:?}", v, &self.current_path);
  }

  /// step through the neighbours loop until we find cycle or finish the loop
  fn neighbours_loop(&mut self) -> Option<Vec<usize>> {
    let sp = self.local_variables.len() - 1;
    let v = self.local_variables[sp].v;
    let scc = self.current_scc.unwrap();

    while let Some(e) = self.local_variables[sp].edges_from_v.next() {
      // self.local_variables[sp].edge_idx += 1;
      let w = e.to;
      if !scc.contains(&w) || self.blocked_succ[v].contains(&w) {
        continue;
      }
      if !self.marked[w] {
        // Recursive call begins
        self.pre_loop(w); // stack push
        if let Some(cycle) = self.neighbours_loop() { // recurse - callee will run post-loop
          self.local_variables[sp].cycle_found = true; // don't pop stack - inner call might emit more
          return Some(cycle);
        }
        // recursive call end - no cycles found
        self.no_cycle(v, w);
      } else if !self.reached[w] {
        // have found a cycle
        self.local_variables[sp].cycle_found = true;
        let start = self.current_path_pos[w];
        debug_assert_ne!(start, usize::MAX);
        // println!("cycle w={}, {:?} -> {:?}", w, &self.current_path, &self.current_path[start..]);
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


impl<'a, I, E> Iterator for CycleIter<'a, I, E>
  where
    I: 'a + Iterator<Item=&'a FnvHashSet<usize>>,
    E: EdgeLookup,
{
  type Item = Vec<usize>;

  fn next(&mut self) -> Option<Self::Item> {
    while self.current_scc.is_some() {
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

// #[derive(Clone)] // FIXME add back in
pub struct CyclicIisIter<'a, I, E: EdgeLookup> {
  one_iis_per_scc: bool,
  cycle_iter: CycleIter<'a, I, E>,
  graph: &'a Graph<E>,
}

impl<'a, I, E> CyclicIisIter<'a, I, E>
  where
    I: 'a + Iterator<Item=&'a FnvHashSet<usize>>,
    E: EdgeLookup,
{
  fn new(graph: &'a Graph<E>, sccs: I, one_iis_per_scc: bool) -> Self
  {
    CyclicIisIter {
      cycle_iter: CycleIter::new(graph, sccs, false),
      one_iis_per_scc,
      graph,
    }
  }
}

impl<'a, I, E> Iterator for CyclicIisIter<'a, I, E>
  where
    I: 'a + Iterator<Item=&'a FnvHashSet<usize>>,
    E: EdgeLookup,
{
  type Item = Iis;

  fn next(&mut self) -> Option<Self::Item> {
    while let Some(cycle) = self.cycle_iter.next() {
      // println!("{:?}", &cycle);
      match self.graph.cycle_infeasible(&cycle) {
        None => continue,
        Some(kind) => {
          // println!("inf: {:?}", &cycle);
          let mut iis = Iis::from_cycle(self.graph, cycle.iter().copied());
          match kind {
            CyclicInfKind::Pure => {}
            CyclicInfKind::Bounds(bi) => {
              iis.add_bounds(bi.lb_node, bi.ub_node);
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

struct CyclesEdgeIter<'a, E> {
  nodes: std::slice::Iter<'a, usize>,
  len: usize,
  first: usize,
  prev: usize,
  graph: &'a Graph<E>,
}

impl<'a, E: EdgeLookup> CyclesEdgeIter<'a, E> {
  fn new(graph: &'a Graph<E>, cycle: &'a [usize]) -> Self {
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

impl<E: EdgeLookup> Iterator for CyclesEdgeIter<'_, E> {
  type Item = Edge;

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(&n) = self.nodes.next() {
      let e = *self.graph.edges.find_edge(self.prev, n);
      self.prev = n;
      self.len -= 1;
      Some(e)
    } else if self.len > 0 {
      let e = *self.graph.edges.find_edge(self.prev, self.first);
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

impl<E: EdgeLookup> Graph<E> {
  fn cycle_edges<'a>(&'a self, nodes: &'a [usize]) -> CyclesEdgeIter<'a, E> {
    CyclesEdgeIter::new(self, nodes)
  }

  fn cycle_infeasible(&self, nodes: &[usize]) -> Option<CyclicInfKind> {
    if self.cycle_edges(nodes).any(|e| e.weight != 0) {
      return Some(CyclicInfKind::Pure);
    }
    if let Some(bi) = self.find_scc_bound_infeas(nodes.iter().copied(), false) {
      return Some(CyclicInfKind::Bounds(bi));
    }
    return None;
  }

  fn iter_cycles<'a, I>(&'a self, sccs: I) -> CycleIter<'a, I::IntoIter, E>
    where
      I: IntoIterator<Item=&'a FnvHashSet<usize>>,
      I::IntoIter: 'a,
  {
    CycleIter::new(self, sccs.into_iter(), false)
  }

  fn iter_cyclic_iis<'a, I>(&'a self, sccs: I) -> CyclicIisIter<'a, I::IntoIter, E>
    where
      I: IntoIterator<Item=&'a FnvHashSet<usize>>,
      I::IntoIter: 'a,
  {
    CyclicIisIter::new(self, sccs.into_iter(), false)
  }
}

//
// #[cfg(test)]
// mod tests {
//   use super::*;
//   use proptest::prelude::*;
//   use crate::test_utils::*;
//   use crate::test_utils::strategy::*;
//   use crate::*;
//
//   enum Tests {}
//   impl Tests {
//     fn check_iis_and_cycles_counts(g: &mut Graph, n_iis: u128, n_cycles: u128) -> TestCaseResult {
//       let num_nodes = g.nodes.len();
//       g.solve();
//       let (sccs, first_inf_scc) = match &g.state {
//         ModelState::InfCycle { sccs, first_inf_scc } => (&*sccs, *first_inf_scc),
//         _ => return Err(TestCaseError::fail("should find infeasible cycles")),
//       };
//       prop_assert_eq!(g.sccs.len(), 0, "SCCs should only be present when feasible");
//       prop_assert_eq!(sccs.len(), 1, "Complete graph has one SCC");
//       let mut cycle_cnt = g.iter_cycles(sccs).count() as u128;
//       prop_assert_eq!(cycle_cnt, n_cycles);
//       let mut iis_cnt = g.iter_cyclic_iis(&sccs[first_inf_scc..]).count() as u128;
//       prop_assert_eq!(iis_cnt, n_iis);
//       Ok(())
//     }
//
//     pub fn count_cycles_and_iis_cycle_graph(g: &mut Graph) -> TestCaseResult {
//       let n_iis = if g.edges.all_edges().any(|e| e.weight != 0) {
//         1
//       } else {
//         0
//       };
//       Self::check_iis_and_cycles_counts(g, n_iis, 1)
//     }
//
//     pub fn count_cycles_and_iis_complete_graph(g: &mut Graph) -> TestCaseResult {
//       let n = g.nodes.len();
//
//       // Number of cycles of length k is `(n choose k) / k`
//       let n_cycles = {
//         let mut c = 0;
//         for k in 2..=n {
//           let mut q: u128 = (n - k + 1) as u128;
//           for x in (n - k + 2)..=n {
//             q *= (x as u128);
//           }
//           c += q / (k as u128);
//         }
//         c
//       };
//
//       // One edge `(i, j)` has non-zero weight, so number of k-cycles o.t.f
//       // `[i, ... seq of length k-2, ..., j]` is `(n - 2 choose k - 2)`
//       let n_iis = {
//         let mut total = 0u128;
//         for k in 2..=n {
//           let mut prod = 1;
//           for i in (n - k + 1)..=(n - 2) { // may be empty if k == 2
//             prod *= i as u128;
//           }
//           total += prod;
//         }
//         total
//       };
//
//       Self::check_iis_and_cycles_counts(g, n_iis, n_cycles)
//     }
//   }
//
//   graph_proptests!(
//     Tests;
//     set_arbitrary_edge_to_one(complete_graph_zero_edges(default_nodes(2..=8)))
//       => count_cycles_and_iis_complete_graph [layout=LayoutAlgo::Fdp];
//     graph_with_conn(default_nodes(2..1000), Cycle::new(), any_edge_weight())
//       => count_cycles_and_iis_cycle_graph [layout=LayoutAlgo::Fdp];
//   );
// }