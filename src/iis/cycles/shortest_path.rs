use super::*;
use crate::iis::Iis;
use crate::edge_storage::{ForwardDir, BackwardDir, EdgeDir};

pub enum ShortestPathAlg {}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Prune {
  // No pruning
  AllDest,
  // Find the best destination and prune others (stop)
  BestDest,
  // Find the best dest less than
  BestDestLessThan(u32),
  // All dest less than
  AllDestLessThan(u32),
}

impl Prune {
  fn all_dests(&self) -> bool {
    use Prune::*;
    matches!(self, AllDest | AllDestLessThan(..))
  }

  fn bound(&self) -> Option<u32> {
    use Prune::*;
    match self {
      BestDestLessThan(bnd) | AllDestLessThan(bnd) => Some(*bnd),
      AllDest | BestDest => None
    }
  }

  fn update_bound(&mut self, f: impl FnOnce(u32) -> u32) {
    use Prune::*;
    match self {
      BestDestLessThan(bnd) | AllDestLessThan(bnd) => {
        *bnd = f(*bnd);
      }
      AllDest | BestDest => {}
    }
  }
}

#[derive(Debug, Copy, Clone)]
enum Label {
  // The source node
  Src { node: usize },
  // Sentinel for values which have not been processed, but are already in the queue.
  Queued,
  // Regular node
  Node { dist_from_src: u32, node: usize, pred: usize },
  // Sentinel for destination which has never been visited
  DestUnvisited { node: usize },
  // Destination node
  Dest { dist_from_src: u32, node: usize, pred: usize },
}

impl Label {
  pub fn node(&self) -> usize {
    match *self {
      Label::Queued => unreachable!(),
      Label::Src { node }
      | Label::Node { node, .. }
      | Label::DestUnvisited { node }
      | Label::Dest { node, .. }
      => node
    }
  }

  pub fn pretty(&self) -> String {
    use Label::*;
    match self {
      Src { node } => format!("S({})", node),
      Node { node, dist_from_src, pred } => format!("N({} <- {}; d={})", node, pred, dist_from_src),
      Queued => "Q".to_string(),
      Dest { node, dist_from_src, pred } => format!("D({} <- {}; d={})", node, pred, dist_from_src),
      DestUnvisited { node } => format!("U({})", node),
    }
  }
}

pub(crate) struct ShortestPaths<D: ?Sized> {
  _dir: std::marker::PhantomData<D>,
  labels: FnvHashMap<usize, Label>,
}

impl<D> ShortestPaths<D> {
  fn new(labels: FnvHashMap<usize, Label>) -> Self {
    ShortestPaths {
      _dir: std::marker::PhantomData,
      labels,
    }
  }

  /// Returns the label corresponding to the destination node.  Panics if `dest` was not a destination node
  /// during labeling, or was never reached due to pruning
  fn dest_label(&self, dest: usize) -> Label {
    use Label::*;
    match self.labels.get(&dest) {
      None | Some(DestUnvisited { .. }) => panic!("BFS terminated before finding path to node {}", dest),
      Some(Queued) => unreachable!("should never reach a node"),
      Some(Node { .. })
      | Some(Src { .. }) => panic!("Node {} is not a destination node", dest),
      Some(l @ Dest { .. }) => *l
    }
  }

  /// Returns length of the shortest path from the source node to this dest node, in number of edges.
  /// Panics if [`ShortestPaths::dest_label`] panics
  pub fn num_edges(&self, dest: usize) -> usize {
    match self.dest_label(dest) {
      Label::Dest { dist_from_src, .. } => dist_from_src as usize,
      _ => unreachable!()
    }
  }

  /// Return an iterator of the nodes in the shortest path.  If the edge direction is
  /// forward, the path is returned in reverse order (destination node first)
  /// If the edge direction is backward, the nodes are iterated in normal order
  /// (source node first).
  pub fn iter_nodes(&self, dest: usize) -> ShortestPath<D> {
    ShortestPath { path: self, label: Some(self.dest_label(dest)) }
  }
}
//
// impl SccShortestPath<ForwardDir> {
//   pub fn to_vec(&self, dest: Option<usize>) -> Vec<usize> {
//     let mut path = self.iter_nodes(dest).collect();
//     path.reverse();
//     path
//   }
// }
//
// impl SccShortestPath<BackwardDir> {
//   pub fn to_vec(&self, dest: Option<usize>) { self.iter_nodes(dest).collect() }
// }

pub(crate) struct ShortestPath<'a, D: ?Sized> {
  path: &'a ShortestPaths<D>,
  label: Option<Label>,
}

impl<D> Iterator for ShortestPath<'_, D> {
  type Item = usize;

  fn next(&mut self) -> Option<Self::Item> {
    use Label::*;
    match self.label.take() {
      Some(l) => match l {
        DestUnvisited { .. } | Queued => unreachable!(),
        Dest { node, pred, .. } | Node { node, pred, .. } => {
          self.label = Some(self.path.labels[&pred]);
          Some(node)
        }
        Src { node } => {
          Some(node)
        }
      }
      None => None
    }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    use Label::*;

    let sz = match self.label {
      Some(l) => match l {
        DestUnvisited { .. } | Queued => unreachable!(),
        Dest { dist_from_src, .. } | Node { dist_from_src, .. } => dist_from_src as usize + 1,
        Src { .. } => 1,
      }
      None => 0,
    };
    (sz, Some(sz))
  }
}

impl<D> ExactSizeIterator for ShortestPath<'_, D> {}


impl FindCyclicIis<ShortestPathAlg> for Graph {
  fn find_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    for scc in sccs { // start from first SCC index found in ModelState
      let iis = self.find_cycle_edge_iis(scc, false, None)
        .or_else(|| self.find_cycle_bound_iis(scc));
      if let Some(iis) = iis {
        return iis;
      }
    }
    unreachable!();
  }

  fn find_smallest_cyclic_iis(&self, sccs: &[FnvHashSet<usize>]) -> Iis {
    let mut smallest_iis_size = None;
    let mut best_iis = None;

    for scc in sccs {
      if let Some(iis) = self.find_cycle_edge_iis(scc, true, smallest_iis_size) {
        smallest_iis_size = Some(iis.len() as u32);
        best_iis = Some(iis);
      }
      if let Some(iis) = self.find_smallest_cycle_bound_iis(scc, smallest_iis_size) {
        smallest_iis_size = Some(iis.len() as u32);
        best_iis = Some(iis);
      }
    }

    best_iis.unwrap()
  }
}

impl<E: EdgeLookup> Graph<E> {
  /// Computes the shortest path from `src` to one or more `dests`.
  ///
  /// Type parameter `D` governs whether edges are traverse forwards `D = ForwardDir` or backwards `D = BackwardDir`.
  ///
  /// Arguments
  /// ---------
  ///   - `scc` contains the nodes of the SCC in which to restrict the search to
  ///   - `src` is the starting node
  ///   - `dests` are the destination nodes.
  ///   - `prune` describes when to terminate the search.
  ///
  fn shortest_path_scc<D, I>(&self,
                             scc: &FnvHashSet<usize>,
                             src: usize,
                             dests: I,
                             prune: Prune,
  ) -> Option<ShortestPaths<D>>
    where
      D: EdgeDir,
      I: IntoIterator<Item=usize>,
      E: crate::edge_storage::Neighbours<D>,
  {

    use Label::*;

    let mut labels = map_with_capacity(scc.len());
    for dest in dests {
      // Can use the `labels` Map to keep store the dests as well
      labels.insert(dest, DestUnvisited { node: dest });
    }

    let n_dests = labels.len();
    if n_dests == 0 { return None; }
    let mut n_dests_found = 0;
    let mut queue = VecDeque::with_capacity(scc.len());

    queue.push_back(Src { node: src });

    'bfs: while let Some(v_label) = queue.pop_front() {
      let (v, new_dist_from_src) = match v_label {
        Src { node } => (node, 1),
        Dest { dist_from_src, node, .. } => (node, dist_from_src + 1),
        Node { dist_from_src, node, .. } => (node, dist_from_src + 1),
        DestUnvisited { .. } | Queued => unreachable!(),
      };

      if let Some(bnd) = prune.bound() {
        if new_dist_from_src >= bnd {
          break 'bfs;
        }
      }

      for w in EdgeLookup::neighbour_nodes::<D>(&self.edges, v).filter(|w| scc.contains(w)) {
        match labels.get_mut(&w) {
          Some(dest_label @ DestUnvisited { .. }) => {
            *dest_label = Dest { dist_from_src: new_dist_from_src, node: w, pred: v };
            n_dests_found += 1;
            if !prune.all_dests() || n_dests_found == n_dests {
              labels.insert(v, v_label);
              break 'bfs;
            } else {
              queue.push_back(*dest_label);
            }
          }
          None => {
            labels.insert(w, Queued);
            queue.push_back(Node { dist_from_src: new_dist_from_src, node: w, pred: v });
          }
          Some(_) => {}
        }
      }
      labels.insert(v, v_label);
    }

    if n_dests_found > 0 {
      Some(ShortestPaths::new(labels))
    } else {
      None
    }
  }


  /// Try to find an IIS which consists only of edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  /// Returns `None` if no such IIS exists in the SCC.
  fn find_cycle_edge_iis(&self, scc: &FnvHashSet<usize>, smallest: bool, prune: Option<u32>) -> Option<Iis> {
    let mut smallest_iis = None;
    let mut path_prune = match (smallest, prune) {
      (false, _) => Prune::BestDest,
      (true, None) => Prune::BestDest,
      (true, Some(max_iis_size)) => Prune::BestDestLessThan(max_iis_size - 1),
    };

    for &n in scc {
      for e in self.edges.successors(n) {
        if e.weight > 0 && scc.contains(&e.to) {
          let p = self.shortest_path_scc::<ForwardDir, _>(
            scc,
            e.to,
            std::iter::once(e.from),
            path_prune,
          );

          if let Some(p) = p {
            let num_path_edges = p.num_edges(e.from);
            let iis = smallest_iis.get_or_insert_with(|| Iis::with_capacity(self, num_path_edges + 1));
            iis.clear();
            iis.edges.insert((e.from, e.to));
            iis.add_backwards_path(p.iter_nodes(e.from), false);
            if !smallest {
              return smallest_iis;
            } else {
              path_prune.update_bound(|_| num_path_edges as u32)
            }
          }
        }
      }
    }
    smallest_iis
  }

  /// Try to find an IIS which consists an Upper bound, Lower bound and edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  fn find_cycle_bound_iis(&self, scc: &FnvHashSet<usize>) -> Option<Iis> {
    self.find_scc_bound_infeas(scc.iter().copied(), false).map(|bi| {
      let src = bi.lb_node;
      let dest = bi.ub_node;

      let p1 = self.shortest_path_scc::<ForwardDir, _>(scc, src, once(dest), Prune::BestDest).unwrap();
      let p2 = self.shortest_path_scc::<BackwardDir, _>(scc, src, once(dest), Prune::BestDest).unwrap();

      let n_edges = p1.num_edges(dest) + p2.num_edges(dest);
      let mut iis = Iis::with_capacity(self, n_edges);

      // p1.iter_nodes(): ub_node <- ... <- lb_node
      iis.add_backwards_path(p1.iter_nodes(dest), true);
      // p2.iter_nodes(): ub_node -> ... -> lb_node
      iis.add_forwards_path(p2.iter_nodes(dest), false);
      debug_assert_eq!(iis.edges.len(), n_edges);
      iis
    })
  }

  /// Try to find the *smallest* IIS which consists an Upper bound, Lower bound and edge-constraints (constraints `t[i] + d[i,j] <= t[j]`)
  /// If `prune` is `Some(k)`, only looks for IIS strictly smaller than `k`
  /// Returns `None` if no such IIS exists in the SCC.
  fn find_smallest_cycle_bound_iis(&self, scc: &FnvHashSet<usize>, prune: Option<u32>) -> Option<Iis> {
    let mut best_iis = None;
    let mut global_path_prune = match prune {
      None => Prune::BestDest,
      Some(bound) => Prune::BestDestLessThan(bound - 2), // IIS will always contain two bounds
    };
    let mut dests = Vec::with_capacity(scc.len());

    let max_lb_node = *scc.iter().max_by_key(|n| self.nodes[**n].lb).unwrap();
    let min_ub = scc.iter().map(|n| self.nodes[*n].ub).min().unwrap();
    let src_nodes = std::iter::once(max_lb_node)
      .chain(scc.iter().copied().filter(|n| n != &max_lb_node));

    for src in src_nodes {
      let lb = self.nodes[src].lb;
      if lb <= min_ub { continue; }
      dests.extend(scc.iter().copied().filter(|&n| self.nodes[n].ub < lb));

      let paths_there = self.shortest_path_scc::<ForwardDir, _>(scc, src, dests.iter().copied(), global_path_prune);

      if let Some(paths_there) = paths_there {
        let min_path_there_len = dests.iter().map(|&dst| paths_there.num_edges(dst)).min().unwrap();
        let mut path_prune = global_path_prune;
        path_prune.update_bound(|n| n - min_path_there_len as u32);
        let paths_back = self.shortest_path_scc::<BackwardDir, _>(scc, src, dests.iter().copied(), path_prune);
        if let Some(paths_back) = paths_back {
          let best_dest = dests.iter().copied()
            .min_by_key(|&dst| paths_there.num_edges(dst) + paths_back.num_edges(dst))
            .unwrap();

          // dst <- i1 ... ik <- src
          let p1 = paths_there.iter_nodes(best_dest);
          // dst -> j1 ... jk -> src
          let p2 = paths_back.iter_nodes(best_dest);
          let n_edges = p2.size_hint().0 + p1.size_hint().0 - 2; /* - 1 - 1 (bounds)  */
          let iis = best_iis.get_or_insert_with(|| Iis::with_capacity(self, n_edges));
          iis.clear();
          iis.add_backwards_path(p1, true);
          iis.add_forwards_path(p2, false);
          debug_assert_eq!(n_edges, iis.edges.len());
          global_path_prune.update_bound(|_| iis.len() as u32 - 2);
        }
      }

      dests.clear();
    }
    best_iis
  }
}

#[cfg(test)]
mod tests {
  #[macro_use]
  use crate::*;

  use crate::test_utils::{*, strategy::SharableStrategy};
  use crate::graph::{ModelState, Graph};
  use proptest::prelude::*;
  use proptest::test_runner::TestCaseResult;
  use crate::test_utils::strategy::{graph_with_conn, default_nodes, set_arbitrary_edge_to_one};
  use crate::iis::cycles::{FindCyclicIis, ShortestPathAlg};
  use crate::viz::LayoutAlgo;

  fn cei_triangular_graph() -> impl SharableStrategy<Value=GraphSpec> {
    (3..200usize)
      .prop_flat_map(|mut size| {
      // ensure the graph has enough nodes that the last row (highest rank) doesn't
      // have any lonely nodes (otherwise we don't have a SCC graph)
      let last_node = size - 1;
      let rank = inverse_triangular_number(last_node);
      if last_node < triangular_number(rank - 1) + 2 {
        size += 1;
      }
      set_arbitrary_edge_to_one(graph_with_conn(default_nodes(size), Triangular(), Just(0)))
    })
  }

  fn cbi_triangular_graph() -> impl SharableStrategy<Value=GraphSpec> {
    (1..=13usize)
      .prop_flat_map(|mut rank| {
      let size = triangular_number(rank) + 1;
      graph_with_conn(prop::collection::vec(Just(NodeData { lb: 0, ub: 4, obj: 0 }), size), Triangular(), Just(0))
    })
      .prop_map(|mut g| {
        g.nodes.last_mut().unwrap().lb = 2;
        g.nodes.first_mut().unwrap().ub = 1;
        g
      })
  }


  struct Tests;

  impl Tests {
    fn cei_triangular_graph_iis_size(g: &mut Graph) -> TestCaseResult {
      g.solve();
      let (sccs, first_inf_scc) = match &g.state {
        ModelState::InfCycle { sccs, first_inf_scc } => (&sccs[..], *first_inf_scc),
        other => test_case_bail!("should find infeasible cycle, found: {:?}", other)
      };
      prop_assert_eq!(sccs.len(), 1, "graph is strongly connected");
      let iis = <Graph as FindCyclicIis<ShortestPathAlg>>::find_smallest_cyclic_iis(g, &sccs[first_inf_scc..]);
      prop_assert_eq!(iis.len(), 3);
      Ok(())
    }

    fn cbi_triangular_graph_iis_size(g: &mut Graph) -> TestCaseResult {
      let t = std::time::Instant::now();

      let iis_size = 3 * inverse_triangular_number(g.nodes.len() - 1) /* num edges */ + 2 /* bounds */;
      g.solve();
      // println!("solve time = {}s", t.elapsed().as_millis() as f64 / 1000.);
      let (sccs, first_inf_scc) = match &g.state {
        ModelState::InfCycle { sccs, first_inf_scc } => (&sccs[..], *first_inf_scc),
        other => test_case_bail!("should find infeasible cycle, found: {:?}", other)
      };
      let t = std::time::Instant::now();
      prop_assert_eq!(sccs.len(), 1, "graph is strongly connected");
      let iis = <Graph as FindCyclicIis<ShortestPathAlg>>::find_smallest_cyclic_iis(g, &sccs[first_inf_scc..]);
      prop_assert_eq!(iis.len(), iis_size);
      // println!("iis time = {}s", t.elapsed().as_millis() as f64 / 1000.);
      let no_iis = Graph::find_smallest_cycle_bound_iis(g, &sccs[first_inf_scc], Some(iis_size as u32));
      prop_assert_eq!(no_iis, None);
      let iis2 = Graph::find_smallest_cycle_bound_iis(g, &sccs[first_inf_scc], Some(iis_size as u32 + 1));
      match iis2 {
        Some(iis2) => prop_assert_eq!(iis2.len(), iis.len()),
        None => test_case_bail!("no iis found")
      }

      Ok(())
    }

    fn mixed_cycle_inf_triangular_graph(g: &mut Graph) -> TestCaseResult {
      g.solve();
      let (sccs, first_inf_scc) = match &g.state {
        ModelState::InfCycle { sccs, first_inf_scc } => (&sccs[..], *first_inf_scc),
        other => test_case_bail!("should find infeasible cycle, found: {:?}", other)
      };
      prop_assert_eq!(sccs.len(), 1, "graph is strongly connected");

      let t = std::time::Instant::now();
      let iis = <Graph as FindCyclicIis<ShortestPathAlg>>::find_smallest_cyclic_iis(g, &sccs[first_inf_scc..]);
      prop_assert_eq!(iis.len(), 3);
      // println!("iis time = {}s", t.elapsed().as_millis() as f64 / 1000.);
      Ok(())
    }
  }

  // graph_test_dbg!(Tests; cbi_triangular_graph_iis_size);

  graph_proptests! {
    Tests;
    // Triangular graphs with a single non-zero edge
    cei_triangular_graph() =>
    cei_triangular_graph_iis_size [layout=LayoutAlgo::Fdp];
    // Triangular graphs with a zero edges and a bound infeasibile (LB is the top of the triangle, UB is the bottom-right)
    cbi_triangular_graph() =>
    cbi_triangular_graph_iis_size [layout=LayoutAlgo::Fdp];
    cbi_triangular_graph().prop_map(|mut g| { g.edges.values_mut().for_each(|w| *w = 1); g })
    => mixed_cycle_inf_triangular_graph [layout=LayoutAlgo::Fdp];
  }
}