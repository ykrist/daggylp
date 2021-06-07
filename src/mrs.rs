use super::graph::*;
use fnv::FnvHashSet;

/// Recursive helper function.  For a node n with an active predecessor p, will add the variable on n and the constraint on the active
/// edge to the same MRS as `p`.  Returns the MRS index for the predecessor.
fn add_edge_to_mrs(graph: &Graph, n: usize, mrs_list: &mut Vec<Mrs>, mrs_index: &mut [Option<usize>]) -> usize {
  if let Some(idx) = mrs_index[n] {
    return idx;
  }

  match graph.nodes[n].active_pred {
    Some(p) => {
      let idx = add_edge_to_mrs(graph, p, mrs_list, mrs_index);
      let node_var = graph.var_from_node_id(n);
      mrs_list[idx].add_var(node_var);
      mrs_list[idx].add_constraint(Constraint::Edge(
        graph.var_from_node_id(p),
        node_var,
      ));
      idx
    }
    None => {
      // this is the root
      let idx = mrs_list.len();
      mrs_list.push(Mrs::new_with_root(graph.var_from_node_id(n)));
      idx
    }
  }
}

/// Compute a spanning tree of active edges for an SCC using Breadth-First Search.
fn scc_spanning_tree_bfs(edges_from: &Vec<Vec<Edge>>, nodes: &mut [Node], scc: &FnvHashSet<usize>, root: usize) {
  let mut queue = std::collections::VecDeque::with_capacity(scc.len());
  queue.push_back(root);
  while let Some(n) = queue.pop_front() {
    for e in &edges_from[n] {
      // Adjacent edges that haven't been visited - node is visited if it has an
      // active predecessor or is the root node.
      if scc.contains(&e.to) && nodes[e.to].active_pred.is_none() && e.to != root {
        nodes[e.to].active_pred = Some(n);
        queue.push_back(e.to);
      }
    }
  }
}

impl Graph {
  pub fn compute_mrs(&mut self) -> Mrs {
    todo!() // TODO write a version which just pools everything into a single allocated vector
  }

  pub fn compute_partitioned_mrs(&mut self) -> Vec<Mrs> {
    self.compute_scc_active_edges();

    let mut visited = vec![None; self.nodes.len()];
    let mut mrs_list = Vec::new();

    for (mut n, node) in self.nodes.iter().enumerate() {
      if matches!(&node.kind, NodeKind::Scc(_)) { continue; }
      // already visited
      if visited[n].is_some() { continue; }
      add_edge_to_mrs(self, n, &mut mrs_list, &mut visited);
    }

    mrs_list
  }

  /// This one's a doozy.
  ///
  /// Lets say we have a set of active edges like this:
  /// ```text
  /// a --> b --> c
  /// ```
  /// where `b` is an artificial SCC node made from the cycle
  /// ```text
  /// b1 --> b2 --> b3 --> b1
  /// ```
  /// First, we find a spanning tree of active edges of the SCC.  To do this we need a root node for the SCC.
  /// If the SCC node is the root (i.e. if there were no active edge `a --> b`) then the root node in SCC will be
  /// the member node with the largest lower bound. Otherwise, it will be whatever SCC member node gave rise to
  /// the edge into SCC `a --> b`.  Say that `a --> b1` was the original edge, and `b2` has the largest LB in the SCC.
  ///
  /// In the first case, the SCC spanning tree would be `b1 --> b2 --> b3`
  /// whereas in the second case, it would be `b2 --> b3 --> b1`
  ///
  /// Then, we replace the active edge `a --> b` with `a --> (tree root)`, and replace any edge like `b --> c` with
  /// `b(i) --> c`, where `b(i)` is the original edge from the SCC to `c`.
  ///
  /// Once this transformation is done, we have a spanning forest of the original, non-SCC nodes.
  fn compute_scc_active_edges(&mut self) { // FIXME should be a check to ensure this is only called once (Maybe a new model state)
    // For every Scc, label the internal active-edge tree
    for scc in &self.sccs {
      let root = match self.nodes[scc.scc_node].active_pred {
        // The fake SCC node is not the root
        Some(n) => match self.find_edge(n, scc.scc_node).kind {
          EdgeKind::SccIn(n) => n,
          _ => unreachable!()
        },
        // The fake SCC node *is* the root
        None => scc.lb_node, // node in the SCC with the max LB
      };
      // Set up the edges *inside* the SCC
      scc_spanning_tree_bfs(&self.edges_from, &mut self.nodes, &scc.nodes, root);
    }

    // For all nodes that have an active edge from an SCC node, move the active edge to come
    // from the relevant SCC-component node, which will now have a tree
    for n in 0..self.nodes.len() {
      if let Some(p) = self.nodes[n].active_pred {
        if matches!(self.nodes[p].kind, NodeKind::Scc(_)) { // this node's pred is a fake SCC node
          // move the active edge to come from the underling real node inside the SCC
          let scc_member = match self.find_edge(p, n).kind {
            EdgeKind::SccOut(scc_member) => scc_member,
            _ => unreachable!()
          };
          self.nodes[n].active_pred = Some(scc_member);
        }
      }
    }
  }


}