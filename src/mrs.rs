use super::graph::*;
use fnv::FnvHashSet;
use std::ops::Range;
use crate::set_with_capacity;
use std::cmp::{min, max};
use crate::test::strategy::node;
use proptest::strategy::W;

#[derive(Debug, Clone)]
pub struct MrsTreeNode {
  node: usize,
  parent_idx: usize,
  incoming_edge_weight: Weight,
  obj: Weight,
  lb: Weight,
  /// `MrsTree.nodes[children_start..child_end]` is this node's children
  children_start: usize,
  /// `MrsTree.nodes[children_start..child_end]` is this node's children
  children_end: usize,
  /// `MrsTree.nodes[children_start..subtree_end] `is the subtree rooted at this node (excluding this node)
  subtree_end: usize,
}

#[derive(Debug, Clone)]
pub struct MrsTree {
  graph_id: u32,
  // Nodes in topological order
  nodes: Vec<MrsTreeNode>,
}

impl GraphId for MrsTree {
  fn graph_id(&self) -> u32 { self.graph_id }
}

impl MrsTree {
  fn build(graph: &Graph, root: usize) -> Self {
    let r = &graph.nodes[root];

    let mut nodes = vec![MrsTreeNode {
      node: root,
      parent_idx: usize::MAX,
      incoming_edge_weight: 0,
      children_start: 1,
      children_end: 1,
      subtree_end: 1,
      lb: r.lb,
      obj: r.obj,
    }];

    let mut tree = MrsTree { graph_id: graph.graph_id(), nodes };
    tree.build_recursive(graph, root);
    debug_assert_eq!(tree.nodes[0].subtree_end, tree.nodes.len());
    tree
  }

  fn build_recursive(&mut self, graph: &Graph, root: usize) {
    let idx_of_root = self.nodes.len() - 1;
    for e in graph.edges.successors(root) {
      if matches!(&e.kind, &EdgeKind::Regular) && graph.nodes[e.to].active_pred == Some(root) {
        let child = &graph.nodes[e.to];

        self.nodes.push(MrsTreeNode {
          node: e.to,
          parent_idx: idx_of_root,
          children_start: idx_of_root,
          children_end: idx_of_root,
          subtree_end: idx_of_root,
          lb: child.lb,
          obj: child.obj,
          incoming_edge_weight: e.weight,

        });
      }
    }
    let root_children_start = idx_of_root + 1;
    let root_children_end = self.nodes.len();
    self.nodes[idx_of_root].children_start = root_children_start;
    self.nodes[idx_of_root].children_end = root_children_end;

    for child in root_children_start..root_children_end {
      self.build_recursive(graph, self.nodes[child].node)
    }
    self.nodes[idx_of_root].subtree_end = self.nodes.len();
  }

  pub fn vars(&self) -> impl Iterator<Item=Var> + '_ {
    self.nodes.iter().map(move |n| self.var_from_node_id(n.node))
  }

  pub fn constrs(&self) -> impl Iterator<Item=Constraint> + '_ {
    std::iter::once(Constraint::Lb(self.var_from_node_id(self.nodes[0].node)))
      .chain(self.nodes.iter().skip(1).map(move |n| { // first node is root node
        Constraint::Edge(self.var_from_node_id(self.nodes[n.parent_idx].node), self.var_from_node_id(n.node))
      }))
  }

  /// Computes the optimal solution of the problem defined by a subtree of the MRS.
  fn forward_label_subtree(&self, soln_buf: &mut [Weight], root_idx: usize) {
    // We can just traverse in topological order here
    let root = &self.nodes[root_idx];
    soln_buf[root_idx] = root.lb;
    for (i, n) in self.nodes.iter().enumerate().skip(root.children_start).take(root.subtree_end - root.children_start) {
      debug_assert!(n.parent_idx == root_idx || (root.children_start..root.subtree_end).contains(&i));
      soln_buf[i] = max(n.lb, soln_buf[n.parent_idx] + n.incoming_edge_weight);
    }
  }
}

/// Compute a spanning tree of active edges for an SCC using Breadth-First Search.
fn scc_spanning_tree_bfs<E: EdgeLookup>(edges: &E, nodes: &mut [Node], scc: &FnvHashSet<usize>, root: usize) {
  let mut queue = std::collections::VecDeque::with_capacity(scc.len());
  queue.push_back(root);
  while let Some(i) = queue.pop_front() {
    for j in edges.successor_nodes(i) {
      // Adjacent edges that haven't been visited - node is visited if it has an
      // active predecessor or is the root node.
      if j != root && scc.contains(&j) {
        match &mut nodes[j].active_pred {
          ap @ None => {
            *ap = Some(i);
            queue.push_back(j);
          },
          Some(_) => {}
        }
      }
    }
  }
}

impl Graph {
  /// For each edge in the tree, how much would the objective change if we removed that edge from the MRS?
  /// Does *NOT* account for constraints which are not in the MRS.
  pub fn edge_sensitivity_analysis(&self, mrs: &MrsTree) -> Vec<((Var, Var), Weight)> {
    // optimal solution for whole tree
    let mut cum_obj_full_tree: Vec<Weight> = mrs.nodes.iter().map(|n| self.nodes[n.node].x).collect();
    // traverse in reverse topological order
    for (i, node) in mrs.nodes.iter().enumerate().rev() {
      cum_obj_full_tree[i] = node.obj * cum_obj_full_tree[i] +
        (node.children_start..node.children_end).map(|j| cum_obj_full_tree[j]).sum::<Weight>();
    }
    // cum_obj_full_tree[i] now holds the cumulative subtree objective (coeff * variable value) at node i

    let mut solution_buf = vec![Weight::MAX; mrs.nodes.len()];

    mrs.nodes.iter().enumerate().skip(1)
      .map(|(i, node)| {
        let parent_node = &mrs.nodes[node.parent_idx];
        mrs.forward_label_subtree(&mut solution_buf, i);
        let cum_obj_subtree = solution_buf[i] * node.obj +
          solution_buf[node.children_start..node.subtree_end].iter()
            .zip(&mrs.nodes[node.children_start..node.subtree_end])
            .map(|(&t, n)| t * n.obj)
            .sum::<Weight>();
        let diff = cum_obj_subtree - cum_obj_full_tree[i];
        debug_assert!(diff <= 0);
        ((self.var_from_node_id(parent_node.node), self.var_from_node_id(node.node)), diff)
      })
      .collect()
  }

  pub fn compute_mrs(&mut self) -> Vec<MrsTree> {
    self.compute_scc_active_edges();
    let mut is_in_mrs = vec![false; self.nodes.len()];

    let roots: Vec<usize> = self.nodes.iter().enumerate()
      .filter_map(|(n, node)|
        if node.obj == 0 || matches!(node.kind, NodeKind::Scc(_)) {
          None
        } else {
          self.mark_path_to_mrs_root(&mut is_in_mrs, n, node)
        }
      )
      .collect();

    roots.into_iter()
      .map(|r| MrsTree::build(self, r))
      .collect()
  }

  /// Marks the path to the MRS root, returning the root if the root has never been visited and None if it has.
  fn mark_path_to_mrs_root(&self, is_in_mrs: &mut [bool], n: usize, node: &Node) -> Option<usize> { // TODO this should happen after the active edge re-labelling
    if !is_in_mrs[n] {
      is_in_mrs[n] = true;
      if let Some(pred) = node.active_pred {
        self.mark_path_to_mrs_root(is_in_mrs, pred, &self.nodes[pred])
      } else {
        Some(n)
      }
    } else {
      None
    }
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
    // Step 1: Mark active edges inside the SCC, and the active edge ENTERING the SCC (if applicable)
    for scc in &self.sccs {
      let (root, root_active_pred) = match self.nodes[scc.scc_node].active_pred {
        // The fake SCC node is not the root
        Some(p) => match self.edges.find_edge(p, scc.scc_node).kind {
          EdgeKind::SccIn(n) => (n, Some(p)),
          EdgeKind::SccToScc { from: p, to: n } => (n, Some(p)),
          _ => unreachable!()
        },
        // The fake SCC node *is* the root
        None => (scc.lb_node, None), // node in the SCC with the max LB
      };
      // Set up the edges *inside* the SCC
      scc_spanning_tree_bfs(&self.edges, &mut self.nodes, &scc.nodes, root);
      self.nodes[root].active_pred = root_active_pred;
    }

    // Step 2: Mark active edges leaving the SCC.
    // Note that if we have an active edge SCC(1) -> SCC(2) during forward-labelling,
    // this edge has already been processed in Step 1 for SCC(2).
    // Therefore, we only need to check active edges SCC -> VAR
    for n in 0..self.nodes.len() {
      match (self.nodes[n].active_pred, self.nodes[n].kind) {
        (Some(p), NodeKind::Var) => {
          if matches!(self.nodes[p].kind, NodeKind::Scc(_)) {
            let scc_member = match self.edges.find_edge(p, n).kind {
              EdgeKind::SccOut(scc_member) => scc_member,
              _ => unreachable!(),
            };
            self.nodes[n].active_pred = Some(scc_member);
          }
        }
        (Some(_), NodeKind::Scc(_)) => self.nodes[n].active_pred = None, // TODO necessary only for debug output?
        _ => {}
      }
    }
  }
}


#[cfg(test)]
mod tests {
  #[macro_use]
  use crate::*;
  use super::*;
  use crate::test::{*, strategy::*};
  use crate::viz::*;
  use SolveStatus::*;
  use InfKind::*;
  use proptest::prelude::*;
  use crate::graph::SolveStatus;

  #[graph_proptest]
  #[input(acyclic_graph(
    prop::collection::vec(nodes(MIN_WEIGHT..MAX_WEIGHT / 2, Just(MAX_WEIGHT), Just(0)), 1..100),
    0..(10 as Weight)
  ))]
  fn trivial_objective_gives_empty(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Optimal);
    prop_assert_eq!(g.compute_mrs().len(), 0);
    Ok(())
  }

  #[graph_proptest]
  #[input(acyclic_graph(
    prop::collection::vec(nodes(MIN_WEIGHT..MAX_WEIGHT/2, Just(MAX_WEIGHT), 1..MAX_WEIGHT), 50..100),
    0..(10 as Weight)
  ))]
  fn disjoint(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Optimal);

    let mut seen_vars = FnvHashSet::default();
    let mut seen_cons = FnvHashSet::default();
    let mrs = g.compute_mrs();
    prop_assert!(mrs.len() > 0, "MRS should not be empty");
    // println!("{:?}", mrs);
    for mrs in mrs {
      for v in mrs.vars() {
        let unseen = seen_vars.insert(v);
        prop_assert!(unseen, "MRS is not disjoint, seen_vars = {:?}", &seen_vars);
      }
      for c in mrs.constrs() {
        let unseen = seen_cons.insert(c);
        prop_assert!(unseen, "MRS is not disjoint, seen_cons = {:?}", &seen_cons);
      }
    }
    Ok(())
  }

  #[graph_test]
  #[config(sccs="hide")]
  #[input("multiple-sccs.f")]
  fn forest(g: &mut Graph) -> GraphTestResult {
    if matches!(g.solve(), SolveStatus::Infeasible(_)) {
      Err(anyhow::anyhow!("infeasible").iis(g.compute_iis(true)))?
    }
    let mut mrs = g.compute_mrs();
    graph_testcase_assert_eq!(mrs.len(), 1);
    let mrs = mrs.pop().unwrap();
    println!("{:?}", &mrs);
    Ok(())
  }

}
