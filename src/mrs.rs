use super::graph::*;
use crate::model_states::{ModelAction, ModelState};
use crate::set_with_capacity;
use fnv::FnvHashSet;
use gvdot::attr::Model;
use std::cmp::{max, min};
use std::ops::Range;

#[derive(Debug, Clone)]
struct MrsTreeNode {
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
  fn graph_id(&self) -> u32 {
    self.graph_id
  }
}

pub trait MrsPathVisitor {
  fn new_path(&mut self, lb_var: Var);

  fn add_edge(&mut self, from: Var, to: Var);

  fn end_path(&mut self);
}

impl MrsTree {
  fn build<E: EdgeLookup>(graph: &Graph<E>, in_mrs: &[bool], root: usize) -> Self {
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

    let mut tree = MrsTree {
      graph_id: graph.graph_id(),
      nodes,
    };
    tree.build_recursive(graph, in_mrs, 0);
    debug_assert_eq!(tree.nodes[0].subtree_end, tree.nodes.len());
    tree
  }

  fn build_recursive<E: EdgeLookup>(
    &mut self,
    graph: &Graph<E>,
    in_mrs: &[bool],
    idx_of_root: usize,
  ) {
    let root = self.nodes[idx_of_root].node;
    debug_assert_eq!(self.nodes[idx_of_root].node, root);
    let root_children_start = self.nodes.len();
    for e in graph.edges.successors(root) {
      if matches!(&e.kind, &EdgeKind::Regular)
        && in_mrs[e.to]
        && graph.nodes[e.to].active_pred == Some(root)
      {
        let child = &graph.nodes[e.to];
        self.nodes.push(MrsTreeNode {
          node: e.to,
          parent_idx: idx_of_root,
          children_start: usize::MAX, // placeholder
          children_end: usize::MAX,   // placeholder
          subtree_end: usize::MAX,    // placeholder
          lb: child.lb,
          obj: child.obj,
          incoming_edge_weight: e.weight,
        });
      }
    }

    let root_children_end = self.nodes.len();
    self.nodes[idx_of_root].children_start = root_children_start;
    self.nodes[idx_of_root].children_end = root_children_end;
    for child in root_children_start..root_children_end {
      self.build_recursive(graph, in_mrs, child)
    }
    self.nodes[idx_of_root].subtree_end = self.nodes.len();
  }

  pub fn obj(&self) -> Weight {
    let root = &self.nodes[0];
    self.compute_obj_dfs(root.lb, root)
  }

  fn compute_obj_dfs(&self, x: Weight, node: &MrsTreeNode) -> Weight {
    let mut total = node.obj * x;
    for child in (node.children_start..node.children_end).map(|idx| &self.nodes[idx]) {
      total += self.compute_obj_dfs(x + child.incoming_edge_weight, child);
    }
    total
  }

  /// Return the variable at the root of the MRS, i.e. the variable whose lower bounds is in the MRS
  pub fn root(&self) -> Var {
    self.var_from_node_id(self.nodes[0].node)
  }

  /// Return the LB of the variable at the root of the MRS
  pub fn root_lb(&self) -> Weight {
    self.nodes[0].lb
  }

  /// Iterate over the variables in this MRS
  pub fn vars(&self) -> impl Iterator<Item = Var> + '_ {
    self
      .nodes
      .iter()
      .map(move |n| self.var_from_node_id(n.node))
  }

  /// Iterate over the variables in this MRS whose objectives are non-zero
  pub fn obj_vars(&self) -> impl Iterator<Item = Var> + '_ {
    self
      .nodes
      .iter()
      .filter(|n| n.obj != 0)
      .map(move |n| self.var_from_node_id(n.node))
  }

  pub fn has_obj_vars(&self) -> bool {
    self.nodes.iter().filter(|n| n.obj != 0).next().is_some()
  }

  /// Iterate over the edge constraints in the MRS
  pub fn edge_constraints(&self) -> impl Iterator<Item = (Var, Var)> + '_ {
    self.nodes.iter().skip(1).map(move |n| {
      // first node is root node
      (
        self.var_from_node_id(self.nodes[n.parent_idx].node),
        self.var_from_node_id(n.node),
      )
    })
  }

  /// Iterate over the edge constraints in the MRS, as Constraint objects.
  pub fn constraints(&self) -> impl Iterator<Item = Constraint> + '_ {
    std::iter::once(Constraint::Lb(self.var_from_node_id(self.nodes[0].node))).chain(
      self
        .edge_constraints()
        .map(|(vi, vj)| Constraint::Edge(vi, vj)),
    )
  }

  /// Computes the optimal solution of the problem defined by a subtree of the MRS.
  fn forward_label_subtree(&self, soln_buf: &mut [Weight], root_idx: usize) {
    // We can just traverse in topological order here
    let root = &self.nodes[root_idx];
    soln_buf[root_idx] = root.lb;
    for (i, n) in self
      .nodes
      .iter()
      .enumerate()
      .skip(root.children_start)
      .take(root.subtree_end - root.children_start)
    {
      debug_assert!(
        n.parent_idx == root_idx || (root.children_start..root.subtree_end).contains(&i)
      );
      soln_buf[i] = max(n.lb, soln_buf[n.parent_idx] + n.incoming_edge_weight);
    }
  }
}

/// Compute a spanning tree of active edges for an SCC using Breadth-First Search.
fn scc_spanning_tree_bfs<E: EdgeLookup>(
  edges: &E,
  nodes: &mut [Node],
  scc: &FnvHashSet<usize>,
  root: usize,
) {
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
          }
          Some(_) => {}
        }
      }
    }
  }
}

impl<E: EdgeLookup> Graph<E> {
  pub fn critical_paths(&mut self) -> Result<Vec<Vec<Var>>, crate::Error> {
    self.check_allowed_action(ModelAction::ComputeOptimalityInfo)?;
    self.compute_scc_active_edges();

    let mut paths = Vec::new();

    for (n, mut node) in self.nodes.iter().enumerate() {
      if !matches!(node.kind, NodeKind::Scc(_)) && node.obj > 0 {
        let mut path = Vec::new();
        path.push(self.var_from_node_id(n));
        while let Some(p) = node.active_pred {
          path.push(self.var_from_node_id(p));
          node = &self.nodes[p];
        }
        path.reverse();
        paths.push(path);
      }
    }
    Ok(paths)
  }

  pub fn visit_critical_paths(
    &mut self,
    mut callback: impl FnMut(&Self, &[Var]),
  ) -> Result<(), crate::Error> {
    self.check_allowed_action(ModelAction::ComputeOptimalityInfo)?;
    self.compute_scc_active_edges();
    let mut var_buf = Vec::new();

    for (n, mut node) in self.nodes.iter().enumerate() {
      if !matches!(node.kind, NodeKind::Scc(_)) && node.obj > 0 {
        var_buf.clear();
        var_buf.push(self.var_from_node_id(n));
        while let Some(p) = node.active_pred {
          var_buf.push(self.var_from_node_id(p));
          node = &self.nodes[p];
        }
        var_buf.reverse();
        callback(self, &var_buf);
      }
    }
    Ok(())
  }

  /// Returns the objective contribution of the tree, and for each edge constraint in the tree, returns:
  /// 1. The edge (v1, v2)
  /// 2. The total subtree objective of the subtree rooted at v2, assuming all constraints in the MRS hold.
  /// 3. The total subtree objective of the subtree rooted at v2, once edge (v1, v2) is removed,
  ///    and all other constraints in the MRS hold.
  ///
  /// Does *NOT* account for constraints which are not in the MRS.
  pub fn edge_sensitivity_analysis(
    &self,
    mrs: &MrsTree,
  ) -> (Weight, Vec<((Var, Var), Weight, Weight)>) {
    // optimal solution for whole tree
    let mut cum_obj_full_tree: Vec<Weight> =
      mrs.nodes.iter().map(|n| self.nodes[n.node].x).collect();
    // traverse in reverse topological order
    for (i, node) in mrs.nodes.iter().enumerate().rev() {
      cum_obj_full_tree[i] = node.obj * cum_obj_full_tree[i]
        + (node.children_start..node.children_end)
          .map(|j| cum_obj_full_tree[j])
          .sum::<Weight>();
    }
    // cum_obj_full_tree[i] now holds the cumulative subtree objective (coeff * variable value) at node i

    let mut solution_buf = vec![Weight::MAX; mrs.nodes.len()];

    let edge_obj_diff = mrs
      .nodes
      .iter()
      .enumerate()
      .skip(1)
      .map(|(i, node)| {
        let parent_node = &mrs.nodes[node.parent_idx];
        mrs.forward_label_subtree(&mut solution_buf, i);
        let cum_obj_subtree = solution_buf[i] * node.obj
          + solution_buf[node.children_start..node.subtree_end]
            .iter()
            .zip(&mrs.nodes[node.children_start..node.subtree_end])
            .map(|(&t, n)| t * n.obj)
            .sum::<Weight>();
        debug_assert!(cum_obj_subtree <= cum_obj_full_tree[i]);
        (
          (
            self.var_from_node_id(parent_node.node),
            self.var_from_node_id(node.node),
          ),
          cum_obj_full_tree[i],
          cum_obj_subtree,
        )
      })
      .collect();

    (cum_obj_full_tree[0], edge_obj_diff)
  }

  pub fn compute_mrs(&mut self) -> Result<Vec<MrsTree>, crate::Error> {
    self.check_allowed_action(ModelAction::ComputeOptimalityInfo)?;
    self.compute_scc_active_edges();
    let mut is_in_mrs = vec![false; self.nodes.len()];

    let roots: Vec<usize> = self
      .nodes
      .iter()
      .enumerate()
      .filter_map(|(n, node)| {
        if node.obj == 0 || matches!(node.kind, NodeKind::Scc(_)) {
          None
        } else {
          self.mark_path_to_mrs_root(&mut is_in_mrs, n, node)
        }
      })
      .collect();

    #[cfg(feature = "viz-extra")]
    {
      self.viz_data.clear_highlighted();
    }

    let forest = roots
      .into_iter()
      .map(|r| {
        let tree = MrsTree::build(self, &is_in_mrs, r);
        #[cfg(feature = "viz-extra")]
        {
          self
            .viz_data
            .highlighted_edges
            .extend(tree.edge_constraints().map(|(vi, vj)| (vi.node, vj.node)));
          self.viz_data.highlighted_nodes.insert(r);
        }
        tree
      })
      .collect();
    Ok(forest)
  }

  /// Marks the path to the MRS root, returning the root if the root has never been visited and None if it has.
  fn mark_path_to_mrs_root(&self, is_in_mrs: &mut [bool], n: usize, node: &Node) -> Option<usize> {
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
  fn compute_scc_active_edges(&mut self) {
    // FIXME should be a check to ensure this is only called once (Maybe a new model state)
    if matches!(self.state, ModelState::Mrs) {
      return;
    }
    // Step 1: Mark active edges inside the SCC, and the active edge ENTERING the SCC (if applicable)
    for scc in &self.sccs {
      let (root, root_active_pred) = match self.nodes[scc.scc_node].active_pred {
        // The fake SCC node is not the root
        Some(p) => match self.edges.find_edge(p, scc.scc_node).kind {
          EdgeKind::SccIn(n) => (n, Some(p)),
          EdgeKind::SccToScc { from: p, to: n } => (n, Some(p)),
          _ => unreachable!(),
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
    self.state = ModelState::Mrs;
  }
}

#[cfg(test)]
mod tests {
  #[macro_use]
  use crate::*;
  use super::*;
  use crate::graph::SolveStatus;
  use crate::test::{strategy::*, *};
  use fnv::FnvHashMap;
  use proptest::prelude::*;
  use InfKind::*;
  use SolveStatus::*;

  #[graph_proptest]
  #[input(acyclic_graph(
  prop::collection::vec(nodes(MIN_WEIGHT..MAX_WEIGHT / 2, Just(MAX_WEIGHT), Just(0)), 1..100),
  0..(10 as Weight)
  ))]
  fn trivial_objective_gives_empty(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Optimal);
    prop_assert_eq!(g.compute_mrs()?.len(), 0);
    Ok(())
  }

  #[graph_proptest]
  #[input(acyclic_graph(
  prop::collection::vec(nodes(MIN_WEIGHT..MAX_WEIGHT / 2, Just(MAX_WEIGHT), 1..MAX_WEIGHT), 50..100),
  0..(10 as Weight)
  ))]
  fn disjoint(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Optimal);

    let mut seen_vars = FnvHashSet::default();
    let mut seen_cons = FnvHashSet::default();
    let mrs = g.compute_mrs()?;
    prop_assert!(mrs.len() > 0, "MRS should not be empty");
    // println!("{:?}", mrs);
    for mrs in mrs {
      for v in mrs.vars() {
        let unseen = seen_vars.insert(v);
        prop_assert!(unseen, "MRS is not disjoint, seen_vars = {:?}", &seen_vars);
      }
      for c in mrs.constraints() {
        let unseen = seen_cons.insert(c);
        prop_assert!(unseen, "MRS is not disjoint, seen_cons = {:?}", &seen_cons);
      }
    }
    Ok(())
  }

  fn build_minimal_graph(tree: &MrsTree) -> (FnvHashMap<(Var, Var), (Var, Var)>, Graph) {
    let mut g = Graph::new();

    let vars: Vec<_> = tree
      .nodes
      .iter()
      .map(|n| g.add_var(n.obj, n.lb, Weight::MAX))
      .collect();

    let mut g = g.finish_nodes();

    let constraints: FnvHashMap<_, _> = tree
      .nodes
      .iter()
      .enumerate()
      .skip(1)
      .map(|(j, n)| {
        let vi = vars[n.parent_idx];
        let vj = vars[j];
        let wi = tree.var_from_node_id(tree.nodes[n.parent_idx].node);
        let wj = tree.var_from_node_id(n.node);
        g.add_constr(vi, n.incoming_edge_weight, vj);
        ((wi, wj), (vi, vj))
      })
      .collect();

    (constraints, g.finish())
  }

  fn sensitivity_analysis(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Optimal);
    let computed_obj = g.compute_obj()?;
    let mut total_obj = 0;

    let mrs = g.compute_mrs()?;
    for mrs_tree in mrs {
      let (obj, edge_sa) = g.edge_sensitivity_analysis(&mrs_tree);
      total_obj += obj;
      let (constr_map, mut subgraph) = build_minimal_graph(&mrs_tree);

      let status = subgraph.solve();
      // subgraph.viz().save_svg("scrap.svg");
      prop_assert_matches!(status, SolveStatus::Optimal);
      let total_obj_mrs = subgraph.compute_obj().unwrap();

      for (e, obj_with_e, obj_without_e) in edge_sa {
        let mut subgraph_without_e = subgraph.clone();
        let (vi, vj) = constr_map[&e];
        subgraph_without_e.remove_edge_constraint(vi, vj);
        let status = subgraph_without_e.solve();
        prop_assert_matches!(status, SolveStatus::Optimal);
        let total_obj_without_e = subgraph_without_e.compute_obj().unwrap();
        prop_assert_eq!(
          obj_with_e - obj_without_e,
          total_obj_mrs - total_obj_without_e,
          "removing edge {} -> {}",
          vi.node,
          vj.node
        );
      }
    }
    prop_assert_eq!(computed_obj, total_obj);
    Ok(())
  }

  #[graph_test]
  #[input("tree.f")]
  #[config(layout = "fdp")]
  #[input("multiple-sccs-mrs.f")]
  fn sa_testcases(g: &mut Graph) -> GraphTestResult {
    sensitivity_analysis(g).into_graph_test()
  }

  #[graph_proptest]
  #[input(acyclic_graph(
  prop::collection::vec(nodes(MIN_WEIGHT..MAX_WEIGHT / 2, Just(MAX_WEIGHT), 0..=1i64), 2..100),
  0..(20 as Weight)
  ))]
  fn sa_proptests(g: &mut Graph) -> GraphProptestResult {
    sensitivity_analysis(g)
  }

  #[graph_proptest]
  #[input(acyclic_graph(
  prop::collection::vec(nodes(MIN_WEIGHT..MAX_WEIGHT / 2, Just(MAX_WEIGHT), 0..=1i64), 2..100),
  0..(20 as Weight)
  ))]
  fn compute_obj(g: &mut Graph) -> GraphProptestResult {
    let status = g.solve();
    prop_assert_matches!(status, SolveStatus::Optimal);
    let mut total_obj_sa = 0;
    let mut total_obj_direct = 0;

    let mrs = g.compute_mrs()?;
    for mrs_tree in mrs {
      let obj_direct = mrs_tree.obj();
      let (obj_sa, _) = g.edge_sensitivity_analysis(&mrs_tree);
      prop_assert_eq!(obj_direct, obj_sa);
    }
    Ok(())
  }

  #[graph_test]
  #[config(sccs = "hide")]
  #[input("multiple-sccs.f")]
  fn forest(g: &mut Graph) -> GraphTestResult {
    if matches!(g.solve(), SolveStatus::Infeasible(_)) {
      g.compute_iis(true);
      anyhow::bail!("infeasible")
    }
    let mut mrs = g.compute_mrs()?;
    graph_testcase_assert_eq!(mrs.len(), 1);
    let mrs = mrs.pop().unwrap();
    Ok(())
  }
}
