use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};
use fnv::{FnvHashSet, FnvHashMap};
use std::cmp::min;
use std::collections::HashMap;
use std::hash::Hash;
use std::collections::hash_map::Entry;
use crate::graph::EdgeKind::{SccOut, SccIn};

fn set_with_capacity<K>(n: usize) -> FnvHashSet<K> {
    FnvHashSet::with_capacity_and_hasher(n, Default::default())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Mrs {
    vars: Vec<Var>,
    constrs: FnvHashSet<Constraint>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Iis {
    constrs: FnvHashSet<Constraint>,
}

type Weight = u64;

#[derive(Debug, Copy, Clone)]
pub enum InfKind {
    Cycle,
    Path
}

#[derive(Debug, Copy, Clone)]
pub enum SolveStatus {
    Infeasible(InfKind),
    Optimal,
}

#[derive(Debug, Clone)]
enum ModelState {
    Init,
    Cycles,
    CycleInfeasible{ sccs: Vec<FnvHashSet<usize>>, inf_idx: usize },
    InfPath(Vec<usize>),
    Optimal,
}

pub enum EdgeKind {
    Regular,
    SccIn(usize),
    SccOut(usize),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edge {
    from : usize,
    to: usize,
    weight : Weight,
    /// used for MRS calculation
    kind: EdgeKind,
}


#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Constraint {
    Ub(Var),
    Lb(Var),
    Edge(Var, Var)
}

#[derive(Hash, Copy, Clone, Debug, Eq, PartialEq)]
pub struct Var {
    graph_id: u32,
    node: usize,
}

enum NodeKind {
    /// Regular node
    Var,
    /// Strongly-connected component Member, contains scc index
    SccMember(usize),
    /// Artificial SCC node, contains scc index
    Scc(usize),
}

impl NodeKind {
    // Should this node kind be ignored during forward-labelling
    fn ignored_fwd_label(&self) -> bool {
        matches!(self, NodeKind::SccMember(_))
    }
}

pub struct Node {
    x: Weight,
    obj: Weight,
    lb: Weight,
    ub: Weight,
    active_pred: Option<usize>,
    kind: NodeKind
}

pub struct SccEdge {
    scc_member: usize,
    from: usize,
    to: usize,

}

pub struct SccInfo {
    nodes: FnvHashSet<usize>,
    lb_node: usize,
    ub_node: usize,
    scc_node: usize,
}

pub struct Graph {
    id: u32,
    nodes: Vec<Node>,
    sccs: Vec<SccInfo>,
    num_active_nodes: usize,
    edges_from: Vec<Vec<Edge>>,
    edges_to: Vec<Vec<Edge>>,
    source_nodes: Vec<usize>,
    parameters: Parameters,
    state: ForwardLabelResult,
}

#[derive(Copy, Clone, Debug)]
pub struct Parameters {
    minimal_cyclic_iis: bool,
    minimal_acyclic_iis: bool,
    size_hint_vars: usize,
    size_hint_constrs: usize,
}

impl Default for Parameters {
    fn default() -> Self {
        Parameters {
            minimal_cyclic_iis: false,
            minimal_acyclic_iis: true,
            size_hint_vars: 0,
            size_hint_constrs: 0,
        }
    }
}

impl Parameters {
    fn build() -> ParamBuilder { ParamBuilder{ params: Default::default() }}
}

pub struct ParamBuilder {
    params: Parameters,
}

impl ParamBuilder {
    pub fn min_cyclic_iis(mut self, val: bool) -> Self {
        self.params.minimal_cyclic_iis = val;
        self
    }

    pub fn min_acyclic_iis(mut self, val: bool) -> Self {
        self.params.minimal_acyclic_iis = val;
        self
    }

    pub fn size_hint(mut self, n_vars: usize, n_constrs: usize) -> Self {
        self.params.size_hint_vars = n_vars;
        self.params.size_hint_constrs = n_constrs;
        self
    }
}

struct GraphBuilder {}
// once it's been finalised, not more adding of edges/nodes.  Only modification allowed is to remove
// an edge.

impl Graph {
    pub fn new_with_params(params: Parameters) -> Self {
        static NEXT_ID : AtomicU32 = AtomicU32::new(0);
        Graph {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            nodes: Vec::with_capacity(params.size_hint_vars),
            sccs: Vec::new(),
            num_active_nodes: 0,
            edges_from: Vec::with_capacity(params.size_hint_constrs),
            edges_to: Vec::with_capacity(params.size_hint_constrs),
            source_nodes: Vec::new(),
            parameters: params,
            state: ModelState::Init,
        }
    }

    pub fn new() -> Self {
        Self::new_with_params(Parameters::default())
    }

    fn reset_nodes(&mut self) {
        for n in self.nodes.iter_mut() {
            n.x = n.lb;
            n.active_pred = None;
        }
    }

    fn add_node(&mut self, obj: Weight, lb: Weight, ub: Weight, kind: NodeKind) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node{ lb, ub, x: lb, obj, active_pred: None, kind});
        self.edges_to.push(Vec::new());
        self.edges_from.push(Vec::new());
        id
    }

    fn var_from_node_id(&self, node: usize) -> Var {
        Var{ graph_id: self.id, node }
    }

    fn add_var(&mut self, obj: Weight, lb: Weight, ub: Weight) -> Var {
        assert!(obj >= 0);
        self.var_from_node_id(self.add_node(obj, lb, ub, NodeKind::Var))
    }

    pub fn add_constr(&mut self, lhs: Var, d: Weight, rhs: Var) {
        assert!(d >= 0);
        assert!(lhs != rhs);
        let e = Edge{
            from: lhs.node,
            to: rhs.node,
            weight: d,
            kind: EdgeKind::Regular
        };
        self.add_edge(e);
    }

    fn remove_edge(&mut self, e: Edge) {
        self.edges_to[e.to].retain(|e| e.from != e.to);
        self.edges_from[e.from].retain(|e| e.to != e.from);
    }

    fn add_edge(&mut self, e: Edge){
        self.edges_to[e.to].push(e);
        self.edges_from[e.from].push(e);
    }

    fn solve(&mut self) -> SolveStatus {
        self.update();
        let state = self.forward_label();
        self.state = state;
        match state {
            ModelState::Cycles => SolveStatus::Optimal,
            ModelState::CyclesFeasible => {
                let sccs = self.find_sccs();
                let inf_idx = sccs.iter().enumerate()
                    .find(|(_, scc)| !self.scc_is_feasible(scc) )
                    .map(|(idx, _)| idx);

                if let Some(inf_idx) = inf_idx {
                    self.state = ModelState::CycleInfeasible { sccs, inf_idx };
                    SolveStatus::Infeasible(InfKind::Cycle)
                } else {
                    self.condense(sccs);
                    self.solve()
                }
            },
            ModelState::InfPath(_) => SolveStatus::Infeasible(InfKind::Path),
            ModelState::Init => unreachable!(),
        }
    }

    fn edge_to_constraint(&self, e: &Edge) -> Constraint {
        Constraint::Edge(self.var_from_node_id(e.from), self.var_from_node_id(e.to))
    }

    fn forward_label(&mut self) -> ModelState {
        self.update();

        let mut num_active_nodes = 0;
        let mut num_visited_nodes = 0;
        let mut stack = vec![];
        let mut violated_ubs = vec![];
        let mut num_visited_pred = vec![0; self.nodes.len()];
        // Preprocessing - find source nodes and count number of active nodes
        for (n, node) in self.nodes.iter().enumerate() {
            if self.edges_to[n].is_empty() {
                stack.push(n);
            }
            if !node.kind.ignored_fwd_label() {
                num_active_nodes += 1;
            }
        }
        let num_active_nodes = num_active_nodes;

        // Solve
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            num_visited_nodes += 1;

            let x = node.x;
            for e in self.edges_from[i] {
                let nxt_node = &mut self.nodes[e.to];
                if node.kind.ignored_fwd_label() {
                    continue; // skip this once
                }
                let nxt_x = x + e.weight;
                if nxt_node.x < nxt_x {
                    nxt_node.x = nxt_x;
                    nxt_node.active_pred = Some(i);
                    if nxt_x > nxt_node.ub {
                        violated_ubs.push(e.to);
                    }
                }
                num_visited_pred[e.to] += 1;
                if num_visited_pred[e.to] == self.edges_to[e.to].len() {
                    stack.push(e.to)
                }
            }

        }

        // Post-processing: check infeasibility kinds
        if num_visited_nodes < num_active_nodes {
            ModelState::Cycles
        } else if !violated_ubs.is_empty() {
            ModelState::InfPath(violated_ubs)
        } else {
            ModelState::Optimal
        }
    }

    /// Find all Strongly-Connected Components with 2 or more nodes
    fn find_sccs(&mut self) -> Vec<FnvHashSet<usize>> {
        debug_assert_eq!(self.sccs.len(), 0);
        todo!()
    }

    // return the shortest path along edges in the SCC
    fn shortest_path_scc(&self, scc: &FnvHashSet<usize>, forwards: bool,  start: usize, end: usize) ->  Vec<usize> {
        todo!()
    }

    fn find_scc_bound_infeas(&self, nodes: impl Iterator<Item=usize>) -> Option<((usize, Weight), (usize, Weight))> {
        let mut nodes = nodes.map(|n| (n, &self.nodes[n]));

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
                min_ub_node = node.n;
            }
            if max_lb < min_ub {
                return Some((((max_lb_node, max_lb), (min_ub_node, min_ub))))
            }
        }

        None
    }

    fn scc_is_feasible(&self, scc: &FnvHashSet<usize>) -> bool {
        for &n in scc {
            for e in &self.edges_from[n] {
                if e.weight > 0 && scc.contains(&e.to) {
                    return false
                }
            }
        }

        self.find_scc_bound_infeas(scc.iter().copied()).is_none()
    }

    fn compute_cyclic_iis(&self, scc: &FnvHashSet<usize>) -> Iis {
        for &n in scc {
            for e in &self.edges_from[n] {
                if e.weight > 0 && scc.contains(&e.to) {
                    let p = self.shortest_path_scc(scc,true, e.to, e.from);

                    let mut constrs = set_with_capacity(p.len());
                    constrs.insert(Constraint::Edge(self.var_from_node_id(e.from), self.var_from_node_id(e.to)));

                    let mut vars = p.into_iter().map(|n| self.var_from_node_id(n));

                    let mut vi = vars.next().unwrap();
                    for vj in vars {
                        constrs.insert(Constraint::Edge(vi, vj));
                        vi = vj;
                    }
                    return Iis{ constrs }
                }
            }
        }

        if let Some(((max_lb_node, _), (min_ub_node, _))) =
            self.find_scc_bound_infeas(scc.iter().copied()) {
            let p1 = self.shortest_path_scc(scc, true, max_lb_node, min_ub_node);
            let p2 = self.shortest_path_scc(scc, true, min_ub_node, max_lb_node);
            let mut constrs = FnvHashSet::default();

            let mut var_cycle = p1.into_iter()
                    .into_iter()
                    .chain(p2.into_iter().skip(1))
                .map(|n| self.var_from_node_id(n));

            let mut vi = var_cycle.next().unwrap();
            for vj in var_cycle.next() {
                constrs.insert(Constraint::Edge(vi, vj));
                vi = vj;
            }
            constrs.insert(Constraint::Lb(self.var_from_node_id(max_lb_node)));
            constrs.insert(Constraint::Ub(self.var_from_node_id(min_ub_node)));
            return Iis{ constrs }
        }
        unreachable!()
    }

    fn condense(&mut self, sccs: Vec<FnvHashSet<usize>>) {
        for scc in sccs {
            let (lb_node, lb) = scc.iter().map(|&n| (n, self.nodes[n].lb))
                .max_by_key(|pair| pair.1).unwrap();
            let (ub_node, ub) = scc.iter().map(|&n| (n, self.nodes[n].ub))
                .min_by_key(|pair| pair.1).unwrap();

            let scc_n = self.sccs.len();
            let scc_node = Node {
                x: lb,
                ub, lb,
                obj: 0,
                kind: NodeKind::Scc(scc_n),
                active_pred: None
            };
            self.nodes.push(scc_node);

            // edges into the SCC
            let mut biggest_in = FnvHashMap::default();
            let mut biggest_out = FnvHashMap::default();
            for e in self.edges_from.iter().flat_map(|edges| edges.iter()) {
                if scc.contains(&e.to) && scc.contains(&e.from) {
                    if !biggest_in.contains_key(&e.from) || biggest_in[&e.from].weight < e.weight {
                        biggest_in.insert(e.from, e)
                    }
                }

                if scc.contains(&e.from) && scc.contains(&e.to) {
                    if !biggest_out.contains_key(&e.from) || biggest_out[&e.from].weight < e.weight {
                        biggest_out.insert(e.from, e)
                    }
                }
            }

            self.edges_from.push(biggest_out.values()
                .map(|e| Edge{
                    from: scc_n,
                    to: e.to,
                    weight: e.weight,
                    kind: SccOut(e.from)
                })
                .collect());

            for e in biggest_in.values() {
                let e = Edge {
                    from: e.from,
                    to: scc_n,
                    weight: e.weight,
                    kind: SccIn(e.to)
                };
                self.edges_from[e.from].push(e);
            }

            for &n in &scc {
                self.nodes[n].kind = NodeKind::SccMember(scc_n);
            }

            self.sccs.push(SccInfo{
                nodes,
                scc_node: scc_n,
                lb_node,
                ub_node,
            })
        }
    }

    pub fn compute_iis(&mut self) -> Iis {
        match &self.state {
            ModelState::InfPath(violated_ubs) => self.compute_path_iis(violated_ubs),
            ModelState::CycleInfeasible { sccs, inf_idx } => {
                self.compute_cyclic_iis(sccs[inf_idx])
            }
            ModelState::Optimal => panic!("cannot compute IIS on feasible model"),
            ModelState::Init => panic!("need to call solve() first"),
            ModelState::Cycles => unreachable!()
        }
    }

    fn find_edge(&self, from: usize, to: usize) -> &Edge {
        todo!()
    }

    fn compute_path_iis(&self, violated_ubs: &[usize]) -> Iis {
        todo!()
    }

    fn compute_mrs(&mut self) -> Mrs {

    }

    fn compute_scc_active_edges(&mut self) {
        fn bfs(edges_from: &Vec<Vec<Edge>>, nodes: &mut [Node], scc: &FnvHashSet<usize>, root: usize) {
            let mut queue = std::collections::VecDeque::with_capacity(scc.len());
            queue.push_back(root);
            while let Some(n) = queue.pop_front() {
                for e in edges_from[n] {
                    // Adjacent edges that haven't been visited - node is visited if it has an
                    // active predecessor or is the root node.
                    if scc.contains(&e.to) && nodes[e.to].active_pred.is_none() && e.to != root  {
                        nodes[e.to].active_pred = Some(n);
                        queue.push_back(e.to);
                    }
                }
            }
        }

        // For every Scc, label the internal active-edge tree
        for scc in &self.sccs {
            let root = match self.nodes[scc.scc_node].active_pred {
                Some(n) => match self.find_edge(n, scc.scc_node).kind {
                    EdgeKind::SccIn(n) => n,
                    _ => unreachable!()
                },
                None => scc.lb_node,
            };
            bfs(&self.edges_from, &mut self.nodes, &scc.nodes, root);
        }

        // For all nodes that have an active edge from an SCC node, move the active edge to come
        // from the relevant SCC-component node, which will now have a tree
        for n in 0..self.nodes.len() {
            if let Some(p) = self.nodes[n].active_pred {
                if matches!(self.nodes[p].kind, NodeKind::Scc(_)) {
                    let p = match self.find_edge(p, n).kind {
                        EdgeKind::SccOut(p) => p,
                        _ => unreachable!()
                    };
                    self.nodes[n].active_pred = Some(p);
                }
            }
        }
    }

    fn compute_partitioned_mrs(&mut self) -> Vec<Mrs> {
        self.compute_scc_active_edges();
        let mut visited = vec![None; self.nodes.len()];
        let mut mrss = Vec::new();


        fn add_edges_to_root(graph: &Graph, n: usize, mrss: &mut Vec<Mrs>, visited: &mut [Option<usize>]) -> usize {
            if let Some(idx) = visited[n] {
                return idx
            }

            match nodes[n].active_pred {
                Some(p) => {
                    let idx = add_edges_to_root(nodes, p, tree_id, next_tree_id);
                    mrss[idx].constrs.insert(Constraint::Edge(
                        graph.var_from_node_id(p),
                        graph.var_from_node_id(n),
                    ));
                    idx
                },
                None => {
                    // this is the root
                    let mut constrs = FnvHashSet::default();
                    constrs.insert(Constraint::Lb(graph.var_from_node_id(n)));
                    let idx = mrss.len();
                    mrss.push(Mrs{ constrs }); // FIXME figure out the variables too
                    idx
                }
            }
        }

        for (mut n, node) in self.nodes.iter().enumerate() {
            if matches!(&node.kind, NodeKind::Scc(_)) { continue }
            // already visited
            if tree_id[n].is_some() { continue }
            add_edges_to_root(&self.nodes, n, &mut tree_id, &mut next_tree_id);
        }


        todo!()
    }
}
