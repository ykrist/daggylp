use super::graph::*;
use fnv::FnvHashMap;
use crate::{map_with_capacity};


fn update_min<T: Ord, U>(dst: &mut Option<(T, U)>, src: (T, U)) -> bool {
  if let Some(v) = dst {
    if &v.0 <= &src.0 {
      return false
    }
  }
  *dst = Some(src);
  true
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct DpState {
  node: usize,
  deadline: Weight,
}



#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
enum DpValueAction {
  /// Before the recursive call to compute value of `state[t+1]`, `state[t]` gets a
  /// cache entry with a `CycleSentinel` entry, so we can detect cycles.  Once the value `state[t+1]`
  /// has been computed, `state[t]` becomes either `Pruned` or `Step`.
  CycleSentinel,
  /// This state cannot possible lead to an optimal path
  Pruned,
  /// A terminal (base-case) state with associated value
  Terminate(u32),
  /// The value of this state, and the next state in the optimal trajectory.
  Step(u32, DpState),
}

use DpValueAction::*;

//      val            i     edge weight (i, j)
///                          j    deadline[j]
type DpCache = FnvHashMap<DpState, DpValueAction>;

struct DpShortestPath<'a> {
  dp_cache: &'a DpCache,
  path_len: u32,
  state: Option<DpState>,
}

impl<'a> DpShortestPath<'a> {
  fn new(cache: &'a DpCache, initial_state: DpState, len: u32) -> Self {
    DpShortestPath {
      dp_cache: cache,
      path_len: len,
      state: Some(initial_state),
    }
  }
}

impl Iterator for DpShortestPath<'_> {
  type Item = usize;

  fn next(&mut self) -> Option<Self::Item> {
    self.state.take()
      .map(|state| {
        let (path_len, next_state) = match self.dp_cache[&state] {
          Terminate(l) => (l, None),
          Step(l, s) => (l, Some(s)),
          Pruned | CycleSentinel => unreachable!()
        };
        self.state = next_state;
        self.path_len = path_len;
        state.node
      })
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let n = self.path_len as usize + 1;
    (n, Some(n))
  }
}

impl Graph {
  pub(crate) fn compute_path_iis(&self, violated_ubs: &[usize]) -> Iis {
    let mut cache  = map_with_capacity(64);

    let mut shortest_path_initial_state = None;

    for &node in violated_ubs {
      let initial_state = DpState{ node, deadline: self.nodes[node].ub };
      match self.dp(&mut cache, initial_state) {
        Pruned => continue,
        Terminate(l) => {
          debug_assert_eq!(l, 0);
          update_min(&mut shortest_path_initial_state, (0, initial_state));
          break
        }
        Step(l, _) => {
          update_min(&mut shortest_path_initial_state, (l, initial_state));
        },
        CycleSentinel => unreachable!(),
      }
    }

    let (path_len, initial_state) = shortest_path_initial_state.unwrap();
    let shortest_path = DpShortestPath::new(&cache, initial_state, path_len);
    Iis::from_path(shortest_path.map(|n| self.var_from_node_id(n)))
  }

  fn dp(&self, cache: &mut DpCache, state: DpState) -> DpValueAction {
    if let Some(val_action) = cache.get(&state) {
      return *val_action
    }

    let n = &self.nodes[state.node];
    let return_val = if state.deadline < n.lb {
      Terminate(0)
    } else if state.deadline > n.ub {
      Pruned
    } else {
      cache.insert(state, DpValueAction::CycleSentinel);

      let mut best_val_action = None;

      for e in &self.edges_to[state.node] {
        let new_state = DpState{ node: e.from, deadline: state.deadline - e.weight };
        let mut val = match self.dp(cache, new_state) {
          Pruned => continue,
          CycleSentinel => continue,
          Terminate(val) => val,
          Step(val, _) => val,
        };

        val += 1; // add cost from the edge we just traverse
        update_min(&mut best_val_action, (val, new_state));
      }

      match best_val_action {
        None => DpValueAction::Pruned,
        Some((val, new_state)) => DpValueAction::Step(val, new_state)
      }
    };

    cache.insert(state, return_val);
    return_val
  }
}
