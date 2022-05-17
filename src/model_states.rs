use super::graph::*;
use crate::Error;
use fnv::FnvHashSet;

#[derive(Debug, Clone)]
pub(crate) enum ModelState {
  Unsolved,
  InfCycle {
    sccs: Vec<FnvHashSet<NodeIdx>>,
    first_inf_scc: usize,
  },
  InfPath(Vec<NodeIdx>),
  Optimal,
  Mrs,
  Dirty {
    rebuild_sccs: bool,
  },
}

pub(crate) enum ModelAction {
  Solve,
  ComputeOptimalityInfo,
  ComputeIis,
}

impl<E: EdgeLookup> Graph<E> {
  pub(crate) fn check_allowed_action(&self, action: ModelAction) -> Result<(), Error> {
    use ModelAction::*;
    use ModelState::*;

    let result = match (action, &self.state) {
      (Solve, _) => Ok(()),

      (ComputeOptimalityInfo, Unsolved)
      | (ComputeOptimalityInfo, Dirty { .. })
      | (ComputeIis, Unsolved)
      | (ComputeIis, Dirty { .. }) => Err("solve the model first"),

      (ComputeOptimalityInfo, InfCycle { .. }) | (ComputeOptimalityInfo, InfPath(..)) => {
        Err("model is infeasible")
      }

      (ComputeOptimalityInfo, Optimal) | (ComputeOptimalityInfo, Mrs) => Ok(()),

      (ComputeIis, InfPath(..)) | (ComputeIis, InfCycle { .. }) => Ok(()),

      (ComputeIis, Optimal) | (ComputeIis, Mrs) => Err("model is feasible"),
    };

    result.map_err(|msg| Error::InvalidAction(msg.to_string()))
  }
}
