use super::graph::*;
use crate::{Result, Error};

pub(crate) enum ModelAction {
  Solve,
  ComputeMrs,
  ComputeIis,
}

impl Graph {
  pub(crate) fn check_allowed_action(&self, action: ModelAction) -> Result<()> {
    use ModelState::*;
    use ModelAction::*;

    let result = match (action, &self.state) {
      (Solve, _)
      => Ok(()),

      (ComputeMrs, Unsolved)
      | (ComputeIis, Unsolved)
      => Err("solve the model first"),

      (ComputeMrs, InfCycle { .. })
      | (ComputeMrs, InfPath(..))
      => Err("model is infeasible"),

      (ComputeMrs, Optimal)
      | (ComputeMrs, Mrs)
      => Ok(()),

      (ComputeIis, InfPath(..))
      | (ComputeIis, InfCycle { .. })
      => Ok(()),

      (ComputeIis, Optimal)
      | (ComputeIis, Mrs)
      => Err("model is feasible"),
    };

    result.map_err(|msg| Error::InvalidAction(msg.to_string()))
  }
}