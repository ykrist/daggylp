use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Copy, Clone)]
pub enum Error {
  CyclicIis,
  FeasibleModel,
  InfeasibleModel,
}

impl Display for Error {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    use Error::*;
    match self {
      CyclicIis => f.write_str("model contains an infeasible cycle"),
      FeasibleModel => f.write_str("model is feasible"),
      InfeasibleModel => f.write_str("model is infeasible"),
    }
  }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
