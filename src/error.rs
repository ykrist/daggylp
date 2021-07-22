use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone)]
pub enum Error {
  InvalidAction(String),
  GraphMismatch,
}

impl Display for Error {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    use Error::*;
    match self {
      InvalidAction(msg) => f.write_str(msg),
      GraphMismatch => f.write_str("Object came from another model"),
    }
  }
}

impl std::error::Error for Error {}

