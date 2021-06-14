mod generators;
mod test_cases;

pub(crate) use generators::*;
pub use test_cases::generate_test_cases;


use std::path::{PathBuf};

pub(crate) fn test_input(name: &str) -> PathBuf {
  let mut path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/inputs"));
  path.push(name);
  path.set_extension("txt");
  path
}


pub(crate) fn test_output(filename: &str) -> PathBuf {
  let mut path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/outputs"));
  path.push(filename);
  path
}
