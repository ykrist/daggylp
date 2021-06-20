mod generators;
mod test_cases;
mod lp;
mod strategy;

pub(crate) use generators::*;
pub use test_cases::generate_test_cases;


use std::path::{PathBuf, Path};

pub(crate) fn test_input_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/inputs"))
}

pub(crate) fn test_output_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/outputs"))
}

pub(crate) fn test_input(name: &str) -> PathBuf {
  let mut path = test_input_dir().to_path_buf();
  path.push(name);
  path.set_extension("txt");
  path
}


pub(crate) fn test_output(filename: &str) -> PathBuf {
  test_output_dir().join(filename)
}
