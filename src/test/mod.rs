use std::path::{PathBuf, Path};
use std::borrow::Cow;
use std::sync::mpsc;
use std::ffi::OsString;
use std::marker::PhantomData;
use std::fmt;

#[cfg(test)]
mod data;
#[cfg(test)]
pub use data::*;

#[cfg(feature = "grb")]
mod lp;

pub mod strategy;
pub use strategy::SharableStrategy;

pub use daggylp_macros::{graph_test, graph_proptest};

mod framework;
pub use framework::*;

pub fn test_manual_input(name: &str) -> PathBuf {
  let input_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/inputs/"));
  let mut p = input_dir.join(name).into_os_string();
  p.push(".txt");
  PathBuf::from(p)
}

pub fn test_input_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/inputs/"))
}

pub fn test_failures_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/failures"))
}

pub fn test_testcase_failures_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/testcase_failures"))
}

pub fn test_regressions_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/regressions"))
}

fn true_stem<'a, P: AsRef<Path>>(path: &'a P) -> Option<&'a str> {
  path.as_ref().file_name()?.to_str()?.split('.').next()
}
