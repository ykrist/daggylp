mod generators;
mod test_cases;
#[cfg(feature = "tests-gurobi")]
mod lp;
pub mod strategy;

pub(crate) use generators::*;

pub use test_cases::generate_test_cases;
use crate::viz::{GraphViz, LayoutAlgo};
use crate::graph::Graph;

use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestError};

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


pub fn proptest_config_cases(cases: u32) -> ProptestConfig {
  ProptestConfig::with_cases(cases)
}

pub fn proptest_config() -> ProptestConfig {
  proptest_config_cases(100)
}

pub struct GraphTestRunner {
  id: &'static str,
  layout: LayoutAlgo,
  config: ProptestConfig,
}

type GraphTestResult = std::result::Result<(), String>;

impl GraphTestRunner {
  pub fn new_with_layout_prog(id: &'static str, layout: LayoutAlgo) -> Self {
    GraphTestRunner {
      id,
      layout,
      config: ProptestConfig::with_cases(1000),
    }
  }

  pub fn new(id: &'static str) -> Self {
    Self::new_with_layout_prog(id, LayoutAlgo::Dot)
  }

  pub fn run(&self, inputs: impl Strategy<Value=GraphSpec>, test: fn(&mut Graph) -> TestCaseResult) {
    use TestError::*;
    let mut runner = prop::test_runner::TestRunner::new(self.config.clone());
    let result = runner.run(&inputs, |g| {
      let mut g = g.build();
      test(&mut g)
    });


    if let Err(Fail(reason, input)) = &result {
      let mut dir = test_output_dir().join("failures");
      std::fs::create_dir_all(&dir).unwrap();
      dir.push(self.id);
      input.save_to_file(dir.with_extension("txt"));
      input.save_svg_with_layout(dir.with_extension("input.svg"), self.layout);
      let mut graph = input.build();
      test(&mut graph).ok();
      graph.viz().save_svg_with_layout(dir.with_extension("svg"), self.layout);
      eprintln!("{}", reason.message());
    }

    match result {
      Err(Abort(reason)) => {
        panic!("test aborted: {}", reason.message());
      }
      Err(Fail(reason, _)) => {
        panic!("test case failure");
      }
      Ok(()) => {}
    }
  }


  pub fn debug(id: &str, test: fn(&mut Graph) -> TestCaseResult) {
    let mut path = test_output_dir().join("failures");
    path.push(id);
    path.set_extension("txt");
    let input = GraphSpec::load_from_file(path);
    let mut graph = input.build();

    match test(&mut graph) {
      Err(TestCaseError::Reject(reason)) => {
        unreachable!()
      }
      Err(TestCaseError::Fail(reason)) => {
        panic!("test case failure: {}", reason.message());
      }
      Ok(()) => {}
    }
  }

  pub fn layout(&mut self, l: LayoutAlgo) {
    self.layout = l;
  }

  pub fn cases(&mut self, n: u32) {
    self.config.cases = n;
  }
}


#[macro_export]
macro_rules! graph_tests {
      ($m:path; $($strategy:expr => $test:ident $([$($kwargs:tt)+])? ; )+) => {
        $(
          #[test]
          fn $test() {
            let mut runner = GraphTestRunner::new_with_layout_prog(stringify!($test), LayoutAlgo::Dot);
            $(
              graph_tests!(@KW runner : $($kwargs)* );
            )*;
            runner.run($strategy, <$m>::$test)
          }
        )*
      };

    (@KW $runner:ident : ) => {};

    (@KW $runner:ident : layout = $layout:expr $( , $($tail:tt)* )? ) => {
      $runner.layout($layout);
      // $(stringify!($($tail)*);)*;
      $(
        graph_tests!(@KW $runner : $($tail)*);
      )*
    };

    (@KW $runner:ident : cases = $n:expr $(, $($tail:tt)* )?) => {
      $runner.cases($n);
      // $(stringify!($($tail)*);)*;
      $(
        graph_tests!(@KW $runner : $($tail)*);
      )*
    };
  }

#[macro_export]
macro_rules! graph_test_dbg {
      ($m:path, $test:ident) => {
        #[test]
        fn dbg() {
          GraphTestRunner::debug(stringify!($ident), <$m>::$test)
        }
      };
  }