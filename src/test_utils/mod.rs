mod generators;
mod test_cases;
#[cfg(feature = "tests-gurobi")]
mod lp;
pub mod strategy;

pub(crate) use generators::*;

pub use test_cases::generate_test_cases;
use crate::viz::{GraphViz, LayoutAlgo};
use crate::graph::Graph;

use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, FileFailurePersistence};
use proptest::prelude::*;
use std::path::{PathBuf, Path};
use serde::{Serialize, Deserialize, de::DeserializeOwned};

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
  input_graph: PathBuf,
  input_meta: PathBuf,
  input_svg: PathBuf,
  output_svg: PathBuf,
}

type GraphTestResult = std::result::Result<(), String>;

#[derive(Deserialize, Serialize, Copy, Clone, Debug)]
struct NulInputMeta;


impl GraphTestRunner {
  pub fn new_with_layout_prog(id: &'static str, layout: LayoutAlgo) -> Self {
    // Ok to leak memory here - is only for testing and we only leak once per test.
    // let path: &'static str = Box::leak(format!("{}/tests/regressions/{}.txt", env!("CARGO_MANIFEST_DIR"), id).into_boxed_str());

    let mut config = ProptestConfig::with_cases(1000);
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Off));
    config.result_cache = prop::test_runner::basic_result_cache;

    let mut path = test_output_dir().join("failures");
    path.push(id);
    let output_svg = path.with_extension("svg");
    let mut path = path.into_os_string();
    path.push("-input");
    let mut path = PathBuf::from(path);
    let input_meta = path.with_extension("json");
    let input_graph = path.with_extension("txt");
    path.set_extension("svg");
    let input_svg = path;

    GraphTestRunner {
      id,
      layout,
      config,
      output_svg,
      input_meta,
      input_svg,
      input_graph
    }
  }

  pub fn new(id: &'static str) -> Self {
    Self::new_with_layout_prog(id, LayoutAlgo::Dot)
  }

  pub fn run(&self, inputs: impl Strategy<Value=(GraphSpec)>, test: impl Fn(&mut Graph) -> TestCaseResult) {
    self.run_with_answer(inputs.prop_map(|g| (g, NulInputMeta)), move |g, _| test(g))
  }

  pub fn run_with_answer<T: Serialize>(&self, inputs: impl Strategy<Value=(GraphSpec, T)>, test: impl Fn(&mut Graph, T) -> TestCaseResult) {
    use TestError::*;
    let mut runner = prop::test_runner::TestRunner::new(self.config.clone());
    let result = runner.run(&inputs, |(g, ans)| {
      let mut g = g.build();
      test(&mut g, ans)
    });


    match result {
      Err(Abort(reason)) => {
        panic!("test aborted: {}", reason.message());
      }
      Err(Fail(reason, (input, ans))) => {
        std::fs::create_dir_all(self.output_svg.parent().unwrap()).unwrap();
        input.save_to_file(&self.input_graph);
        input.save_svg_with_layout(&self.input_svg, self.layout);
        std::fs::write(&self.input_meta, serde_json::to_string(&ans).unwrap());
        let mut graph = input.build();
        test(&mut graph, ans).ok();
        graph.viz().save_svg_with_layout(&self.output_svg, self.layout);

        eprintln!("{}", reason.message());
        panic!("test case failure");
      }
      Ok(()) => {}
    }
  }

  pub fn debug(&self, test: impl Fn(&mut Graph) -> TestCaseResult) {
    self.debug_with_answer(|g, _ : NulInputMeta| test(g))
  }

  pub fn debug_with_answer<'a, T: DeserializeOwned>(&self, test: impl Fn(&mut Graph, T) -> TestCaseResult) {
    let input = GraphSpec::load_from_file(&self.input_svg).pretty_unwrap();
    let ans_json = std::fs::read_to_string(&self.input_meta).unwrap();
    let ans = serde_json::from_str(&ans_json).unwrap();

    let mut graph = input.build();

    match test(&mut graph, ans) {
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
macro_rules! test_case_fail {
    ($($args:tt)*) => {
      Err(proptest::test_runner::TestCaseError::fail(format!($($args)*)))
    };
}

#[macro_export]
macro_rules! test_case_bail {
    ($($args:tt)*) => {
      return test_case_fail!($($args)*)
    };
}

#[macro_export]
macro_rules! graph_tests {
      ($m:path; $($strategy:expr => $test:ident $([$($kwargs:tt)+])? ; )+) => {
        $(
          #[test]
          fn $test() {
            let mut runner = GraphTestRunner::new(stringify!($test));
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
      ($m:path; $test:ident (ans)) => {
        #[test]
        fn dbg() {
          GraphTestRunner::debug(stringify!($test), <$m>::$test)
        }
      };

      ($m:path; $test:ident) => {
        #[test]
        fn dbg() {
          GraphTestRunner::debug(stringify!($test), <$m>::$test)
        }
      };
  }

pub trait PrettyUnwrap {
  type Value;
  fn pretty_unwrap(self) -> Self::Value;
}

impl<T> PrettyUnwrap for anyhow::Result<T> {
  type Value = T;
  fn pretty_unwrap(self) -> Self::Value {
    match self {
      Ok(v) => v,
      Err(e) => {
        eprintln!("ERROR:\n{:?}", e);
        panic!("unrecoverable error: see above");
      }
    }
  }
}


