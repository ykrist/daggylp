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
use crate::test_utils::strategy::SccKind;

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


pub trait TestInput: Sized {
  type Meta;
  type TestFunction;

  fn save_meta(&self, path: impl AsRef<Path>);

  fn load_meta(g: GraphSpec, path: impl AsRef<Path>) -> Self;

  fn split(self) -> (GraphSpec, Self::Meta);

  fn run_test(test: &Self::TestFunction, input: (&mut Graph, Self::Meta)) -> TestCaseResult;
}


impl TestInput for GraphSpec {
  type Meta = ();
  type TestFunction = fn(&mut Graph) -> TestCaseResult;

  fn save_meta(&self, _: impl AsRef<Path>) {}

  fn load_meta(g: GraphSpec, _: impl AsRef<Path>) -> Self { g }

  fn split(self) -> (GraphSpec, Self::Meta) {
    (self, ())
  }

  fn run_test(test: &Self::TestFunction, input: (&mut Graph, ())) -> TestCaseResult {
    (test)(input.0)
  }
}

impl<M: Serialize + DeserializeOwned> TestInput for (GraphSpec, M) {
  type Meta = M;
  type TestFunction = fn(&mut Graph, M) -> TestCaseResult;

  fn save_meta(&self, path: impl AsRef<Path>) {
    std::fs::write(path, serde_json::to_string_pretty(&self.1).unwrap()).unwrap();
  }

  fn load_meta(g: GraphSpec, path: impl AsRef<Path>) -> Self {
    (g, serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap())
  }

  fn split(self) -> (GraphSpec, Self::Meta) {
    self
  }

  fn run_test(test: &Self::TestFunction, input: (&mut Graph, Self::Meta)) -> TestCaseResult {
    (test)(input.0, input.1)
  }
}


impl GraphTestRunner {
  pub fn new_with_layout_prog(id: &'static str, layout: LayoutAlgo) -> Self {
    // Ok to leak memory here - is only for testing and we only leak once per test.
    // let path: &'static str = Box::leak(format!("{}/tests/regressions/{}.txt", env!("CARGO_MANIFEST_DIR"), id).into_boxed_str());

    let mut config = ProptestConfig::with_cases(1000);
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Off));
    // TODO keep track of regressions ourselves, since we have a unified input format
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

  pub fn run<V, S>(&self, strategy: S, test: V::TestFunction)
    where
      V: TestInput,
      S: Strategy<Value=V>,
  {
    use TestError::*;
    // let mut runner = prop::test_runner::TestRunner::new(self.config.clone());
    let mut runner = prop::test_runner::TestRunner::new_with_rng(self.config.clone(), prop::test_runner::TestRng::deterministic_rng(self.config.rng_algorithm));
    let result = runner.run(&strategy, |input| {
      let (mut graph_spec, meta) = input.split();
      let mut g = graph_spec.build();
      V::run_test(&test, (&mut g, meta))
    });


    match result {
      Err(Abort(reason)) => {
        panic!("test aborted: {}", reason.message());
      }
      Err(Fail(reason, input)) => {
        std::fs::create_dir_all(self.output_svg.parent().unwrap()).unwrap();
        input.save_meta(&self.input_meta);
        let (input, meta) = input.split();
        input.save_to_file(&self.input_graph);
        input.save_svg_with_layout(&self.input_svg, self.layout);
        let mut graph = input.build();
        V::run_test(&test, (&mut graph, meta)).ok();
        graph.viz().save_svg_with_layout(&self.output_svg, self.layout);
        eprintln!("{}", reason.message());
        panic!("test case failure");
      }
      Ok(()) => {}
    }
  }


  pub fn debug<V>(&self, test: V::TestFunction)
    where
      V: TestInput,
  {
    let input = GraphSpec::load_from_file(&self.input_graph).pretty_unwrap();
    let (input, meta) = V::load_meta(input, &self.input_meta).split();
    let mut graph = input.build();

    match V::run_test(&test, (&mut graph, meta)) {
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
      ($m:path; $test:ident) => {
        graph_test_dbg!($m; $test crate::test_utils::GraphSpec);
      };

      ($m:path; $test:ident _) => {
        graph_test_dbg!($m; $test (crate::test_utils::GraphSpec, _));
      };

      ($m:path; $test:ident $t:ty) => {
        #[test]
        fn dbg() {
          GraphTestRunner::new(stringify!($test)).debug::<$t>(<$m>::$test);
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


