mod generators;
mod test_cases;
#[cfg(feature = "tests-gurobi")]
mod lp;
pub mod strategy;

pub(crate) use generators::*;

pub use test_cases::generate_test_cases;
use crate::viz::{GraphViz, LayoutAlgo};
use crate::graph::Graph;

use proptest::test_runner::{TestCaseError, TestCaseResult, TestError, FileFailurePersistence, TestRunner as PropTestRunner};
use proptest::prelude::*;
use std::path::{PathBuf, Path};
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use crate::test_utils::strategy::SccKind;
use glob::MatchOptions;
use std::borrow::Cow;
use crate::test_utils::GraphTestMode::Regression;
use std::sync::mpsc;
pub use strategy::SharableStrategy;

use anyhow::Context;
use std::ffi::OsString;
use std::marker::PhantomData;
use std::fmt;

// pub(crate) fn test_input_dir() -> &'static Path {
//   Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/inputs"))
// }

pub(crate) fn test_failures_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/failures"))
}

pub(crate) fn test_regressions_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/regressions"))
}

fn is_unit_ty<M: 'static>() -> bool {
  use std::any::TypeId;
  TypeId::of::<M>() == TypeId::of::<()>()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestManifest {
  #[serde(skip)]
  path: PathBuf,
  /// Path (relative to the inputs.json) in which meta-information resides, if it exists
  meta: String,
  /// Path (relative to the inputs.json) in which the input graph resides
  input: String,
  /// Path (relative to the inputs.json) in which the input graph resides
  input_graph: String,
  /// Path (relative to the inputs.json) in which the input graph resides
  output_graph: String,
}

impl TestManifest {
  fn new(test_id: &str, mode: GraphTestMode) -> Self {
    let (mut path, input_prefix, output_prefix) = match &mode {
      GraphTestMode::Regression(i) => {
        let p = test_regressions_dir().join(test_id);
        std::fs::create_dir_all(&p).unwrap();
        let input_prefix = format!("{}", i);
        let output_prefix = input_prefix.clone();
        (p, input_prefix, output_prefix)
      }
      GraphTestMode::Proptest => {
        let input_prefix = test_id.to_string();
        let output_prefix = input_prefix.clone();
        (test_failures_dir().to_path_buf(), input_prefix, output_prefix)
      }
      GraphTestMode::Debug => {
        (test_failures_dir().to_path_buf(), test_id.to_string(), format!("{}.debug", test_id))
      }
    };
    path.push(format!("{}.manifest.json", &input_prefix));

    TestManifest {
      path,
      meta: format!("{}.meta.input.json", &input_prefix),
      input: format!("{}.input.txt", &input_prefix),
      input_graph: format!("{}.input.svg", &input_prefix),
      output_graph: format!("{}.output.svg", &output_prefix),
    }
  }

  fn from_file(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
    let path = path.into();
    let contents = std::fs::read_to_string(&path)
      .with_context(|| format!("read manifest {:?}", &path))?;
    let mut manifest : Self = serde_json::from_str(&contents)?;
    manifest.path = path;
    Ok(manifest)
  }

  fn to_file(&self) -> anyhow::Result<()> {
    std::fs::write(&self.path, serde_json::to_string_pretty(&self)?)
      .with_context(|| format!("write manifest {:?}", &self.path))
  }

  fn migrate_input_files(&self, dest: &Self) -> anyhow::Result<()> {
    use std::fs::copy;
    let mut src_path = self.path.to_path_buf();
    let mut dest_path = dest.path.to_path_buf();

    for (src, dst, missing_ok) in &[
      (&self.input, &dest.input, false),
      (&self.input_graph, &dest.input_graph, false),
      (&self.meta, &dest.meta, true),
    ] {
      src_path.set_file_name(src);
      dest_path.set_file_name(dst);
      if !*missing_ok || src_path.exists() {
        std::fs::rename(&src_path, &dest_path)
          .with_context(|| format!("copying {:?} to {:?}", &src_path, &dest_path))?;
      }
    }
    Ok(())
  }

  fn write_meta<V: TestInput>(&self, meta: &V::Meta) -> anyhow::Result<()> {
    if !is_unit_ty::<V::Meta>() {
      let contents = serde_json::to_string_pretty(meta)?;
      let path = self.path.with_file_name(&self.meta);
      std::fs::write(&path, contents).with_context(|| format!("manifest path {:?}", &path))?;
    }
    Ok(())
  }

  fn write_input(&self, g: &GraphSpec) -> anyhow::Result<()> {
    g.save_to_file(self.path.with_file_name(&self.input));
    Ok(())
  }

  fn write_input_graph(&self, g: &GraphSpec, layout: LayoutAlgo) -> anyhow::Result<()> {
    g.save_svg_with_layout(self.path.with_file_name(&self.input_graph), layout);
    Ok(())
  }

  fn write_output_graph(&self, g: &Graph, layout: LayoutAlgo) -> anyhow::Result<()> {
    g.viz().save_svg_with_layout(self.path.with_file_name(&self.output_graph), layout);
    Ok(())
  }

  fn load_input(&self) -> anyhow::Result<GraphSpec> {
    GraphSpec::load_from_file(self.path.with_file_name(&self.input))
  }

  fn load_meta<V: TestInput>(&self) -> anyhow::Result<V::Meta> {
    if let Some(meta) = V::constant_meta() {
      return Ok(meta);
    }
    let path = self.path.with_file_name(&self.meta);
    let contents = std::fs::read_to_string(&path)
      .with_context(|| format!("read meta from file {:?}", &path))?;
    Ok(serde_json::from_str(&contents)?)
  }
}

pub struct GraphTestRunner<'a> {
  id: &'a str,
  cpus: u32,
  skip_regressions: bool,
  deterministic: bool,
  layout: LayoutAlgo,
  config: ProptestConfig,
}

type GraphTestResult = std::result::Result<(), String>;


pub trait TestInput: Sized {
  type Meta: Serialize + DeserializeOwned + 'static;
  type TestFunction: Copy;

  fn constant_meta() -> Option<Self::Meta> { None }

  fn split(self) -> (GraphSpec, Self::Meta);

  fn run_test(test: Self::TestFunction, input: (&mut Graph, Self::Meta)) -> TestCaseResult;
}


impl TestInput for GraphSpec {
  type Meta = ();
  type TestFunction = fn(&mut Graph) -> TestCaseResult;

  fn constant_meta() -> Option<Self::Meta> { Some(()) }

  fn split(self) -> (GraphSpec, Self::Meta) {
    (self, ())
  }

  fn run_test(test: Self::TestFunction, input: (&mut Graph, ())) -> TestCaseResult {
    (test)(input.0)
  }
}

impl<M: Serialize + DeserializeOwned + 'static> TestInput for (GraphSpec, M) {
  type Meta = M;
  type TestFunction = fn(&mut Graph, M) -> TestCaseResult;

  fn split(self) -> (GraphSpec, Self::Meta) {
    self
  }

  fn run_test(test: Self::TestFunction, input: (&mut Graph, Self::Meta)) -> TestCaseResult {
    (test)(input.0, input.1)
  }
}

#[derive(Debug, Copy, Clone, Eq, PartialOrd, PartialEq)]
enum GraphTestMode {
  Proptest,
  Regression(u32),
  Debug,
}

impl<'a> GraphTestRunner<'a> {
  pub fn new_with_layout_prog(id: &'a str, layout: LayoutAlgo) -> Self {
    let mut config = ProptestConfig::with_cases(1000);
    // keep track of regressions ourselves, since we have a unified input format
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Off));
    config.result_cache = prop::test_runner::basic_result_cache;
    config.max_shrink_iters = 50_000;
    config.max_shrink_time = 30_000; // in millis
    GraphTestRunner {
      id,
      cpus: 1,
      skip_regressions: false,
      deterministic: false,
      layout,
      config,
    }
  }

  pub fn new(id: &'a str) -> Self {
    Self::new_with_layout_prog(id, LayoutAlgo::Dot)
  }

  fn manifest(&self, mode: GraphTestMode) -> TestManifest {
    TestManifest::new(self.id, mode)
  }

  fn handle_test_result<V: TestInput>(&self, result: Result<(), TestError<V>>, test: V::TestFunction) {
    use TestError::*;
    match result {
      Err(Abort(reason)) => {
        panic!("test aborted: {}", reason.message());
      }
      Err(Fail(reason, input)) => {
        let manifest = self.manifest(GraphTestMode::Proptest);
        std::fs::create_dir_all(manifest.path.parent().unwrap()).unwrap();
        let (input, meta) = input.split();
        manifest.to_file().unwrap();
        manifest.write_input(&input).unwrap();
        manifest.write_input_graph(&input, self.layout).unwrap();
        manifest.write_meta::<V>(&meta).unwrap();
        let mut graph = input.build();
        manifest.write_output_graph(&graph, self.layout).unwrap();
        V::run_test(test, (&mut graph, meta)).ok();
        manifest.write_output_graph(&graph, self.layout).unwrap();
        eprintln!("{}", reason.message());
        panic!("test case failure");
      }
      Ok(()) => {}
    }
  }

  pub fn run<V, S>(&self, strategy: S, test: V::TestFunction)
    where
      V: TestInput + Send + 'static, // TODO could remove Send + 'static and let caller determine if they want to run parallel
      V::TestFunction: Send + 'static,
      S: SharableStrategy<Value=V>,
  {
    if !self.skip_regressions {
      self.run_regressions::<V>(test);
    }

    if self.cpus > 1 {
      return self.run_parallel(strategy, test)
    }

    let mut runner = if self.deterministic {
      prop::test_runner::TestRunner::new_with_rng(
        self.config.clone(),
        prop::test_runner::TestRng::deterministic_rng(self.config.rng_algorithm)
      )
    } else {
      prop::test_runner::TestRunner::new(self.config.clone())
    };
    let result = runner.run(&strategy, |input| {
      let (mut graph_spec, meta) = input.split();
      let mut g = graph_spec.build();
      V::run_test(test, (&mut g, meta))
    });
    self.handle_test_result(result, test);
  }

  fn run_parallel<V, S>(&self, strategy: S, test: V::TestFunction)
    where
      V: TestInput + Send + 'static,
      V::TestFunction: Send + 'static,
      S: SharableStrategy<Value=V>,
  {
    use std::sync::mpsc;
    use TestError::*;

    if self.deterministic {
      eprintln!("warning: running with multiple CPUS doesn't make sense, running non-deterministically.");
    }

    let worker_threads : Vec<_> = (0..self.cpus).map(|_| {
      let config = self.config.clone();
      let strategy = strategy.clone();

      std::thread::spawn(move || {
        let mut runner = PropTestRunner::new(config);
        runner.run(&strategy, |input| {
          let (mut graph_spec, meta) = input.split();
          let mut g = graph_spec.build();
          V::run_test(test, (&mut g, meta))
        })
      })
    }).collect();

    let mut test_result = None;
    let mut thread_panic = None;
    for t in worker_threads {
      match t.join() {
        Ok(r) => {
          if test_result.is_none() {
            test_result = Some(r);
          }
        },
        Err(panic) => thread_panic = Some(panic),
      }
    }

    if let Some(panic) = thread_panic {
      std::panic::resume_unwind(panic);
    }

    self.handle_test_result(test_result.unwrap(), test);
  }

  pub fn skip_regressions(&mut self, skip: bool) {
    self.skip_regressions = skip;
  }

  pub fn deterministic(&mut self, det: bool) { self.deterministic = det; }

  pub fn cpus(&mut self, cpus: u32) {
    assert!(cpus > 0);
    self.cpus = cpus;
  }
  pub fn layout(&mut self, l: LayoutAlgo) {
    self.layout = l;
  }

  pub fn cases(&mut self, n: u32) {
    self.config.cases = n;
  }


  fn find_regressions<V: TestInput>(&self) -> anyhow::Result<Vec<TestManifest>> {
    let search_path = test_regressions_dir().join(self.id);
    let mut search_path = search_path.into_os_string().into_string().unwrap();
    search_path.push_str("/*.manifest.json");
    glob::glob(&search_path).unwrap()
      .map(|p| {
        let idx = true_stem(&p?).unwrap().parse()?;
        Ok(self.manifest(Regression(idx)))
      })
      .collect()
  }

  pub fn run_regressions<V: TestInput>(&self, test: V::TestFunction) {
    for manifest in self.find_regressions::<V>().unwrap() {
      println!("running regression: {:?}", &manifest.path);
      self.run_with_existing::<V>(manifest, test);
    }
    println!("finished running regressions")
  }

  pub fn debug<V>(&self, test: V::TestFunction)
    where
      V: TestInput,
  {
    self.run_with_existing::<V>(self.manifest(GraphTestMode::Debug), test);
  }

  fn run_with_existing<V>(&self, manifest: TestManifest, test: V::TestFunction)
    where
      V: TestInput,
  {
    let input = manifest.load_input().unwrap();
    let meta = manifest.load_meta::<V>().unwrap();
    let mut graph = input.build();

    match V::run_test(test, (&mut graph, meta)) {
      Err(TestCaseError::Reject(reason)) => {
        unreachable!()
      }
      Err(TestCaseError::Fail(reason)) => {
        manifest.write_input_graph(&input, self.layout);
        manifest.write_output_graph(&graph, self.layout);
        panic!("test case failure: {}", reason.message());
      }
      Ok(()) => {}
    }
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
      $(
        graph_tests!(@KW $runner : $($tail)*);
      )*
    };

    (@KW $runner:ident : cases = $n:expr $(, $($tail:tt)* )?) => {
      $runner.cases($n);
      $(
        graph_tests!(@KW $runner : $($tail)*);
      )*
    };

    (@KW $runner:ident : skip_regressions $(, $($tail:tt)* )?) => {
      $runner.skip_regressions();
      $(
        graph_tests!(@KW $runner : $($tail)*);
      )*
    };

    (@KW $runner:ident : deterministic $(, $($tail:tt)* )?) => {
      $runner.deterministic();
      $(
        graph_tests!(@KW $runner : $($tail)*);
      )*
    };

    (@KW $runner:ident : parallel = $n:literal $(, $($tail:tt)* )?) => {
      $runner.cpus($n);
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

#[macro_export]
macro_rules! prop_assert_matches {
    ($e:expr, $( $pattern:pat )|+ $( if $guard: expr )?) => {{
        let e = $e;
        let patt_s = stringify!($( $pattern )|+ $( if $guard )?);
        proptest::prop_assert!(
            matches!($e, $( $pattern )|+ $( if $guard )?),
            "assertion failed: `{:?}` does not match {}`",
            e, patt_s);
    }};
}

fn true_stem<'a, P: AsRef<Path>>(path: &'a P) -> Option<&'a str> {
  path.as_ref().file_name()?.to_str()?.split('.').next()
}

pub fn mark_failed_as_regression() -> anyhow::Result<()> {
  let mut pattern = test_failures_dir().to_str().unwrap().to_string();
  pattern.push_str("/*.manifest.json");

  for p in glob::glob(&pattern)? {
    let p = p?;
    let test_id = true_stem(&p).unwrap();
    let src = TestManifest::from_file(&p)?;
    let mut patt = test_regressions_dir().join(test_id).into_os_string().into_string().unwrap();
    patt.push_str("/*.manifest.json");
    let mut idx = 0;
    for file in glob::glob(&patt)? {
      let existing_index = true_stem(&file?)
          .map(|s| s.parse::<u32>().ok()).flatten();
      if existing_index == Some(idx) {
        idx += 1;
      } else {
        break;
      }
    }
    let dest = TestManifest::new(test_id, GraphTestMode::Regression(idx));
    src.migrate_input_files(&dest)?;
    std::fs::remove_file(src.path)?;
    dest.to_file()?;
  }

  Ok(())
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


