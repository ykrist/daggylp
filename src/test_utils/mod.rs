mod generators;
mod test_cases;
#[cfg(feature = "tests-gurobi")]
mod lp;
pub mod strategy;

pub(crate) use generators::*;

pub use test_cases::generate_test_cases;
use crate::viz::{GraphViz, LayoutAlgo, SccViz, VizConfig};
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
use crate::iis::Iis;

pub(crate) fn test_manual_input(name: &str) -> PathBuf {
  let input_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/inputs/"));
  let mut p = input_dir.join(name).into_os_string();
  p.push(".txt");
  PathBuf::from(p)
}

pub(crate) fn test_failures_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/failures"))
}

pub(crate) fn test_testcase_failures_dir() -> &'static Path {
  Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/testcase_failures"))
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

  fn write_meta<F: GraphTestFn>(&self, meta: &F::Meta) -> anyhow::Result<()> {
    if F::constant_meta().is_none() {
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

  fn write_input_graph(&self, g: &GraphSpec, config: VizConfig) -> anyhow::Result<()> {
    g.viz().configure(config).save_svg(self.path.with_file_name(&self.input_graph));
    Ok(())
  }

  fn write_output_graph(&self, g: &Graph, config: VizConfig) -> anyhow::Result<()> {
    g.viz().configure(config).save_svg(self.path.with_file_name(&self.output_graph));
    Ok(())
  }

  fn load_input(&self) -> anyhow::Result<GraphSpec> {
    GraphSpec::load_from_file(self.path.with_file_name(&self.input))
  }

  fn load_meta<F: GraphTestFn>(&self) -> anyhow::Result<F::Meta> {
    if let Some(meta) = F::constant_meta() {
      return Ok(meta);
    }
    let path = self.path.with_file_name(&self.meta);
    let contents = std::fs::read_to_string(&path)
      .with_context(|| format!("read meta from file {:?}", &path))?;
    Ok(serde_json::from_str(&contents)?)
  }
}


type GraphTestResult = std::result::Result<(), String>;

pub struct SimpleTest {}
pub struct TestWithMeta<M>(std::marker::PhantomData<M>);
pub struct TestWithData;
pub struct TestWithDataAndMeta<M>(std::marker::PhantomData<M>);

pub trait GraphTestFn: Sized {
  type Meta: Serialize + DeserializeOwned + 'static;
  type SValue: Send + 'static;
  type Function: Send + Copy + 'static;

  fn constant_meta() -> Option<Self::Meta> { None }

  fn split_off_input(sval: Self::SValue) -> (GraphSpec, Self::Meta);

  fn run_test(test: Self::Function, graph: &mut Graph, data: &GraphSpec, meta: Self::Meta) -> TestCaseResult;
}

impl GraphTestFn for SimpleTest {
  type Meta = ();
  type SValue = GraphSpec;
  type Function = fn(&mut Graph) -> TestCaseResult;

  fn constant_meta() -> Option<()> { Some(()) }

  fn split_off_input(sval: Self::SValue) -> (GraphSpec, ()) { (sval, ()) }

  fn run_test(test: Self::Function, graph: &mut Graph, _: &GraphSpec, _: Self::Meta) -> TestCaseResult {
    (test)(graph)
  }
}

impl<M: Serialize + DeserializeOwned + Send + 'static> GraphTestFn for TestWithMeta<M> {
  type Meta = M;
  type SValue = (GraphSpec, M);
  type Function = fn(&mut Graph, M) -> TestCaseResult;

  fn split_off_input(sval: Self::SValue) -> (GraphSpec, M) { sval }

  fn run_test(test: Self::Function, graph: &mut Graph, _: &GraphSpec, meta: Self::Meta) -> TestCaseResult {
    (test)(graph, meta)
  }
}

impl GraphTestFn for TestWithData {
  type Meta = ();
  type SValue = GraphSpec;
  type Function = fn(&mut Graph, &GraphSpec) -> TestCaseResult;

  fn constant_meta() -> Option<()> { Some(()) }

  fn split_off_input(sval: Self::SValue) -> (GraphSpec, ()) { (sval, ()) }

  fn run_test(test: Self::Function, graph: &mut Graph, data: &GraphSpec, _: Self::Meta) -> TestCaseResult {
    (test)(graph, data)
  }
}


impl<M: Serialize + DeserializeOwned + Send + 'static> GraphTestFn for TestWithDataAndMeta<M> {
  type Meta = M;
  type SValue = (GraphSpec, M);
  type Function = fn(&mut Graph, &GraphSpec, M) -> TestCaseResult;

  fn split_off_input(sval: Self::SValue) -> (GraphSpec, M) { sval }

  fn run_test(test: Self::Function, graph: &mut Graph, data: &GraphSpec, meta: Self::Meta) -> TestCaseResult {
    (test)(graph, data,  meta)
  }
}

#[derive(Debug, Copy, Clone, Eq, PartialOrd, PartialEq)]
enum GraphTestMode {
  Proptest,
  Regression(u32),
  Debug,
}

pub struct GraphProptestRunner<'a> {
  id: &'a str,
  cpus: u32,
  skip_regressions: bool,
  deterministic: bool,
  pub viz_config: VizConfig,
  config: ProptestConfig,
}

impl<'a> GraphProptestRunner<'a> {
  pub fn new_with_layout_prog(id: &'a str, layout: LayoutAlgo) -> Self {
    let mut config = ProptestConfig::with_cases(1000);
    // keep track of regressions ourselves, since we have a unified input format
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Off));
    config.result_cache = prop::test_runner::basic_result_cache;
    config.max_shrink_iters = 50_000;
    config.max_shrink_time = 30_000; // in millis
    GraphProptestRunner {
      id,
      cpus: 1,
      skip_regressions: false,
      deterministic: false,
      viz_config: Default::default(),
      config,
    }
  }

  pub fn new(id: &'a str) -> Self {
    Self::new_with_layout_prog(id, LayoutAlgo::Dot)
  }

  fn manifest(&self, mode: GraphTestMode) -> TestManifest {
    TestManifest::new(self.id, mode)
  }

  fn handle_test_result<F: GraphTestFn>(&self, result: Result<(), TestError<F::SValue>>, test: F::Function) {
    use TestError::*;
    match result {
      Err(Abort(reason)) => {
        panic!("test aborted: {}", reason.message());
      }
      Err(Fail(reason, input)) => {
        let manifest = self.manifest(GraphTestMode::Proptest);
        std::fs::create_dir_all(manifest.path.parent().unwrap()).unwrap();
        let (input, meta) = F::split_off_input(input);
        manifest.to_file().unwrap();
        manifest.write_input(&input).unwrap();
        manifest.write_input_graph(&input, self.viz_config).unwrap();
        manifest.write_meta::<F>(&meta).unwrap();
        let mut graph = input.build();
        manifest.write_output_graph(&graph, self.viz_config).unwrap();
        F::run_test(test, &mut graph, &input, meta).ok();
        manifest.write_output_graph(&graph, self.viz_config).unwrap();
        eprintln!("{}", reason.message());
        panic!("test case failure");
      }
      Ok(()) => {}
    }
  }

  pub fn run<F, S>(&self, strategy: S, test: F::Function)
    where
      F: GraphTestFn,
      S: SharableStrategy<Value=F::SValue>,
  {
    if !self.skip_regressions {
      self.run_regressions::<F>(test);
    }

    if self.cpus > 1 {
      return self.run_parallel::<F, S>(strategy, test)
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
      let (mut graph_spec, meta) = F::split_off_input(input);
      let mut g = graph_spec.build();
      F::run_test(test, &mut g, &graph_spec,  meta)
    });
    self.handle_test_result::<F>(result, test);
  }

  fn run_parallel<F, S>(&self, strategy: S, test: F::Function)
    where
      F: GraphTestFn,
      S: SharableStrategy<Value=F::SValue>,
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
          let (mut graph_spec, meta) = F::split_off_input(input);
          let mut g = graph_spec.build();
          F::run_test(test, &mut g, &graph_spec,  meta)
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

    self.handle_test_result::<F>(test_result.unwrap(), test);
  }

  pub fn skip_regressions(&mut self, skip: bool) {
    self.skip_regressions = skip;
  }

  pub fn deterministic(&mut self, det: bool) { self.deterministic = det; }

  pub fn cpus(&mut self, cpus: u32) {
    assert!(cpus > 0);
    self.cpus = cpus;
  }

  pub fn cases(&mut self, n: u32) {
    self.config.cases = n;
  }

  pub fn viz_config(&mut self) -> &mut VizConfig {
    &mut self.viz_config
  }

  fn find_regressions(&self) -> anyhow::Result<Vec<TestManifest>> {
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

  pub fn run_regressions<F: GraphTestFn>(&self, test: F::Function) {
    for manifest in self.find_regressions().unwrap() {
      println!("running regression: {:?}", &manifest.path);
      self.run_with_existing::<F>(manifest, test);
    }
    println!("finished running regressions")
  }

  pub fn debug<F>(&self, test: F::Function)
    where
      F: GraphTestFn,
  {
    self.run_with_existing::<F>(self.manifest(GraphTestMode::Debug), test);
  }

  fn run_with_existing<F>(&self, manifest: TestManifest, test: F::Function)
    where
      F: GraphTestFn
  {
    let input = manifest.load_input().unwrap();
    let meta = manifest.load_meta::<F>().unwrap();
    let mut graph = input.build();

    match F::run_test(test, &mut graph, &input, meta) {
      Err(TestCaseError::Reject(reason)) => {
        unreachable!()
      }
      Err(TestCaseError::Fail(reason)) => {
        manifest.write_input_graph(&input, self.viz_config);
        manifest.write_output_graph(&graph, self.viz_config);
        panic!("test case failure: {}", reason.message());
      }
      Ok(()) => {}
    }
  }
}

pub type GraphTestcaseResult = Result<(), ErrorWithGraphContext>;

pub trait TestcaseTest {
  type Function: Copy + Sized + 'static;
  type TestInput;
  type Meta;

  fn split(raw: Self::TestInput) -> (&'static str, Self::Meta);

  fn run(test: Self::Function, graph: &mut Graph, data: &GraphSpec, input: Self::Meta) -> GraphTestcaseResult;
}


impl TestcaseTest for SimpleTest {
  type Function = fn(&mut Graph) -> GraphTestcaseResult;
  type TestInput = &'static str;
  type Meta = ();
  fn split(raw: Self::TestInput) -> (&'static str, Self::Meta) { (raw, ()) }

  fn run(test: Self::Function, graph: &mut Graph, _: &GraphSpec, _: Self::Meta) -> GraphTestcaseResult {
    (test)(graph)
  }
}


impl<M: 'static> TestcaseTest for TestWithMeta<M> {
  type Function = fn(&mut Graph, M) -> GraphTestcaseResult;
  type TestInput = (&'static str, M);
  type Meta = M;
  fn split(raw: Self::TestInput) -> (&'static str, Self::Meta) { raw }

  fn run(test: Self::Function, graph: &mut Graph, _: &GraphSpec, meta: Self::Meta) -> GraphTestcaseResult {
    (test)(graph, meta)
  }
}



#[derive(Debug)]
pub struct ErrorWithGraphContext {
  error: anyhow::Error,
  iis: Option<Iis>
}

pub trait GraphContext: Sized {
  type Output;

  fn iis(self, iis: Iis) -> Self::Output;
}

impl GraphContext for ErrorWithGraphContext {
  type Output = Self;

  fn iis(mut self, iis: Iis) -> Self::Output {
    self.iis = Some(iis);
    self
  }
}

impl GraphContext for anyhow::Error {
  type Output = ErrorWithGraphContext;

  fn iis(mut self, iis: Iis) -> Self::Output {
    ErrorWithGraphContext {
      error: self,
      iis: Some(iis),
    }
  }
}

impl<T, E> GraphContext for Result<T, E>
  where
    T: Sized,
    E: GraphContext<Output=ErrorWithGraphContext>
{
  type Output = Result<T, ErrorWithGraphContext>;

  fn iis(mut self, iis: Iis) -> Self::Output {
    self.map_err(|e| e.iis(iis))
  }
}


impl<E: Into<anyhow::Error>> From<E> for ErrorWithGraphContext {
  fn from(error: E) -> Self {
    ErrorWithGraphContext{ error: error.into(), iis: None }
  }
}

pub struct GraphTestcaseRunner<'a, T: TestcaseTest> {
  id: &'a str,
  pub viz_config: VizConfig,
  test: T::Function
}


impl<'a, T: TestcaseTest> GraphTestcaseRunner<'a, T> {
  pub fn new_with_layout_prog(id: &'a str, test: T::Function, layout: LayoutAlgo) -> Self {
    GraphTestcaseRunner {
      id,
      viz_config: Default::default(),
      test,
    }
  }

  pub fn new(id: &'a str, test: T::Function) -> Self {
    Self::new_with_layout_prog(id, test, LayoutAlgo::Dot)
  }

  pub fn run(&self, input: T::TestInput)
  {
    let (input_basename, meta) = T::split(input);
    let data =  GraphSpec::load_from_file(test_manual_input(input_basename)).unwrap();
    let mut graph = data.build();
    let result = T::run(self.test, &mut graph, &data, meta);
    if let Err(ErrorWithGraphContext{ error, iis }) = result {
      let mut path = test_testcase_failures_dir().join(format!("{}.input.svg", self.id));
      data.viz().configure(self.viz_config).save_svg(&path);
      path.set_file_name(format!("{}.output.svg", self.id));
      let mut gv = graph.viz().configure(self.viz_config);
      if let Some(iis) = iis.as_ref() {
        gv = gv.iis(iis);
      }
      gv.save_svg(path);
      eprintln!("test case failure (input {}):\n{:?}", input_basename, error);
      panic!("test failed");
    }
  }

  pub fn viz_config(&mut self) -> &mut VizConfig {
    &mut self.viz_config
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
macro_rules! graph_testcase_assert {
  ($cond:expr $(,)?) => {{
    graph_testcase_assert!($cond, "assertion failed: {}: {}", file!(), line!())
  }};

  ($cond:expr, $($arg:tt)+) => {{
    let cond = $cond;
    if !cond {
      return Err($crate::test_utils::ErrorWithGraphContext::from(anyhow::anyhow!($($arg)*)))
    }
  }};
}

#[macro_export]
macro_rules! graph_testcase_assert_eq {
  ($a:expr, $b:expr $(,)?) => {{
    graph_testcase_assert!($a == $b, "values not equal: {:?} != {:?}", &$a, &$b)
  }};
}

#[macro_export]
macro_rules! graph_testcase_assert_ne {
  ($a:expr, $b:expr $(,)?) => {{
    graph_testcase_assert!($a != $b, "values are equal: {:?} == {:?}", &$a, &$b)
  }};
}

#[macro_export]
macro_rules! graph_testcases {
  ($m:path; $(
    $test_function:ident$(($($test_type_spec:tt)*))? $([$($input:tt)*])+
  )+) => {
    $(
      #[test]
      fn $test_function() {
        let mut runner = GraphTestcaseRunner::<'_, graph_proptests!(@TT_SPEC $($($test_type_spec)*)*)>::new(stringify!($test_function), <$m>::$test_function);

        $(
          graph_testcases!(@INPUT runner $($input)*);
        )*

      }
    )*
  };

  (@INPUT $runner:ident $input:literal $(; $($kw:tt)*)?) => {
    $(graph_testcases!(@KW $runner : $($kw)*))*
    $runner.run($input);
  };

  (@INPUT $runner:ident $input:literal, $meta:expr $(; $($kw:tt)*)?) => {
    stringify!($(graph_testcases!(@KW $runner : $($kw)*))*);
    $(graph_testcases!(@KW $runner : $($kw)*))*;
    $runner.run(($input, $meta));
  };

  (@KW $runner:ident : ) => {};

  (@KW $runner:ident : layout = $layout:expr $( , $($tail:tt)* )? ) => {
    $runner.viz_config().layout($layout);
    $(
      graph_testcases!(@KW $runner : $($tail)*);
    )*
  };

  (@KW $runner:ident : sccs = $scc:expr $( , $($tail:tt)* )? ) => {
    $runner.viz_config().sccs($scc);
    $(
      graph_testcases!(@KW $runner : $($tail)*);
    )*
  };
}


#[macro_export]
macro_rules! graph_proptests {
  ($m:path; $($strategy:expr => $test:ident$(($($test_type_spec:tt)*))? $([$($kwargs:tt)+])? ; )+) => {
    $(
      #[test]
      fn $test() {
        let mut runner = GraphProptestRunner::new(stringify!($test));
        $(
          graph_proptests!(@KW runner : $($kwargs)* );
        )*;
        runner.run::<graph_proptests!(@TT_SPEC $($($test_type_spec)*)*), _>($strategy, <$m>::$test)
      }
    )*
  };

  (@TT_SPEC ) => {
    $crate::test_utils::SimpleTest
  };


  (@TT_SPEC data) => {
    $crate::test_utils::TestWithData
  };

  (@TT_SPEC meta) => {
    $crate::test_utils::TestWithMeta<_>
  };

  (@TT_SPEC data , meta) => {
    $crate::test_utils::TestWithDataAndMeta<_>
  };


  (@KW $runner:ident : ) => {};

  (@KW $runner:ident : layout = $layout:expr $( , $($tail:tt)* )? ) => {
    $runner.viz_config.layout($layout);
    $(
      graph_proptests!(@KW $runner : $($tail)*);
    )*
  };

  (@KW $runner:ident : sccs = $scc:expr $( , $($tail:tt)* )? ) => {
    $runner.viz_config().sccs($scc);
    $(
      graph_proptests!(@KW $runner : $($tail)*);
    )*
  };

  (@KW $runner:ident : cases = $n:expr $(, $($tail:tt)* )?) => {
    $runner.cases($n);
    $(
      graph_proptests!(@KW $runner : $($tail)*);
    )*
  };

  (@KW $runner:ident : skip_regressions $(, $($tail:tt)* )?) => {
    $runner.skip_regressions();
    $(
      graph_proptests!(@KW $runner : $($tail)*);
    )*
  };

  (@KW $runner:ident : deterministic $(, $($tail:tt)* )?) => {
    $runner.deterministic();
    $(
      graph_proptests!(@KW $runner : $($tail)*);
    )*
  };

  (@KW $runner:ident : parallel = $n:literal $(, $($tail:tt)* )?) => {
    $runner.cpus($n);
    $(
      graph_proptests!(@KW $runner : $($tail)*);
    )*
  };
}

#[macro_export]
macro_rules! graph_proptest_dbg {
      ($m:path; $test:ident$(($($test_type_spec:tt)*))?) => {
        #[test]
        fn dbg() {
          GraphProptestRunner::new(stringify!($test))
          .debug::<graph_proptests!(@TT_SPEC $($($test_type_spec)*)*)>(<$m>::$test);
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


