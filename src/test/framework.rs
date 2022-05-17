use std::borrow::Cow;
use std::ffi::OsString;
use std::fmt;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::Context;
use proptest::prelude::*;
use proptest::test_runner::{FileFailurePersistence, TestError, TestRunner as PropTestRunner};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::*;
use crate::graph::Graph;
use crate::iis::Iis;
use crate::test::strategy::SccKind;
use crate::viz::{GraphViz, SccViz, VizConfig};

pub use daggylp_macros::{graph_proptest, graph_test};
pub use data::*;
pub use strategy::SharableStrategy;

pub fn get_test_id(module_path: &str, ident: &str) -> String {
  let mut s = String::with_capacity(module_path.len());
  let mut path = module_path.split("::").skip(1).peekable();

  while let Some(part) = path.next() {
    if path.peek().is_some() {
      s.push_str(part);
      s.push('_');
    }
  }
  s.push_str(ident);
  s
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
        (
          test_failures_dir().to_path_buf(),
          input_prefix,
          output_prefix,
        )
      }
      GraphTestMode::Debug => (
        test_failures_dir().to_path_buf(),
        test_id.to_string(),
        format!("{}.debug", test_id),
      ),
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
    let contents =
      std::fs::read_to_string(&path).with_context(|| format!("read manifest {:?}", &path))?;
    let mut manifest: Self = serde_json::from_str(&contents)?;
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

  fn write_meta<F: GraphProptest>(&self, meta: &F::Meta) -> anyhow::Result<()> {
    if F::constant_meta().is_none() {
      let contents = serde_json::to_string_pretty(meta)?;
      let path = self.path.with_file_name(&self.meta);
      std::fs::write(&path, contents).with_context(|| format!("manifest path {:?}", &path))?;
    }
    Ok(())
  }

  fn write_input(&self, g: &GraphData) -> anyhow::Result<()> {
    g.save_to_file(self.path.with_file_name(&self.input));
    Ok(())
  }

  fn write_input_graph(&self, g: &GraphData, config: VizConfig) -> anyhow::Result<()> {
    g.viz()
      .configure(config)
      .render(self.path.with_file_name(&self.input_graph));
    Ok(())
  }

  fn write_output_graph(&self, g: &Graph, config: VizConfig) -> anyhow::Result<()> {
    g.viz()
      .configure(config)
      .render(self.path.with_file_name(&self.output_graph));
    Ok(())
  }

  fn load_input(&self) -> anyhow::Result<GraphData> {
    GraphData::load_from_file(self.path.with_file_name(&self.input))
  }

  fn load_meta<F: GraphProptest>(&self) -> anyhow::Result<F::Meta> {
    if let Some(meta) = F::constant_meta() {
      return Ok(meta);
    }
    let path = self.path.with_file_name(&self.meta);
    let contents =
      std::fs::read_to_string(&path).with_context(|| format!("read meta from file {:?}", &path))?;
    Ok(serde_json::from_str(&contents)?)
  }
}

pub struct SimpleTest<R>(pub fn(&mut Graph) -> R);

pub struct TestWithMeta<R, M>(pub fn(&mut Graph, M) -> R);

pub struct TestWithData<R>(pub fn(&mut Graph, &GraphData) -> R);

pub struct ComplexTest<R, M>(pub fn(&mut Graph, &GraphData, M) -> R);

macro_rules! impl_copy_clone {
    ([$($param:tt)*] $($ty:tt)+) => {
      impl <$($param)*> Clone for $($ty)* {
        fn clone(&self) -> Self { Self(self.0) }
      }
      impl<$($param)*> Copy for $($ty)* {}
    };
}

impl_copy_clone!([R] SimpleTest<R>);
impl_copy_clone!([R, M] TestWithMeta<R, M>);
impl_copy_clone!([R] TestWithData<R>);
impl_copy_clone!([R, M] ComplexTest<R, M>);

pub type GraphProptestResult = prop::test_runner::TestCaseResult;

pub trait GraphProptest: Sized + Copy + Send + Sync + 'static {
  type Meta: Serialize + DeserializeOwned + 'static;
  type SValue: Send + 'static;

  fn constant_meta() -> Option<Self::Meta> {
    None
  }

  fn split_input(sval: Self::SValue) -> (GraphData, Self::Meta);

  fn run(&self, graph: &mut Graph, data: &GraphData, meta: Self::Meta) -> GraphProptestResult;
}

impl GraphProptest for SimpleTest<GraphProptestResult> {
  type Meta = ();
  type SValue = GraphData;

  fn constant_meta() -> Option<()> {
    Some(())
  }

  fn split_input(sval: Self::SValue) -> (GraphData, Self::Meta) {
    (sval, ())
  }

  fn run(&self, graph: &mut Graph, _: &GraphData, _: Self::Meta) -> GraphProptestResult {
    self.0(graph)
  }
}

impl<M: Serialize + DeserializeOwned + Send + 'static> GraphProptest
  for TestWithMeta<GraphProptestResult, M>
{
  type Meta = M;
  type SValue = (GraphData, M);

  fn split_input(sval: Self::SValue) -> (GraphData, Self::Meta) {
    sval
  }

  fn run(&self, graph: &mut Graph, _: &GraphData, meta: Self::Meta) -> GraphProptestResult {
    self.0(graph, meta)
  }
}

impl GraphProptest for TestWithData<GraphProptestResult> {
  type Meta = ();
  type SValue = GraphData;

  fn constant_meta() -> Option<()> {
    Some(())
  }

  fn split_input(sval: Self::SValue) -> (GraphData, Self::Meta) {
    (sval, ())
  }

  fn run(&self, graph: &mut Graph, data: &GraphData, _: Self::Meta) -> GraphProptestResult {
    self.0(graph, data)
  }
}

impl<M: Serialize + DeserializeOwned + Send + 'static> GraphProptest
  for ComplexTest<GraphProptestResult, M>
{
  type Meta = M;
  type SValue = (GraphData, M);

  fn split_input(sval: Self::SValue) -> (GraphData, Self::Meta) {
    sval
  }

  fn run(&self, graph: &mut Graph, data: &GraphData, meta: Self::Meta) -> GraphProptestResult {
    self.0(graph, data, meta)
  }
}

#[derive(Debug, Copy, Clone, Eq, PartialOrd, PartialEq)]
enum GraphTestMode {
  Proptest,
  Regression(u32),
  Debug,
}

pub struct GraphProptestRunner<F> {
  id: String,
  test: F,
  cpus: u32,
  skip_regressions: bool,
  deterministic: bool,
  pub viz_config: VizConfig,
  config: ProptestConfig,
  run_count: u32,
}

impl<F: GraphProptest> GraphProptestRunner<F> {
  pub fn new(id: String, test: F) -> Self {
    let mut config = ProptestConfig::with_cases(1000);
    // keep track of regressions ourselves, since we have a unified input format
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Off));
    config.result_cache = prop::test_runner::basic_result_cache;
    config.max_shrink_iters = 50_000;
    config.max_shrink_time = 30_000; // in millis
    GraphProptestRunner {
      id,
      test,
      cpus: 1,
      skip_regressions: false,
      deterministic: false,
      viz_config: Default::default(),
      config,
      run_count: 0,
    }
  }

  fn manifest(&self, mode: GraphTestMode) -> TestManifest {
    TestManifest::new(&self.id, mode)
  }

  fn handle_test_result(&self, result: Result<(), TestError<F::SValue>>) {
    use TestError::*;
    if result.is_ok() {
      return;
    }

    if self.run_count > 0 {
      println!(
        "Strategy #{} triggered a failure ({} previous runs)",
        self.run_count + 1,
        self.run_count
      )
    }

    match result {
      Err(Abort(reason)) => {
        panic!("test aborted: {}", reason.message());
      }
      Err(Fail(reason, input)) => {
        let manifest = self.manifest(GraphTestMode::Proptest);
        std::fs::create_dir_all(manifest.path.parent().unwrap()).unwrap();
        let (input, meta) = F::split_input(input);
        manifest.to_file().unwrap();
        manifest.write_input(&input).unwrap();
        manifest.write_input_graph(&input, self.viz_config).unwrap();
        manifest.write_meta::<F>(&meta).unwrap();
        let mut graph = input.build();
        manifest
          .write_output_graph(&graph, self.viz_config)
          .unwrap();
        self.test.run(&mut graph, &input, meta).ok();
        manifest
          .write_output_graph(&graph, self.viz_config)
          .unwrap();
        eprintln!("{}", reason.message());
        panic!("test case failure");
      }
      Ok(()) => unreachable!(),
    }
  }

  pub fn run<S>(&mut self, strategy: S)
  where
    S: SharableStrategy<Value = F::SValue>,
  {
    if !self.skip_regressions {
      self.run_regressions();
    }

    if self.cpus > 1 {
      return self.run_parallel(strategy);
    }

    let mut runner = if self.deterministic {
      prop::test_runner::TestRunner::new_with_rng(
        self.config.clone(),
        prop::test_runner::TestRng::deterministic_rng(self.config.rng_algorithm),
      )
    } else {
      prop::test_runner::TestRunner::new(self.config.clone())
    };
    let result = runner.run(&strategy, |input| {
      let (mut graph_spec, meta) = F::split_input(input);
      let mut g = graph_spec.build();
      self.test.run(&mut g, &graph_spec, meta)
    });
    self.handle_test_result(result);
    self.run_count += 1;
  }

  fn run_parallel<S>(&self, strategy: S)
  where
    S: SharableStrategy<Value = F::SValue>,
  {
    use std::sync::mpsc;
    use TestError::*;

    if self.deterministic {
      eprintln!(
        "warning: running with multiple CPUS doesn't make sense, running non-deterministically."
      );
    }

    let worker_threads: Vec<_> = (0..self.cpus)
      .map(|_| {
        let config = self.config.clone();
        let strategy = strategy.clone();
        let test = self.test;

        std::thread::spawn(move || {
          let mut runner = PropTestRunner::new(config);
          runner.run(&strategy, |input| {
            let (mut graph_spec, meta) = F::split_input(input);
            let mut g = graph_spec.build();
            test.run(&mut g, &graph_spec, meta)
          })
        })
      })
      .collect();

    let mut test_result = None;
    let mut thread_panic = None;
    for t in worker_threads {
      match t.join() {
        Ok(r) => {
          if test_result.is_none() {
            test_result = Some(r);
          }
        }
        Err(panic) => thread_panic = Some(panic),
      }
    }

    if let Some(panic) = thread_panic {
      std::panic::resume_unwind(panic);
    }

    self.handle_test_result(test_result.unwrap());
  }

  pub fn skip_regressions(&mut self, skip: bool) {
    self.skip_regressions = skip;
  }

  pub fn deterministic(&mut self, det: bool) {
    self.deterministic = det;
  }

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
    let search_path = test_regressions_dir().join(&self.id);
    let mut search_path = search_path.into_os_string().into_string().unwrap();
    search_path.push_str("/*.manifest.json");
    glob::glob(&search_path)
      .unwrap()
      .map(|p| {
        let idx = true_stem(&p?).unwrap().parse()?;
        Ok(self.manifest(GraphTestMode::Regression(idx)))
      })
      .collect()
  }

  pub fn run_regressions(&self) {
    for manifest in self.find_regressions().unwrap() {
      println!("running regression: {:?}", &manifest.path);
      self.run_with_existing(manifest);
    }
    println!("finished running regressions")
  }

  pub fn debug(&self) {
    self.run_with_existing(self.manifest(GraphTestMode::Debug));
  }

  fn run_with_existing(&self, manifest: TestManifest) {
    let input = manifest.load_input().unwrap();
    let meta = manifest.load_meta::<F>().unwrap();
    let mut graph = input.build();

    match self.test.run(&mut graph, &input, meta) {
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

pub type GraphTestResult = anyhow::Result<()>;

pub trait GraphTest {
  type TestInput;
  type Meta: Clone;

  fn split_meta(raw: Self::TestInput) -> (&'static str, Self::Meta);

  fn run(&self, graph: &mut Graph, data: &GraphData, input: Self::Meta) -> GraphTestResult;
}

impl GraphTest for SimpleTest<GraphTestResult> {
  type Meta = ();
  type TestInput = &'static str;

  fn split_meta(raw: Self::TestInput) -> (&'static str, Self::Meta) {
    (raw, ())
  }

  fn run(&self, graph: &mut Graph, _: &GraphData, _: Self::Meta) -> GraphTestResult {
    (self.0)(graph)
  }
}

impl GraphTest for TestWithData<GraphTestResult> {
  type Meta = ();
  type TestInput = &'static str;

  fn split_meta(raw: Self::TestInput) -> (&'static str, Self::Meta) {
    (raw, ())
  }

  fn run(&self, graph: &mut Graph, data: &GraphData, _: Self::Meta) -> GraphTestResult {
    (self.0)(graph, data)
  }
}

impl<M: Clone> GraphTest for TestWithMeta<GraphTestResult, M> {
  type Meta = M;
  type TestInput = (&'static str, M);

  fn split_meta(raw: Self::TestInput) -> (&'static str, Self::Meta) {
    raw
  }

  fn run(&self, graph: &mut Graph, _: &GraphData, meta: Self::Meta) -> GraphTestResult {
    (self.0)(graph, meta)
  }
}

impl<M: Clone> GraphTest for ComplexTest<GraphTestResult, M> {
  type Meta = M;
  type TestInput = (&'static str, M);

  fn split_meta(raw: Self::TestInput) -> (&'static str, Self::Meta) {
    raw
  }

  fn run(&self, graph: &mut Graph, data: &GraphData, meta: Self::Meta) -> GraphTestResult {
    (self.0)(graph, data, meta)
  }
}

pub struct GraphTestRunner<F> {
  id: String,
  pub viz_config: VizConfig,
  test: F,
}

fn glob_graph_inputs(input_pattern: &str) -> anyhow::Result<Vec<PathBuf>> {
  let mut patt = test_input_dir().to_str().unwrap().to_string();
  patt.push('/');
  patt.push_str(input_pattern);
  patt.push_str(".txt");
  let paths: Result<Vec<_>, _> = glob::glob(&patt)?.collect();
  let paths = paths?;
  if paths.is_empty() {
    anyhow::bail!("no inputs matching pattern `{}` found", &patt)
  }
  Ok(paths)
}

impl<F: GraphTest> GraphTestRunner<F> {
  pub fn new(id: String, test: F) -> Self {
    GraphTestRunner {
      id,
      viz_config: Default::default(),
      test,
    }
  }

  pub fn run(&self, input: F::TestInput) {
    std::fs::create_dir_all(test_testcase_failures_dir()).unwrap();

    let (input_patt, meta) = F::split_meta(input);
    println!("searching `{}`", input_patt);
    for input in glob_graph_inputs(input_patt).pretty_unwrap() {
      println!("input {:?}", input);
      let data = GraphData::load_from_file(&input).unwrap();
      let mut graph = data.build();
      let result = self.test.run(&mut graph, &data, meta.clone());
      if let Err(error) = result {
        let mut path = test_testcase_failures_dir().join(format!("{}.input.svg", self.id));
        data.viz().configure(self.viz_config).render(&path);
        path.set_file_name(format!("{}.output.svg", self.id));
        graph.viz().configure(self.viz_config).render(path);
        eprintln!(
          "test case failure (input {:?}):\n{:?}",
          input.file_stem().unwrap(),
          error
        );
        panic!("test failed");
      }
    }
  }

  pub fn viz_config(&mut self) -> &mut VizConfig {
    &mut self.viz_config
  }
}

pub trait ResultConv {
  fn into_graph_test(self) -> GraphTestResult;
}

impl ResultConv for GraphProptestResult {
  fn into_graph_test(self) -> GraphTestResult {
    match self {
      Ok(()) => Ok(()),
      Err(TestCaseError::Reject(reason)) => anyhow::bail!("test aborted: {}", reason),
      Err(TestCaseError::Fail(reason)) => anyhow::bail!("test failed: {}", reason),
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
macro_rules! graph_testcase_assert {
  ($cond:expr $(,)?) => {{
    graph_testcase_assert!($cond, "assertion failed: {}: {}", file!(), line!())
  }};

  ($cond:expr, $($arg:tt)+) => {{
    let cond = $cond;
    if !cond {
      return Err(anyhow::anyhow!($($arg)*))
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

pub fn mark_failed_as_regression() -> anyhow::Result<()> {
  let mut pattern = test_failures_dir().to_str().unwrap().to_string();
  pattern.push_str("/*.manifest.json");
  println!("searching `{}`", pattern);
  for p in glob::glob(&pattern).unwrap() {
    let p = p?;
    let test_id = true_stem(&p).unwrap();
    let src = TestManifest::from_file(&p).unwrap();
    let mut patt = test_regressions_dir()
      .join(test_id)
      .into_os_string()
      .into_string()
      .unwrap();
    patt.push_str("/*.manifest.json");
    let mut idx = 0;
    for file in glob::glob(&patt)? {
      let existing_index = true_stem(&file?).map(|s| s.parse::<u32>().ok()).flatten();
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

#[test]
fn test_id_from_module_path() {
  assert_eq!(&get_test_id("daggylp::scc::tests", "foobar"), "scc_foobar");
  assert_eq!(
    &get_test_id("daggylp::iis::cycles::enumeration::tests", "foobar"),
    "iis_cycles_enumeration_foobar"
  );
}
