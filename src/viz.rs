use dot::*;
use super::graph::*;
use std::path::Path;
use crate::iis::Iis;
use fnv::FnvHashSet;
use std::fmt;

#[derive(Debug, Copy, Clone)]
pub enum SccViz {
  Hide,
  Show,
  Collapse,
}

#[derive(Debug, Copy, Clone)]
pub enum LayoutAlgo {
  Dot,
  Neato,
  Fdp,
}

impl LayoutAlgo {
  fn prog_name(&self) -> &'static str {
    use LayoutAlgo::*;
    match self {
      Dot => "dot",
      Neato => "neato",
      Fdp => "fdp",
    }
  }
}

#[derive(Debug, Clone, Copy)]
pub struct VizConfig {
  scc: SccViz,
  show_edge_weights: bool,
  layout: LayoutAlgo,
}

impl Default for VizConfig {
  fn default() -> Self {
    VizConfig { scc: SccViz::Show, show_edge_weights: true, layout: LayoutAlgo::Dot }
  }
}

impl VizConfig {
  pub fn sccs(&mut self, scc: SccViz) {
    self.scc = scc;
  }

  pub fn show_edge_weights(&mut self, show: bool) {
    self.show_edge_weights = show;
  }

  pub fn layout(&mut self, layout: LayoutAlgo) {
    self.layout = layout;
  }
}

#[cfg(feature = "viz-extra")]
#[derive(Debug, Default, Clone)]
pub struct VizData {
  pub highlighted_edges: FnvHashSet<(usize, usize)>,
  pub highlighted_nodes: FnvHashSet<usize>,
  pub last_solve: Option<SolveStatus>,
}

#[cfg(feature = "viz-extra")]
impl VizData {
  pub fn clear_highlighted(&mut self) {
    self.highlighted_edges.clear();
    self.highlighted_nodes.clear();
  }
}


#[derive(Clone)]
pub(crate) struct VizGraph<'a> {
  graph: &'a Graph,
  config: VizConfig,
  var_names: Option<&'a dyn Fn(Var) -> String>,
  edge_fmt: Option<&'a dyn Fn(Var, Var, Weight) -> String>,
}


impl Graph {
  pub(crate) fn viz(&self) -> VizGraph<'_> {
    VizGraph {
      graph: self,
      config: Default::default(),
      var_names: None,
      edge_fmt: None,
    }
  }
}


impl<'a> VizGraph<'a> {
  pub fn fmt_vars(mut self, f: &'a dyn Fn(Var) -> String) -> Self {
    self.var_names = Some(f);
    self
  }

  pub fn fmt_edges(mut self, f: &'a dyn Fn(Var, Var, Weight) -> String) -> Self {
    self.edge_fmt = Some(f);
    self
  }
}


pub(crate) trait GraphViz<'a, N, E>: GraphWalk<'a, N, E> + Labeller<'a, N, E> + Sized
  where
    N: Clone + 'a,
    E: Clone + 'a,
{
  fn config(&self) -> &VizConfig;
  fn config_mut(&mut self) -> &mut VizConfig;

  fn save_as_dot(&'a self, path: impl AsRef<Path>) {
    let mut file = std::fs::File::create(path).map(std::io::BufWriter::new).unwrap();
    render(self, &mut file).unwrap();
  }

  fn save_svg(&'a self, path: impl AsRef<Path>) {
    use std::process::{Command, Stdio};
    use std::io::Write;
    let mut dot_contents = Vec::with_capacity(1000);
    render(self, &mut dot_contents).unwrap();

    let mut gv = std::process::Command::new(self.config().layout.prog_name())
      .arg(format!("-o{}", path.as_ref().as_os_str().to_str().expect("printable filename")))
      .arg("-Grankdir=LR")
      .arg("-Tsvg")
      .stdin(Stdio::piped())
      .stdout(Stdio::inherit())
      .stderr(Stdio::inherit())
      .spawn()
      .unwrap();

    gv.stdin.take().unwrap().write_all(&dot_contents).unwrap();
    gv.wait().unwrap();
  }

  fn configure(mut self, c: VizConfig) -> Self {
    *self.config_mut() = c;
    self
  }

  fn configure_with<F: FnOnce(&mut VizConfig)>(mut self, f: F) -> Self {
    f(self.config_mut());
    self
  }
}

const MRS_COLOR : &'static str = "blueviolet";
const IIS_COLOR : &'static str = "orangered";
const ACTIVE_EDGE_COLOR : &'static str = "lightseagreen";
const DEFAULT_EDGE_COLOR : &'static str = "grey";
const DEFAULT_NODE_COLOR : &'static str = "black";

impl<'a> GraphWalk<'a, usize, Edge> for VizGraph<'a> {
  fn nodes(&'a self) -> Nodes<'a, usize> {
    fn hide_scc(n: &Node) -> bool {
      !matches!(&n.kind, NodeKind::Scc(_))
    }
    fn show_scc(_: &Node) -> bool {
      true
    }
    fn collapse_scc(n: &Node) -> bool {
      !matches!(&n.kind, NodeKind::SccMember(_))
    }
    let filter_func = match self.config.scc {
      SccViz::Collapse => collapse_scc,
      SccViz::Hide => hide_scc,
      SccViz::Show => show_scc,
    };

    self.graph.nodes.iter().enumerate()
      .filter_map(|(n, node)|
        if filter_func(node) { Some(n) } else { None }
      )
      .collect::<Vec<_>>()
      .into()
  }

  fn edges(&'a self) -> Edges<'a, Edge> {
    fn hide_scc(_: &Graph, e: &Edge) -> bool {
      matches!(&e.kind, EdgeKind::Regular)
    }
    fn show_scc(_: &Graph, _: &Edge) -> bool {
      true
    }
    fn collapse_scc(g: &Graph, e: &Edge) -> bool {
      !(matches!(&g.nodes[e.to].kind, NodeKind::SccMember(_))
        || matches!(&g.nodes[e.from].kind, NodeKind::SccMember(_)))
    }
    let filter_func = match self.config.scc {
      SccViz::Collapse => collapse_scc,
      SccViz::Hide => hide_scc,
      SccViz::Show => show_scc,
    };

    self.graph.edges.all_edges()
      .filter(|e| filter_func(self.graph, e))
      .copied()
      .collect::<Vec<_>>()
      .into()
  }

  fn source(&'a self, e: &Edge) -> usize {
    e.from
  }

  fn target(&'a self, e: &Edge) -> usize {
    e.to
  }
}

impl<'a> Labeller<'a, usize, Edge> for VizGraph<'a> {
  fn graph_id(&'a self) -> Id<'a> { Id::new("debug").unwrap() }

  fn node_id(&'a self, n: &usize) -> Id<'a> { Id::new(format!("n{}", n)).unwrap() }

  fn node_label(&'a self, n: &usize) -> LabelText {
    let node = &self.graph.nodes[*n];

    let name = match node.kind {
      NodeKind::Scc(k) => format!("SCC[{}]", k),
      _ => if let Some(f) = self.var_names {
        f(self.graph.var_from_node_id(*n))
      } else {
        format!("X[{}]", n)
      }
    };


    let mut border_color = None;
    #[cfg(feature= "viz-extra")] {
      match (self.graph.viz_data.last_solve, self.graph.viz_data.highlighted_nodes.contains(n)) {
        (Some(SolveStatus::Optimal), true) => border_color = Some(MRS_COLOR),
        (Some(SolveStatus::Infeasible(_)), true) => border_color = Some(IIS_COLOR),
        _ => {},
      }
    }
    let border_color = border_color.unwrap_or(DEFAULT_NODE_COLOR);

    let bg_color = match &self.graph.nodes[*n].kind {
      &NodeKind::SccMember(k) => format!("/pastel19/{}", (k % 8) + 1),
      NodeKind::Scc(_) => "/pastel28/8".to_string(),
      NodeKind::Var => "/pastel19/9".to_string(),
    };

    let scc_member_obj_row = match (node.kind, node.obj) {
      // (NodeKind::SccMember(k), 0) => format!(r#"<TR><TD COLSPAN="3">SCC[{}]</TD></TR>"#, k),
      (NodeKind::SccMember(k), obj) => format!(r#"<TR><TD>{}</TD><TD COLSPAN="2">SCC[{}]</TD></TR>"#, obj, k),
      // (_, 0) => "".to_string(),
      (_, obj) => format!(r#"<TR><TD COLSPAN="3">{}</TD></TR>"#, obj),
    };

    let html = format!(
      concat!(
      r#"<FONT FACE="fantasque sans mono">"#,
      r#"<TABLE BORDER="0" CELLSPACING="0" CELLBORDER="1" BGCOLOR="{}" COLOR="{}">"#,
      r#"<TR><TD COLSPAN="3">{}</TD></TR>"#,
      "{}",
      r#"<TR><TD>{}</TD><TD>{}</TD><TD>{}</TD></TR>"#,
      r#"</TABLE>"#,
      r#"</FONT>"#
      ),
      bg_color, border_color, name,
      scc_member_obj_row,
      node.lb, node.x, node.ub,
    );
    LabelText::html(html)
  }

  fn edge_color(&'a self, e: &Edge) -> Option<LabelText<'a>> {
    let mut color = None;

    #[cfg(feature = "viz-extra")] {
      match (self.graph.viz_data.last_solve, self.graph.viz_data.highlighted_edges.contains(&(e.from, e.to))) {
        (Some(SolveStatus::Infeasible(_)), true) => { color = Some(IIS_COLOR); },
        (Some(SolveStatus::Optimal), true) => { color = Some(MRS_COLOR); },
        _ => {}
      }
    }

    if color.is_none() {
      match self.graph.nodes[e.to].active_pred {
        Some(pred) if pred == e.from => color = Some(ACTIVE_EDGE_COLOR),
        _ => {},
      }
    }

    Some(LabelText::escaped(color.unwrap_or("grey")))
  }


  fn edge_label(&self, e: &Edge) -> LabelText {
    let html = if let Some(fmt) = self.edge_fmt {
      format!(r#"<FONT FACE="fantasque sans mono">{}</FONT>"#,
              fmt(
                self.graph.var_from_node_id(e.from),
                self.graph.var_from_node_id(e.to),
                e.weight
              ))
    } else {
      format!(r#"<FONT FACE="fantasque sans mono">{}</FONT>"#, e.weight)
    };
    LabelText::html(html)

  }

  fn node_shape(&self, _: &usize) -> Option<LabelText> {
    Some(LabelText::escaped("plaintext"))
  }
}

impl<'a> GraphViz<'a, usize, Edge> for VizGraph<'a> {
  fn config(&self) -> &VizConfig {
    &self.config
  }
  fn config_mut(&mut self) -> &mut VizConfig {
    &mut self.config
  }
}


#[cfg(test)]
pub use viz_graph_data::VizGraphSpec;

#[cfg(test)]
mod viz_graph_data {
  use super::*;
  use crate::test::GraphData;

  #[derive(Copy, Clone, Debug)]
  pub struct VizGraphSpec<'a> {
    graph: &'a GraphData,
    config: VizConfig,
  }


  impl GraphData {
    pub(crate) fn viz(&self) -> VizGraphSpec<'_> {
      VizGraphSpec {
        graph: self,
        config: Default::default(),
      }
    }
  }


  impl<'a> GraphWalk<'a, usize, (usize, usize)> for VizGraphSpec<'a> {
    fn nodes(&'a self) -> Nodes<'a, usize> {
      (0..self.graph.nodes.len()).collect::<Vec<_>>().into()
    }

    fn edges(&'a self) -> Edges<'a, (usize, usize)> {
      self.graph.edges.keys().copied().collect::<Vec<_>>().into()
    }

    fn source(&'a self, e: &(usize, usize)) -> usize {
      e.0
    }

    fn target(&'a self, e: &(usize, usize)) -> usize {
      e.1
    }
  }

  impl<'a> Labeller<'a, usize, (usize, usize)> for VizGraphSpec<'a> {
    fn graph_id(&'a self) -> Id<'a> { Id::new("debug").unwrap() }

    fn node_id(&'a self, n: &usize) -> Id<'a> { Id::new(format!("n{}", n)).unwrap() }

    fn node_label(&'a self, n: &usize) -> LabelText {
      let node = &self.graph.nodes[*n];
      LabelText::escaped(format!("{}\n[{},{}]", n, node.lb, node.ub))
    }

    fn edge_label(&self, e: &(usize, usize)) -> LabelText {
      LabelText::escaped(format!("{}", self.graph.edges[e]))
    }
  }


  impl<'a> GraphViz<'a, usize, (usize, usize)> for VizGraphSpec<'a> {
    fn config(&self) -> &VizConfig {
      &self.config
    }
    fn config_mut(&mut self) -> &mut VizConfig {
      &mut self.config
    }
  }
}


// #[cfg(test)]
// mod tests {
// use super::*;
// use crate::test::*;
//
// #[test]
// fn viz() {
//   let g = GraphSpec::load_from_file(test_input("simple-f")).pretty_unwrap().build();
//   g.viz().save_svg(test_output("test.svg"));
// }
//
// #[test]
// fn viz_generator() {
//   let g = GraphSpec::load_from_file(test_input("simple-f")).pretty_unwrap();
//   g.save_svg(test_output("test.generator.svg"));
// }
// }