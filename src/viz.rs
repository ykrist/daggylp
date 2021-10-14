use super::graph::*;
use std::path::Path;
use crate::iis::Iis;
use fnv::FnvHashSet;
use std::fmt;
use std::io;
use gvdot::{SetAttribute, GraphComponent};

#[derive(Debug, Copy, Clone)]
pub enum SccViz {
  Hide,
  Show,
  Collapse,
}


#[derive(Debug, Clone, Copy)]
pub struct VizConfig {
  scc: SccViz,
  show_edge_weights: bool,
  layout: gvdot::Layout,
}

impl Default for VizConfig {
  fn default() -> Self {
    VizConfig { scc: SccViz::Show, show_edge_weights: true, layout: gvdot::Layout::Dot }
  }
}

impl VizConfig {
  pub fn sccs(&mut self, scc: SccViz) {
    self.scc = scc;
  }

  pub fn show_edge_weights(&mut self, show: bool) {
    self.show_edge_weights = show;
  }

  pub fn layout(&mut self, layout: gvdot::Layout) {
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
pub struct VizGraph<'a, E> {
  graph: &'a Graph<E>,
  config: VizConfig,
  var_names: Option<&'a dyn Fn(Var) -> String>,
  edge_fmt: Option<&'a dyn Fn(Var, Var, Weight) -> String>,
}


impl<E: EdgeLookup> Graph<E> {
  pub fn viz(&self) -> VizGraph<'_, E> {
    VizGraph {
      graph: self,
      config: Default::default(),
      var_names: None,
      edge_fmt: None,
    }
  }
}


impl<'a, E> VizGraph<'a, E> {
  pub fn fmt_vars(mut self, f: &'a dyn Fn(Var) -> String) -> Self {
    self.var_names = Some(f);
    self
  }

  pub fn fmt_edges(mut self, f: &'a dyn Fn(Var, Var, Weight) -> String) -> Self {
    self.edge_fmt = Some(f);
    self
  }
}


pub trait GraphViz : Sized
{
  fn config(&self) -> &VizConfig;
  fn config_mut(&mut self) -> &mut VizConfig;

  fn visit<W: io::Write>(&self, g: &mut gvdot::Graph<W>) -> io::Result<()>;

  fn render(&self, path: impl AsRef<Path>) -> io::Result<()> {
    let mut g = gvdot::Graph::new()
      .directed()
      .strict(true)
      .stream_to_gv(self.config().layout, path)?
      .attr(gvdot::attr::RankDir, gvdot::val::RankDir::LR)?;
    self.visit(&mut g)?;
    let status = g.wait()?;
    assert!(status.success());
    Ok(())
  }

  fn save_dot(&self, path: impl AsRef<Path>) -> io::Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut g = gvdot::Graph::new()
      .directed()
      .strict(true)
      .create(gvdot::StrId::default(), file)?
      .attr(gvdot::attr::RankDir, gvdot::val::RankDir::LR)?;

    self.visit(&mut g)?;
    g.finish()
  }

  fn configure_with<F: FnOnce(&mut VizConfig)>(mut self, f: F) -> Self {
    f(self.config_mut());
    self
  }
  fn configure(mut self, c: VizConfig) -> Self {
    *self.config_mut() = c;
    self
  }
}

const MRS_COLOR : &'static str = "blueviolet";
const IIS_COLOR : &'static str = "orangered";
const ACTIVE_EDGE_COLOR : &'static str = "lightseagreen";
const DEFAULT_EDGE_COLOR : &'static str = "grey";
const DEFAULT_NODE_COLOR : &'static str = "black";

impl<E: EdgeLookup> GraphViz for VizGraph<'_, E>  {
  fn config(&self) -> &VizConfig {
    &self.config
  }
  fn config_mut(&mut self) -> &mut VizConfig {
    &mut self.config
  }
  fn visit<W: io::Write>(&self, g: &mut gvdot::Graph<W>) -> io::Result<()> {
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


    for (n, node) in self.graph.nodes.iter().enumerate().filter(|(n, node)| filter_func(node)) {
      let name = match node.kind {
        NodeKind::Scc(k) => format!("SCC[{}]", k),
        _ => if let Some(f) = self.var_names {
          f(self.graph.var_from_node_id(n))
        } else {
          format!("X[{}]", n)
        }
      };

      let mut border_color = None;
      #[cfg(feature= "viz-extra")] {
        match (self.graph.viz_data.last_solve, self.graph.viz_data.highlighted_nodes.contains(&n)) {
          (Some(SolveStatus::Optimal), true) => border_color = Some(MRS_COLOR),
          (Some(SolveStatus::Infeasible(_)), true) => border_color = Some(IIS_COLOR),
          _ => {},
        }
      }
      let border_color = border_color.unwrap_or(DEFAULT_NODE_COLOR);

      let bg_color = match &self.graph.nodes[n].kind {
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
      g.add_node(n)?
        .attr(gvdot::attr::Shape, gvdot::val::Shape::Plaintext)?
        .attr(gvdot::attr::Label, gvdot::attr::html(&html))?;
    }

    fn hide_scc_edges<E>(_: &Graph<E>, e: &Edge) -> bool {
      matches!(&e.kind, EdgeKind::Regular)
    }
    fn show_scc_edges<E>(_: &Graph<E>, _: &Edge) -> bool {
      true
    }
    fn collapse_scc_edges<E>(g: &Graph<E>, e: &Edge) -> bool {
      !(matches!(&g.nodes[e.to].kind, NodeKind::SccMember(_))
        || matches!(&g.nodes[e.from].kind, NodeKind::SccMember(_)))
    }
    let filter_func = match self.config.scc {
      SccViz::Collapse => collapse_scc_edges,
      SccViz::Hide => hide_scc_edges,
      SccViz::Show => show_scc_edges,
    };

    for e in self.graph.edges.all_edges().filter(|e| filter_func(self.graph, e)) {
      let e: &Edge = e;
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
      let color = color.unwrap_or("grey");
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
      g.add_edge(e.from, e.to)?
        .attr(gvdot::attr::Label, gvdot::attr::html(&html))?
        .attr(gvdot::attr::Color, color)?;

    }
    Ok(())
  }
}


#[cfg(test)]
pub use viz_graph_data::VizGraphData;

#[cfg(test)]
mod viz_graph_data {
  use super::*;
  use crate::test::GraphData;

  #[derive(Copy, Clone, Debug)]
  pub struct VizGraphData<'a> {
    graph: &'a GraphData,
    config: VizConfig,
  }

  impl GraphData {
    pub(crate) fn viz(&self) -> VizGraphData<'_> {
      VizGraphData {
        graph: self,
        config: Default::default(),
      }
    }
  }

  impl<'a> GraphViz for VizGraphData<'a> {
    fn config(&self) -> &VizConfig {
      &self.config
    }
    fn config_mut(&mut self) -> &mut VizConfig {
      &mut self.config
    }

    fn visit<W: io::Write>(&self, g: &mut gvdot::Graph<W>) -> io::Result<()> {
      for n in 0..self.graph.nodes.len() {
        g.add_node(n)?;
      }
      for &(i,j) in self.graph.edges.keys() {
        g.add_edge(i, j)?;
      }
      Ok(())
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