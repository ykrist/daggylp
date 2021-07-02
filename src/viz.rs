use dot::*;
use super::graph::*;
use std::path::Path;
use crate::test_utils::{GraphSpec};


impl Graph {
  pub (crate) fn viz(&self) -> VizGraph<'_> {
    VizGraph {
      graph: self,
      var_names: None,
      scc_members: true,
      edge_weights: true,
    }
  }
}

#[derive(Copy, Clone)]
pub(crate) struct VizGraph<'a> {
  graph: &'a Graph,
  var_names: Option<&'a dyn Fn(Var) -> String>,
  scc_members: bool,
  edge_weights: bool,
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

pub(crate) trait GraphViz<'a, N, E> : GraphWalk<'a, N, E>  + Labeller<'a, N, E> + Sized
  where
    N: Clone + 'a,
    E: Clone + 'a,
{
  fn save_as_dot(&'a self, path: impl AsRef<Path>) {
    let mut file = std::fs::File::create(path).map(std::io::BufWriter::new).unwrap();
    render(self, &mut file).unwrap();
  }

  fn save_svg_with_layout(&'a self, path: impl AsRef<Path>, layout: LayoutAlgo) {
    use std::process::{Command, Stdio};
    use std::io::Write;
    let mut dot_contents = Vec::with_capacity(1000);
    render(self, &mut dot_contents).unwrap();

    let mut gv = std::process::Command::new(layout.prog_name())
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

  fn save_svg(&'a self, path: impl AsRef<Path>) {
    self.save_svg_with_layout(path, LayoutAlgo::Dot);
  }
}

impl<'a> VizGraph<'a> {
  pub(crate) fn hide_scc_members(mut self) -> Self {
    self.scc_members = false;
    self
  }

  pub(crate) fn hide_edge_weights(mut self) -> Self {
    self.edge_weights = false;
    self
  }

  pub(crate) fn fmt_nodes(mut self, f: &'a dyn Fn(Var) -> String) -> Self {
    self.var_names = Some(f);
    self
  }
}

impl<'a> GraphWalk<'a, usize, Edge> for VizGraph<'a> {
  fn nodes(&'a self) -> Nodes<'a, usize> {
    if !self.scc_members {
      self.graph.nodes.iter().enumerate()
        .filter_map(|(n, node)| if node.kind.is_scc_member() { None } else { Some(n) } )
        .collect::<Vec<_>>()
        .into()
    } else {
      (0..self.graph.nodes.len()).collect::<Vec<_>>().into()
    }
  }

  fn edges(&'a self) -> Edges<'a, Edge> {
    let edges = (0..self.graph.nodes.len())
      .flat_map(|n| self.graph.edges.successors(n)).copied();
    if !self.scc_members {
      edges.filter(|e| !self.graph.nodes[e.to].kind.is_scc_member() && !self.graph.nodes[e.from].kind.is_scc_member() )
        .collect::<Vec<_>>()
        .into()
    } else {
      edges.collect::<Vec<_>>().into()
    }
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
    let name = if let Some(f) = self.var_names {
      f(self.graph.var_from_node_id(*n))
    } else {
      match node.kind {
        NodeKind::Var => format!("X[{}]", n),
        NodeKind::Scc(k) => format!("SCC[{}]", k),
        NodeKind::SccMember(k) => format!("X[{}|{}]", n, k),
      }
    };
    let var = match &self.graph.state {
      ModelState::Optimal | ModelState::Mrs => format!("{} = {}", &name, node.x),
      _ => name
    };

    LabelText::escaped(format!("{}\n[{},{}]", var, node.lb, node.ub))
  }

  fn edge_color(&'a self, e: &Edge) -> Option<LabelText<'a>> {
    let c = match self.graph.nodes[e.to].active_pred {
      Some(pred) if pred == e.from => "red",
      _ => "black",
    };
    Some(LabelText::escaped(c))
  }

  fn edge_label(&self, e: &Edge) -> LabelText {
    LabelText::escaped(format!("{}", e.weight))
  }

  fn node_shape(&self, n: &usize) -> Option<LabelText> {
    let shp = match &self.graph.nodes[*n].kind {
      NodeKind::Var => "circle",
      NodeKind::SccMember(_) => "doublecircle",
      NodeKind::Scc(_) => "square"
    };
    Some(LabelText::escaped(shp))
  }
}

impl<'a> GraphViz<'a, usize, Edge> for VizGraph<'a> {}

impl<'a> GraphWalk<'a,  usize, (usize, usize)> for GraphSpec {
  fn nodes(&'a self) -> Nodes<'a, usize> {
    (0..self.nodes.len()).collect::<Vec<_>>().into()
  }

  fn edges(&'a self) -> Edges<'a, (usize, usize)> {
    self.edges.keys().copied().collect::<Vec<_>>().into()
  }

  fn source(&'a self, e: &(usize, usize)) -> usize {
    e.0
  }

  fn target(&'a self, e: &(usize, usize)) -> usize {
    e.1
  }
}

impl<'a> Labeller<'a, usize, (usize, usize)> for GraphSpec {
  fn graph_id(&'a self) -> Id<'a> { Id::new("debug").unwrap() }

  fn node_id(&'a self, n: &usize) -> Id<'a> { Id::new(format!("n{}", n)).unwrap() }

  fn node_label(&'a self, n: &usize) -> LabelText {
    let node = &self.nodes[*n];
    LabelText::escaped(format!("{}\n[{},{}]", n, node.lb, node.ub))
  }

  fn edge_label(&self, e: &(usize, usize)) -> LabelText {
    LabelText::escaped(format!("{}", self.edges[e]))
  }
}


impl<'a> GraphViz<'a, usize, (usize, usize)> for GraphSpec {}

#[cfg(test)]
mod tests {
  // use super::*;
  // use crate::test_utils::*;
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
}