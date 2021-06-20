use crate::graph::Graph;
use grb::prelude::*;
use crate::map_with_capacity;
use fnv::FnvHashMap;

pub struct Lp {
  pub vars: Vec<grb::Var>,
  pub constr: FnvHashMap<(grb::Var, grb::Var), grb::Constr>,
  pub model: Model,
}

impl Lp {
  pub fn from_graph(graph: &Graph) -> Self {
    let mut vars = Vec::with_capacity(graph.nodes.len());
    let mut constr = map_with_capacity(graph.nodes.len());
    let mut model = Model::new("graph").unwrap();

    for (i, node) in graph.nodes.iter().enumerate() {
      vars.push(add_ctsvar!(model, bounds: node.lb..node.ub, obj: node.obj).unwrap());
    }
    for (i, vi) in vars.iter().copied().enumerate() {
      for e in &graph.edges_from[i] {
        let vj = vars[e.to];
        constr.insert((vi, vj), model.add_constr("", c!(vi + e.weight <= vj)).unwrap());
      }
    }

    model.update().unwrap();

    Lp { vars, constr, model }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::test_utils::{GraphSpec, test_input, test_input_dir, test_output};
  use crate::graph::*;
  use std::path::Path;
  use crate::viz::GraphViz;

  fn compare_feas_with_lp(path: impl AsRef<Path>) {
    let mut graph = GraphSpec::load_from_file(path).build();
    for n in graph.nodes.iter_mut() {
      n.obj = 1;
    }

    let mut lp = Lp::from_graph(&graph);

    assert_eq!(graph.solve(), SolveStatus::Optimal);
    lp.model.optimize().unwrap();
    assert_eq!(lp.model.status(), Ok(Status::Optimal));

    for (n, (lp_var, node)) in lp.vars.iter().zip(&graph.nodes).enumerate() {
      let lp_x = lp.model.get_obj_attr(attr::X, lp_var).unwrap().round() as Weight;
      if lp_x != node.x {
        graph.viz().save_svg(test_output("test-failed.svg"));
        panic!("Difference at node {} {:?}:\n\tLP = {} != {} = Graph",n, node,  lp_x, node.x)
      }
    }
  }

  #[test]
  fn feasible_test_cases() {
    let patt = test_input_dir().join("*-f.txt");
    dbg!(&patt);
    for p in glob::glob(patt.to_str().unwrap()).unwrap() {
      let p = p.unwrap();
      println!("checking {:?}", &p);
      compare_feas_with_lp(p);
    }
  }
}