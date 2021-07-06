use crate::graph::{Graph, Weight};
use grb::prelude::*;
use crate::map_with_capacity;
use fnv::FnvHashMap;
use crate::test_utils::*;
use crate::graph::*;

thread_local!{
  static ENV: grb::Env = {
      let mut env = grb::Env::empty().unwrap();
      env.set(param::OutputFlag, 0).unwrap();
      env.set(param::Threads, 1).unwrap();
      env.start().unwrap()
  };
}

pub struct Lp {
  pub vars: Vec<grb::Var>,
  pub constr: FnvHashMap<(usize, usize), grb::Constr>,
  pub model: Model,
}

impl Lp {
  pub fn build(graph: &GraphSpec) -> Self {
    let mut vars = Vec::with_capacity(graph.nodes.len());
    let mut constr = map_with_capacity(graph.nodes.len());
    let mut model = ENV.with(|env| Model::with_env("graph", env).unwrap());
    // let mut model = Model::new("graph").unwrap();

    for (i, node) in graph.nodes.iter().enumerate() {
      vars.push(add_ctsvar!(model, bounds: node.lb..node.ub, obj: node.obj).unwrap());
    }
    for (&(i,j), &w) in &graph.edges {
      constr.insert((i, j), model.add_constr("", c!(vars[i] + w <= vars[j])).unwrap());
    }

    model.update().unwrap();
    Lp { vars, constr, model }
  }

  fn solve(&mut self) -> grb::Result<LpSolution> {
    self.model.optimize()?;
    match self.model.status()? {
      Status::Infeasible => {
        self.model.compute_iis()?;
        let mut edges = Vec::new();
        for (&(i,j), c) in &self.constr {
          if self.model.get_obj_attr(attr::IISConstr, c)? > 0 {
            edges.push((i,j));
          }
        }
        let mut ubs = Vec::new();
        let mut lbs = Vec::new();
        for (i, v) in self.vars.iter().enumerate() {
          if self.model.get_obj_attr(attr::IISUB, v)? > 0 { ubs.push(i); }
          if self.model.get_obj_attr(attr::IISLB, v)? > 0 { lbs.push(i); }
        }
        ubs.sort();
        lbs.sort();
        edges.sort();
        Ok(LpSolution::Infeasible { lbs, ubs, edges })
      },
      Status::Optimal => {
        let soln = self.model.get_obj_attr_batch(attr::X, self.vars.iter().copied())?
          .into_iter().map(|x| x.round() as Weight)
          .collect();
        Ok(LpSolution::Optimal(soln))
      },
      other => panic!("unexpected model status: {:?}", other),
    }
  }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
enum LpSolution {
  Optimal(Vec<Weight>),
  Infeasible{
    edges: Vec<(usize, usize)>,
    lbs: Vec<usize>,
    ubs: Vec<usize>,
  }
}


//
// #[cfg(test)]
// mod tests {
//   use super::*;
//   #[macro_use]
//   use crate::*;
//   use crate::test_utils::*;
//   use crate::test_utils::strategy::*;
//   use crate::graph::*;
//   use proptest::prelude::*;

  // fn graph_without_solution() -> impl SharableStrategy<Value=GraphSpec> {
  //   (3..=4usize)
  //     .prop_flat_map(|size| graph(size, any_nodes(), 0..MAX_EDGE_WEIGHT))
  //     .prop_map(|g| {
  //       // let s = Lp::build(&g).solve().unwrap();
  //       g
  //     })
  // }

  // fn compare_feas_with_lp(path: impl AsRef<Path>) {
  //   let mut graph = GraphSpec::load_from_file(path).build();
  //   for n in graph.nodes.iter_mut() {
  //     n.obj = 1;
  //   }
  //
  //   let mut lp = Lp::from_graph(&graph);
  //
  //   assert_eq!(graph.solve(), SolveStatus::Optimal);
  //   lp.model.optimize().unwrap();
  //   assert_eq!(lp.model.status(), Ok(Status::Optimal));
  //
  //   for (n, (lp_var, node)) in lp.vars.iter().zip(&graph.nodes).enumerate() {
  //     let lp_x = lp.model.get_obj_attr(attr::X, lp_var).unwrap().round() as Weight;
  //     if lp_x != node.x {
  //       graph.viz().save_svg(test_output("test-failed.svg"));
  //       panic!("Difference at node {} {:?}:\n\tLP = {} != {} = Graph",n, node,  lp_x, node.x)
  //     }
  //   }
  // }
  //
  // #[test]
  // fn feasible_test_cases() {
  //   let patt = test_input_dir().join("*-f.txt");
  //   dbg!(&patt);
  //   for p in glob::glob(patt.to_str().unwrap()).unwrap() {
  //     let p = p.unwrap();
  //     println!("checking {:?}", &p);
  //     compare_feas_with_lp(p);
  //   }
  // }

//   struct Tests;
//   impl Tests {
//     fn compare_daggylp_with_gurobi(g: &mut Graph, data: &GraphSpec) -> TestCaseResult {
//       let mut lp = Lp::build(data);
//       let s = lp.solve().unwrap();
//       match (g.solve(), s) {
//         (SolveStatus::Optimal, LpSolution::Optimal(solution)) => {
//           let graph_soln : Vec<_> = g.nodes.iter().filter(|n: &&Node| matches!(n.kind, NodeKind::Var)).map(|n| n.x).collect();
//           prop_assert_eq!(graph_soln, solution);
//         },
//         (SolveStatus::Infeasible(_), LpSolution::Infeasible { lbs, ubs, edges }) => {
//
//         }
//         (g_status, lp_status) => {
//           test_case_bail!("DLP status: {:?} != {:?} Gurobi status ", g_status, lp_status)
//         }
//       }
//
//       Ok(())
//     }
//   }
//
//   // graph_test_dbg!{ Tests; compare_daggylp_with_gurobi _ }
//
//   graph_proptests!{
//     Tests;
//     graph(any_nodes(3..300), any_edge_weight()) => compare_daggylp_with_gurobi(data) [cases=500, parallel=4, layout=LayoutAlgo::Fdp];
//   }
//
//
// }