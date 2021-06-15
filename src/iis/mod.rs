use crate::set_with_capacity;
use crate::graph::*;
use fnv::FnvHashSet;
use crate::model_states::ModelAction;

mod cycles;
mod path;


#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct Iis {
  pub(crate) constrs: FnvHashSet<Constraint>,
}

impl Iis {
  pub(crate) fn clear(&mut self) {
    self.constrs.clear()
  }

  pub(crate) fn with_capacity(size: usize) -> Self {
    Iis{ constrs: set_with_capacity(size) }
  }
  /// First variable should **NOT** be the same as the last
  pub(crate) fn from_cycle(vars: impl IntoIterator<Item=Var>) -> Self {
    let mut vars = vars.into_iter();
    let mut constrs = set_with_capacity(vars.size_hint().0);
    let first : Var = vars.next().unwrap();
    let mut i = first;
    for j in vars {
      constrs.insert(Constraint::Edge(i, j));
      i = j;
    }
    constrs.insert(Constraint::Edge(i, first));
    Iis{ constrs }
  }

  /// First variable should **NOT** be the same as the last
  pub(crate) fn from_cycle_backwards(vars: impl IntoIterator<Item=Var>) -> Self {
    let mut vars = vars.into_iter();
    let mut constrs = set_with_capacity(vars.size_hint().0);
    let last : Var = vars.next().unwrap();
    let mut j = last;
    for i in vars {
      constrs.insert(Constraint::Edge(i, j));
      j = i;
    }
    constrs.insert(Constraint::Edge(last, j));
    Iis{ constrs }
  }


  /// Path should be in order ( start -> ... -> end )
  /// If `bounds` is `true`, add the lower bound of start and upper bound of end to the IIS.
  pub(crate) fn add_forwards_path(&mut self, vars: impl IntoIterator<Item=Var>, bounds: bool) {
    let mut vars = vars.into_iter();
    self.constrs.reserve(vars.size_hint().0);
    let mut i : Var = vars.next().unwrap();
    if bounds {
      self.add_constraint(Constraint::Lb(i));
    }
    for j in vars {
      self.add_constraint(Constraint::Edge(i, j));
      i = j;
    }
    if bounds {
      self.add_constraint(Constraint::Ub(i));
    }
  }

  /// Path should be in reverse order ( end <- ... <- start)
  ///   /// If `bounds` is `true`, add the lower bound of start and upper bound of end to the IIS.
  pub(crate) fn add_backwards_path(&mut self, vars: impl IntoIterator<Item=Var>, bounds: bool) {
    let mut vars = vars.into_iter();
    self.constrs.reserve(vars.size_hint().0);
    let mut j : Var = vars.next().unwrap();
    if bounds {
      self.add_constraint(Constraint::Ub(j));
    }
    for i in vars {
      self.add_constraint(Constraint::Edge(i, j));
      j = i;
    }
    if bounds {
      self.add_constraint(Constraint::Lb(j));
    }
  }


  pub(crate) fn add_constraint(&mut self, c: Constraint) {
    let unique = self.constrs.insert(c);
    debug_assert!(unique)
  }

  pub fn len(&self) -> usize { self.constrs.len() }
}

impl Graph {

  pub fn compute_iis(&mut self) -> Iis {
    self.check_allowed_action(ModelAction::ComputeIis).unwrap();
    match &self.state {
      ModelState::InfPath(violated_ubs) => {
        self.compute_path_iis(violated_ubs)
      },
      ModelState::InfCycle { sccs, first_inf_scc } => {
        self.compute_cyclic_iis(&sccs[*first_inf_scc..])
      }
      _ => unreachable!()
    }
  }

}