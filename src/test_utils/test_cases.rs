use std::io::Result as IoResult;
use super::*;

pub fn generate_test_cases() -> IoResult<()> {
  let k8 = GraphSpec::new(
    8,
    IdenticalNodes{ lb: 0, ub: 10, obj: 0 },
    AllEdges(0)
  );
  k8.save_to_file(test_input("k8-f.txt"));

  {
    let mut k8 = k8.clone();
    k8.nodes[0].ub = 5;
    k8.nodes[7].lb = 6;
    k8.save_to_file(test_input("k8-cbi.txt"))
  }

  {
    let mut k8 = k8.clone();
    k8.edges.insert((0,6), 1);
    k8.save_to_file(test_input("k8-cei.txt"))
  }

  let k4 = GraphSpec::new(
    4,
    IdenticalNodes{ lb: 0, ub: 10, obj: 0 },
    AllEdges(0)
  );
  k4.save_to_file(test_input("k4-f.txt"));

 Ok(())
}