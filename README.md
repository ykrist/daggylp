# DAGgy LP

A linear-time linear programming solver which solves problems of the form

```txt
min sum(c[i] * x[i] ; for all i)
x[i] - x[j] >= d[i, j]
a[i] <= x[i] <= b[i] 
```

using a labelling algorithm on a directed graph.  Supports computation of minimally responsible/responsible subsets, useful for Combinatorial Benders Decomposition.

This code has not been cleaned up, beware.
