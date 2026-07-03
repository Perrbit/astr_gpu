# Use layered validation for full GPU migration

Full ASTR GPU migration will use a layered validation matrix covering build/runtime contracts, module equivalence, time integration, multi-rank correctness, performance/residency profiling, and staged physics expansion. Each layer must have an explicit oracle, such as same-topology CPU/GPU statistics, field comparison at output boundaries, single-rank versus multi-rank GPU statistics, or Nsight transfer budgets.

**Consequences**

Validation scripts and reports should become reusable project assets rather than temporary commands. New GPU modules or physics paths must declare which validation layer they satisfy before being treated as part of the full-GPU architecture.
