# Validate single-rank GPU before GPU MPI

The first-stage GPU acceptance target is one MPI rank on one GPU. Multi-rank halo exchange and CUDA-aware MPI are deferred until the single-rank TGV path passes CPU/GPU numerical equivalence, because adding cross-rank data motion before RHS, filter, and RK equivalence would make error localization unnecessarily difficult.
