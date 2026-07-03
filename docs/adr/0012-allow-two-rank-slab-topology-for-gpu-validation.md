# Allow two-rank slab topology for GPU validation

The first multi-rank validation may override ASTR's legacy 3D MPI auto-decomposition through an explicit validation control such as `ASTR_FORCE_MPI_TOPOLOGY=2,1,1`, forcing `isize=2`, `jsize=1`, and `ksize=1` for both CPU and GPU oracle runs. This reflects the available two-GPU environment and creates a practical same-topology validation path before larger `2x2x2` or multi-node topologies are available.

**Consequences**

This is a scoped validation bridge, not a general input-file topology system. It must not change the default CPU decomposition behavior, and future multi-GPU validation should formalize topology control before expanding to `1x2x1`, `1x1x2`, `2x2x2`, or multi-node runs.
