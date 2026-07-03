# Keep file output as a CPU-owned boundary

Full ASTR GPU migration will keep HDF5 flowfield output, checkpoint/restart files, slice/list/monitor output, and controller reload as CPU-owned output/control boundaries in the near and medium term. Whole-field D2H transfers at these explicit boundaries are allowed, while per-kernel or per-module whole-field transfers remain outside the GPU-authoritative compute-loop architecture.

**Consequences**

GPU migration work should focus on resident compute-loop data, kernels, halo exchange, and reductions. Output-boundary optimization may later reduce or pipeline D2H volume, but direct GPU HDF5/checkpoint writing is not a core requirement for the next architecture phase.
