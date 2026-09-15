# ASTR A800 TGV FP64 Scaling Matrix Plan

_Approved short-run performance campaign, 2026-09-13._

## Objective

The campaign measures topology-aware strong and weak scaling of the complete
ASTR CUDA Fortran solver on one node with four A800 GPUs. It is a performance
campaign rather than a long-time TGV physical-validation run.

All runs use FP64 state and arithmetic, the full five-component `qwork_d`
filter workspace, explicit sixth-order central derivatives, explicit
tenth-order filtering, RK3, explicit post-kernel synchronization, and the
`pinned-overlap` host-staged MPI halo backend.

## Timing contract

Each matrix entry performs one process-level warm-up followed by five
independent measured runs. Every measured run advances 22 complete RK steps,
discards the first two in-process steps, and retains 20 steps. Checkpoint and
full-field output are disabled during timing. The report uses the median RK
time within each run and then the median of the five run medians.

The fixed timing step is `1.0e-4`. It is chosen for stable short sampling and
does not define a production TGV trajectory.

## Strong scaling

The global grid remains `512x512x512`. NP=1 uses `1x1x1`; NP=2 tests all three
slabs; NP=4 tests the three four-way slabs and the three two-axis
decompositions. For each NP, the fastest five-run median is reported as the
topology-optimized result. Every topology remains in the raw table.

Strong speedup and efficiency are

$$
S_p = \frac{T_1}{T_p}, \qquad E_p = \frac{T_1}{pT_p}.
$$

## Weak scaling

Each MPI rank owns approximately `256x256x256` points. The global grid is the
local grid multiplied by the selected MPI topology. NP=2 and NP=4 again test
every topology permutation.

Weak efficiency and aggregate throughput ratio are

$$
E_p^{\mathrm{weak}} = \frac{T_1}{T_p}, \qquad
R_p = \frac{pT_1}{T_p}.
$$

The anisotropic global domains are fixed-local-work performance workloads.
Their trajectories must not be compared with the DLR TGV physical reference.

## Failure handling and provenance

Each topology is isolated. A failed entry records its exit code and log path,
then the campaign continues. The job records the source commit, clean source
status, executable hash, compiler and MPI versions, HDF5 configuration, GPU
topology, peak memory, GPU utilization, and the explicit `full` workspace log
marker. The clean solver source snapshot and the uncommitted benchmark-control
scripts are stored separately, and the job records SHA256 hashes for both
control scripts. Results are written below
`/data/user/hd56000/weiph/results/a800_tgv_scaling_matrix_<job-id>`.
