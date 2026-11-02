# M12 CPU continuation for first-repair diagnosis

## Purpose and Scope

Continue the failed four-GPU M12 state on CPU resources, without waiting for A800
allocation. This is a non-exact numerical diagnosis, not a production run or a
claim that CPU and GPU lose positivity at exactly the same step.

The feature/gpu_dev changes are normal source changes, not a temporary M12-only
override. The CPU upper-y bc52 compatibility closure now matches the existing GPU
incoming-wave classification and relaxation normalization for the non-reacting
five-variable ideal-gas system. Other face normals and reacting closures are not
changed. The admitted profile bc22 path also uses the GPU outlet classification
and backflow recovery. Its compatibility-stage wall/filter/RHS ordering is aligned.
GPU numerical kernels are unchanged.

Non-exact restart permits CPU/GPU migration while preserving the remaining
numerical/controller contract. Random blowing still requires matching compiler
identity, RNG signature, state packet size, rank count and topology. Build paths
may differ in non-exact mode; exact mode still requires the original compiler
options. A new CPU bc52 algorithm identity prevents silently treating old CPU
boundary formulas as exact continuation.

## Local Verification

- Root-CMake FP64 CUDA build with runtime CPU/GPU selection.
- Two-rank GPU-to-CPU and CPU-to-GPU 5+7-step continuation, both compatibility and
  incoming_only: restored physical fields and latent RNG packets checked.
- Twelve-step CPU/GPU field comparison with normal outflow and manufactured
  subsonic/supersonic backflow in both modes. Maximum scaled conservative-field
  difference across the six two-rank cases: 2.50e-15; limit: 2e-10.
- Additional four-rank 4x1x1 compatibility comparison passed the same limit.
- Exact cross-backend migration remains rejected.
- CPU first-repair reporting and CPU non-exact restore with field output disabled
  passed. The diagnostic stops before repair and writes one rank-local report.
- Seven upper-face analytic probe modes at NP=1/2: 14 runs. Uniform/non-gradient
  modes additionally compare the new CPU characteristic closure with the GPU RHS;
  outgoing-gradient modes retain the independent GPU analytic check.

Primary tests: `tests/gpu_validation/test_output_cross_backend_restart.py`,
`run_nscbc_physics_gate.py`, `nscbc_physics_probe.cuf`.

The uncorrected compatibility comparison failed first at the outlet/wall corner
(density scaled difference approximately 5.44e-6). Aligning the wall/filter order
reduced the remaining difference to approximately 3.79e-9, localized at the upper
face. Aligning CPU bc52 removed this remaining discrepancy. These short tests do
not establish the cause of the original long-run instability.

## Frozen Platform Deployment

Base: `bc73c5c2f41ebc5b26e5b1e41e1db3c96133ac9e`, matching the failed GPU executable.
Apply only `tests/gpu_validation/patches/m12_bc73_cpu_diagnostic.patch`; it modifies
`src/bc.F90`, `src/mainloop.F90` and `src/output_runtime.F90`. Do not deploy the
whole current branch or modify the original deployment.

| Item | CPU diagnostic setting |
| --- | --- |
| Grid intervals | 2250 x 260 x 480 |
| MPI topology | 4 x 1 x 1; preserve rank-local blowing RNG |
| Convection / diffusion | 743e MP7-LD / 643e |
| Boundaries | 11 prof / 22 / 41 Tw=10 / 52 compatibility / periodic z |
| Time step | 0.02, unchanged |
| Start / upper bound | Completed step 40000 / completed step 40450 |
| Repair policy | lcracon=t; stop before the first CPU repair |
| Precision / restart | FP64 / exact_restart=f |
| Resources | One CPU node, four ranks, one CPU core/rank, 256 GiB, two-day limit |
| Outputs | Logs and first-repair report only; no rendering, field series or new checkpoints |

`ASTR_CPU_CRASHFIX_DIAGNOSTIC=stop` reports each affected rank's first candidate
in k/j/i scan order, including global/local indices, coordinates, q, total RHS,
primitive caches and internal energy when evaluable. Exit 93 with the diagnostic
marker denotes an intentional capture, not a successful physical simulation.
The trigger uses the existing crashfix thresholds and is not necessarily the
first mathematically negative point. No first-repair event within the window
does not prove GPU correctness.

Use `ASTR_GPU_BENCHMARK_NO_FIELD_IO=1` and `restart_output='override'` together:
restore remains enabled, while generated-grid writes and subsequent field output
are disabled. Small login-node preflight precedes CPU job submission. No large
checkpoint download or full-grid run on the login node is required.
