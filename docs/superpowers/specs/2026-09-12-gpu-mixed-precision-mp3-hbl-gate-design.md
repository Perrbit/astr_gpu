# GPU Mixed-Precision MP3 Cartesian Viscous HBL Gate Design

_ASTR CUDA Fortran MP3-HBL1 characteristic-flux admission, 2026-09-12_

## Decision

Extend the experimental MP3 `characteristic_flux` workspace to one exact
Cartesian viscous hypersonic-boundary-layer configuration. The admitted case is
the existing S2-C3 three-dimensional `bl` path with profile inflow, simple
outflow, a lower wall, an upper farfield boundary, periodic z, selective Roe
characteristic reconstruction, and sixth-order explicit diffusion.

This extension changes only storage of the five-component characteristic
interface-flux workspace from FP64 to FP32. Roe averages, characteristic
matrices, MP7 reconstruction, the Ducros sensor, viscous terms, authoritative
state, right-hand side, Runge-Kutta update, MPI halos, diagnostics, and output
remain FP64.

The FP64 path remains the production default. Passing this gate establishes a
bounded local numerical and memory-safety result. It does not promote mixed
precision to production use and does not establish long-time HBL or SBLI
physics.

## Exact runtime eligibility

The new eligibility branch must call the existing
`gpu_s2_hbl_selective_roe_diffusion_supported()` capability. It therefore
requires all of the following:

- `flowtype='bl'` and `ndims=3`;
- Cartesian geometry with `lihomo=f`, `ljhomo=f`, and `lkhomo=t`;
- `conschm='543e'`, `difschm='643e'`, and `recon_schem=3`;
- `lchardecomp=t`, `diffterm=t`, and `lfilter=f`;
- nondimensional, single-component laminar flow with `turbmode='none'`;
- profile inflow with `turbinf='prof'`;
- `bctype(1:6)=11,21,41,51,1,1`;
- no sponge on any face;
- `numq=5`, `num_species=0`, and `num_modequ=0`;
- `rkscheme='rk3'`;
- an MPI topology already admitted by the S2-C3 capability.

The existing all-periodic and S0-B0 x-physical MP3 branches remain unchanged.
S2-C4 `12/21/41/52`, NSCBC, sponge, filtering, CURVE, species, chemistry, and
other physical-boundary combinations remain ineligible. An ineligible request
must abort instead of silently using a different workspace or boundary path.

## Kernel and data flow

Add FP32-workspace counterparts for the six existing `xyphysical` kernels:

- `characteristic_upwind_flux_x_xyphysical_global_sp_kernel`;
- `characteristic_upwind_rhs_x_xyphysical_global_sp_kernel`;
- `characteristic_upwind_flux_y_xyphysical_global_sp_kernel`;
- `characteristic_upwind_rhs_y_xyphysical_global_sp_kernel`;
- `characteristic_upwind_flux_z_xyphysical_global_sp_kernel`;
- `characteristic_upwind_rhs_z_xyphysical_global_sp_kernel`.

Each writer must retain `real(8) :: fh(5)` and the existing FP64 reconstruction
call. It casts only the final interface flux when storing to
`flux_characteristic_work_sp_d`. Each reader promotes both adjacent FP32 fluxes
to FP64 before subtraction and accumulation into `qrhs_d`.

The index bounds, physical-owner clipping, MPI interior bounds, and direction
arguments must match the corresponding FP64 kernels exactly. The existing
direction-local workspace allocation is reused; no second FP32 array or FP64
mirror is added. Every new kernel launch remains followed by
`sync_after_kernel`.

In `mainloop_gpu`, the mixed branch is added only within the existing
`characteristic_xyphysical_case` dispatch. The original FP64 kernels remain the
fallback when the candidate is not active.

## Validation architecture

Use three independently prepared but identical cases:

1. CPU FP64 reference;
2. GPU FP64 reference;
3. GPU `mixed_workspace/characteristic_flux` candidate.

CPU FP64 versus GPU FP64 verifies that the existing S2-C3 implementation remains
within its established tolerance. GPU FP64 versus MP3-HBL1 isolates the error
introduced by FP32 characteristic-flux storage. A CPU-to-candidate comparison
may be reported, but it must not replace either of these two attribution gates.

All field comparisons use the complete state after a completed Runge-Kutta
step. Intermediate boundary-filter or boundary-RHS HDF5 states are not valid
references. The comparison includes all physical boundary planes and all
available primitive and conservative fields.

## Calibration and frozen tolerances

Run one explicit calibration case at `192x192x8`, two time steps, NP=1. Record
the maximum absolute and relative differences for:

- GPU FP64 versus MP3-HBL1 complete fields;
- GPU FP64 versus MP3-HBL1 statistics;
- GPU FP64 versus MP3-HBL1 raw shock sensor.

Freeze the formal tolerances before the MPI matrix. The calibration script must
write the selected values to a standalone tolerance file and refuse to
overwrite an existing file unless explicitly requested. Formal matrix and
sanitizer runs must not enable calibration or adjust a threshold after failure.

The raw sensor is not required to remain bitwise identical after the first
Runge-Kutta stage. Although sensor arithmetic remains FP64, the candidate state
has already received an FP32-workspace flux update. Raw sensor values therefore
use a frozen tolerance. The binary shock mask must have zero mismatches because
a changed mask selects a different numerical path.

## Formal numerical matrix

| Gate | Grid and steps | Decomposition | Required comparisons |
| --- | --- | --- | --- |
| MP3-HBL1-CAL | `192x192x8`, 2 steps | NP=1 | Calibration only; freeze field, statistics, and raw-sensor tolerances |
| MP3-HBL1-N1 | `192x192x8`, 2 steps | NP=1 | CPU/GPU FP64 baseline and GPU FP64/MP3 complete comparisons |
| MP3-HBL1-X2 | `192x192x8`, 2 steps | NP=2, `2x1x1` | x-boundary ownership and x halo |
| MP3-HBL1-Y2 | `192x192x8`, 2 steps | NP=2, `1x2x1` | y-wall/farfield ownership and y halo |
| MP3-HBL1-Z2 | `192x192x8`, 2 steps | NP=2, `1x1x2` | periodic z halo |

Each formal run must verify:

- the unique ASTR completion marker and finite outputs;
- the expected MPI topology;
- CPU FP64 versus GPU FP64 within its pre-existing S2-C3 thresholds;
- GPU FP64 versus MP3-HBL1 within the frozen thresholds;
- zero binary shock-mask mismatches;
- `ASTR_GPU_PRECISION_MODE=mixed_workspace`;
- `ASTR_GPU_MIXED_CANDIDATE=characteristic_flux`;
- `ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux`.

The FP64 GPU reference must also be checked to ensure that no mixed candidate
was activated accidentally.

## Memory-safety and storage gates

Run Compute Sanitizer memcheck for NP=2 `2x1x1` and NP=2 `1x2x1`. These cases
exercise internal MPI interfaces and both x and y physical-boundary ownership.
The sanitizer driver may disable CUDA-aware OpenMPI probing through the existing
local memcheck environment, but it must not change solver data placement or
numerical configuration.

Acceptance requires normal completion and `ERROR SUMMARY: 0 errors` from every
rank in both runs. Missing rank reports, nonzero sanitizer exit status,
non-finite fields, or a missing candidate marker are failures.

The analytical allocation contract remains:

```text
mixed bytes = halo_points * 5 * 4
FP64 bytes  = halo_points * 5 * 8
```

The mixed workspace must save exactly 50 percent relative to the corresponding
FP64 workspace, with no simultaneous FP64 characteristic-flux allocation.

## Performance evidence

Performance is descriptive in MP3-HBL1. If timing is run, use five interleaved
GPU FP64 and candidate repetitions and report complete-RK median, spread,
sampled peak memory, and utilization. No speedup threshold is imposed because
this gate broadens physical coverage rather than promoting the candidate.

The timing result must not be generalized to production HBL or SBLI grids. A
local slowdown does not invalidate the numerical gate, while a local speedup
does not establish A800 production performance.

## Failure policy

Stop and report without changing CPU code or relaxing MP3 tolerances when any of
the following occurs:

- the CPU FP64 versus GPU FP64 S2-C3 baseline fails;
- a discrepancy localizes to a physical boundary or MPI ownership plane;
- the binary shock mask changes;
- a field, statistic, or sensor exceeds its frozen tolerance;
- an ineligible case enters the mixed path;
- a sanitizer error or non-finite value appears.

A defect discovered in the CPU or FP64 GPU physical path requires a separate
user decision before repair. It must not be folded into the mixed-precision
change.

## Acceptance boundary and documentation

Passing build, contract, calibration, all four formal numerical runs, both
sanitizer runs, and storage accounting yields the classification
`hbl-cartesian-local-pass-not-promoted`.

That classification establishes only short-time consistency for FP32 storage of
the final characteristic interface flux in the exact Cartesian viscous S2-C3
case. It excludes long-time HBL statistics, SBLI shock motion, separation,
NSCBC, `bctype=52`, sponge, filtering, CURVE, chemistry, multi-component flow,
and production performance.

After evidence exists, synchronize measured tolerances, observed maxima,
sanitizer results, and classification into:

- `tests/gpu_validation/README.md`;
- `documents/GPU_VALIDATION_MATRIX.md`;
- `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`;
- `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`;
- the original MP3 characteristic-flux design and implementation plan.

No status document may describe MP3-HBL1 as passing before the complete runtime
evidence matrix has finished.
