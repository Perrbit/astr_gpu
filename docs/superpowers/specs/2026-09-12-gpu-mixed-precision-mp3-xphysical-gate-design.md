# GPU Mixed-Precision MP3 X-Physical Gate Design

_ASTR CUDA Fortran characteristic-flux physical-boundary admission, 2026-09-12_

## Decision

Extend the experimental MP3 `characteristic_flux` workspace from the completed
all-periodic Shu-Osher gate to the existing S0-B0 finite-domain Shu-Osher case.
The admitted case has x physical zero-extrapolation boundaries
`bctype(1:2)=50`, periodic y/z boundaries, `543e` MP7 reconstruction with
selective Roe characteristic decomposition, no diffusion, and no filtering.

This is the smallest physical-boundary extension that exercises the existing
x-face MP5 degradation and Ducros-mask boundary closure. It is not the
`11/21` OpenShock inflow/outflow case. It does not admit NSCBC, viscous HBL,
SBLI, CURVE, species, chemistry, or a general nonperiodic configuration.

The FP64 path remains the production default. Passing this gate changes the MP3
classification only from periodic-local evidence to x-physical-local evidence;
it does not promote mixed precision to the production default.

## Runtime eligibility

The existing periodic admission remains unchanged. The
`characteristic_flux` candidate gains one alternative eligibility branch that
requires the existing `gpu_shock_characteristic_s0b0_xphysical_supported()`
case capability. That branch fixes all of the following conditions:

- `flowtype='shuosher'`;
- `conschm='543e'`, `difschm='643e'`, and `recon_schem=3`;
- `lchardecomp=t`, `diffterm=f`, and `lfilter=f`;
- `lihomo=f`, `ljhomo=t`, and `lkhomo=t`;
- `bctype(1:2)=50` and `bctype(3:6)=1`;
- `numq=5`, `num_species=0`, and `num_modequ=0`;
- `rkscheme='rk3'`;
- NP=1 or NP=2 with topology `2x1x1`, as bounded by the existing case gate.

An MP3 request outside the periodic branch or this exact S0-B0 branch aborts.
It must not silently allocate the FP64 characteristic workspace or enter a
different boundary implementation.

## Kernel and data-flow changes

Add FP32-workspace counterparts for the existing x-physical writer and reader:

- `characteristic_upwind_flux_x_physical_global_sp_kernel`;
- `characteristic_upwind_rhs_x_physical_global_sp_kernel`.

The writer retains FP64 Roe averages, characteristic matrices, MP7/MP5
reconstruction, and `real(8) :: fh(5)`. It casts only the final five interface
flux values when storing them in `flux_characteristic_work_sp_d`. The reader
promotes both adjacent FP32 values to FP64 before forming the flux difference
and accumulating into FP64 `qrhs_d`.

The y/z directions continue to use the existing periodic MP3 kernels. The
solution halo, FP64 Ducros sensor halo, integer shock mask, MPI payloads, RK
state, diagnostics, and HDF5 output do not change. The characteristic flux
workspace remains direction-local and is not exchanged through MPI. Every new
writer and reader launch is followed by `sync_after_kernel`.

## Validation driver

Add a dedicated MP3 S0-B0 driver instead of modifying the existing FP64
CPU/GPU gate. Each run creates three independent case directories:

1. CPU FP64 reference;
2. GPU FP64 reference;
3. GPU `mixed_workspace/characteristic_flux` candidate.

The driver must verify the unique completion marker and the following MP3 log
contract before comparing results:

```text
ASTR_GPU_PRECISION_MODE=mixed_workspace
ASTR_GPU_MIXED_CANDIDATE=characteristic_flux
ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux
```

The numerical matrix contains:

| Gate | Grid and steps | Decomposition | Required comparison |
| --- | --- | --- | --- |
| MP3-XP1 | `400x8x8`, 3 steps | NP=1 | Full field, statistics, raw sensor, exact mask |
| MP3-XP2 | `400x8x8`, 3 steps | NP=2, `2x1x1` | Same quantities with rankwise sensor comparison |

CPU versus GPU FP64 retains the established S0-B0 tolerances. GPU FP64 versus
MP3 initially runs in an explicit calibration mode. The frozen field and
statistics absolute thresholds are selected as ten times the observed maxima,
rounded upward on the `1-2-5` engineering sequence, with the existing `2e-6`
periodic threshold as a lower bound and `1e-5` as a hard ceiling. The threshold
is frozen before MP3-XP2 and must not be relaxed after a failure.

Because MP3 does not alter sensing, GPU FP64 and MP3 must have bitwise-identical
raw sensors and zero shock-mask mismatches. The complete HDF5 comparison must
include both physical boundary planes; trimming x boundary points is forbidden.

## Safety gate

Run Compute Sanitizer memcheck on the NP=2 `2x1x1` case so both physical owners
and the internal MPI interface are exercised. Acceptance requires:

- exit status zero from every rank;
- `ERROR SUMMARY: 0 errors` for every rank;
- the MP3 active-workspace log marker;
- a normal ASTR completion marker;
- no non-finite field or statistic.

The existing analytical allocation gate remains valid: the five-component
characteristic workspace must use exactly half the bytes of its FP64 form, with
no simultaneous FP64 mirror.

## Failure policy and acceptance boundary

Stop and report rather than changing the CPU reference when the existing S0-B0
FP64 gate fails, a boundary-plane discrepancy is localized, sensor or mask
equality fails, non-finite values appear, or the candidate exceeds the frozen
threshold. A source defect in the CPU or FP64 GPU boundary path requires a
separate user decision before repair.

Passing MP3-XP1, MP3-XP2, build, contract, and memcheck gates establishes only
that FP32 storage of the final characteristic interface flux is numerically
admissible for the bounded S0-B0 x-physical case. It does not establish
accuracy for `11/21` inflow/outflow, `12/22` NSCBC, wall boundaries, diffusion,
filtering, curvilinear geometry, long-time shock motion, or physical SBLI.

## Documentation updates after execution

After evidence is generated, synchronize the measured thresholds, maxima,
sanitizer result, and final classification into:

- `tests/gpu_validation/README.md`;
- `documents/GPU_VALIDATION_MATRIX.md`;
- `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`;
- `documents/ASTR_FULL_GPU_ARCHITECTURE_PLAN.md`;
- the original MP3 characteristic-flux design and implementation plan.

No document may describe the x-physical gate as passing before the runtime
evidence exists.
