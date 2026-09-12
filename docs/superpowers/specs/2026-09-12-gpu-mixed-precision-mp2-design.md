# GPU Mixed-Precision Phase MP2 Design

## Status and decision

Phase MP2 extends the existing `mixed_workspace` experiment to diagnostic
derivatives and viscous-flux temporary arrays. It does not promote mixed precision to the
production default. Local validation can classify candidates as usable or
rejected, but A800 promotion is explicitly deferred.

The approved implementation is candidate-specialized storage. An active FP32
candidate replaces its FP64 workspace rather than adding a full-size mirror.
This is the only design that measures real memory reduction without changing
the type of authoritative solver state.

## Runtime contract

The existing precision selector remains:

```text
ASTR_GPU_PRECISION_MODE=fp64|mixed_workspace
```

Mixed mode gains an exactly-one-candidate selector:

```text
ASTR_GPU_MIXED_CANDIDATE=flux|derivative|viscous_flux
```

Rules:

- an unset candidate preserves backward compatibility and selects `flux`;
- comma-separated or otherwise combined candidates are invalid;
- all MPI ranks must select the same value, checked with collective minimum
  and maximum reductions;
- an unknown or inconsistent value aborts before GPU work begins;
- setting a non-default candidate while precision mode is `fp64` is an error,
  because silently ignoring it would make benchmark provenance ambiguous;
- a requested candidate that is ineligible for the current case is a hard
  error rather than an FP64 fallback;
- rank zero reports precision mode, requested candidate, active candidate,
  and allocated bytes.

`fp64` with no candidate variable remains behaviorally unchanged.

## Precision ownership

The following remain FP64 in every MP2 configuration:

- `q_d`, `qrhs_d`, `qsave_d`, primitive variables, and Runge-Kutta updates;
- `jacob_d`, `dxi_d`, coordinates, and boundary normals;
- thermodynamic and transport-property evaluation;
- physical boundary state construction and NSCBC/GCBC characteristic
  projection;
- CFL evaluation, diagnostics reductions, checkpoints, and HDF5 output;
- all MPI message buffers and MPI payloads.

Only one of the following workspace pairs may become FP32:

| Candidate | FP32 arrays | Arrays remaining FP64 |
| --- | --- | --- |
| `flux` | `flux_work_sp_d` | all MP2 arrays |
| `derivative` | `dvel_sp_d`, `dtmp_sp_d` | `sigma_d`, `qflux_d` |
| `viscous_flux` | `sigma_sp_d`, `qflux_sp_d` | `dvel_d`, `dtmp_d` |

The corresponding FP64 arrays are not allocated when their FP32 candidate is
active. This mutual exclusion is required for honest byte accounting.

## Derivative candidate

### Write path

The five existing Cartesian/physical-boundary gradcal variants retain FP64
stencil arithmetic and metric projection. Their candidate-specific kernels
cast only the final nine velocity-gradient components and three temperature-
gradient components to FP32 on store.

### Read path

The current stored-diffusion implementation does not consume `dvel/dtmp`.
Its five diffusion-flux kernels recompute gradients directly from
`vel_d/tmp_d` through `gradient_at*()`. The derivative candidate therefore
targets only the existing statistics and diagnostic consumers. Candidate-
specific kernels load FP32 derivative values and explicitly promote each value
to FP64 before wall projection, accumulation, or reduction. This includes:

- enstrophy and dissipation diagnostics;
- Channel and S1/HBL scalar diagnostics;
- compact wall-moment statistics;
- the gradcal CPU/GPU comparison output.

This candidate is a memory-footprint experiment rather than a timestep-
performance candidate. Replacing the solver's direct gradient recomputation
with stored derivatives would change the numerical execution path and is not
part of MP2.

No whole-field conversion kernel or FP64 derivative mirror is permitted.
Shock-sensitive cases are ineligible for this MP2 candidate because the shock
sensor also consumes `dvel`; that interaction belongs to MP3.

## Viscous-flux candidate

### Write and read path

The five diffusion-flux variants retain FP64 viscosity, conductivity, stress,
and heat-flux algebra and cast the final six `sigma` plus three `qflux`
components to FP32 on store. Candidate-specific stored-diffusion RHS kernels
promote values to FP64 before metric projection, differencing, and accumulation
into `qrhs_d`.

### Halo path

`sigma/qflux` require halo exchange. The current generic field transport is
typed as FP64 and cannot accept FP32 arrays. MP2 therefore adds specialized
FP32 pack and unpack kernels:

1. pack FP32 field values into the existing FP64 device send buffers;
2. explicitly synchronize;
3. preserve the existing FP64 device-to-host, MPI, and host-to-device path;
4. unpack FP64 receive buffers into FP32 field halos;
5. explicitly synchronize.

Local periodic halo refresh uses corresponding FP32 kernels. Existing private
MPI tags, neighbor ordering, message counts, and host-staged transport remain
unchanged. No FP32 MPI payload is introduced.

## Eligibility

Both MP2 candidates require `diffterm=t`, explicit `643e` diffusion, five
conservative variables, `num_species=0`, RK3, and the existing validated GPU
case capability. `derivative` is intended only for flow paths that consume the
stored derivative workspace through statistics or diagnostics. It rejects
characteristic/shock-sensor routes so it cannot alter the MP3 mask through
FP32 gradients.

Initial admitted flow families are:

- periodic viscous TGV for field, statistics, MPI, memory, and timing gates;
- Cartesian Sutherland HBL for wall friction and heat-flux gates;
- the validated CURVE-C23 viscous HBL and acoustic/uniform slices for geometry,
  wall, thermal, and NSCBC coupling gates.

Eligibility is expanded only after a flow family passes its own FP64-reference
validation. A successful TGV test does not admit physical boundaries.

## Synchronization and failure behavior

Every new kernel launch is followed by the existing explicit
`sync_after_kernel` call. MP2 does not introduce streams, overlap, asynchronous
conversion, or selective synchronization.

Allocation failure, unsupported candidate/case pairing, non-finite values,
MPI-rank disagreement, or an unexpected physical discrepancy is fatal. The
implementation must not relax a threshold or silently fall back to FP64.
Rollback is `ASTR_GPU_PRECISION_MODE=fp64`, or `mixed_workspace` with the
previous `flux` candidate.

## Validation sequence

Each candidate is completed independently in this order:

1. source-contract tests fail before implementation and pass afterward;
2. runtime probe validates default, each accepted value, invalid value,
   precision/candidate mismatch, and MPI-rank disagreement;
3. NVHPC builds the probe and complete `astr` target;
4. periodic viscous TGV compares FP64 and candidate fields and statistics;
5. HBL compares fields, wall friction, wall heat flux, and profile diagnostics;
6. CURVE compares uniform preservation, acoustic response, wall quantities,
   and viscous NSCBC behavior at the same RK phase;
7. NP=2 and NP=4 compare fields/statistics with FP64 halo payloads;
8. Compute Sanitizer reports zero invalid-access errors;
9. five interleaved local runs report complete-RK median and spread;
10. allocation logs verify the predicted byte reduction and no FP64 mirror.

Tolerance values are frozen from pilot evidence and expected FP32 workspace
rounding. FP64 equivalence gates are not copied blindly, but unexplained drift,
non-finite values, changed stability, or inconsistent wall/thermal trends stop
the phase for user review.

## Documentation and classification

Results update the architecture plan, current-status report, validation matrix,
and GPU validation README. Each candidate receives one of two local outcomes:

- `local-pass-not-promoted`: all local gates pass and measured memory savings
  are real, regardless of whether local timing improves;
- `rejected`: a software, numerical, physical, MPI, sanitizer, or memory gate
  fails.

Production promotion remains unavailable until the deferred A800 campaign.

## Exclusions

MP2 does not convert characteristic flux workspaces, shock sensors, filter
workspaces, authoritative state, halo payloads, checkpoint formats, or physical
models. It does not enable combined FP32 candidates, FP16, BF16, TF32, Tensor
Cores, fast math, compact schemes, chemistry, or unrelated CPU refactoring.
