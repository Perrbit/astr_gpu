# GPU Upwind Flux-Pair Fusion Design

## Scope

This optimization applies only to the all-periodic, physical-space `543e`
WENO7/MP7 route. It does not alter physical-boundary, characteristic-space,
shock-sensor, diffusion, filter, MPI halo, or Runge-Kutta semantics.

The runtime selector is:

```text
ASTR_GPU_FLUX_PAIR_MODE=split|fused
```

`split` remains the default and rollback path. The value is checked
collectively across MPI ranks; unknown or rank-inconsistent values are fatal.

## Numerical contract

The original implementation launches separate positive and negative
Steger-Warming reconstruction kernels. Each sign independently reconstructs
seven stencil states. The fused implementation evaluates the eight unique
cell offsets needed by the pair once, forms both split contributions, and
writes the same scalar interface flux workspace.

FP64 arithmetic order is preserved inside each reconstructed sign. In
`mixed_workspace` mode, the final store preserves the previous rounding order:

```fortran
real(real(real(fh_plus, 4), 8) + fh_minus, 4)
```

The authoritative state and RHS remain FP64. One explicit device
synchronization follows each fused flux kernel, followed by the existing RHS
kernel and its explicit synchronization.

## Local evidence

RTX 4000 Ada, TGV `128^3`, WENO7, five interleaved FP64 repeats:

| Mode | Median complete RK time | Spread |
|---|---:|---:|
| split | `1.219539164 s` | `0.160%` |
| fused | `0.845358029 s` | `0.117%` |

The measured complete-RK reduction is `30.682%`, or `1.44263x` speedup.

The corresponding `256^3` run used five interleaved repeats with two retained
complete-RK samples per repeat. Split and fused median times were
`8.678503483 s` and `6.052904717 s`; spreads were `1.223%` and `1.136%`.
The fused reduction was `30.254%`, or `1.43378x` speedup. GPU 0 used about
`9.0 GiB` and reached `100%` sampled utilization during the run.

Nsight Systems reduces periodic flux-kernel instances from `540` to `270` and
their total duration from `7.109023053 s` to `4.866302954 s` over six complete
RK advances. Total CUDA launch/synchronization API calls fall from `984` to
`714`.

Nsight Compute reports `128` registers per thread for both implementations.
The split baseline already spills, so zero spill is not a valid acceptance
gate. Comparing one fused pair with two split signs, local-memory sectors fall
from approximately `23.76 million` to `16.97 million`, a `28.57%` reduction.

WENO7 and MP7 split-versus-fused field differences are at FP64 roundoff, with
maximum observed conservative-variable difference `5.68e-14`. Statistics are
identical at `1e-14`; WENO7 NP=2 and NP=4 topology checks pass; Compute
Sanitizer reports zero invalid-access errors.

At `256^3`, the FP64 one-step split/fused maximum conservative-variable
difference is `1.14e-13` and all statistics are identical. The mixed-workspace
pair has a sparse FP32 rounding difference with maximum `q5=4.20e-9`; both
mixed variants independently remain within the MP1 FP64-reference gate
(`q5=8.72e-7 < 2e-6`, statistics `4.88e-13 < 5e-11`). Therefore the comparison
driver uses separate split/fused field gates: `1e-12` for FP64 and `1e-8` for
mixed workspace.

## Promotion boundary

The implementation is locally admitted as an opt-in performance candidate.
It must remain non-default until A800 NP=1/2/4 timing and long TGV/restart
checks pass. Any unsupported route continues through the original split
kernels even when the environment requests `fused`.
