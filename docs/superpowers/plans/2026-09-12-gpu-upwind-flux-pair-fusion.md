# GPU Upwind Flux-Pair Fusion Plan

## Objective

Remove duplicated positive/negative state evaluation in the dominant periodic
physical-space upwind flux kernels without changing the FP64 state, explicit
synchronization policy, fixed launch geometry, or unsupported solver routes.

## Implementation and local admission

- [x] Add an MPI-consistent `ASTR_GPU_FLUX_PAIR_MODE=split|fused` runtime contract with `split` as default.
- [x] Add FP64 and FP32 x/y/z fused flux-pair kernels using eight unique stencil offsets.
- [x] Preserve mixed-workspace store rounding and existing FP64 RHS accumulation.
- [x] Keep physical-boundary, characteristic, shock-sensor, diffusion, and filter paths unchanged.
- [x] Preserve one explicit device synchronization after every launched kernel.
- [x] Add source-contract, runtime-probe, field/statistics comparison, benchmark, and summarizer tests.
- [x] Validate WENO7 and MP7 split/fused equivalence locally.
- [x] Validate WENO7 NP=2 `2x1x1` and NP=4 `2x2x1` locally.
- [x] Run Compute Sanitizer with zero invalid-access errors.
- [x] Profile the split and fused routes with Nsight Systems and Nsight Compute.

## Production admission

- [ ] Repeat interleaved complete-RK timing on A800 for NP=1/2/4.
- [ ] Confirm no regression in long TGV statistics.
- [ ] Confirm checkpoint/restart continuity with the fused mode selected after restart.
- [ ] Decide whether to promote `fused` or retain it as an opt-in candidate.

Rollback requires only `ASTR_GPU_FLUX_PAIR_MODE=split` or removal of the
environment variable; the original kernels remain compiled and testable.
