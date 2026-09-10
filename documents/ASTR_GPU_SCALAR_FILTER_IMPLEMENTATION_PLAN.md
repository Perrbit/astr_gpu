# GPU Single-Component Filter Workspace Implementation Plan

## Objective

Add an optional single-component workspace for the explicit tenth-order GPU
filter while retaining the existing full-state `qwork_d` implementation as the
default and reference path.

## Runtime contract

- `ASTR_GPU_FILTER_WORKSPACE=full` remains the default.
- `ASTR_GPU_FILTER_WORKSPACE=scalar` allocates one haloed three-dimensional
  device workspace and filters conservative variables sequentially.
- Every kernel launch remains followed by the existing explicit synchronization
  and CUDA error check.
- All MPI ranks must select the same workspace mode.
- When `lfilter=f`, the scalar filter is inactive and the implementation retains
  the full compatibility workspace required by existing NSCBC or sponge paths.
  Runtime output makes this fallback explicit.

## Implementation steps

1. Add a source-contract test for mode parsing, conditional allocation, scalar
   kernels, scalar halo exchange, and full-mode retention.
2. Add runtime mode selection and conditional device allocation in
   `commarray_gpu`.
3. Add x, y, z and final-copy scalar filter kernels without changing the full
   ping-pong kernels.
4. Add component-scoped halo refresh/exchange using the existing field halo
   transport protocol.
5. Centralize filter orchestration so RK-first statistics and normal RK stages
   use the same selected backend.
6. Propagate the mode through TGV validation drivers and document commands.
7. Verify compilation, source-contract tests, NP=1 field/statistics equivalence,
   and representative multi-rank x/y/z decompositions.

## Acceptance criteria

- Default `full` results and tests remain unchanged.
- `scalar` matches the CPU and full GPU paths within the existing field and
  statistics tolerances.
- NP=2 x-, y-, and z-slab scalar runs complete with correct halo behavior.
- Runtime output identifies the selected mode and allocated workspace size.
- Invalid or inconsistent rank-local mode values terminate before time stepping.

## 2026-09-10 local admission result

Completed:

- NVHPC 26.1 CUDA build;
- runtime `full|scalar` selection with cross-rank consistency checking;
- conditional full-state or single-scalar allocation;
- component-scoped x/y/z filter kernels and y/z halo transport;
- centralized filter dispatch shared by RK-first statistics and RK stages;
- NP=1 TGV 1-step and 10-step statistics and field comparison;
- NP=2 x/y/z slabs, NP=4 `2x2x1`, and NP=8 `2x2x2` statistics comparison;
- x/y/z `bctype=41` physical-boundary field comparison;
- NP=1 full/scalar bitwise field equality after one step;
- Compute Sanitizer memcheck with zero reported errors.
- five-step filtered LDC and Channel field/statistics comparisons;
- five-step curvilinear `bctype=42` comparisons for x-wall NP=1 and y-wall NP=2;
- explicit `FILTER_WORKSPACE` forwarding in validation, timing, segmented
  production, memcheck, and A800 campaign drivers;
- compute-node dependency and host-staged MPI gates in both A800 job drivers.

The runtime rejection gate was also exercised with an invalid workspace value:
the executable returned nonzero after printing the accepted `full|scalar`
contract and before reporting any time-step cost. The scalar-specific `32^3`
TGV Compute Sanitizer run completed with `ERROR SUMMARY: 0 errors`.

Measured on the `128^3`, `hm=5`, NP=1 TGV case:

- full workspace: `107,424,760` bytes;
- scalar workspace: `21,484,952` bytes;
- reduction: `80%`.

The fresh local `128^3`, NP=1, explicit-sync timing used one process warm-up and
five measured runs, with 19 retained complete-RK samples per run. The full and
scalar medians were `0.081408774 s` and `0.078219996 s`; scalar was `3.916%`
faster in this workstation gate. Peak sampled process memory decreased from
`1686 MiB` to `1604 MiB`. This is an RTX 4000 Ada result and is not transferable
to A800.

A two-advance `64^3` Nsight Systems trace counted `188/332` total kernel
launches for full/scalar and `262/262` total MPI calls. The scalar path therefore
increases GPU launch count while leaving NP=1 process-level MPI call count
unchanged. Multi-rank filter halo message counts remain topology dependent and
must be measured on A800.

The first Channel field gate exposed a validation-script phase mismatch rather
than a filter error: CPU ordinary HDF output was compared with the GPU complete
RK state. `run_channel_phased_compare.sh` now requests the same CPU complete-RK
snapshot used by the LDC and CURVE gates. The corrected scalar comparison passes
with `q5 L_inf=4.6185277824406512e-14`.

Still required before A800 production admission:

- repeat the full/scalar complete-RK and launch accounting on A800;
- confirm `512^3` per-rank memory headroom for NP=1/2/4;
- admit scalar for production only if numerical, runtime, and restart gates pass.
