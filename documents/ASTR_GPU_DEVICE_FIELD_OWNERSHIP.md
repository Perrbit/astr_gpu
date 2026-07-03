# ASTR GPU Device Field Ownership

This document records the current CUDA Fortran device-field ownership contract for the TGV GPU path and the required expansion rules for future full-ASTR GPU migration.

## 1. Current Scope

Current implementation location:

```text
src_gpu/commarray_gpu.cuf
src_gpu/commvar_gpu.cuf
src_gpu/gpu_runtime.cuf
```

The current GPU path is authoritative inside the compute loop. Whole-field host/device transfers are allowed at initialization and explicit output/checkpoint boundaries only.

## 2. Current Device Fields

| Field | Current allocation | Owner | Producer | Consumer | Host transfer policy |
|---|---|---|---|---|---|
| `q_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:numq)` | GPU compute loop | initial upload, RK update, halo exchange, filter | primitive refresh, convection, filter, halo, output-boundary download | upload at initialization; download only at output/checkpoint/debug boundary |
| `qwork_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:numq)` | GPU filter path | filter ping-pong kernels | filter y/z halo and copy-back kernels | no routine whole-field host transfer |
| `qrhs_d` | `(0:im,0:jm,0:km,1:numq)` | GPU RHS path | zero, convection, diffusion | RK update | no routine whole-field host transfer |
| `qsave_d` | `(0:im,0:jm,0:km,1:numq)` | GPU RK path | RK save kernels | RK update kernels | no routine whole-field host transfer |
| `rho_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm)` | GPU primitive state | initial upload, primitive refresh | statistics, output-boundary download | upload at initialization; download only at output/checkpoint/debug boundary |
| `vel_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3)` | GPU primitive state | initial upload, primitive refresh | gradcal, convection, diffusion, statistics | same as `rho_d` |
| `prs_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm)` | GPU primitive state | initial upload, primitive refresh | convection, output-boundary download | same as `rho_d` |
| `tmp_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm)` | GPU primitive/thermo state | initial upload, primitive refresh | gradcal, diffusion, dissipation statistics | same as `rho_d` |
| `jacob_d` | `(0:im,0:jm,0:km)` | GPU geometry state | initial upload | convection, diffusion, RK update | upload after geometry is ready; no routine download |
| `dxi_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3,1:3)` | GPU metric state | initial upload | gradcal, convection, diffusion | upload after geometry is ready; no routine download |
| `dvel_d` | `(0:im,0:jm,0:km,1:3,1:3)` | GPU gradient state | `gradcal_gpu` | diffusion, enstrophy, dissipation | no routine whole-field host transfer |
| `dtmp_d` | `(0:im,0:jm,0:km,1:3)` | GPU gradient state | `gradcal_gpu` or diffusion helper paths | diffusion | no routine whole-field host transfer |
| `sigma_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:6)` | GPU diffusion flux state | diffusion flux kernel | diffusion stored RHS, diffusion halo exchange | halo-buffer transfer only in current L0 transport |
| `qflux_d` | `(-hm:im+hm,-hm:jm+hm,-hm:km+hm,1:3)` | GPU diffusion heat flux state | diffusion flux kernel | diffusion stored RHS, diffusion halo exchange | halo-buffer transfer only in current L0 transport |

## 3. Current Boundary Transfers

Allowed whole-field transfers:

- `copy_flow_to_gpu()`: initialization and explicit CPU-to-GPU synchronization boundary.
- `copy_flow_from_gpu()`: explicit CPU-owned output/checkpoint/debug boundary.

Allowed partial transfers:

- `copy_output_boundary_from_gpu()`: CPU-owned output/checkpoint boundary helper that refreshes primitive boundary faces after GPU filter/qswap preparation, matching the CPU path where `qswap` updates primitive boundary regions but does not recompute all interior primitive fields before `writechkpt`;
- statistics partial-sum or scalar reduction transfers;
- L0 host-staged halo buffers;
- debug-only field comparison hooks marked as validation code.

Disallowed inside normal GPU execution:

- per-kernel whole-field D2H/H2D bridges;
- CPU recomputation of GPU-owned solver fields inside the GPU path;
- hidden output-boundary downloads inside solver kernels.

## 4. Expansion Rules

When adding new physics, each new field must define:

- owner: CPU-owned, GPU-owned, or output-boundary-only;
- allocation extent, including halo width and variable dimension;
- producer and consumer modules;
- halo semantic: qswap-compatible, dataswap-compatible, reduction-only, or no halo;
- output policy: CPU-owned output boundary or GPU-resident diagnostic;
- validation oracle.

Future field groups:

| Phase | Field group | Required ownership decision |
|---|---|---|
| Non-reacting generalization | boundary metadata, non-TGV initialization scratch | define whether boundary application is GPU-owned or CPU boundary |
| Species | species conservative fields, mass fractions, diffusion coefficients | extend `numq` layout and halo semantics before chemistry |
| Turbulence | turbulence variables and model source scratch | decide wall/near-wall diagnostic ownership |
| Chemistry | species source terms, thermochemistry tables/mechanisms | keep batched source execution isolated from flow solver ownership |
| Immersed boundary | masks, interpolation data, forcing fields | keep preprocessing boundary explicit; runtime forcing should become GPU-owned only after validation |

## 5. Acceptance Criteria

The ownership table is enforceable when:

- every new device array is listed before use in a solver path;
- `src/` still calls only backend-neutral GPU facades;
- full-field transfers are visible in facade names or validation hooks;
- Nsight residency checks distinguish halo-buffer transfers from whole-field output-boundary transfers.
