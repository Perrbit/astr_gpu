# ASTR GPU Multi-Rank Porting Plan

## 1. Scope

This plan extends the current GPU-resident single-rank TGV path to the first multi-rank GPU validation path.

First acceptance target:

- Case: `examples/Taylor_Green_Vortex`
- Environment: single node, two visible GPUs
- MPI/GPU mapping: two MPI ranks, one rank per GPU
- Decomposition: `isize=2`, `jsize=1`, `ksize=1`
- Physics: homogeneous periodic TGV, `numq=5`, no immersed boundary, no species, no turbulence, no chemistry
- Derivative/filter policy: explicit sixth-order central difference and explicit tenth-order central filter
- Output scope: file output, checkpoint, and HDF5 `flowfield` GPU porting are intentionally out of scope

The implementation must keep full field variables resident on GPU during the compute loop. Host transfers are allowed for small statistics reductions, host-staged halo buffers, and CPU-owned output boundaries.

## 2. Confirmed Decisions

- Use a host-staged halo-buffer baseline, not full-field D2H and not mandatory CUDA-aware MPI.
- Match CPU `parallel:qswap` semantics: exchange `hm+1` layers of `q_d(1:5)` per direction, use `hm` layers for exterior halos, and use the extra interface plane for averaging.
- Design the halo exchange interface for x/y/z topology generality even though the first validation only exercises `2x1x1`.
- Bind GPU devices by node-local MPI rank: `device_id = mod(node_local_rank, visible_device_count)`.
- Require two visible GPUs for the first two-rank validation; do not silently oversubscribe one GPU.
- Use blocking `MPI_Sendrecv` for the first correctness path.
- Use a private fixed MPI tag range for GPU halo exchange; do not mutate `parallel:mpitag`.
- Use same-topology validation: primary pass/fail is `mpirun -np 2` CPU vs `mpirun -np 2` GPU with the same `2x1x1` decomposition.
- Compute multi-rank statistics by local GPU numerator reductions followed by host MPI global reduction. Do not average already-normalized rank-local values.
- Keep public GPU facade names backend-neutral even if the first implementation is CUDA Fortran.

Related ADRs:

- [`0011-use-host-staged-qswap-compatible-halo-exchange.md`](../docs/adr/0011-use-host-staged-qswap-compatible-halo-exchange.md)
- [`0012-allow-two-rank-slab-topology-for-gpu-validation.md`](../docs/adr/0012-allow-two-rank-slab-topology-for-gpu-validation.md)

## 3. CPU Reference Behavior

The GPU path must follow CPU exchange cadence from `src/mainloop.F90`.

For TGV without immersed boundary, every RK substep follows:

```text
if lfilter:
    filterq

qrhs = 0
boucon
qswap
gradcal

if rkstep == 1:
    qsave = q * jacob
    rkfirst

rhscal
RK update q
spongefilter
updatefvar
```

Therefore the multi-rank GPU exchange must occur once per RK substep after filter/boundary handling and before `gradcal`/RHS assembly. There is no unconditional final exchange after the RK update.

CPU `parallel:qswap` exchanges `hm+1` layers, not only `hm` layers. For example, in x:

```text
left packet:  q(0:hm,:,:,:)
right packet: q(im-hm:im,:,:,:)
```

After receiving:

```text
q(im+1:im+hm,:,:,:) = received_right(1:hm,:,:,:)
q(im,:,:,:)         = 0.5 * (q(im,:,:,:) + received_right(0,:,:,:))

q(-hm:-1,:,:,:) = received_left(-hm:-1,:,:,:)
q(0,:,:,:)      = 0.5 * (q(0,:,:,:) + received_left(0,:,:,:))
```

The y and z directions have equivalent semantics.

## 4. Target Architecture

CPU-side `src/` remains an orchestration layer and may import only the GPU facade.

Proposed facade calls:

```fortran
call gpu_bind_device()
call gpu_exchange_solution_halo()
call gpu_write_flowstate()
```

Proposed GPU implementation module:

```text
src_gpu/device_runtime_gpu.cuf
src_gpu/halo_exchange_gpu.cuf
```

Public naming must stay backend-neutral:

```fortran
module halo_exchange_gpu
  subroutine exchange_solution_halo_gpu()
end module
```

Avoid public names such as `cuda_exchange_*`, `tgv_2gpu_*`, or `exchange_x_only_*`.

Required file scope:

```text
Modify:
  src/parallel.F90
  src/mainloop.F90
  src/CMakeLists.txt
  src_gpu/gpu_runtime.cuf
  src_gpu/commarray_gpu.cuf
  src_gpu/qswap_gpu.cuf
  src_gpu/mainloop_gpu.cuf
  src_gpu/statistic_gpu.cuf
  tests/gpu_validation/prepare_tgv_case.py
  tests/gpu_validation/compare_flowstate.py
  tests/gpu_validation/compare_flowfield_h5.py

Add:
  src_gpu/device_runtime_gpu.cuf
  src_gpu/halo_exchange_gpu.cuf
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

## 5. Device Binding

At GPU initialization, each MPI rank must derive a node-local rank using a shared-memory MPI communicator.

Policy:

```text
device_id = mod(node_local_rank, visible_device_count)
```

First validation guard:

```text
mpisize == 2
node-local rank count == 2
visible_device_count >= 2
```

If fewer than two GPUs are visible, stop with a clear error. Do not silently place both ranks on one GPU.

## 6. MPI Topology

Current CPU 3D auto-decomposition does not accept `mpisize=2` for 3D because it prefers all three decomposition directions to have more than one rank. The first multi-rank validation needs a scoped override that can be applied to both CPU and GPU oracle runs:

```text
if ASTR_FORCE_MPI_TOPOLOGY=2,1,1 and mpisize == 2 and ndims == 3:
    isize = 2
    jsize = 1
    ksize = 1
```

This must not change default CPU decomposition. It is only a validation bridge for same-topology CPU/GPU comparison in the two-GPU path.

## 7. Halo Exchange Design

Module responsibilities:

```text
src_gpu/qswap_gpu.cuf
  local qswap-compatible periodic kernels
  reusable primitive halo refresh helper
  no MPI topology decisions

src_gpu/halo_exchange_gpu.cuf
  topology-general exchange entry point
  x/y/z MPI pack, host-staged exchange, and unpack
  calls qswap_gpu helpers when a direction is single-rank homogeneous
  calls primitive halo refresh after q_d halo/interface updates
```

The first implementation uses host-staged buffers:

```text
GPU pack q_d packet -> device send buffer
D2H device send buffer -> host send buffer
MPI_Sendrecv host send buffer -> host recv buffer
H2D host recv buffer -> device recv buffer
GPU unpack device recv buffer -> q_d halo/interface plane
GPU refresh primitive halo
```

Transport width:

```text
hm + 1 layers
```

Primary solution variables:

```text
q_d(1:5) only
```

Primitive fields are never exchanged directly. After unpack:

```text
q_d -> rho_d, vel_d, prs_d, tmp_d
```

Diffusion has an additional CPU-compatibility requirement. The CPU diffusion
path computes `sigma` and `qflux`, then calls `dataswap(sigma)` and
`dataswap(qflux)` before differentiating the diffusive flux. Therefore the GPU
path must also exchange `sigma_d(1:6)` and `qflux_d(1:3)` halos after
`diffusion_flux_global_kernel` and before stored diffusion RHS kernels. This is
not a primitive-field exchange and it does not change the main q-residency
model.

Filter has a separate CPU-compatibility requirement. CPU `filterq` calls
`dataswap(q,direction=...)`, not `qswap`, before directional filtering. For the
first x-slab GPU path, the x filter pre-pass therefore exchanges exactly `hm`
layers with no interface-plane averaging. The `hm+1` qswap-compatible exchange
is still used before `gradcal`/RHS.

Directional behavior:

```text
if isize > 1: exchange x through MPI
else if lihomo: local qswap-compatible x periodic handling

if jsize > 1: exchange y through MPI
else if ljhomo: local qswap-compatible y periodic handling

if ksize > 1: exchange z through MPI
else if lkhomo: local qswap-compatible z periodic handling
```

The first validation will exercise x MPI exchange and y/z local periodic handling.

MPI tags are private to this module. First implementation tag allocation:

```text
x first sendrecv: 21001
x second sendrecv: 21002
y first sendrecv: 21003
y second sendrecv: 21004
z first sendrecv: 21005
z second sendrecv: 21006
```

These tags must not use or modify `parallel:mpitag`.

## 8. Statistics

Single-rank GPU statistics currently normalize inside the scalar routines. Multi-rank statistics must instead reduce numerators globally.

Required pattern:

```text
GPU local numerator reduction
D2H scalar or partial-sum data
MPI_Allreduce global numerator sum
global normalization by ia * ja * ka
I/O rank writes flowstate.dat and GPU statistic artifacts
```

Do not average rank-local normalized values.

Required quantities:

- `kenergy`: global sum of `rho * (u^2 + v^2 + w^2)`
- `enstophy`: global sum of `rho * omega^2`
- `dissipation`: global sum of local dissipation density

The GPU artifact files remain useful for debugging:

```text
gpu_kenergy.dat
gpu_enstophy.dat
gpu_dissipation.dat
```

In multi-rank runs these files must be written only by the I/O rank and must contain global-reduced values. Do not write per-rank local artifact files under these names.

## 9. Validation Matrix

Minimum validation set:

| ID | Case | MPI ranks | Topology | GPU | Module | Tolerance | Required result |
|---|---|---:|---|---:|---|---|---|
| MR-L0-CPU | TGV | 2 | `2x1x1` | 0 | CPU topology override/reference | exit 0 | CPU run produces `flowstate.dat` |
| MR-L0-GPU | TGV | 2 | `2x1x1` | 2 | device binding | rank 0/1 bind distinct GPUs | GPU run starts, no oversubscription |
| MR-L1-HALO-X | TGV fixed field | 2 | `2x1x1` | 2 | qswap-compatible x halo | `1e-10` | x halo and interface planes match CPU `qswap` |
| MR-L2-STATS | TGV | 2 | `2x1x1` | 2 | native statistics | `flowstate.dat` abs/rel <= `1e-10` | `np=2` GPU matches `np=2` CPU |
| MR-L2-FIELD | TGV | 2 | `2x1x1` | 2 | full-field output comparison | `flowfield.h5` q `L_inf` <= `1e-10` | GPU field output matches CPU when output D2H is used |
| MR-L3-PROFILE | TGV | 2 | `2x1x1` | 2 | transfer audit | no full-field D2H in compute loop | only halo buffers/statistics/output boundaries cross host |

Single-rank CPU output may be recorded as context, but it is not the primary pass/fail oracle for multi-rank GPU correctness.

## 10. Implementation Phases

### Phase 1: Topology and Device Binding

- [x] Add scoped validation-controlled topology override in MPI decomposition, so both CPU and GPU validation can force `2x1x1`.
- [x] Add node-local GPU binding in `src_gpu/device_runtime_gpu.cuf` and expose it through `src_gpu/gpu_runtime.cuf`.
- [x] Add new GPU source files to `src/CMakeLists.txt`.
- [x] Guard first multi-rank GPU path so local MPI ranks cannot silently oversubscribe visible GPUs.
- [x] Validate each rank binds a distinct GPU.

### Phase 2: Buffer Allocation

- [x] Allocate direction-aware device send/recv buffers for `hm+1` `q_d(1:5)` packets.
- [x] Allocate matching host send/recv buffers.
- [x] Keep allocation reusable across RK substeps.
- [x] Allocate reusable x-direction `hm` filter-halo buffers and diffusion-field halo buffers.

### Phase 3: Pack/Unpack Kernels

- [x] Implement x pack kernels for q and diffusion fields.
- [x] Implement x unpack kernels with q interface-plane averaging where CPU `qswap` requires it.
- [x] Implement x filter-halo unpack without interface-plane averaging where CPU `dataswap` requires it.
- [x] Reuse existing directional block policy where applicable.
- [x] Synchronize and check after every kernel.

### Phase 4: Blocking Host-Staged MPI Exchange

- [x] Implement blocking `MPI_Sendrecv` for the x direction matching CPU `qswap` and `dataswap` semantics.
- [x] First validation exercises x MPI exchange with y/z local periodic handling.
- [x] Use private fixed MPI tags for GPU halo exchange; do not mutate `parallel:mpitag`.

### Phase 5: Primitive Halo Refresh

- [x] Refresh `rho_d`, `vel_d`, `prs_d`, and `tmp_d` after qswap-compatible solution exchange.
- [x] Use full halo refresh for the first correctness path.

### Phase 6: Mainloop Integration

- [x] Replace single-rank qswap call with topology-general `gpu_exchange_solution_halo()`.
- [x] Call once per RK substep after filter/boundary handling and before GPU `gradcal`/RHS.
- [x] Do not add unconditional final exchange after RK update.
- [x] Add filter pre-pass x halo exchange using CPU `dataswap` semantics.

### Phase 7: Multi-Rank Statistics

- [x] Split GPU statistic routines into numerator reduction and normalization.
- [x] Add MPI global reduction.
- [x] I/O rank writes native `flowstate.dat` and global-reduced GPU statistic artifacts.

### Phase 8: Validation and Profiling

- [x] Add `run_tgv_mpirank2_stats_compare.sh`.
- [x] Add optional field comparison using current CPU-owned output path.
- [x] Extend `prepare_tgv_case.py` and validation drivers with `FEQCHKPT`, defaulting to `MAXSTEP`, so field comparison still writes output while no-checkpoint profiling can avoid CPU-owned full-field D2H boundaries.
- [x] Add an `nsys` no-checkpoint compute-loop transfer audit to prove no full-field D2H occurs in the compute loop.

## 11. Risks

- `hm+1` exchange is easy to implement incorrectly as `hm`; this would pass some halo-fill checks but fail interface-plane averaging.
- CPU and GPU rank topology must match exactly for validation; comparing `np=2` GPU against `np=1` CPU is not the primary oracle.
- Statistics cannot average rank-local normalized values.
- Host-staged buffers preserve portability but are not the final performance path.
- Two-rank slab override must remain scoped to GPU validation and must not silently alter CPU behavior.
- Multi-rank oversubscription on a two-GPU workstation proves topology and halo correctness only; it is not a performance or deployment proof for one-rank-per-GPU clusters.

## 12. Deferred Work

- CUDA-aware or HIP-aware MPI device-buffer exchange.
- Nonblocking MPI and compute/communication overlap.
- Real one-rank-per-GPU multi-card runs beyond the local two-GPU oversubscription smoke tests, and multi-node runs.
- Species, turbulence, chemistry, immersed boundary, and non-periodic boundary support.
- GPU porting of checkpoint, file output, or HDF5 `flowfield` writing.

## 13. Cross-Review Findings To Address Before Implementation

- The primary oracle is same-topology `np=2` CPU vs GPU. Therefore topology override must be validation-controlled, for example `ASTR_FORCE_MPI_TOPOLOGY=2,1,1`, and must apply to both CPU and GPU validation runs.
- `src/CMakeLists.txt` must include every new `.cuf` file, including `device_runtime_gpu.cuf` and `halo_exchange_gpu.cuf`.
- Device binding needs an explicit implementation location. Use `src_gpu/device_runtime_gpu.cuf` and expose it through `src_gpu/gpu_runtime.cuf`.
- `src_gpu/qswap_gpu.cuf` and `src_gpu/halo_exchange_gpu.cuf` need a strict boundary: local periodic helpers and primitive refresh in `qswap_gpu`; topology-general MPI exchange in `halo_exchange_gpu`.
- `src/mainloop.F90` must replace the GPU branch's `gpu_qswap_single_rank()` call with a topology-general facade call.
- `src_gpu/mainloop_gpu.cuf` must replace direct `periodic_copy_kernel` calls in the GPU RK loop with the same topology-general exchange path when multi-rank is active.
- `src_gpu/statistic_gpu.cuf` must split local numerator reduction from global normalization. Multi-rank artifact files must be I/O-rank-only and global-reduced.
- Validation scripts must support `FEQCHKPT`, defaulting to `MAXSTEP`, so no-checkpoint profiling can avoid CPU-owned full-field output boundaries.
- GPU halo exchange must use the private fixed MPI tag range documented in this plan and must not mutate `parallel:mpitag`.

## 14. Implementation Log

### 2026-07-02

- Implemented `ASTR_FORCE_MPI_TOPOLOGY=i,j,k` in `src/parallel.F90`; verified CPU `mpirun -np 2` TGV with `ASTR_FORCE_MPI_TOPOLOGY=2,1,1` exits normally and reports `mpi size= 2 x 1 x 1`.
- Added `src_gpu/device_runtime_gpu.cuf`; verified GPU `mpirun -np 2` binds rank 0 to GPU 0 and rank 1 to GPU 1 before stopping at the expected single-rank compute guard.
- Added `src_gpu/halo_exchange_gpu.cuf` and routed `src/mainloop.F90` plus `src_gpu/mainloop_gpu.cuf` through `gpu_exchange_solution_halo()` / `exchange_solution_halo_gpu()`. Current implementation preserves single-rank `qswap_single_rank_gpu()` behavior and explicitly stops for multi-rank halo exchange until Phase 2-5 are implemented.
- Extended `tests/gpu_validation/prepare_tgv_case.py` and existing single-rank validation drivers with `FEQCHKPT`.
- Regression checks passed:
  - CPU build: `cmake --build build_cpu_probe -j4`
  - GPU build: `cmake --build build_gpu_probe -j4`
  - Single-rank TGV statistics, `MAXSTEP=10`: pass, max `kenergy` diff `4.6726511548911276e-14`
  - Single-rank TGV statistics, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`

### 2026-07-03

- Implemented first x-slab host-staged GPU halo exchange:
  - qswap-compatible `q_d(1:5)` exchange uses `hm+1` layers and interface-plane averaging.
  - filter pre-pass exchange uses CPU `dataswap(q,direction=1)` semantics: `hm` layers, no interface-plane averaging.
  - diffusion field exchange covers `sigma_d(1:6)` and `qflux_d(1:3)` after `diffusion_flux_global_kernel`, matching CPU `dataswap(sigma)` / `dataswap(qflux)`.
- Corrected GPU stencil helpers that used local `idx_periodic(i,im)` in x. Multi-rank x stencils now read exchanged halo values instead of wrapping inside the local slab.
- Removed the unconditional final GPU qswap after RK update to match CPU exchange cadence.
- Added validation drivers:
  - `tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh`
  - `tests/gpu_validation/run_tgv_mpirank2_field_compare.sh`
- Regression checks passed:
  - CPU build: `cmake --build build_cpu_probe -j4`
  - GPU build: `cmake --build build_gpu_probe -j4`
  - Multi-rank TGV stats, `np=2`, `2x1x1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`
  - Multi-rank TGV field, `np=2`, `2x1x1`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `5.8832938520936295e-12`
  - Single-rank TGV stats, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`
- Rebuilt post-git CPU/GPU probe directories from the root `CMakeLists.txt` and reran the `np=2`, `2x1x1` validation:
  - CPU configure/build: `cmake -B build_cpu_probe -DCMAKE_Fortran_COMPILER=mpif90 .`; `cmake --build build_cpu_probe -j4`: pass.
  - GPU configure/build: `cmake -B build_gpu_probe -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON .`; `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `8.5597761517730575e-17`.
  - Multi-rank TGV field, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `5.8832938520936295e-12`.
- Completed the `np=2`, `2x1x1` no-checkpoint `nsys` transfer audit under `tests/gpu_validation/out/tgv_mpirank2_nsys_nochk_postgit`:
  - Command shape: `ASTR_FORCE_MPI_TOPOLOGY=2,1,1 nsys profile --trace=cuda,mpi --stats=true --force-overwrite=true -o ../nsys_mpirank2_nochk mpirun -np 2 .../build_gpu_probe/bin/astr run datin/input.tgv`.
  - Case settings: `MAXSTEP=1`, `FEQCHKPT=99`, `LFILTER=t`, `DIFFTERM=t`, `SCHEME=643e`.
  - `Device-to-Host` transfers: count `116`, total `380.607 MB`, max `4.793 MB`.
  - `Host-to-Device` transfers: count `188`, total `867.120 MB`, max `104.333 MB`; the largest H2D copies are initialization of resident device arrays such as geometry/device field storage.
  - No `Device-to-Host` event reached local full-field scale. For this `2x1x1` case, one local interior scalar field is about `8.39 MB`; the observed maximum D2H `4.793 MB` is halo-buffer scale, not full-field scale.
  - ASTR may still create the initial CPU-owned `outdat/flowfield.h5`; this audit is scoped to checkpoint-disabled GPU compute-loop transfers and does not change the project decision that file output/checkpoint/HDF5 field writing are outside the current GPU-porting scope.
- Extended the first multi-rank GPU halo path from x-slab to y-slab:
  - Added `q_d(1:5)` y-direction MPI qswap-compatible exchange for `1x2x1`, using `hm+1` layers and `j=0/j=jm` interface averaging to match CPU `qswap`.
  - Added y-direction filter support for the ping-pong filter path. After `filter_x_global_kernel`, `qwork_d` now exchanges y halos before the raw-halo y filter kernel is launched; this avoids local `idx_periodic(j,jm)` wrapping across a multi-rank y slab.
  - Added y-direction diffusion-field halo exchange for `sigma_d(1:6)` and `qflux_d(1:3)`.
  - Corrected diffusion-field x/local halo semantics from qswap-like `hm+1` plus interface averaging to CPU `dataswap` semantics: exactly `hm` layers and no interface averaging. This applies to `sigma_d` and `qflux_d`, not to solution `q_d`.
- Regression checks passed after the y-slab extension:
  - GPU build: `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `np=2`, `1x2x1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9226621123257246e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `8.5489341300482025e-17`.
  - Multi-rank TGV field, `np=2`, `1x2x1`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `6.1106675275368616e-12`.
  - X-slab regression stats, `np=2`, `2x1x1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
  - X-slab regression field, `np=2`, `2x1x1`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `5.8832938520936295e-12`.
  - Single-rank stats regression, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
- Extended the first multi-rank GPU halo path from x/y single-direction slabs to z-slab:
  - Added `q_d(1:5)` z-direction MPI qswap-compatible exchange for `1x1x2`, using `hm+1` layers and `k=0/k=km` interface averaging to match CPU `qswap`.
  - Added z-direction filter support for the ping-pong filter path. After the y filter writes back to `q_d`, `q_d` now exchanges z halos before the raw-halo z filter kernel is launched; this avoids local `idx_periodic(k,km)` wrapping across a multi-rank z slab.
  - Added z-direction diffusion-field halo exchange for `sigma_d(1:6)` and `qflux_d(1:3)`, using CPU `dataswap` semantics: exactly `hm` layers and no interface averaging.
  - Relaxed the first-stage topology guard from x/y-only to single-direction slab topologies. At this checkpoint, combined-direction decompositions remained intentionally blocked until corner/edge halo ordering was checked and validated.
- Regression checks passed after the z-slab extension:
  - GPU build: `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `np=2`, `1x1x2`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9240498911065060e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `5.7571135358980285e-17`.
  - Multi-rank TGV field, `np=2`, `1x1x2`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `6.2243543652584776e-12`.
  - X-slab regression stats, `np=2`, `2x1x1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
  - Y-slab regression stats, `np=2`, `1x2x1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9226621123257246e-14`.
  - Single-rank stats regression, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
- Added the first two-direction multi-rank correctness path for `NP=4`, `TOPOLOGY=2,2,1` on a two-GPU workstation:
  - Validation drivers now accept `NP` as an alias for `MPI_NP`, while preserving `MPI_NP` compatibility.
  - GPU device binding now uses `device_id = mod(node_local_rank, visible_device_count)`. On the current two-GPU machine, ranks `0/2` bind to GPU 0 and ranks `1/3` bind to GPU 1. This is explicitly logged as oversubscription correctness mode.
  - The first-stage topology guard now allows the x-y combined topology `2x2x1`; z-combined topologies were still blocked at this checkpoint and are validated separately below.
  - This validates MPI rank topology, x+y halo composition, filter ping-pong ordering, diffusion field `dataswap` ordering, and fixed GPU MPI tags under four ranks. It does not validate real four-GPU or eight-GPU performance.
- Regression checks passed after the `2x2x1` extension:
  - GPU build: `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `NP=4`, `TOPOLOGY=2,2,1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - Multi-rank TGV field, `NP=4`, `TOPOLOGY=2,2,1`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `6.5654148784233257e-12`.
  - X-slab regression stats, `np=2`, `2x1x1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
- Added the second two-direction multi-rank correctness path for `NP=4`, `TOPOLOGY=2,1,2` on a two-GPU workstation:
  - The first-stage topology guard now allows the x-z combined topology `2x1x2`.
  - The y direction remains local periodic in this case. X and z halo exchanges are composed in the same order as CPU `qswap`/`dataswap`.
  - Corner halo packets are not required for the current explicit TGV stencil path: x-direction stencil reads x halos at interior `k`, z-direction stencil reads z halos at interior `i`, and diffusion/filter paths follow the same axis-local dependency.
  - Y-z combined topology `1x2x2` and full `2x2x2` were still blocked at this checkpoint and are validated separately.
- Regression checks passed after the `2x1x2` extension:
  - GPU build: `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `NP=4`, `TOPOLOGY=2,1,2`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - Multi-rank TGV field, `NP=4`, `TOPOLOGY=2,1,2`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `5.9685589803848416e-12`.
  - X-y combined regression stats, `NP=4`, `TOPOLOGY=2,2,1`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
- Added the third two-direction multi-rank correctness path for `NP=4`, `TOPOLOGY=1,2,2` on a two-GPU workstation:
  - The first-stage topology guard now allows the y-z combined topology `1x2x2`.
  - The x direction remains local periodic in this case. Y and z halo exchanges are composed in the same order as CPU `qswap`/`dataswap`.
  - Corner halo packets are still not required for the current explicit TGV stencil path: y-direction stencil reads y halos at interior `k`, z-direction stencil reads z halos at interior `j`, and diffusion/filter paths follow the same axis-local dependency.
  - Full three-direction topology `2x2x2` was still blocked at this checkpoint and is validated separately below.
- Regression checks passed after the `1x2x2` extension:
  - GPU build: `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `NP=4`, `TOPOLOGY=1,2,2`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9226621123257246e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - Multi-rank TGV field, `NP=4`, `TOPOLOGY=1,2,2`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `5.9685589803848416e-12`.
  - X-z combined regression stats, `NP=4`, `TOPOLOGY=2,1,2`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9254376698872875e-14`.
- Added the full three-direction multi-rank correctness path for `NP=8`, `TOPOLOGY=2,2,2` on a two-GPU workstation:
  - Removed the last first-stage combined-topology guard for the TGV CUDA Fortran path.
  - The full topology composes x, y, and z face-halo exchanges in the same order as CPU `qswap` and `dataswap`.
  - Corner/edge halo packets are not required for the current explicit TGV stencil path because every derivative/filter application is axis-local and reads only the corresponding face halo at interior coordinates in the other axes.
  - The validation is an eight-rank, two-GPU oversubscription correctness proof. It does not validate real eight-GPU performance, CUDA-aware MPI, or multi-node deployment.
- Regression checks passed after the `2x2x2` extension:
  - GPU build: `cmake --build build_gpu_probe -j4`: pass.
  - Multi-rank TGV stats, `NP=8`, `TOPOLOGY=2,2,2`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9240498911065060e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - Multi-rank TGV field, `NP=8`, `TOPOLOGY=2,2,2`, `MAXSTEP=1 FEQCHKPT=1`: pass, max reconstructed `q5` L-infinity `6.4517280407017097e-12`.
  - Y-z combined regression stats, `NP=4`, `TOPOLOGY=1,2,2`, `MAXSTEP=2 FEQCHKPT=99`: pass, max `kenergy` diff `2.9226621123257246e-14`.
- Ran higher-rank topology smoke tests on the current two-GPU workstation:
  - Scope: native `flowstate.dat` statistics only, `MAXSTEP=1`, `FEQCHKPT=99`, `LFILTER=t`, `DIFFTERM=t`. These runs exercise halo/topology composition under oversubscription; they are not performance tests and do not replace future one-rank-per-GPU validation on real multi-card hardware.
  - `NP=16`, `TOPOLOGY=4,2,2`: pass, max `kenergy` diff `1.1227130336521896e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - `NP=16`, `TOPOLOGY=2,4,2`: pass, max `kenergy` diff `1.1227130336521896e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - `NP=16`, `TOPOLOGY=2,2,4`: pass, max `kenergy` diff `1.1227130336521896e-14`, max `enstophy` diff `8.3266726846886741e-15`, max `dissipation` diff `4.2392304944183223e-17`.
  - `NP=32`, `TOPOLOGY=4,4,2`: pass, max `kenergy` diff `1.1241008124329710e-14`, max `enstophy` diff `8.2711615334574162e-15`, max `dissipation` diff `4.2392304944183223e-17`.
