# ASTR GPU HaloTransport Sketch

This document defines the current GPU halo exchange contract and the target transport abstraction for future NVIDIA CUDA, HIP/DCU, and multi-node work.

## 1. Current Implementation

Current CUDA Fortran location:

```text
src_gpu/qswap_gpu.cuf
src_gpu/halo_exchange_gpu.cuf
```

Current transport level:

```text
L0 host_staged_blocking
```

The current path packs halo data on device, copies only halo buffers to host, performs blocking `MPI_SENDRECV`, copies received halo buffers back to device, then unpacks on device. It intentionally keeps every kernel followed by explicit synchronization.

## 2. Semantic Layer

HaloTransport must preserve solver semantics independently of the transport backend.

| Semantic operation | Current routine | Width | Field class | Notes |
|---|---|---|---|---|
| Solution qswap | `exchange_solution_halo_gpu()` | `hm+1` for MPI solution faces | `q_d` | qswap-compatible exchange followed by primitive halo refresh |
| Local periodic qswap | `qswap_local_x/y/z_gpu()` | local periodic faces | `q_d`, primitive halo refresh | used when a direction has no MPI split and is homogeneous |
| Filter solution halo | `exchange_solution_filter_halo_gpu()` | `hm` | `q_d` | dataswap-compatible filter halo, currently x plus local homogeneous handling |
| Filter y work halo | `exchange_filter_y_work_halo_gpu()` | `hm` | `qwork_d` | required by ping-pong y filtering |
| Filter z halo | `exchange_filter_z_halo_gpu()` | `hm` | `q_d` or filter work path | required by z filtering |
| Diffusion field halo | `exchange_diffusion_flux_halo_gpu()` | `hm` | `sigma_d`, `qflux_d` | dataswap-compatible generic field halo |
| Generic field halo | `exchange_field_halo_gpu(field,nvar)` | `hm` | device field with `nvar` components | future extension point for additional fields |

The semantic layer owns:

- which fields are exchanged;
- halo width;
- qswap versus dataswap behavior;
- local periodic fallback;
- post-exchange primitive refresh where required;
- private GPU tag range.

The semantic layer must not own:

- host-staged versus device-aware transport choice;
- pinned versus pageable host buffer choice;
- blocking versus nonblocking scheduling;
- CUDA versus HIP implementation details.

## 3. Transport Backends

Target backend levels:

| Level | Backend | Purpose | Required before use |
|---|---|---|---|
| L0 | host-staged blocking | portable correctness reference | current implementation |
| L1 | host-staged nonblocking | reduce blocking wait and prepare overlap | request lifecycle and completion points |
| L2 | pinned host-staged | reduce host/device staging overhead | pinned buffer ownership and reuse policy |
| L3 | CUDA-aware or HIP-aware MPI | remove host staging where supported | device-pointer MPI validation and fallback path |
| L4 | topology-aware multi-node transport | production scale-out | node/rank/GPU binding and multi-node profile evidence |

L0 must remain available after higher transport levels are added. It is the portable correctness baseline for CUDA, HIP/DCU, and debugging.

## 4. Proposed Interface Shape

Near-term Fortran-facing facade:

```fortran
call gpu_halo_exchange_solution()
call gpu_halo_exchange_filter_solution()
call gpu_halo_exchange_filter_work(direction)
call gpu_halo_exchange_diffusion_flux()
call gpu_halo_exchange_field(field_id, nvar, semantic)
```

Backend-internal transport hook:

```fortran
call halo_transport_exchange(direction, width, nvar, send_left, send_right, recv_left, recv_right, semantic)
```

The exact Fortran API can evolve, but the split must remain:

```text
solver/mainloop -> semantic exchange facade -> transport backend
```

No solver kernel should directly call MPI or know whether the transport is L0, L1, L2, L3, or L4.

## 5. Current Validation Evidence

Validated correctness evidence includes:

- `2x1x1`, `1x2x1`, `1x1x2`;
- `2x2x1`, `2x1x2`, `1x2x2`;
- `2x2x2`;
- selected high-rank oversubscription smoke tests.

The latest full core matrix rerun covered:

- stats and field: `NP=2 TOPOLOGY=2,1,1`;
- stats and field: `NP=2 TOPOLOGY=1,2,1`;
- stats and field: `NP=2 TOPOLOGY=1,1,2`;
- stats and field: `NP=4 TOPOLOGY=2,2,1`;
- stats and field: `NP=4 TOPOLOGY=2,1,2`;
- stats and field: `NP=4 TOPOLOGY=1,2,2`;
- stats and field: `NP=8 TOPOLOGY=2,2,2`.

This validates that the current L0 semantic path still runs after the reusable matrix driver was added. Higher-rank oversubscription smoke tests remain useful but are not performance evidence.

## 6. Risks

- Mixing semantic rules with transport optimization will make HIP/DCU migration harder.
- Treating two-GPU oversubscription as performance evidence is invalid; it is correctness smoke only.
- CUDA-aware MPI must not become the only backend, because future DCU/HIP and conservative cluster deployments may need L0/L1 fallback.
- Interface-plane averaging and `hm` versus `hm+1` width must remain explicit; hiding it behind a generic byte transport is unsafe.

## 7. Acceptance Criteria

HaloTransport abstraction is ready for implementation when:

- every exchange call is classified by semantic operation and width;
- L0 can be selected explicitly as the default backend;
- a full core topology matrix passes after the interface split;
- Nsight profiles distinguish host-staged halo transfers from whole-field output-boundary transfers;
- the interface names are backend-neutral and do not encode CUDA-specific assumptions.
