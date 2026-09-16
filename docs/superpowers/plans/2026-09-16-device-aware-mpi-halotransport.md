# Device-Aware MPI HaloTransport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task. Do not use subagents or create another worktree. Work in the current `feature/gpu_dev` checkout.

**Goal:** Add an opt-in, fail-closed device-aware MPI HaloTransport backend that removes host staging while preserving every existing ASTR halo semantic and the host-staged correctness fallback.

**Architecture:** Solver code continues to call semantic solution, filter, diffusion, and generic-field halo facades. Device packing, halo width, interface averaging, tags, physical endpoints, and primitive refresh remain in `halo_exchange_gpu.cuf`; only contiguous device-buffer transport is delegated to a backend. The first implementation is CUDA-aware MPI, but the public mode is named `device-aware` so a future HIP/DCU adapter can implement the same contract through `ISO_C_BINDING` without changing solver semantics.

**Tech Stack:** CUDA Fortran, NVHPC, Open MPI/HPC-X, UCX, CMake, Python `unittest`, Compute Sanitizer, Nsight Systems, existing ASTR GPU validation drivers.

## Global Constraints

- Build only through `/home/dell/workspace/astr_gpu/CMakeLists.txt`.
- Keep FP64 state, MPI payloads, halo widths, tags, field ordering, interface averaging, and post-exchange primitive refresh unchanged.
- Keep explicit post-kernel synchronization for the first device-aware implementation.
- Keep `pageable`, `pinned`, `pinned-overlap`, and `pinned-pipeline` available. Device-aware MPI is opt-in and must not become the only backend.
- Do not trust `MPIX_Query_cuda_support()` as the sole admission check. Representative large-message exact-payload tests are mandatory.
- All ranks must select the same backend collectively before the first exchange. No rank may fall back after MPI requests have been posted.
- A failed probe or numerical gate stops downstream performance testing.
- Local dual-GPU results establish implementation correctness only. A800 and multi-node results establish performance and scale-out behavior.
- Do not change CPU halo routines in `src/parallel.F90`.
- Do not commit or push unless separately requested.

## Execution Status (2026-09-16)

The blocking-correctness device-aware backend is implemented and admitted on
the Zhongke A800 single-node HPC-X stack as an explicit opt-in. Solution,
full/scalar filter, FP64 diffusion, mixed-storage diffusion, shock-sensor,
sponge, species, and generic-field facades all exchange their existing packed
FP64 device buffers without host staging. Existing host backends remain
available and the runtime capability check is fail-closed.

The A800 payload qualification job `460370` passed exact large-message,
protocol, and Compute Sanitizer gates with `cuda_ipc/cuda`. Solver jobs `460439`
and `460441` passed `128^3` periodic TGV with full filter and diffusion for all
NP=2 slab and NP=4 plane topologies. Five-step pinned/device-aware diagnostics
were bitwise identical, every tested topology recorded CUDA IPC, sanitizer
reported zero errors, and no field HDF5 was written. The local no-IPC
`cuda_copy` fallback passed the same x/y/z numerical and sanitizer gates.

This is an A800 single-node correctness and memory-safety admission, not the
final performance promotion. The backend remains
`ASTR_GPU_HALO_TRANSPORT=device-aware` opt-in until the repeated `256^3/512^3`
timing matrix and broader non-periodic case matrix pass. Multi-node CUDA-aware
MPI and the HIP/DCU adapter remain pending.

---

## Scope And Stop Gates

The work is divided into three independently reviewable stages:

1. **Local qualification and transport core:** prove that the selected local MPI configuration handles the actual ASTR device-buffer sizes exactly.
2. **Local solver integration:** add device-aware solution, filter, diffusion, and generic-field halo paths and close the full local correctness matrix.
3. **A800 admission:** compare device-aware MPI against the existing `pinned-pipeline` backend under the same executable, inputs, node, topology, and repeat policy.

Stop after Stage 1 if no MPI configuration passes both representative payload tests and Compute Sanitizer. In that case retain the probe evidence and do not install a selectable solver backend.

## File Map

### Create

- `src_gpu/device_mpi_transport_gpu.cuf`: CUDA-aware MPI exchange state and device-buffer request lifecycle only.
- `tests/gpu_validation/test_device_aware_halo_contract.py`: static contracts for backend selection, device buffers, synchronization, and fallback preservation.
- `tests/gpu_validation/run_cuda_aware_mpi_qualification.sh`: local two-GPU payload, sanitizer, and transport-configuration gate.
- `tests/gpu_validation/summarize_cuda_aware_mpi_qualification.py`: fail-closed machine-readable qualification summary.
- `tests/gpu_validation/run_tgv_device_aware_field_compare.sh`: CPU-independent pinned versus device-aware field/statistics comparison.
- `tests/gpu_validation/run_tgv_device_aware_benchmark.sh`: interleaved local complete-RK benchmark.
- `tests/gpu_validation/run_zhongke_a800_device_aware_admission.sbatch`: A800 preflight, correctness, strong-scaling, and phase-timing admission job.

### Modify

- `src/CMakeLists.txt`: compile the CUDA device transport module and retain the excluded qualification probe.
- `src_gpu/halo_transport_gpu.cuf`: parse `device-aware`, perform collective mode agreement, and expose backend state without changing host-staged routines.
- `src_gpu/halo_exchange_gpu.cuf`: dispatch already-packed device buffers to the selected transport while preserving semantic handling.
- `tests/gpu_validation/halo_cuda_aware_probe.cuf`: cover actual solution/filter/diffusion field counts and both blocking and nonblocking request paths.
- `tests/gpu_validation/README.md`: record exact local and A800 reproduction commands and evidence boundaries.
- `documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md`: promote device-aware MPI from a generic L3 idea to the qualified optional-backend contract.
- `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`: record only completed gates and measured results.
- `documents/GPU_VALIDATION_MATRIX.md`: add separate qualification, correctness, sanitizer, and performance rows.

---

### Task 1: Freeze The Device-Buffer Qualification Contract

**Files:**
- Modify: `tests/gpu_validation/halo_cuda_aware_probe.cuf`
- Create: `tests/gpu_validation/test_device_aware_halo_contract.py`
- Create: `tests/gpu_validation/run_cuda_aware_mpi_qualification.sh`
- Create: `tests/gpu_validation/summarize_cuda_aware_mpi_qualification.py`

**Interfaces:**
- Consumes: current `halo_cuda_aware_probe` exact-payload logic and private tags `21001:21006`.
- Produces: `qualification.json` with `payload_status`, `sanitizer_status`, `mpi_stack`, `transport`, `message_bytes`, and `qualified`.

- [x] **Step 1: Add a failing source contract.**

Require the probe to test `nvar=[1,3,5,6,9]`, `width=[hm,hm+1]=[5,6]`, periodic peers, `MPI_PROC_NULL`, `MPI_Sendrecv`, and paired `MPI_Irecv/MPI_Isend/MPI_Waitall`. Require the qualification script to include 30, 45, 60, and 90 MiB payload classes.

Run:

```bash
python3 -m unittest tests/gpu_validation/test_device_aware_halo_contract.py -v
```

Expected: FAIL because the current probe omits 5/9-component and nonblocking coverage.

- [x] **Step 2: Extend the probe without adding solver behavior.**

Keep the receive sentinel at `-999.d0`, explicit mismatch indices, CUDA synchronization around MPI, and collective abort on any mismatch. Log the MPI capability query as diagnostic metadata but do not convert a return value of 1 into a pass.

- [x] **Step 3: Implement the qualification driver.**

The driver must run each candidate transport in a fresh `mpirun` process, bind rank 0/1 to GPU 0/1, and save command, environment, executable hash, payload result, and sanitizer result. Candidate configurations are:

```text
UCX CUDA IPC enabled
UCX CUDA copy with IPC disabled
OB1 CUDA shared-memory transport, only if the installed MPI exposes it
```

The summary sets `qualified=true` only when one identical configuration passes every payload size and has zero Compute Sanitizer invalid-access errors on both ranks.

- [x] **Step 4: Build and run the local admission gate.**

Run:

```bash
cmake --build /home/dell/workspace/astr_gpu/build_gpu_probe \
  --target halo_cuda_aware_probe -j2
bash /home/dell/workspace/astr_gpu/tests/gpu_validation/run_cuda_aware_mpi_qualification.sh
```

Expected: a deterministic PASS or STOP result. A payload mismatch, unchanged sentinel, MPI error, CUDA error, or sanitizer error is a STOP result, not a warning.

**Execution result (2026-09-16): local IPC rejected; A800 IPC admitted.**

The RTX 4000 Ada workstation did not qualify UCX CUDA IPC, but its explicit
`UCX_TLS=self,sm,cuda_copy` path passed exact payload, solver equivalence, and
sanitizer checks. This local limitation does not admit IPC. On the target A800
HPC-X 2.22.1/UCX 1.18.0 stack, job `460370` passed both IPC and no-IPC payload
matrices, protocol proof, and the narrow-suppression sanitizer gate. The A800
result unblocked Tasks 2-4 for that specific stack.

---

### Task 2: Add A Backend-Neutral Device Transport Core

**Files:**
- Create: `src_gpu/device_mpi_transport_gpu.cuf`
- Modify: `src_gpu/halo_transport_gpu.cuf`
- Modify: `src/CMakeLists.txt`
- Test: `tests/gpu_validation/test_device_aware_halo_contract.py`

**Interfaces:**
- Consumes: contiguous FP64 device send/receive buffers, peer ranks, element count, two tags, and `MPI_COMM_WORLD`.
- Produces: `exchange_device_pair()` and `device_aware_transport_enabled`.

- [x] **Step 1: Add failing selector and interface tests.**

Require the accepted runtime value to be exactly `device-aware`. Require collective agreement across ranks and a startup log containing backend, MPI library version, and selected transport profile. Reject `cuda-aware` as a public mode name.

- [x] **Step 2: Implement a blocking-correctness device pair.**

The initial interface is:

```fortran
subroutine exchange_device_pair(send_low,send_high,recv_high,recv_low, &
                                count,low,high,tag_first,tag_second)
  real(8),device,contiguous,intent(in) :: send_low(:,:,:,:),send_high(:,:,:,:)
  real(8),device,contiguous,intent(inout) :: recv_high(:,:,:,:),recv_low(:,:,:,:)
  integer,intent(in) :: count,low,high,tag_first,tag_second
end subroutine exchange_device_pair
```

Post both receives, post both sends, and complete all four requests. Validate all MPI return codes. Do not add stream-aware MPI extensions in this task.

- [x] **Step 3: Extend transport selection without disturbing host paths.**

`begin_transport_setup()` must map `device-aware` to a distinct mode. It must not register host buffers or allocate pipeline CUDA events for this mode. Existing four host-staged mode values and fallback behavior remain unchanged.

- [x] **Step 4: Build the solver and transport tests.**

Run:

```bash
cmake --build /home/dell/workspace/astr_gpu/build_gpu_probe \
  --target astr halo_transport_test halo_transport_setup_test -j2
python3 -m unittest tests/gpu_validation/test_device_aware_halo_contract.py -v
```

Expected: build PASS and all transport contracts PASS.

---

### Task 3: Integrate Solution Halo Without Host Staging

**Files:**
- Modify: `src_gpu/halo_exchange_gpu.cuf`
- Test: `tests/gpu_validation/test_device_aware_halo_contract.py`
- Create: `tests/gpu_validation/run_tgv_device_aware_field_compare.sh`

**Interfaces:**
- Consumes: `exchange_device_pair()` and existing `x/y/z_send_*_d`, `x/y/z_recv_*_d` buffers.
- Produces: device-aware `exchange_solution_halo_gpu()` for x, y, and z decomposition.

- [x] **Step 1: Add failing x/y/z semantic contracts.**

Require `hm+1`, five FP64 variables, tags `21001:21006`, existing pack kernels, existing unpack kernels, interface-plane averaging, and primitive-halo refresh. Require no assignment between `_d` and `_h` buffers inside the device-aware branch.

- [x] **Step 2: Implement explicit-synchronization solution exchange.**

For each active MPI axis:

```text
pack device buffers
explicitly synchronize and check the pack kernels
exchange device buffers through MPI
launch the existing unpack kernel
explicitly synchronize and check the unpack kernel
refresh primitive halos using the existing semantic path
```

Do not modify local periodic or physical-boundary handling.

- [x] **Step 3: Compare pinned and device-aware fields locally.**

Run TGV with NP=2 for `2x1x1`, `1x2x1`, and `1x1x2`, then NP=4/8 oversubscription correctness for `2x2x1` and `2x2x2`. Compare every conservative field and existing TGV statistic with the pinned FP64 reference. Require the existing FP64 tolerances and zero topology/tag mismatches.

- [x] **Step 4: Run Compute Sanitizer on x/y/z NP=2.**

Expected: two rank reports per topology and `ERROR SUMMARY: 0 errors`. Any MPI-library device-pointer error blocks Task 4.

---

### Task 4: Extend Filter, Diffusion, And Generic Field Halos

**Files:**
- Modify: `src_gpu/halo_exchange_gpu.cuf`
- Test: `tests/gpu_validation/test_device_aware_halo_contract.py`
- Modify: `tests/gpu_validation/run_tgv_device_aware_field_compare.sh`

**Interfaces:**
- Consumes: validated solution device transport.
- Produces: device-aware `hm` exchanges for filter workspaces, fused nine-component diffusion flux, shock sensor, sponge, species, and generic fields.

- [x] **Step 1: Add failing family-completeness tests.**

Require explicit dispatch coverage for solution, filter solution, scalar filter, full filter work, diffusion FP64, diffusion FP32-storage pack/unpack, shock sensor, sponge, and generic fields. Mixed-precision fields continue to communicate through FP64 device transport buffers.

- [x] **Step 2: Implement filter exchanges.**

Reuse existing device pack/unpack buffers and kernels. Preserve ping-pong ordering and `hm` width. Do not send the full `q_d` or `qwork_d` allocation.

- [x] **Step 3: Implement fused diffusion and generic-field exchanges.**

Preserve the existing nine-component fused diffusion layout. Use the existing `nvar` argument for generic fields and reject counts larger than the allocated device transport capacity.

- [ ] **Step 4: Run the complete local correctness matrix.**

Cover TGV, Shu-Osher, Cartesian HBL, one CURVE case, one wall-family case, shock sensor, and chemistry-flow smoke where already supported. Compare pinned and device-aware outputs; do not use CPU/GPU agreement as a substitute for backend-to-backend equivalence.

Periodic TGV is complete locally and on A800 for NP=2 slabs and NP=4 planes.
The remaining named physical-case families are still required before claiming
case-independent production coverage.

---

### Task 5: Recover Communication And Computation Overlap

**Files:**
- Modify: `src_gpu/device_mpi_transport_gpu.cuf`
- Modify: `src_gpu/halo_exchange_gpu.cuf`
- Modify: `src_gpu/mainloop_gpu.cuf`
- Test: `tests/gpu_validation/test_device_aware_halo_contract.py`

**Interfaces:**
- Consumes: validated blocking-correctness device-aware exchange.
- Produces: nonblocking device-aware request contexts compatible with the current independent interior-work callbacks.

- [ ] **Step 1: Add request-lifecycle tests.**

Require exactly four requests per active axis, receive-before-send ordering, no nested pair on one axis, `MPI_Testall` progress, final `MPI_Waitall`, and no unpack before completion.

- [ ] **Step 2: Add nonblocking device-buffer operations.**

Keep pack readiness conservative: synchronize the communication-stream event before entering standard MPI. Do not introduce `MPIX_Stream`, CUDA Graphs, or vendor-specific stream extensions.

- [ ] **Step 3: Reuse existing independent interior work.**

Allow one eligible solution or diffusion exchange to progress while its already-validated independent interior kernel group runs. Keep explicit synchronization at the dependency boundary.

- [ ] **Step 4: Re-run Tasks 3 and 4 gates.**

Expected: identical fields/statistics and clean sanitizer reports before performance measurement.

---

### Task 6: Measure Local Benefit Without Making Production Claims

**Files:**
- Create: `tests/gpu_validation/run_tgv_device_aware_benchmark.sh`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`

**Interfaces:**
- Consumes: pinned-pipeline and device-aware backends from one frozen executable.
- Produces: five-repeat complete-RK and internal-phase timing summaries.

- [ ] **Step 1: Define the interleaved local matrix.**

Use TGV `256^3`, NP=2, topologies `2x1x1`, `1x2x1`, and `1x1x2`. Alternate pinned-pipeline and device-aware runs to reduce thermal and background-load bias. Keep full FP64 filtering, explicit synchronization, no field HDF5, and compact statistics.

- [ ] **Step 2: Run five retained repeats per backend and topology.**

Record complete-RK median, relative spread, solution halo, diffusion halo, filter, peak memory, and GPU utilization.

- [ ] **Step 3: Classify the local result.**

Use the following gate:

```text
correctness and sanitizer pass
complete-RK median improves by at least 5 percent
repeat spread is at most 3 percent
at least one exposed halo phase decreases
```

A pass is `local-pass-not-promoted`; it is not A800 or multi-node evidence.

---

### Task 7: Admit Or Reject On A800

**Files:**
- Create: `tests/gpu_validation/run_zhongke_a800_device_aware_admission.sbatch`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`
- Modify: `documents/GPU_VALIDATION_MATRIX.md`

**Interfaces:**
- Consumes: frozen source commit, executable hash, local qualification artifacts, and current A800 pinned-pipeline baseline.
- Produces: an A800-specific backend decision without changing other platforms.

- [x] **Step 1: Run an A800 compute-node preflight.**

Verify dynamic libraries, MPI/UCX configuration, four visible GPUs, rank binding, executable hash, CRLF absence, and no field HDF5. Run the large device-buffer probe before the solver matrix.

Completed by probe job `460370` and solver jobs `460439/460441` for one-node
NP=2/4. Rank-local prebinding selected devices 0-3, dependencies resolved in
the job module environment, and no field HDF5 was produced.

- [ ] **Step 2: Run correctness on NP=2 and NP=4.**

Use `2x1x1`, `1x2x1`, `1x1x2`, `2x2x1`, `2x1x2`, and `1x2x2`. Compare pinned-pipeline and device-aware fields/statistics from the same executable.

The listed topology matrix passed pinned versus device-aware statistics and
sanitizer gates. A800 full-field output was intentionally disabled after shared
filesystem HDF5 dominated the earlier jobs; local same-phase full-field gates
remain the field-level evidence.

- [ ] **Step 3: Run the performance matrix.**

Use `512^3` TGV, five repeats, NP=1/2/4, all NP=2/4 topologies, internal phase timing, and no field HDF5. Nsight Systems is additional evidence only after the platform enables CUDA tracing.

- [ ] **Step 4: Make an A800 backend decision.**

Promote device-aware MPI only when correctness, sanitizer, stability, complete-RK speed, and scaling efficiency all pass. Otherwise retain it as an experimental opt-in or reject it on that MPI stack while preserving the host-staged backend.

---

### Task 8: Prepare HIP/DCU Without Porting It Prematurely

**Files:**
- Modify: `documents/ASTR_GPU_HALOTRANSPORT_SKETCH.md`
- Modify: `docs/adr/0015-use-pluggable-halotransport-backends.md`

**Interfaces:**
- Consumes: the proven device-aware transport contract.
- Produces: a portable adapter boundary for a future HIP/DCU implementation.

- [ ] **Step 1: Freeze the backend-neutral contract.**

Document device pointer, FP64 element count, peers, tags, communicator, request lifecycle, readiness, completion, and failure semantics. Do not expose CUDA stream or event types through the semantic facade.

- [ ] **Step 2: Define the future C ABI adapter.**

The HIP/DCU adapter will accept opaque device pointers and `MPI_Fint`, convert the communicator with `MPI_Comm_f2c`, and call the vendor's GPU-aware MPI from HIP C++. CUDA Fortran pointer extraction remains confined to the CUDA adapter.

- [ ] **Step 3: Reuse the same qualification matrix.**

ROCm-aware MPI or a domestic vendor MPI must pass the same payload sizes, endpoint rules, field counts, topology matrix, numerical checks, and fallback requirements. Vendor capability strings are metadata, not acceptance evidence.

## Completion Criteria

The plan is complete only when:

- host-staged and device-aware paths coexist behind the same semantic facades;
- every solution, filter, diffusion, sensor, sponge, and generic-field exchange preserves its current width, layout, tags, and endpoint behavior;
- local large-message qualification, full backend-to-backend numerical comparison, and sanitizer gates pass;
- local performance is classified without production claims;
- A800 independently admits or rejects the backend using its own MPI stack;
- the HIP/DCU adapter boundary is documented without introducing CUDA names into solver-facing interfaces.
