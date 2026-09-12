# GPU Compact Production Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add restartable, GPU-resident compact production statistics for three-dimensional non-reacting `bl` and `swbli` cases without full-field host copies on sampling steps.

**Architecture:** A new CUDA Fortran module owns rank-local plane and wall raw-moment accumulators. `gpu_runtime` remains the only interface visible from `src/`; versioned rank sidecars are checkpointed transactionally and assembled offline into Favre means and stresses.

**Tech Stack:** CUDA Fortran/NVHPC, MPI Fortran, CMake, Python 3, NumPy, h5py, pytest, Compute Sanitizer, Nsight Systems.

## Global Constraints

- Build through `/home/dell/workspace/astr_gpu/CMakeLists.txt`; `use_gpu` remains a runtime choice.
- Keep CPU/GPU implementations separate and limit `src/` edits to facade calls and ownership gates.
- Explicitly synchronize after every new GPU kernel.
- Do not change finite differences, filters, halos, boundaries, or CPU statistics formulas.
- Do not repair the known `tu3`/`tu2` mean-flow I/O defect without separate approval.
- Admit only 3D single-species non-reacting `bl`/`swbli`, periodic homogeneous z, physical y-min wall, and static Cartesian or curvilinear extrusion.
- Unsupported configurations fail before the first sample, with no full-field host fallback.
- Sample only at `nstep>0 .and. lavg .and. mod(nstep,feqavg)==0`, once per completed step.
- Non-checkpoint samples perform no D2H/H2D transfer; restart initially requires the same topology.
- Do not stage, commit, or push unless separately requested. Commit commands below are review checkpoints only.
- Leave unrelated manuals, presentations, rendered files, and unpublished case material untouched.

---

## File Map

- Create `src_gpu/production_statistics_gpu.cuf`: capability gate, device storage, kernels, sidecar I/O, restart.
- Modify `src_gpu/gpu_runtime.cuf`: lifecycle, sample, and checkpoint facade.
- Modify `src/CMakeLists.txt`: module and CUDA probe targets.
- Modify `src/initialisation.F90`: suppress CPU full-field mean restore under compact ownership.
- Modify `src/mainloop.F90`: sample and checkpoint transaction calls.
- Create `scripts/gpu_statistics/compact_statistics.py`: schema, validation, merge, derivation.
- Create `scripts/gpu_statistics/assemble_compact_statistics.py`: CLI and compact HDF5 output.
- Create `tests/gpu_validation/compact_statistics_probe.cuf`: manufactured CUDA checks.
- Create `tests/gpu_validation/compact_statistics_host_reference.py`: independent quadrature oracle.
- Create `tests/gpu_validation/test_compact_statistics.py`: schema/math/merge tests.
- Create `tests/gpu_validation/test_compact_statistics_gpu_contract.py`: source contracts.
- Create `tests/gpu_validation/run_compact_statistics_probe.sh`, `run_compact_statistics_matrix.sh`, and `run_compact_statistics_profile.sh`.
- Modify `tests/gpu_validation/README.md` and `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`.

---

### Task 1: Freeze the sidecar schema and raw-moment mathematics

**Files:**
- Create: `scripts/gpu_statistics/compact_statistics.py`
- Create: `tests/gpu_validation/test_compact_statistics.py`

**Interfaces:**
- Produces: `CompactHeader`, `read_rank_file(path)`, `write_rank_file(path,state)`, `derive_statistics(state)`, `merge_rank_states(states)`.
- Binary format: little-endian stream, magic `ASTRCST1`, version 1, fixed-width integer/real fields, Fortran-order arrays.

- [x] **Step 1: Write failing schema and derivation tests**

Construct synthetic nonuniform-z slabs. Assert `rho_mean=S_rho/(Ns*Lz)`, Favre means `S_rho_f/S_rho`, and stresses `S_rho_uiuj/S_rho-ui_tilde*uj_tilde`. Test concatenated windows and reject bad magic/version/endianness, zero measure/density, missing or duplicate ranks, and incompatible generations/topologies.

- [x] **Step 2: Verify the tests fail**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py`

Expected: import failure because the schema module does not exist.

- [x] **Step 3: Implement the exact stream layout**

Use this order in Python and Fortran:

```text
char[8] magic; int32 version,endian_marker,real_bytes,has_wall
int32 global_dims[3],topology[3],rank,rank_coords[3],offsets[3],local_dims[3]
int64 checkpoint_step; float64 checkpoint_time
int64 sample_count,sampling_start_step; float64 sampling_start_time
float64 plane_measure[ni,nj],plane_sum[ni,nj,12]
float64 wall_measure[ni],wall_sum[ni,3] when has_wall=1
```

Return the six raw Favre stresses, six means, wall quantities, measures, metadata, and untouched raw moments.

- [x] **Step 4: Run the unit tests**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py`

Expected: all schema, merge, and derivation tests pass.

- [x] **Step 5: Review checkpoint**

Proposed commit: `test(gpu): define compact statistics schema`.

---

### Task 2: Implement offline deterministic assembly

**Files:**
- Create: `scripts/gpu_statistics/assemble_compact_statistics.py`
- Modify: `tests/gpu_validation/test_compact_statistics.py`

**Interfaces:**
- Consumes Task 1 readers/merge/derivation.
- Produces `assemble(input_dir,output_h5,metadata_path)` and CLI options `--input-dir`, `--output-h5`, `--metadata`.

- [x] **Step 1: Add failing NP=4 assembly tests**

Build synthetic `2x2x1` files. Verify z slabs sum, equal x/y overlap nodes collapse to one owner, an overlap perturbation above `1e-10` fails, and a missing rank fails.

- [x] **Step 2: Verify the CLI tests fail**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py -k assemble`

Expected: assembler import failure.

- [x] **Step 3: Implement assembly and HDF5 output**

Map nodes with `(ig0+i,jg0+j)`; compare duplicates at `atol=rtol=1e-10`; retain the lowest rank owner. Write `/metadata`, `/measure`, `/raw`, `/mean`, `/stress`, and `/wall` groups plus an ASCII report containing dimensions, topology, sample window, files, and measure/density extrema.

- [x] **Step 4: Run tests and CLI help**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py && python3 scripts/gpu_statistics/assemble_compact_statistics.py --help`

Expected: tests pass and help exits zero.

- [x] **Step 5: Review checkpoint**

Proposed commit: `feat(gpu): add compact statistics assembler`.

---

### Task 3: Add capability ownership and physical line measures

**Files:**
- Create: `src_gpu/production_statistics_gpu.cuf`
- Create: `tests/gpu_validation/compact_statistics_probe.cuf`
- Create: `tests/gpu_validation/run_compact_statistics_probe.sh`
- Create: `tests/gpu_validation/test_compact_statistics_gpu_contract.py`
- Modify: `src/CMakeLists.txt`

**Interfaces:**
- Produces `compact_statistics_requested()`, `initialize_compact_statistics_gpu()`, `release_compact_statistics_gpu()`.
- Owns `plane_measure_d(0:im,0:jm)`, `plane_sum_d(0:im,0:jm,1:12)`; wall arrays exist only on `jrk==0`.

- [x] **Step 1: Write failing source contracts and CUDA measure checks**

Check all capability conditions, `k=1:km` segment ownership, y-wall allocation, and `sync_after_kernel`. Probe uniform, nonuniform, and warped static extrusions.

- [x] **Step 2: Verify failure**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics_gpu_contract.py && bash tests/gpu_validation/run_compact_statistics_probe.sh`

Expected: missing module/target failures.

- [x] **Step 3: Implement allocation and measure kernel**

One `(i,j)` thread sums `sqrt(dx*dx+dy*dy+dz*dz)` over local segments `k=1:km`. Use block `(32,16,1)`, synchronize explicitly, and abort collectively for invalid capability or non-positive measure.

- [x] **Step 4: Build and run**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics_gpu_contract.py && bash tests/gpu_validation/run_compact_statistics_probe.sh`

Expected: all three geometry measures pass.

- [x] **Step 5: Review checkpoint**

Proposed commit: `feat(gpu): allocate compact statistics measures`.

---

### Task 4: Accumulate 12 plane raw moments

**Files:**
- Modify: `src_gpu/production_statistics_gpu.cuf`
- Modify: `tests/gpu_validation/compact_statistics_probe.cuf`
- Create: `tests/gpu_validation/compact_statistics_host_reference.py`
- Modify: `tests/gpu_validation/test_compact_statistics.py`

**Interfaces:**
- Produces `accumulate_compact_statistics_gpu()` and `compact_statistics_sample_count()`.
- Consumes resident `rho_d`, `vel_d`, `prs_d`, `tmp_d`, `x_d`.

- [x] **Step 1: Add failing manufactured tests for all 12 channels**

Use analytic z-varying fields and an independent host trapezoid. Check measure, sample count, two-window concatenation, and every raw moment at `1e-12`.

- [x] **Step 2: Verify the probe fails**

Run: `bash tests/gpu_validation/run_compact_statistics_probe.sh`

Expected: plane-moment assertions fail.

- [x] **Step 3: Implement one-output-thread reduction**

For each segment trapezoid the vector `[rho,rho*u,rho*v,rho*w,rho*T,p,rho*u*u,rho*v*v,rho*w*w,rho*u*v,rho*u*w,rho*v*w]`. Increment the host `integer(int64)` sample count only after explicit kernel synchronization; record sampling-start metadata on the first sample.

- [x] **Step 4: Run probe and Python oracle**

Run: `bash tests/gpu_validation/run_compact_statistics_probe.sh && python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py`

Expected: all plane checks pass.

- [x] **Step 5: Review checkpoint**

Proposed commit: `feat(gpu): accumulate compact plane moments`.

---

### Task 5: Add geometry-projected wall statistics

**Files:**
- Modify: `src_gpu/production_statistics_gpu.cuf`
- Modify: `tests/gpu_validation/compact_statistics_probe.cuf`
- Modify: `tests/gpu_validation/compact_statistics_host_reference.py`

**Interfaces:**
- Owns `wall_measure_d(0:im)`, `wall_sum_d(0:im,1:3)`.
- Consumes `bnorm_j0_d`, geometry, primitive/gradient fields, and perfect-gas transport constants.

- [x] **Step 1: Add failing Cartesian and warped-wall tests**

Use linear velocity/temperature fields. Compare pressure, `t_s dot tau dot n`, `kappa*grad(T) dot n`, signs, and segment quadrature with an independent host calculation.

- [x] **Step 2: Verify failure**

Run: `bash tests/gpu_validation/run_compact_statistics_probe.sh`

Expected: wall assertions fail.

- [x] **Step 3: Implement projected wall kernel**

Construct the increasing-i tangent with forward/backward global-end differences and centered interior differences. Project against `bnorm_j0_d`, normalize, form the Newtonian stress from `dvel_d`, reproduce the existing Sutherland/nondimensional conductivity branches, and trapezoid over z. Stop on degenerate tangents or measures and synchronize after launch.

- [x] **Step 4: Run manufactured wall validation**

Run: `bash tests/gpu_validation/run_compact_statistics_probe.sh`

Expected: Cartesian and warped values pass at `atol=rtol=1e-12`.

- [x] **Step 5: Review checkpoint**

Proposed commit: `feat(gpu): add projected wall statistics`.

---

### Task 6: Integrate lifecycle and suppress CPU mean arrays

**Files:**
- Modify: `src_gpu/gpu_runtime.cuf`
- Modify: `src/initialisation.F90`
- Modify: `src/mainloop.F90`
- Modify: `tests/gpu_validation/test_compact_statistics_gpu_contract.py`

**Interfaces:**
- Adds facade routines `gpu_compact_statistics_requested()` and `gpu_accumulate_compact_statistics()`.
- Initializes/releases through `gpu_after_flowinit` and `gpu_before_finalize`.

- [x] **Step 1: Write failing phase/ownership contracts**

Require compact ownership to suppress `readmeanflow` and CPU `meanflowcal`. Require sampling after `gpu_prepare_rkfirst_stats` and before `gpu_restore_stats_snapshot`, with no `copy_flow_from_gpu`.

- [x] **Step 2: Verify failure**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics_gpu_contract.py`

Expected: facade/phase checks fail.

- [x] **Step 3: Add minimal CPU-facing integration**

In `flowinit`, call `readmeanflow` only when compact GPU ownership is false. In the GPU main-loop branch, call `gpu_accumulate_compact_statistics` immediately after existing GPU flow statistics and before q-snapshot restoration. Keep the sample and duplicate-step guards inside the new module.

- [x] **Step 4: Build CPU and GPU**

Run: `cmake --build build_cpu_probe -j && cmake --build build_gpu_probe -j && python3 -m pytest -q tests/gpu_validation/test_compact_statistics_gpu_contract.py`

Expected: both builds pass; `lavg=f` creates no compact state.

- [x] **Step 5: Review checkpoint**

Proposed commit: `feat(gpu): integrate compact statistics lifecycle`.

---

### Task 7: Add transactional checkpoint and same-topology restart

**Files:**
- Modify: `src_gpu/production_statistics_gpu.cuf`
- Modify: `src_gpu/gpu_runtime.cuf`
- Modify: `src/mainloop.F90`
- Modify: both compact-statistics test files.

**Interfaces:**
- Produces `prepare_compact_statistics_checkpoint()`, `commit_compact_statistics_checkpoint()`, `restore_compact_statistics_checkpoint()`.
- Uses `outdat/compact_stats.rankXXXXXXXX.bin.tmp` and published `.bin`.

- [x] **Step 1: Add failing restart/transaction tests**

Test exact raw-state continuation and reject missing sidecars with `nsamples>0`, step/time/topology/offset mismatch, truncation, and stale temporary files.

- [x] **Step 2: Verify failure**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py -k 'restart or transaction'`

Expected: checkpoint routines are absent.

- [x] **Step 3: Implement prepare/write/commit ordering**

Prepare synchronizes, copies compact arrays only, writes/flushed/closes `.tmp`, then barriers. Main-loop ordering is prepare, existing `writechkpt()`, commit. Commit rotates the old sidecar to `bakup/`, atomically renames through checked libc `rename`, then barriers. Any rank failure aborts all ranks.

- [x] **Step 4: Restore during GPU initialization**

When `nsamples>0`, require and validate the sidecar against flow checkpoint metadata and current decomposition, upload sums, and synchronize. When fresh, initialize zero state and refuse a conflicting published generation.

- [x] **Step 5: Run restart unit tests**

Run: `python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py -k 'restart or transaction'`

Expected: synthetic uninterrupted and restored raw sums, measures, and sample metadata agree exactly; every incompatible-generation case is rejected.

- [x] **Step 6: Review checkpoint**

Proposed commit: `feat(gpu): checkpoint compact statistics`.

---

### Task 8: Validate required solver/topology matrix

**Files:**
- Create: `tests/gpu_validation/run_compact_statistics_matrix.sh`
- Modify: `tests/gpu_validation/compact_statistics_host_reference.py`
- Modify: `src/validation_io.F90`
- Modify: `src/mainloop.F90`
- Modify: `tests/gpu_validation/README.md`

**Interfaces:**
- Case selectors: `static-np1`, `dynamic-np1`, `curve-filter-np4`, `z-slab-np2`, `xyz-np8`, `restart-np1`, `all`.

- [x] **Step 1: Complete independent host extraction**

Write validation-only CPU snapshots immediately after sampled `meanflowcal`, then
read their primitive fields, gradients, geometry, and wall normals and independently
apply the approved physical-segment quadrature. Do not import production quadrature
code. The snapshot path is disabled unless `ASTR_VALIDATION_COMPACT_PREFIX` is set.

- [x] **Step 2: Implement the matrix driver**

Cover NP=1 static BL, NP=1 dynamic BL, NP=4 `2x2x1` filtered CURVE dynamic BL, NP=2 `1x1x2`, NP=8 `2x2x2` smoke, and NP=1 restart.

- [x] **Step 3: Run low-cost cases**

Run: `bash tests/gpu_validation/run_compact_statistics_matrix.sh --case static-np1 && bash tests/gpu_validation/run_compact_statistics_matrix.sh --case dynamic-np1 && bash tests/gpu_validation/run_compact_statistics_matrix.sh --case z-slab-np2`

Expected: raw moments, derived fields, wall data, measures, and metadata pass `atol=rtol=1e-10`.

- [x] **Step 4: Run CURVE/MPI/restart cases**

Run: `bash tests/gpu_validation/run_compact_statistics_matrix.sh --case curve-filter-np4 && bash tests/gpu_validation/run_compact_statistics_matrix.sh --case xyz-np8 && bash tests/gpu_validation/run_compact_statistics_matrix.sh --case restart-np1`

Expected: all pass; NP=8 is a communication/assembly smoke test, not a scalability claim.

- [x] **Step 5: Review checkpoint**

Proposed commit: `test(gpu): validate compact production statistics`.

---

### Task 9: Close safety, residency, performance, and documentation gates

**Files:**
- Create: `tests/gpu_validation/run_compact_statistics_profile.sh`
- Modify: `tests/gpu_validation/README.md`
- Modify: `documents/ASTR_GPU_CURRENT_STATUS_AND_NEXT_TARGETS.md`

**Interfaces:**
- Emits Compute Sanitizer log, Nsight Systems report/statistics, and matched timing summary under caller-supplied `RESULT_DIR`.

- [x] **Step 1: Implement profile-driver safeguards**

Refuse nonempty output directories. Record executable/git hashes, GPU model, topology, grid, `feqavg`, warmup/measured steps, and median complete-RK time.

- [x] **Step 2: Run memory safety**

Run: `bash tests/gpu_validation/run_compact_statistics_profile.sh --mode memcheck`

Expected: zero CUDA API, bounds, race, or leak errors.

- [x] **Step 3: Audit residency**

Run: `bash tests/gpu_validation/run_compact_statistics_profile.sh --mode residency`

Expected: no full-field transfer on non-checkpoint samples; checkpoint traffic contains compact state only.

- [x] **Step 4: Measure overhead**

Run: `bash tests/gpu_validation/run_compact_statistics_profile.sh --mode performance`

Expected: at `feqavg=1`, matched statistics-on median complete-RK time is at most 10% above statistics-off. A larger cost triggers profiling, not relaxed numerical tolerances.

- [x] **Step 5: Update status from evidence**

Mark D4 complete only after Tasks 1-9 pass. Keep D3 open for long precursor convergence and probe time-series decorrelation; numerical equivalence does not prove developed turbulence.

- [x] **Step 6: Run final checks**

Run:

```bash
python3 -m pytest -q tests/gpu_validation/test_compact_statistics.py tests/gpu_validation/test_compact_statistics_gpu_contract.py
bash -n tests/gpu_validation/run_compact_statistics_probe.sh tests/gpu_validation/run_compact_statistics_matrix.sh tests/gpu_validation/run_compact_statistics_profile.sh
git diff --check
git status --short
```

Expected: tests pass, scripts parse, no whitespace errors, and only intended D2/D4 plus pre-existing unrelated files are listed.

- [x] **Step 7: Review checkpoint**

Proposed commit: `feat(gpu): add compact production statistics`.

---

## Execution Order and Stop Conditions

Execute Tasks 1-7 serially. Start solver matrices only after the standalone CUDA probe, both builds, and restart units pass; start profiling only after numerical equivalence passes.

Stop and report immediately if the independent oracle disagrees with the approved quadrature, an existing CPU formula must change, a case violates capability assumptions, checkpoint generation identity cannot be proved, a non-checkpoint sample causes full-field transfer, or any required comparison exceeds `atol=rtol=1e-10`.

Build success, a smoke run, or CPU/GPU agreement alone does not establish physical convergence of the turbulent precursor.
