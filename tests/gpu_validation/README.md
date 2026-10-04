# GPU Validation

## Output Redesign Foundations

`output_config.F90` implements a candidate configuration parser and rank-0
read/typed-broadcast interface. `ASTR_OUTPUT_CONFIG=<file>` now opts into the
restricted completed-step checkpoint runtime. Its admitted cases and pending
gates are recorded in the output redesign plan, sections 10.9 through 10.18.
Without this variable,
existing runs retain the existing controller and checkpoint paths.
See `documents/ASTR_OUTPUT_RESTART_REDESIGN_PLAN.md`, sections 6.2 and 8.1.

Build through the root CMake project with `BUILD_TESTING=ON`:

```bash
cmake -S . -B <build> <existing compiler and dependency options>
cmake --build <build> --target output_config_probe output_config_collective_probe astr
ASTR_OUTPUT_CONFIG_PROBE=<absolute-build>/bin/output_config_probe \
ASTR_OUTPUT_COLLECTIVE_PROBE=<absolute-build>/bin/output_config_collective_probe \
ASTR_OUTPUT_MPIEXEC=<matching-mpiexec> \
python3 -m pytest -q tests/gpu_validation/test_output_config.py
```

The parser requires the four namelist groups in order, with each group name
and terminator on its own line. Tests cover valid configurations, LF/CRLF,
invalid values, unknown/missing/duplicate groups, slice deduplication and global
index bounds, unchanged options on failure, and NP=2 broadcasts/rejections.
These checks do not exercise HDF5 writes, retention, restart, or resource peaks.

`run_complete_step_clock.py` runs isolated 16^3 TGV CPU/GPU, NP=1/2 cases using
a CUDA-capable development executable. It checks fixed timesteps and live
controller changes from 0.001 to 0.002 to 0.0005, recording the actual accepted
changes rather than assuming when the input is consumed. Its PTY driver uses
an atomic controller replacement and stops the launched process group on error.

```bash
python3 tests/gpu_validation/run_complete_step_clock.py \
  --executable <absolute-build>/bin/astr --mpiexec <matching-mpiexec> \
  --output <new-test-directory>
```

Each completed-state clock is checked against accumulated used timesteps and
every rank's in-situ sample header. The existing inclusive loop executes 12
advances for `maxstep=11`; the test does not change that convention. This is a
clock regression, not a new-format restart or reacting-flow validation.

### Real Completed-Step TGV Restart

```bash
python3 tests/gpu_validation/run_output_restart_validation.py \
  --executable <absolute-build>/bin/astr --mpiexec <matching-mpiexec> \
  --output <new-test-directory> --backends cpu gpu --ranks 1 2
```

This bounded 16^3 test compares 12 complete advances with 5+7 restart advances.
It checks final conserved/primitives/halo FP64 representations, control and schedule
bytes, shared geometry, per-step output-on/off samples, and TGV diagnostic records.
Use `--mode time` for physical-time scheduling (interval 0.004); default step
interval is 5. Both paths retain two batches and always save the normal final state.
The CPU NP=2 branch also rejects changed primary input and corrupted shared resources.
The source restart directory is never modified; corruptions use a separate copy.
No legacy flowfield may be produced in this new mode. Test directories must stay
below 64 MiB; controlled host buffers are capped at 64 MiB. These are local test
budgets, not production recommendations or measurements of library memory peaks.

The geometry check also verifies bitwise positive-zero padding in `rank_extras`,
using each partition's geometry-valid face ranges and periodic directions. It
does not clip small flow values or valid geometry. `--case channel --axis y`
covers wall-exterior padding alongside inter-rank metric halos. Run
`python3 -m pytest -q tests/gpu_validation/test_output_geometry_padding.py` for
the checker regression, including tiny valid values that must remain unchanged.

Use `--backends cpu --no-samples` with a CPU-only, non-testing executable: final
exact restart, control, geometry and text statistics are checked, but no claim is
made about per-step output-switch field equality in that run. The testing build
provides that separate field check. This is not AIR5, boundary or repartition
validation.

Add `--statistics` to include formal pointwise and regional Reynolds/Favre state
in the checkpoint transaction. This mode compares periodic checkpoints against
final-only checkpoints, not a fully disabled writer. Final state, accumulated
statistics and exported statistics are exact comparisons. Rendering is disabled;
the fixed statistics window is [0.0005, 0.0115]. `--initial-restart` tests restore
from step 0 rather than step 5, including first-RK and sampling deduplication.
`--schedule-checks` adds changed-configuration rejection and an explicit override
whose new targets start at the restart state. Time mode also checks that an
unrepresentably small output interval fails instead of looping indefinitely.

### Dynamic Inlet Completed-Step Restart

The `dynamic` fixture is a nonreacting 16^3 CURVE flat plate with `bl/intp`,
dt=6e-6, `543e` convection, viscosity and tenth-order filtering. Source frames
have uniform 1e-5 spacing and nonpolynomial time dependence. It checks exact
same-backend flow/cache/control/geometry restart, not CPU/GPU bitwise equivalence
or physical convergence. Original grid, profile and inlet sources are removed
from the restored case; all frozen frames must remain available in resources.

```bash
python3 tests/gpu_validation/run_output_restart_validation.py \
  --executable <absolute-build>/bin/astr --mpiexec <matching-mpiexec> \
  --output <new-test-directory> --case dynamic --backends cpu --ranks 1 2 \
  --axis y --restart-step 3 --no-samples
```

Use a matching CPU-only executable for that command. Repeat with a CUDA
executable and `--backends gpu --axis x` for GPU. `--restart-step 5` crosses
the first frame rollover before saving; `--initial-restart` restores step 0.
Targeted extra cases are `--inflow-count 70 --legacy-statistics` for CPU NP=2 x,
`--filter-workspace full --legacy-statistics` for GPU NP=2 y, and `--mode time`
for physical-time scheduling. Formal statistics (`--statistics`) remain TGV-only.
The local driver limits source counts to 12:256; this is not the solver limit.

Each batch has one role-8 `inflow.h5` collectively written by inlet ranks.
Source frames and their index are frozen once per run, not once per checkpoint.
Runtime source sequences must be contiguous regular single-link files, contain
at least four frames and use five-digit indices. Growing live source sequences
are not admitted. Soft/external HDF5 links, virtual datasets and external raw
storage are rejected rather than copied with hidden dependencies.

```bash
python3 -m pytest -q tests/gpu_validation/test_inflow_resource_filesystem.py
ASTR_OUTPUT_H5CC=<matching-hdf5-c-wrapper> \
  python3 -m pytest -q tests/gpu_validation/test_checkpoint_hdf5_resources.py
```

The HDF5 checker is compiled from production C source and needs matching MPI
headers/libraries. Root CMake resolves MPI C alongside MPI Fortran; configure
`MPI_C_COMPILER` explicitly if the wrappers are not installed together.
Current guarded CPU/GPU reports are under `out/or3_dynamic_*_20261001/`; exact
paths and matrix coverage are in plan section 10.17. The driver records file
bytes and provider-specific explicit host-buffer bounds, including persistent
source fingerprints. The 64 MiB per-directory and per-rank controlled-buffer
limits do not measure RSS, HDF5/MPI or compiler temporary peaks. Unsupported
providers fail the estimator rather than receiving an invented bound.
Nonreacting external initialization `ninit=1/2/3` is registered as described
below. AIR5 still requires `ninit=0`; legacy output is unchanged.

### External Initialization Resources

`--initial-dimension 1/2/3` generates a bounded smooth positive-density/temperature
initial field and selects the original one-, two- or three-dimensional reader.
The one-dimensional profile varies in x, the two-dimensional field in xy; these
are the original reader conventions, not a selectable plane orientation.
Initialization is not a substitute for checkpoint restore or classic TGV
physical validation. Source files are copied once to resources and referenced
by every batch. Restore tests remove the original datin file and require exact
same-backend state/control/geometry and selected statistics continuation.

```bash
python3 tests/gpu_validation/run_output_restart_validation.py \
  --executable <absolute-build>/bin/astr --mpiexec <matching-mpiexec> \
  --output <new-test-directory> --backends cpu --ranks 1 2 \
  --initial-dimension 2 --axis y --no-samples
```

Use matching CPU/CUDA binaries for each backend. Targeted matrix: dimension 1
with x decomposition, dimension 2 with y, dimension 3 with z, each CPU/GPU
NP=1/2; CPU dimension 2 NP=2 with `--statistics`; dimension 3 with `--case dynamic`
and y decomposition, adding GPU full filter/compact statistics; GPU dimension 3
NP=1 with `--initial-restart`. Source file sizes and controlled flow buffers are
included in summary.json; each test directory remains below 64 MiB.
These tests compare periodic against final-only checkpoints, not a completely
disabled writer. The CPU NP=2 branch rejects corrupted initial resources and
HDF5 external dependencies. Root CPU-only/CUDA builds and 16 positive short
cases passed; exact evidence paths and limits are in plan section 10.18.
The AIR5 external reader is not opened by these checks because it does not
provide the required species/two-temperature/compensation initialization state.

### Parallel State File Prototype

The statistics payload adds role 3 with 34 explicit FP64 components: window,
previous sample, accumulated weights, Reynolds/Favre means and central moments,
and a 0/1 previous-sample flag. `test_statistics_state_continuation` checks the
CPU pack/HDF5/unpack/continue path at NP=1/2 x/y partitions, plus role rejection.
The 9x7x5 probe remains within 2 MiB per rank and 4 MiB per directory.
The primitive tests are distinct from the real-solver transaction tests above.
Optional shared integer metadata stores regional statistics and sampling identity.
GPU half-open point statistics use role 4 and explicit contiguous host staging;
role 3 remains the CPU nodal layout. Neither layout currently admits repartition.

`insitu_velocity_statistics_probe` also checks invalid packed-state rejection
without changing existing state. `run_insitu_sample_validation.py --statistics
--statistics-roundtrip` exercises GPU packing, release, exact restore, invalid
flag rejection, and continued statistics, in addition to the existing stream
roundtrip checks. Its added pair of test buffers is capped at 64 MiB. No renderer
or production run is needed for this check.

`checkpoint_state_io.F90` stores owned global physical nodes and original-rank
halo/duplicate values separately, without averaging the latter. The restricted
TGV runtime uses it inside a published bundle; the primitive alone is not a
complete checkpoint. Repartition reads fill physical nodes only; halos and
other persistent solver state still need a separate restoration lifecycle.

```bash
cmake --build <build> --target checkpoint_state_probe
ASTR_CHECKPOINT_STATE_PROBE=<absolute-build>/bin/checkpoint_state_probe \
ASTR_OUTPUT_MPIEXEC=<matching-mpiexec> \
python3 -m pytest -x -q tests/gpu_validation/test_checkpoint_state.py
```

The approved synthetic matrix is 9x7x5 nodes, 5/11 components, NP=1/2, x/y
partitions and halo widths 0/1. No GPU or physical simulation is launched.
Each rank's explicit test arrays/buffers are capped at 2 MiB and each test
directory at 4 MiB; these are not production defaults or bounds on all MPI/HDF5
internal allocations. Exact reads compare FP64 representations including signed
zero. Distinct duplicate values prevent false passes caused by equal copies.
Tests include 1-to-2, 2-to-1 and x-to-y/y-to-x reads, zero-length extras,
wrong metadata/type, missing datasets, and refusing to overwrite an existing
file. They do not yet check checksums, bundle publication, retention or actual
flow restart equivalence.

`checkpoint_bundle_probe` and `test_checkpoint_bundle.py` add bounded manifest
and completion-record integrity checks around the real state file. Build this
probe through the same root CMake and run with the same `ASTR_OUTPUT_MPIEXEC`;
`ASTR_CHECKPOINT_BUNDLE_PROBE` overrides the executable location. Tests reject
unsealed directories, missing payloads/markers, single-byte mutations, empty or
trailing completion records and repeated sealing. CRC64 detects accidental
corruption, not malicious modification. These checks do not yet implement
retention or crash durability. Additional publication cases test exclusive
directory rename and atomic LATEST replacement, including existing destinations,
unsealed candidates, stale temporary pointers and a directory blocking LATEST.
Old batch contents must remain unchanged on both success and failure. Linux
renameat2 availability is checked at build time; unsupported filesystems fail
closed without falling back to an overwrite-capable directory rename.

Retention cases exercise four sequential publications with keep=1/2 using a
single live Fortran ledger. Earlier-run/restart-source directories are not
registered and must remain byte-identical. PROTECT excludes a batch from the
ordinary count. Unknown files, symlink payloads and hardlinked payloads must
stop retirement without invalidating the protected old payload or newest batch.
The small Linux filesystem helper uses directory-relative no-follow operations
and no recursive deletion. This still does not exercise real solver restart,
automatic archive reference management or power-loss durability.

Shared-resource cases use a run/resources directory and a checksummed RESOURCES
table inside run/checkpoints/batch. They verify moving the entire run, missing
or modified resources, and rejecting file/directory symlinks. The reference
table must be included in MANIFEST. These synthetic payloads do not validate
mesh semantics, automatic immutable snapshots or dynamic-inlet source freezing.

The state-file schema is now 2, with an explicit payload role. The perfect-gas
provider uses role 2 for five conserved fields plus rho/u/v/w/p/T caches, not
eleven reacting-flow conserved fields. Three NP=1/2 x/y provider checks use
independent synthetic cached values, require bitwise restore, and reject reading
the same component count under the wrong role. These are storage-interface
checks, not the approved 16^3 TGV continuous-versus-restart acceptance matrix.

## Production Build Without Validation Sources

`BUILD_TESTING` defaults to `ON`, preserving the development targets. Configure
with `-DBUILD_TESTING=OFF` to build `astr` without any source from `tests/`.
This disables the added `bcad/bciv/bcst/bcrh/bcgc` validation commands and the
standalone probes. Invoking a disabled command fails with an explicit message.
Original CPU built-in tests and the root `examples/` configuration are retained.
Keep `examples/` in a production source distribution.

CUDA-aware MPI detection and AIR5 production definitions remain independent of
`BUILD_TESTING`. The optional MPI completion tracing interposer requires
`BUILD_TESTING=ON` because its implementation lives under `tests/`.
Existing development build caches with testing disabled must be reconfigured
with `-DBUILD_TESTING=ON` before building probes or running the added commands.

An opt-in integration check configures a source copy with no `tests/` directory,
covering CPU/CUDA and AIR5 off/on. It also checks development targets:

```bash
ASTR_TEST_CMAKE_COMPILER=/path/to/nvfortran \
HDF5_ROOT=/path/to/hdf5 \
ASTR_TEST_CMAKE_BUILD=1 \
python -m pytest -q tests/gpu_validation/test_cmake_production_decoupling.py
```

Set the matching MPI wrapper and compiler runtime paths in the environment.
Omit `ASTR_TEST_CMAKE_BUILD=1` for configure-only checks. This checks build
separation, not numerical or physical validation.

The source copy includes tracked production inputs and the explicit
`PENDING_PRODUCTION_INPUTS` list of approved additions, using their current
working-tree contents; it excludes other untracked files and local outputs. With builds enabled,
`test_release_install_prefixes` also builds CPU/CUDA nonreacting binaries and
checks configured and overridden install prefixes, separate 2D/3D TGV inputs,
and an unchanged source tree. Configure/build/install logs remain under the
pytest temporary directory. These checks do not qualify a remote runtime stack.

## Nonreacting GPU Checkpoint Regression

`run_release_filter_restart.py` checks 100-step TGV CPU/GPU, full/scalar,
and step-50 restart equivalence with caller-supplied binaries.
`run_release_boundary_filter_restart.py` checks channel or CURVE against a
completed 100-step scalar reference, including the matching statistics window.
Channel restart replaces its text log; CURVE appends it. Neither comparison
changes the numerical acceptance tolerance.

`check_exact_restart_rejection.py` consumes the channel helper's `restart/`
directory (including `checkpoint_step50/`) and checks rejection of missing,
incomplete, corrupt, mismatched and unsupported rank-local checkpoint files,
backup failure, and warned legacy loading. `check_exact_restart_sequence.py`
checks sequence output through a two-step/four-step channel restart.
All tools require a new output directory and an explicit GPU executable.
These checks do not qualify AIR5, topology migration or physical convergence.

## AIR5 Characteristic Top Acoustic Gate

`ASTR_AIR5_SOURCE_MODE=frozen` explicitly disables chemistry and V-T sources
for validation. Default `coupled`, `chemical`, and `vt` semantics are unchanged.
Frozen ROS-2 calls validate inputs but preserve state and compensation bits,
without consuming saved carry through zero-increment additions. Outer boundary,
halo and primitive preparation remain active. This is not reacting validation.

Root-CMake target `air5_frozen_source_probe` checks zero sources/Jacobians,
nonzero-carry preservation, nonequilibrium temperatures and unchanged default
coupled selection; CUDA builds exercise the device path too.
`run_air5_characteristic_uniform.py --source-mode frozen` runs the CPU uniform gate.
`run_air5_characteristic_acoustic.py` prepares/runs the approved pulse experiment;
`--prepare-only` and `--run-prepared` separate inspection from execution.
It rejects reused run directories and changed prepared executables.
Its single-run reflection result is not complete acoustic acceptance: the
extended-domain, relaxation sensitivity and refinement gates remain separate.
`check_air5_characteristic_acoustic_matrix.py` requires `--baseline`,
`--half-tau`, `--double-tau`, `--extended`, `--coarse`, `--half-dt` and
`--prescribed` case directories. It checks configuration agreement and recomputes
the gates from raw completed results before applying pairwise thresholds.
See `documents/ASTR_AIR5_CHARACTERISTIC_ACOUSTIC_GATE.md` for fixed thresholds.

## Complete-Step CFL Diagnostics

`cfl_spectrum_probe` is a root-CMake CPU/GPU target for reversed-velocity,
nonorthogonal-metric and inactive-direction arithmetic checks.
`ASTR_CFL_DIAGNOSTICS=on|off` controls reporting only;
`ASTR_CFL_PROFILE_Y=1` adds global computational-j plane maxima. Options must
agree across ranks. AIR5 uses the local frozen two-temperature sound speed.
`current CFL` is the sum of directional maxima, not proof of viscous or
chemical stability. `ASTR_CFL` records also include the pointwise-sum maximum,
global locations and complete-step physical time.

`check_air5_cfl.py CASE --output REPORT` checks a completed uniform-Cartesian
AIR5 replay against saved second-half chemistry states or the next checkpoint.
It fails when a matched state is unavailable. `check_cfl_freestream_matrix.py`
checks the fixed 16-cubed regression matrix and its flow gates, not arbitrary
cases. `run_curvilinear_freestream_compare.sh` accepts `U1/U2/U3` and
`AMPLITUDE=0` for reversed-flow Cartesian controls; defaults are unchanged.

Evidence and remaining gates are recorded in
`documents/ASTR_CFL_AND_AIR5_TOP_REPAIR_PLAN.md`. The first pre-chemistry incident
restart snapshot is not a matching CPU/GPU boundary phase; do not use it in
place of a complete-step state or silently relax its comparison tolerance.

## AIR5 Stage Mass-Closure Diagnostics

`check_air5_mass_closure_stages.py` reads saved physical-node q snapshots; it
does not integrate, normalize, clip, or modify the solver state. It reports
FP64 sequential `sum(rho_s/rho)-1` against the existing transport threshold
`128*epsilon`, and uses NumPy long double to diagnose `sum(rho_s)-rho` without
the same FP64 summation rounding. The report records the available mantissa
precision and missing stages explicitly. It is a diagnostic, not a pass gate.

```sh
python3 tests/gpu_validation/check_air5_mass_closure_stages.py \
  --case CASE_DIR --rank 1 --node 11 46 5 --steps 1090 1265 \
  --report closure.json
python3 -m pytest -q tests/gpu_validation/test_air5_mass_closure_stages.py
```

The 2026-09-25 Mach4 replay identifies composition-sum threshold crossing,
not negative species, at step1266. Primitive-reuse on/off produced 14 identical
binary snapshots. Increment-form RK and combined ROS-2 increment candidates
were then tested but failed the 0.5 ns / 400-update window at steps1273 and1359.
Their solver changes were withdrawn; frozen executables, reports and the
combined patch remain under `out/air5_*increment*_20260925/`. Short probe and
CPU/GPU equivalence passes must not be reported as long-window acceptance.

## AIR5 Same-State Reuse And Packed Chemistry Diagnostics

Both candidates are opt-in and preserve FP64 and explicit synchronization:
`ASTR_AIR5_PRIMITIVE_REUSE=chemistry` reuses only chemistry-recovered strict
interior primitives immediately after the half-step (`off` forces recovery).
`ASTR_AIR5_CHEMISTRY_REDUCTIONS=packed` combines like-operation diagnostics
without removing the global failure barrier (`baseline` is the default).
There is no persistent full-domain cache or cross-stage validity flag.

The bounded replay accepts `--primitive-reuse chemistry` and
`--chemistry-reductions packed` explicitly. Its environment sanitization does
not forward arbitrary inherited ASTR options. The result records both flags.

`run_air5_chemistry_performance_gate.py --executable EXE --output NEW_DIR`
checks off/A/B/combined periodic reacting TGV against CPU for NP1 and NP2
x/y/z, including state, chemistry constraints, elements and conservation.
Add `--modes combined --filters full scalar --trace-restart` for both filter
workspaces and strictly zero/trace composition restart cases.
These are small numerical gates, not long-time SBLI physical validation.
When only GPU code changes, `--reference-matrix PREVIOUS_MATRIX` reruns GPU
cases and compares against both existing CPU and GPU snapshots with unchanged
tolerances. Inputs are copied from the matching previous GPU case. For
supplements alone use `--supplement-from PREVIOUS_GPU_CASE`; this skips the
matrix and does not claim it was rerun. Zero-species supplements retain the
TGV constraints but require nonnegative, rather than strictly positive,
species at the deliberately zero initial state. Any negative value fails.

For an isolated MPI failure gate, compile the test-only interposer and pass it
to the preceding driver with `--fault-library`:

```bash
mpicc -std=c11 -Wall -Wextra -shared -fPIC \
  tests/gpu_validation/air5_packed_failure_inject.c -ldl \
  -o tests/gpu_validation/out/air5_packed_failure_inject.so
```

It injects status 99 at rank 1's packed chemistry reduction through the
OpenMPI Fortran binding. The driver requires an explicit chemistry failure,
exit 99, a pre-chemistry snapshot and no post-chemistry/transport snapshots.
It must never be linked into ASTR or used in performance/production runs.

After correctness gates, `run_air5_flux_ab.py --chemistry-candidates` runs
baseline/A/B/combined sequentially with alternating order, NP1/NP2, three
rounds each, one warmup and three measured complete steps. Existing two-way
flux A/B mode remains the default. Keep Nsys and sanitizer runs separate.

## AIR5 Complete-Step Performance Diagnosis

The GPU symmetric flux demand-pruning candidate is validated in
`out/air5_flux_opt_*_20260925/`. See the performance plan for numerical gates
and measured complete-step gains. The default `full_state` mode is unchanged.
`run_air5_flux_ab.py --baseline CHECKPOINT_CASE --reference OLD_EXE
--candidate NEW_EXE --output NEW_DIRECTORY` performs three interleaved A/B
rounds for NP1 and NP2, one warmup plus three measured steps per run. It
rejects advanced field output, snapshots and executable/checkpoint drift.
Do not interpret a faster two-GPU time as improved strong-scaling efficiency.

`ASTR_COMPLETE_STEP_TIMING=1` times `crashcheck + time_integration_rk` on
every rank, including reacting chemistry and transport. It is disabled by
default and adds no barrier. It excludes startup and subsequent controller,
CFL and checkpoint handling; it is not the old transport-only RK timer.

```bash
python tests/gpu_validation/run_air5_performance_diagnosis.py \
  --baseline tests/gpu_validation/out/air5_layered_multistep_20260924/gpu_dt2/gpu \
  --output tests/gpu_validation/out/air5_performance_new
```

The driver freezes FP64, explicit synchronization, `symmetric_species`,
`layered`, dt=2 ns, three independent rounds, one warmup and three measured
steps. Each step uses the maximum rank duration. The reported group median
is over round means, not a best-case sample. Startup rewrites the initial
checkpoint outside the measured window; the driver rejects advanced field
output or validation snapshots during timing.

Use `run_air5_sbli_domain_replay.py --nsys` for separate GPU CUDA/MPI traces,
or `--np 1 --ncu-kernel REGEX` for one matching kernel launch. Neither
profiler run supplies benchmark speedups. These options are mutually
exclusive with memcheck. A completed bounded replay is not physical
acceptance. This small Mach4 checkpoint has no incident shock.

## AIR5 Symmetric Flux Feasibility Audit

Final bounded repair gate (2026-09-25): 48 candidate and 3 default runs in
`out/air5_symmetric_final_*_20260925/` pass the frozen trace, low-N2, energy,
species and x/y/z slab gates. Candidate evidence: 1440 bitwise-identical plane
pairs, 32 clean rank memcheck logs, max conservation 2.920763e-14 and own-species
error 1.571109e-15. The default remains unchanged; no long-time physics claim.

`check_air5_boundary_flux_balance.py --root REPLAY_ROOT` consumes the final
face records enabled by `run_air5_sbli_domain_replay.py --symmetric-face-probe`.
It checks the summed owned convective RHS against net physical-boundary flux
and verifies actual neighbor payloads bitwise. It does not test total source/
diffusion conservation. CPU/GPU results in
`out/air5_symmetric_boundary_balance_{cpu,gpu}_20260925/` pass all three RK
stages (maximum residual 1.389391e-16); the GPU run has two clean memcheck logs.
CPU/GPU replay fields, the prior matched-time half-step replay, 120 projection
permutations, the symmetric cache probe, and 52 checker tests also pass.
The notes below retain the earlier implementation history.

GPU nonperiodic update (2026-09-25): single-sided budgets, physical-node exclusion
and null-neighbor handling now match CPU. `run_air5_sbli_domain_replay.py` accepts
`--convection-limiter symmetric_species` and `--memcheck` (two clean GPU-rank logs
required in addition to successful completion). Existing step1090 replay:
`out/air5_symmetric_gpu_boundary_late_20260925/cpu_gpu.json` passes 20 same-phase
fields at atol1e-9/rtol1e-10. The 2 ns and two 1 ns GPU runs have 18/36 valid
state snapshots and four clean memcheck logs. The half-step directory's
`matched_window.json` reports inlet-rank max u difference 2.883809e-5 m/s.
Boundary flux accounting and the final complete regression matrix remain open.
The following CPU-only note records the preceding implementation stage.

CPU nonperiodic candidate (2026-09-25): physical faces now use only the active
cell's RK budget; the missing side's finite bounds follow from mass closure.
Null MPI neighbors never overwrite physical faces or invoke periodic wrap.
Eight CPU periodic controls pass. Existing Mach4 step1090, one 2 ns NP2 CPU
replay completes with 18 valid state snapshots and max mass closure 9.002067e-16
in `out/air5_symmetric_cpu_boundary_late_20260925/`. This does not close boundary
flux accounting, CPU/GPU equivalence, dt/2 or memory-safety gates. GPU continues
to reject this option with physical boundaries until its implementation lands.

Latest cache gate (2026-09-25): the optional symmetric path now caches all five
species as rho_s/rho on CPU and GPU without changing q or the legacy default.
`chemistry_flow_gpu_probe` checks each trace species at 0, 1e-30, 1e-16 and 1e-12,
own-species roundoff, exact zeros, unchanged q and restoration of legacy mode;
`out/air5_species_cache_probe_memcheck_20260925.log` reports zero errors.
The rebuilt main program passes 24 candidate and 3 default runs in
`out/air5_symmetric_cache_primary_20260925/` and
`out/air5_cache_scan_*_20260925/`: 720 candidate plane pairs are bitwise equal,
16 candidate rank memcheck logs are clean, maximum scaled conservation residual
is 2.534431e-15 and own-species difference is 1.571109e-15.
This is a periodic frozen gate, not nonperiodic or reacting-flow acceptance.
The runner now supports `--axis x|y|z`: it rotates grid dimensions, input fields
and velocity, then canonicalizes diagnostic arrays and vector components.
It checks the actual slab seam and passes the corresponding topology to the
bitwise plane checker. In `out/air5_symmetric_{y,z}_slab_20260925/`, each axis
passes contact/low-N2/energy/species on CPU NP1/NP2 and GPU NP2. Across both axes:
24 runs, 720 identical plane pairs, 16 clean rank memcheck logs, maximum scaled
conservation residual 2.920763e-14 and own-species difference 1.571109e-15.
The energy and species constraints activate at the actual MPI seam. These are
periodic slab tests, not mixed-axis topologies or physical-boundary validation.

Current gate: the periodic GPU mainloop passes the 24-run CPU/GPU small numerical
matrix plus three default controls. The initial error700 under memcheck was
isolated to the device `sum(ll-lr,dim=2)` path: an initialization sync alone did
not fix it, but an explicit fixed-loop reduction did. All eight candidate GPU
NP2 cases now have two clean memcheck logs each and retain all numerical gates.
Do not infer physical-boundary, y/z decomposition or SBLI acceptance from this.
Use `--memcheck` to instrument each GPU rank; the runner requires one clean
sanitizer log per rank as well as all ordinary numerical gates and stops on error.
Physical boundaries remain unsupported in the new path.

Build `air5_flux_gpu_probe` through the root CUDA/AIR5 CMake build. It reuses
`air5_transport_roundoff_probe.F90` with `AIR5_FLUX_GPU_PROBE` and launches
the device projection through `air5_flux_gpu_probe_wrappers.cuf`. Every call
compares CPU/GPU status and flux, then runs the original probe assertions.
The default CPU-only target remains unchanged. Example:

```sh
cmake --build build_gpu_probe --target air5_flux_gpu_probe air5_transport_roundoff_probe -j 2
build_gpu_probe/bin/air5_transport_roundoff_probe
compute-sanitizer --tool memcheck --error-exitcode 99 build_gpu_probe/bin/air5_flux_gpu_probe
```

The 2026-09-25 local run passes with zero memcheck errors. This is a projection
helper gate, not the symmetric limiter GPU mainloop or MPI gate.
It was rebuilt and rechecked with the production 128-register device-callee cap;
`out/air5_flux_gpu_probe_reg128_memcheck_20260925.log` also reports zero errors.

`run_air5_layered_stress_gate.py --backends cpu --convection-limiter symmetric_species`
exercises the experimental CPU periodic path. Keep `--require-contact-equilibrium`
for contact/low-N2 fixtures. The default runs both CPU and GPU; the new
limiter rejects physical boundaries rather than silently using the old algorithm.
It additionally checks each species difference against
its own initial maximum, without a bulk absolute floor. The 2026-09-25 contact
scan passed on CPU NP1/NP2, but the species stress fixture failed NP2 periodic
conservation. That failure was subsequently fixed by including the projected
output operands in the arithmetic residual bound, without changing the outer
acceptance thresholds. The 18-run `air5_operand_matrix_*_20260925` CPU matrix
passes. The subsequent `air5_owner_matrix_*_20260925` matrix plus
`air5_symmetric_owner_energy_interior_20260925` passes with canonical CPU face
ownership and strict receiver-side budgets. Across 16 candidate runs, 432 full
plane pairs are bitwise identical. Two old-default controls also pass.
The energy fixture first exposed an internal-face vibrational margin of
`-1.21e-27` after re-evaluation; the restricted beta now retreats by the existing
machine-roundoff gamma128 factor and is projected and checked again. Physical
state tolerances are unchanged. GPU and nonperiodic validation remain open.

`--symmetric-face-probe` writes independent
`validation/air5.symmetric_faces.rankNNNNNNNN.txt` files. Compare both rank files
with `check_air5_consistent_residual.py --symmetric-face-log <rank0> <rank1>`
`--local-intervals 24 --np 2 --report <output.json>`. It samples the x-interface
line at j=k=0; it does not certify every point of a three-dimensional interface.
The same flag now also writes `air5.shared_faces.axis*.bin` files containing
seven int32 header entries and four contiguous float64 plane payloads (11 flux
components plus beta). `check_shared_face_payloads` compares every sent/received
entry by its uint64 bit pattern for all three axes and stages. Checker tests
cover x/y/z rank mappings and reject one-ULP corruption; actual solver runs in
this matrix still use x-slab, not y/z MPI decomposition.

`check_air5_consistent_residual.py --contact-case <recorded-case> --np 1`
`--contact-feasibility --report <output.json>` reads the existing first-stage
periodic contact snapshots. Use `--np 2` for the existing x-slab pair.
It verifies the reconstructed central-flux divergence against `conv_raw`, then
checks mass and gas-constant-weighted flux feasibility in two different boxes:
individual low/high correction segments and sufficient six-face cell budgets.
It does not advance the flow, constrain both energies, or accept a new limiter.
The frozen actual-solver gates are in `ASTR_AIR5_MACH4_SBLI_PLAN.md`.

## Conservative Boundary MPI Configuration

`run_boundary_config_mpi_gate.py --out <new-directory>` validates the root-owned
`ASTR_CONSERVATIVE_BOUNDARY_FILE` loader with the top-level CPU/CUDA executables
and NP=1/2/4. Forty checks cover disabled, valid, malformed, missing and root-only
configuration. Non-root ranks in root-only checks receive an invalid local path
but must receive the valid rank-0 state. Logs and binary hashes are retained.
The `astr test bcfg` entry invokes this loader; `astr run` does not yet enable it.

`python3 tests/gpu_validation/run_boundary_stage_gate.py --out <new-directory>`
executes `astr test bcst` with a generated configuration in the CPU and CUDA
builds, then repeats the CUDA run under memcheck. CPU initialization seeds the
state; the CUDA build applies the actual GPU stage to the same perturbed state
as the CPU. The probe compares the full conservative array and selected
primitive face strips at NP=1, including evolving wall density, inlet ghost
energy updates, and top-face corner priority. It does not advance an RHS/RK
step or validate multi-rank staging. The NP1 sanitizer environment disables
MPI GPU-pointer probing (`OMPI_MCA_opal_cuda_support=false`); application CUDA
checks remain enabled. Logs, binary/config hashes, environment overrides, and
a fail-closed summary are retained in the output directory. Parser tests:
`python3 -m unittest discover -s tests/gpu_validation -p test_boundary_stage_gate.py`.
This is configuration distribution, not flow-field MPI or GPU execution validation.

## Overview

The stretched-metric free-stream gate in `astr test bcrh` now passes with the
user-approved optional metric-consistent flux-splitting epsilon. The probe also
retains the failing fixed-epsilon state as a control and verifies that omitted
and explicitly false options agree. Production defaults remain unchanged;
the new option is not yet enabled in the production reference-mode integration.
See `ASTR_OPENSBLI_VALIDATION_GOAL.md` for evidence and remaining physics gates.

The same RHS gate now compares CPU/GPU with metric-consistent epsilon enabled
on a smooth nonuniform state and x/y density-pressure jumps. These are static
operator comparisons, not evolved shock accuracy or MPI/RK validation. Both
physical-space and all-interface Roe paths must pass independently.

This directory contains validation drivers for the current ASTR CUDA Fortran port.

Current validated scope:

- Taylor-Green Vortex, 3D extruded `2dvort`, generated-velocity HIT, forced 3-D LDC slices, and forced 3-D RTI explicit validation variants
- x/y/z zero-extrapolation and symmetry slices
- CPU-compatible Cartesian wall-family slices: `bctype=41` x/y/z, `bctype=42` x/y, `bctype=411` y, and `bctype=421` y
- channel `bctype=41` y-wall source/statistics path and RTI y-fixed source path
- single-rank and multi-rank MPI decompositions, including correctness smoke tests under two-GPU oversubscription
- one physical GPU, two physical GPUs, and two-GPU oversubscription correctness smoke tests
- q(1:5) only
- explicit sixth-order central difference
- explicit tenth-order central filter
- current implemented GPU convection paths are explicit central `643e` and controlled single-rank explicit upwind `543e` with `recon_schem=-1` (first order), `1` (WENO7), or `3` (MP7)
- detect-only crash check
- LF runtime input files
- runtime `use_gpu=t/f`; GPU support is compiled with `ASTR_WITH_CUDA=ON`
- file output, checkpoint, and HDF5 field writing remain CPU-owned output boundaries

Validation order:

1. L0 build and smoke
2. L1 module-level CPU/GPU field diff
3. L2 one-step and ten-step integration diff
4. L3 multi-rank topology correctness
5. L4 TGV physical diagnostics
6. L5 performance and residency profiling

Do not report GPU speedup until correctness, multi-rank topology coverage, and residency profiling all pass. Two-GPU oversubscription runs are correctness smoke tests only, not performance evidence.

## Build Contract

### Compressible Similarity Input Gate

The inlet and similarity-field generators now use the temperature integral in
the wall-normal velocity mapping (approved correction, 2026-09-07). Run:

```bash
python3 -m unittest discover -s tests/gpu_validation -p test_compressible_blasius_profile.py
python3 tests/gpu_validation/check_blasius_mass_continuity.py
```

These checks cover input generation and steady mass continuity, not CFD
convergence. Previously generated similarity inputs contain the old velocity
mapping and must be regenerated for new physical validation. Historical CPU/GPU
comparisons remain comparisons of their original inputs. See
`documents/ASTR_OPENSBLI_KATZER_STARTUP_AUDIT.md` for scope and remaining gates.

### Optional Perfect-Gas Transport Gate

`ASTR_PERFECT_GAS_PRANDTL` and `ASTR_SUTHERLAND_TEMPERATURE_K` optionally
override Pr=0.72 and S=110.3 K in non-COMB nondimensional runs. Rank zero
reads and broadcasts the values; invalid values fail instead of falling back.
GPU kernels consume the same runtime parameters as the CPU path.

```bash
python3 -m unittest discover -s tests/gpu_validation -p test_transport_config.py
python3 tests/gpu_validation/run_transport_config_compare.py --out tests/gpu_validation/out/transport_config_gate
```

Build both probe executables first. The output directory must not exist.
The integration gate runs two-step TGV field/statistics comparisons for
default/reference transport at NP=1/2 and four negative MPI startup checks.
It fixes the local MPI-IO component to `sharedfp=individual`, records executable
hashes and logs, and rejects any failed check. It is not SBLI physical validation.

### Conservative Boundary Algebra Gate

The optional boundary operators are not yet connected to solver dispatch.
Build their probes through the top-level CMake project:

```bash
cmake --build build_gpu_probe --target boundary_contract_cpu_probe boundary_contract_gpu_probe -j4
ASTR_BOUNDARY_PROBE_EXE=build_gpu_probe/bin/boundary_contract_cpu_probe python3 -m unittest discover -s tests/gpu_validation -p test_perfect_gas_boundary.py
ASTR_BOUNDARY_PROBE_EXE=build_gpu_probe/bin/boundary_contract_gpu_probe python3 -m unittest discover -s tests/gpu_validation -p test_perfect_gas_boundary.py
```

Five tests cover eleven inputs and compare against independently evaluated
conservative/EOS relations. The standalone default additionally compiles a
gfortran bounds/FPE-checking probe; `ASTR_TEST_BOUNDARY_CUDA=1` selects NVHPC CUDA.
For sanitization, run `compute-sanitizer --tool memcheck --error-exitcode 1
build_gpu_probe/bin/boundary_contract_gpu_probe 5` (mode 1 for the inlet and
mode 7 for rejection of an invalid outer ghost temperature).
These probes do not qualify corner ordering, MPI halos, or RK integration.

### Conservative Boundary Face Gate

`conservative_boundary_config` reads a schema-1 namelist with `split_x`,
`q_left(5)`, and `q_right(5)`. The input generator now writes this file, but
the corresponding mainloop mode is not yet enabled. Do not launch it by
substituting legacy SBLI boundary codes.

```bash
cmake --build build_gpu_probe --target conservative_boundary_config_probe boundary_faces_cpu_probe boundary_faces_gpu_probe -j4
ASTR_BOUNDARY_CONFIG_PROBE_EXE=build_gpu_probe/bin/conservative_boundary_config_probe python3 -m unittest discover -s tests/gpu_validation -p test_conservative_boundary_config.py
ASTR_BOUNDARY_FACES_PROBE_EXE=build_gpu_probe/bin/boundary_faces_cpu_probe python3 -m unittest discover -s tests/gpu_validation -p test_boundary_faces.py
ASTR_BOUNDARY_FACES_PROBE_EXE=build_gpu_probe/bin/boundary_faces_gpu_probe python3 -m unittest discover -s tests/gpu_validation -p test_boundary_faces.py
```

Without probe environment variables these tests build isolated gfortran probes.
The face probe argument is a bit mask: inlet=1, outlet=2, wall=4, top=8.
Mask 15 applies all faces in that order. These are local-face/EOS/halo/corner
tests, not MPI or RK validation. The GPU caller synchronizes and checks status
after every face kernel.

### CMake Configuration

Build from the repository root `CMakeLists.txt`.

CPU baseline:

```bash
cmake -B build_cpu_probe -S /home/dell/workspace/astr_gpu -DCMAKE_Fortran_COMPILER=mpif90
cmake --build build_cpu_probe -j4
```

CUDA-capable binary:

```bash
cmake -B build_gpu_probe -S /home/dell/workspace/astr_gpu -DCMAKE_Fortran_COMPILER=mpif90 -DASTR_WITH_CUDA=ON
cmake --build build_gpu_probe -j4
```

`ASTR_WITH_CUDA=ON` compiles GPU support into the binary. The actual CPU/GPU path is selected at runtime by the input-file `use_gpu` flag.

## Line Ending Check

Run:

```bash
find examples -type f \( -name 'input.*' -o -name 'controller' \) -print0 \
  | xargs -0 file | rg 'CRLF'
```

Expected: no output.

## Native TGV Statistics Compare

Run CPU/GPU `flowstate.dat` statistics comparison with isolated case copies:

```bash
MAXSTEP=10 LFILTER=f DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_stats_compare_643e_diff_no_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_stats_compare_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh
```

The driver rewrites the copied TGV input to `643e,643e` so CPU and GPU use the explicit sixth-order central derivative policy. It also appends runtime `use_gpu=t` only for the GPU case.

Current expected result: both commands print `status: pass` with `flowstate.dat` differences within `1e-10`. In the GPU-resident path, `flowstate.dat` is written directly from GPU reductions rather than by CPU `statistic.F90`.

## Full-Field HDF5 Compare

Run CPU/GPU `outdat/flowfield.h5` comparison with reconstructed conservative variables:

```bash
MAXSTEP=10 LFILTER=f DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_field_compare_643e_diff_no_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_field_compare_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh
```

The driver compares primitive datasets `ro,u1,u2,u3,p,t`, reconstructs `q1:q5` from the HDF5 output, and records `ro/p/t` extrema. Current expected result: both commands print `status: pass` with `L_inf` within `1e-10`.

## TGV Physical Diagnostics

Generate CPU/GPU diagnostic plots from native `flowstate.dat` output:

```bash
python3 scripts/gpu_validation/plot_tgv_diagnostics.py
```

The script writes EPS and JPEG figures plus `tgv_diagnostics_summary.txt` under `tests/gpu_validation/out/tgv_diagnostics_l3` by default. It currently covers `kenergy`, `enstophy`, and `dissipation`.

## GPU Kinetic-Energy Scalar Reduction

Validate the GPU-resident TGV statistics slices:

```bash
MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

python3 tests/gpu_validation/compare_gpu_kenergy.py \
  --cpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/cpu \
  --gpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu_kenergy_compare.txt \
  --atol 1e-12 --rtol 1e-12

python3 tests/gpu_validation/compare_gpu_enstophy.py \
  --cpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/cpu \
  --gpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu_enstophy_compare.txt \
  --atol 1e-12 --rtol 1e-12

python3 tests/gpu_validation/compare_gpu_dissipation.py \
  --cpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/cpu \
  --gpu tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_dissipation_643e_diff_filter_10/gpu_dissipation_compare.txt \
  --atol 1e-12 --rtol 1e-12
```

The GPU run writes native `flowstate.dat` plus `gpu_kenergy.dat`, `gpu_enstophy.dat`, and `gpu_dissipation.dat` from device-side reductions. The time loop keeps full flow variables resident on the GPU. File output, checkpoint, and HDF5 `flowfield` writing remain CPU-owned and are intentionally out of current GPU-porting scope; full-field D2H at those boundaries is acceptable.

For the explicit 10th-order filter, the GPU path uses ping-pong storage and halo stencil kernels. In single-rank homogeneous y/z directions it refreshes local halos for `qwork_d` and `q_d` before launching the y/z halo filter kernels; in multi-rank directions it uses the corresponding MPI halo exchange before the same halo filter kernels.

The full-state filter workspace remains the default. Set
`ASTR_GPU_FILTER_WORKSPACE=scalar` to use one haloed scalar workspace and process
the five conservative variables sequentially. The scalar path exchanges the
full x-filter halo once, then exchanges only the active scalar y/z halo for each
component. Every kernel retains the explicit synchronization contract. All MPI
ranks must select the same mode; invalid or inconsistent values abort before
time stepping.

The comparison drivers accept `FILTER_WORKSPACE=full|scalar`, for example:

```bash
FILTER_WORKSPACE=scalar MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_tgv_field_compare.sh

FILTER_WORKSPACE=scalar \
  tests/gpu_validation/run_tgv_gpu_memcheck.sh
```

This mode remains optional rather than the production default. The local
admission matrix now covers TGV NP=1 and x/y/z NP=2 slabs, LDC, Channel, and
curvilinear `bctype=42`. The fresh `128^3` five-run complete-RK medians are
`0.081408774/0.078219996 s` for full/scalar, while sampled process memory is
`1686/1604 MiB`. A800 admission remains separate and is recorded in
`documents/ASTR_GPU_SCALAR_FILTER_IMPLEMENTATION_PLAN.md`.

Channel field comparisons use the CPU complete-RK snapshot. Comparing the
ordinary CPU HDF output with the GPU complete-RK state mixes output phases and
is not a valid filter-equivalence gate.

An invalid `ASTR_GPU_FILTER_WORKSPACE` value is rejected collectively before
time stepping. The scalar `32^3` TGV memcheck reports zero Compute Sanitizer
errors.

## GPU Single-Rank Qswap

Validate the first-stage GPU periodic halo and periodic-plane averaging path:

```bash
MAXSTEP=1 LFILTER=f DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_qswap_643e_diff_no_filter_1 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_qswap_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_qswap_field_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh
```

Current expected result: all commands print `status: pass`. The full-field check has reconstructed conservative-variable `q5` `L_inf` around `8.73e-12`.

## GPU-Resident Loop Check

Validate the current GPU-resident TGV path:

```bash
MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_resident_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_resident_field_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_field_compare.sh
```

For a no-checkpoint runtime copy check, profile a short GPU case with `feqchkpt` greater than `maxstep`. The current 3-step profile under `tests/gpu_validation/out/tgv_gpu_resident_nsys_3_nochk` showed D2H copies only from reduction partial sums: 18 copies, each 65,536 bytes, and no full-field D2H in the compute loop.

## Phase A 2dvort Validation

Validate the first non-TGV explicit case. The driver starts from `examples/Vortex_Transport/datin/input.2dvort`, rewrites it to a 3D extruded periodic case, and forces explicit `643e,643e`:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_phasea_smoke \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_2dvort_phasea_compare.sh
```

Current expected result: both `flowstate.dat` and `flowfield.h5` comparisons print `status: pass`. The default filtered Phase A thresholds are `STATS_ATOL=1e-9`, `STATS_RTOL=1e-10`, `FIELD_ATOL=1e-9`, and `FIELD_RTOL=1e-10`. The most recent filtered run had max stats differences `maxq4=7.8810021050800005e-10` and `maxq5=2.7473134878164274e-11`; final field reconstructed errors were `q2=2.6846691536519529e-10`, `q3=2.6549052794524672e-10`, `q4=7.9986820163237011e-10`, and `q5=3.3623592798903701e-10`.

A stricter filtered field threshold of `1e-10` is still too tight for this Phase A case because CPU produces about `8e-10` of roundoff in the physically zero `u3/q4` component while GPU keeps that component exactly zero. The no-filter strict check below still passes to roundoff, so this is tracked as filtered non-TGV roundoff sensitivity rather than a baseline RHS/RK/copy-back failure.

Run the no-filter strict isolation check:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_phasea_no_filter_1 \
  MAXSTEP=1 FEQCHKPT=1 LFILTER=f STATS_ATOL=1e-10 FIELD_ATOL=1e-10 \
  tests/gpu_validation/run_2dvort_phasea_compare.sh
```

Current expected result: strict no-filter comparison prints `status: pass` for both reports. This isolates the RHS/RK/non-TGV initialization path from the explicit-filter roundoff sensitivity; the most recent no-filter field errors were at roundoff scale, with reconstructed `q5` `L_inf=7.1054273576010019e-15`.

Run the first `2dvort` multi-rank matrix:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_mpirank_matrix_np2 \
  tests/gpu_validation/run_2dvort_mpirank_matrix.sh
```

The matrix covers `NP=2` with `2x1x1`, `1x2x1`, and `1x1x2`. No-filter cases use `STATS_ATOL=1e-10` and `FIELD_ATOL=1e-10`. Filtered cases use `STATS_ATOL=1e-9` and `FILTER_FIELD_ATOL=5e-9`; the field tolerance is separated from the native statistics tolerance because interface-adjacent filtered HDF5 reconstructed energy differs by about `4.7e-9` while `flowstate.dat` remains within `1e-9`.

The optional `NP=4` core matrix can be run with:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_mpirank_matrix_np4 \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' FILTER_FIELD_ATOL=6e-9 \
  tests/gpu_validation/run_2dvort_mpirank_matrix.sh
```

The higher filtered field tolerance is required by the `2x2x1` reconstructed `q5` interface-adjacent difference, about `5.2e-9`; native filtered statistics still use `1e-9`.

The optional `NP=8` `2x2x2` statistics-only smoke can be run with:

```bash
OUT_DIR=tests/gpu_validation/out/2dvort_mpirank_matrix_np8_stats_smoke \
  MATRIX='8:2,2,2' RUN_FIELD=f FEQCHKPT_FIELD=99 \
  tests/gpu_validation/run_2dvort_mpirank_matrix.sh
```

This is a halo-routing smoke under two-GPU oversubscription. It does not compare HDF5 fields and should not be used as performance evidence.

## Phase A HIT Validation

Validate the generated-velocity HIT case. The driver starts from the TGV input template, rewrites `flowtype=hit`, generates `datin/velocity.h5`, and compares CPU/GPU `flowstate.dat` statistics:

```bash
OUT_DIR=tests/gpu_validation/out/hit_phasea_np1_20steps \
  MAXSTEP=20 FEQCHKPT=99 GRID=64,64,64 COMPARE_FIELD=f \
  tests/gpu_validation/run_hit_phasea_compare.sh
```

The generated velocity is an ABC-style periodic field. `hitini` must refresh halos before its divergence diagnostic because `grad()` consumes halo data; with the current code the log reports divergence `avg/min/max = 0`.

Validate HIT x-slab multi-rank HDF5 hyperslab reading and halo exchange:

```bash
OUT_DIR=tests/gpu_validation/out/hit_phasea_np2_xslab_5steps \
  MAXSTEP=5 FEQCHKPT=99 GRID=64,64,64 NP=2 TOPOLOGY=2,1,1 COMPARE_FIELD=f \
  tests/gpu_validation/run_hit_phasea_compare.sh
```

Validate HIT combined-direction halo-routing smoke under two-GPU oversubscription:

```bash
OUT_DIR=tests/gpu_validation/out/hit_phasea_np8_2x2x2_5steps \
  MAXSTEP=5 FEQCHKPT=99 GRID=64,64,64 NP=8 TOPOLOGY=2,2,2 COMPARE_FIELD=f \
  tests/gpu_validation/run_hit_phasea_compare.sh
```

Current expected result: all three commands print `status: pass` with `kenergy`, `enstophy`, and `dissipation` differences at about `1e-15` or below. HIT field-output comparison is intentionally disabled by default because HDF5 flowfield output remains a CPU-owned boundary in the current project scope.

## Phase B Zeroextrap Boundary Validation

Validate the first finite-domain boundary slice. The driver starts from the TGV template, sets `lihomo=f,ljhomo=t,lkhomo=t`, uses x-direction `bctype=50,50`, enables diffusion and the explicit filter, and compares CPU/GPU statistics plus `flowfield.h5` output:

```bash
OUT_DIR=tests/gpu_validation/out/xextrap_phaseb_np1_5steps_filter_diffusion \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: both `flowstate.dat` and field comparison print `status: pass` with differences at roundoff scale. This validation depends on the CPU `parallelini` fix that sets single-rank non-homogeneous active ranges to interior nodes, the GPU x-zeroextrap boundary kernel, x-zeroextrap RHS active-range guards, x-physical GPU `gradcal`, x-physical diffusion flux/RHS support, and CPU-compatible explicit-filter primitive timing. In the CPU code, `filterq` updates `q` but does not immediately refresh all primitive fields; the GPU x-zeroextrap filtered path therefore synchronizes primitive fields before filtering each RK substep, then preserves interior primitive fields after filtering while refreshing only boundary/halo primitive slices required by CPU `qswap`. `LFILTER=f` and `DIFFTERM=f` remain useful regression variants for isolating convection/RK from filter and diffusion effects.

The same driver can validate the y/z zeroextrap slices by setting `ZERO_AXIS=y` or `ZERO_AXIS=z`. These slices validate boundary application, active ranges, y/z physical `gradcal`, y/z physical convection RHS, y/z physical diffusion flux/RHS, explicit filter ping-pong halo semantics, CPU-compatible filtered primitive timing, and single-rank halo routing.

```bash
OUT_DIR=tests/gpu_validation/out/yzero_phaseb_filter_5steps \
  ZERO_AXIS=y MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/zzero_phaseb_filter_5steps \
  ZERO_AXIS=z MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: both y and z commands print `status: pass` for `flowstate.dat` and field comparison. Single-rank finite-domain slices remain useful isolation tests for the physical y/z boundary kernels and filtered primitive timing.

Validate the current multi-rank Phase B boundary matrix:

```bash
OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix \
  tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh
```

The default matrix validates all currently enabled multi-rank Phase B routes, including physical-direction decomposition for x/y/z zeroextrap. Physical-direction convection/diffusion stencils use physical one-sided templates only on true `MPI_PROC_NULL` faces and use halo-backed sixth-order central stencils on MPI internal interfaces.

```text
x:2:1,2,1 x:2:1,1,2 x:2:2,1,1
y:2:2,1,1 y:2:1,1,2 y:2:1,2,1
z:2:2,1,1 z:2:1,2,1 z:2:1,1,2
x:4:1,2,2 x:4:2,2,1 x:4:2,1,2
y:4:2,1,2 y:4:2,2,1 y:4:1,2,2
z:4:2,2,1 z:4:2,1,2 z:4:1,2,2
```

Current expected result: every matrix entry prints `status: pass` for `flowstate.dat` and field comparison at `STATS_ATOL=1e-9` and `FIELD_ATOL=1e-8`. Physical-direction decomposition is enabled for x-zero, y-zero, and z-zero in the default NP=2/NP=4 matrix.

Run the longer Phase B stability checks with:

```bash
OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix_5steps \
  MAXSTEP=5 FEQCHKPT=5 \
  tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix_20steps_stats \
  MAXSTEP=20 FEQCHKPT=99 RUN_FIELD=f \
  tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh
```

Current expected result: the 5-step matrix passes statistics and field comparison; the 20-step matrix passes statistics-only. The latest 5-step field run had max reconstructed `q5` error `7.3896444519050419e-13`; the latest 20-step statistics-only run had max differences `kenergy=4.4231285301066237e-13`, `enstophy=3.54605234065275e-13`, and `dissipation=4.427014310692812e-14`.

## Phase C Symmetry Boundary Slices

Validate CPU-compatible Cartesian `bctype=60,60` symmetry with the TGV template:

```bash
OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_x_5steps \
  BC_KIND=symmetry ZERO_AXIS=x NP=1 TOPOLOGY=1,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_y_5steps \
  BC_KIND=symmetry ZERO_AXIS=y NP=1 TOPOLOGY=1,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np1_z_5steps \
  BC_KIND=symmetry ZERO_AXIS=z NP=1 TOPOLOGY=1,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate the multi-rank Phase C matrix:

```bash
OUT_DIR=tests/gpu_validation/out/symmetry_phasec_mpirank_matrix \
  MAXSTEP=1 FEQCHKPT=1 RUN_FIELD=t \
  tests/gpu_validation/run_symmetry_phasec_mpirank_matrix.sh
```

The current expected result is 18/18 matrix entries passing across x/y/z symmetry and NP=2/NP=4 topologies. Direct five-step physical-direction checks also pass:

```bash
OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np2_yphysical_5steps \
  BC_KIND=symmetry ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/symmetry_phasec_np2_zphysical_5steps \
  BC_KIND=symmetry ZERO_AXIS=z NP=2 TOPOLOGY=1,1,2 \
  MAXSTEP=5 FEQCHKPT=5 GRID=64,64,64 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

This validates the CPU-compatible Cartesian symmetry path only. It does not imply general curvilinear symmetry normals, wall boundaries, farfield, NSCBC, immersed boundary, species, turbulence, chemistry, compact schemes, or GPU HDF5/checkpoint writing.

## Phase D Artificial TGV `bctype=41` Wall Slices

Use the TGV input template as an artificial boundary-path probe for isothermal no-slip walls in one Cartesian direction. This is not a physical TGV wall validation. It deliberately breaks periodicity in one Cartesian direction, sets the two faces in that direction to `bctype=41` with fixed wall temperature, keeps the other two directions periodic, and compares CPU/GPU boundary/filter/diffusion behavior.

```bash
WALL_AXIS=x MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=y MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=z MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh
```

Validate physical-direction MPI decomposition with:

```bash
WALL_AXIS=x NP=2 TOPOLOGY=2,1,1 MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=y NP=2 TOPOLOGY=1,2,1 MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh

WALL_AXIS=z NP=2 TOPOLOGY=1,1,2 MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t \
  tests/gpu_validation/run_wall41_phased_compare.sh
```

Current expected result: all six commands pass `flowstate.dat` at `1e-9` and `flowfield.h5` at `1e-8`. The latest single-rank 5-step field maxima were reconstructed `q5=5.68e-13` for x, `6.25e-13` for y, and `4.55e-13` for z. The latest NP=2 physical-direction 5-step field maxima were reconstructed `q5=5.12e-13` for x, `5.68e-13` for y, and `5.68e-13` for z.

Validate the reusable wall41 MPI matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/wall41_phased_mpirank_matrix_np2_np4_1step \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=t \
  tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh
```

The default matrix covers 18 entries across x/y/z wall41 and NP=2/NP=4 topologies, including physical-direction and transverse decompositions. Current expected result: `matrix_summary.txt` has 18 pass lines, and each entry passes `flowstate.dat` at `1e-9` and `flowfield.h5` at `1e-8`.

Run the current NP=8 oversubscription smoke with:

```bash
OUT_DIR=tests/gpu_validation/out/wall41_phased_mpirank_matrix_np8_1step \
  MATRIX='x:8:2,2,2 y:8:2,2,2 z:8:2,2,2' \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=t \
  tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/wall41_phased_mpirank_matrix_np8_5step_stats \
  MATRIX='x:8:2,2,2 y:8:2,2,2 z:8:2,2,2' \
  MAXSTEP=5 FEQCHKPT=99 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=f \
  tests/gpu_validation/run_wall41_phased_mpirank_matrix.sh
```

The summary from the latest NP=8 runs has three pass lines. The one-step field smoke passes for all three wall axes, and the five-step statistics-only smoke passes with max statistic differences below `5e-14`. This is two-GPU oversubscription correctness evidence, not performance scaling evidence.

The x/y/z statistics are not expected to match each other because the artificial wall direction changes which TGV structures are cut by no-slip/isothermal faces. This validates only the current explicit, Cartesian, no-species, no-turbulence, no-wall-blowing wall41 slices; it does not imply general wall-boundary physics, wall models, curvilinear wall normals, compact schemes, species, chemistry, or GPU HDF5/checkpoint writing.

## Phase D Channel `bctype=41` Wall Slice

Validate the first wall-bounded channel slice from `examples/Channel/datin/input.chl`. The current GPU support is intentionally narrow: y-direction `bctype=41,41` isothermal no-slip walls, x/z periodic, `numq=5`, no species, no turbulence model, explicit `643e,643e`, RK3, explicit filter/diffusion only.

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_np1_filter_diff_1step \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-8 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: one-step `flowstate.dat` and `flowfield.h5` comparisons print `status: pass`. This validates the GPU `bctype=41` y-wall kernel, y-physical gradient/diffusion/convection path, explicit filter ping-pong path, channel statistics (`massflux`, `fbcx`, `forcex`, `wrms`), and the GPU channel body-force source with variables resident on the device except for scalar reductions and CPU-owned output.

Run the short feedback check with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_np1_filter_diff_2steps_green \
  MAXSTEP=2 FEQCHKPT=2 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: statistics and field comparison pass at `1e-8`. The channel source path now reads `jacob_d` directly inside the GPU kernel, matching the rest of the GPU RHS kernels and avoiding the previous device-array lower-bound mismatch.

Run the current single-rank feedback gate with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_feedback_filter_np1_20steps_after_source_fix \
  MAXSTEP=20 FEQCHKPT=20 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-8 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: statistics and field comparison pass at `1e-8`; the most recent run had final differences near roundoff (`massflux=2.20e-13`, `forcex=3.91e-16`, reconstructed `q5=2.84e-13`).

For longer source/wall/diffusion validation without the channel feedback loop, use a fixed body force:

```bash
OUT_DIR=tests/gpu_validation/out/channel_fixedforce_nofilter_np1_100steps_dt5e4_after_source_fix \
  MAXSTEP=100 FEQCHKPT=100 GRID=32,32,32 DELTAT=5.d-4 \
  LFILTER=f DIFFTERM=t CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  COMPARE_STATS=t COMPARE_FIELD=t STATS_ATOL=1e-8 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_channel_phased_compare.sh
```

Current expected result: statistics and field comparison pass at `1e-8`; the latest 100-step run had reconstructed `q5=6.04e-14` and `u1=4.00e-15`. Do not use `LFILTER=t` beyond the current 20-step gate as a long-step convergence test on this small channel setup: the CPU baseline itself crashes around 23 filtered steps at `DELTAT=5.d-4`, and around 20 filtered steps for smaller `DELTAT`. That is a CPU baseline/filter-frequency stability limit, not a GPU equivalence failure.

For larger multi-rank long-step statistics-only validation, use the original channel grid size and timestep with fixed force and field output disabled:

```bash
OUT_DIR=tests/gpu_validation/out/channel_long_np2_128_100steps_stats \
  MATRIX='2:2,1,1 2:1,2,1 2:1,1,2' \
  GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 FEQCHKPT=9999 \
  LFILTER=f DIFFTERM=t RUN_FIELD=f \
  CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  STATS_ATOL=1e-8 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_long_np4_128_100steps_stats \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' \
  GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 FEQCHKPT=9999 \
  LFILTER=f DIFFTERM=t RUN_FIELD=f \
  CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  STATS_ATOL=1e-8 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_long_np8_128_100steps_stats \
  MATRIX='8:2,2,2' \
  GRID=128,128,128 DELTAT=7.5d-4 MAXSTEP=100 FEQCHKPT=9999 \
  LFILTER=f DIFFTERM=t RUN_FIELD=f \
  CHANNEL_FORCE_MODE=fixed CHANNEL_FORCE_FIXED=1.d-4 \
  STATS_ATOL=1e-8 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

Current expected result: NP=2 `2x1x1`, `1x2x1`, `1x1x2`; NP=4 `2x2x1`, `2x1x2`, `1x2x2`; and NP=8 `2x2x2` all print `status: pass` in `flowstate_compare.txt`. The latest run had maximum observed `massflux=4.998e-13`, `wrms=4.979e-15`, and `forcex=0` differences. This is a long-step CPU/GPU statistics equivalence gate under two-GPU oversubscription, not a production scaling result.

For reproducible wall-clock benchmark reporting, use:

```bash
OUT_DIR=tests/gpu_validation/out/channel_128_benchmark \
  tests/gpu_validation/run_channel_128_benchmark.sh
```

The benchmark driver runs the `128^3`, `DELTAT=7.5d-4`, `MAXSTEP=100`, fixed-force channel case by default. It records explicit CPU/GPU wall times in `benchmark_times.tsv`, writes speedups relative to the NP=1 CPU baseline in `benchmark_summary.md`, and keeps per-topology `flowstate_compare.txt` reports when `RUN_CPU_FOR_ALL=t`.

Validate the current `NP=2` channel slab matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np2_1step \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The default matrix covers `2x1x1`, `1x2x1`, and `1x1x2`. The `1x2x1` entry is the key physical-direction decomposition check: the lower and upper y-wall kernels must apply only on true physical faces, while the internal y interface uses halo-backed central stencils.

Run the short multi-rank feedback matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np2_2steps \
  MAXSTEP=2 FEQCHKPT=2 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

Current expected result: all three entries pass statistics at `1e-8` and field comparison at `1e-6`.

Validate the `NP=4` combined-direction channel matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np4_1step \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np4_2steps \
  MATRIX='4:2,2,1 4:2,1,2 4:1,2,2' \
  MAXSTEP=2 FEQCHKPT=2 STATS_ATOL=2e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The `MAXSTEP=1` matrix is strict at `STATS_ATOL=1e-8` and `FIELD_ATOL=1e-8`. The two-step matrix uses `STATS_ATOL=2e-8` because the channel feedback loop produces `~1.2e-8` mass-flux/force tail differences under combined decompositions.

Validate the `NP=8` `2x2x2` channel smoke with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np8_1step \
  MATRIX='8:2,2,2' \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_matrix_np8_2steps \
  MATRIX='8:2,2,2' \
  MAXSTEP=2 FEQCHKPT=2 STATS_ATOL=2e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The `NP=8` run is a correctness smoke under two-GPU oversubscription. It is not performance evidence and should not be used to infer production scaling.

Validate the `NP=27` `3x3x3` fully interior-rank channel smoke with:

```bash
OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_np27_3x3x3_1step \
  MATRIX='27:3,3,3' \
  MAXSTEP=1 FEQCHKPT=1 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/channel_phased_mpirank_np27_3x3x3_2steps \
  MATRIX='27:3,3,3' \
  MAXSTEP=2 FEQCHKPT=2 STATS_ATOL=3e-8 FIELD_ATOL=1e-6 \
  tests/gpu_validation/run_channel_phased_mpirank_matrix.sh
```

The `3x3x3` topology creates ranks fully wrapped by MPI neighbors in x/y/z, including y-interior ranks that do not touch either wall. This is a high-oversubscription correctness smoke on the current two-GPU machine, not performance evidence.

## Phase E Adiabatic Wall `bctype=42` Slices

Validate CPU-compatible Cartesian `bctype=42,42` no-slip adiabatic walls for x/y directions only. The CPU implementation `noslip_adibatic` currently implements `ndir=1..4`; z-direction wall42 is therefore outside the CPU-supported scope and is intentionally rejected by the validation driver.

```bash
OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_x_5steps \
  BC_KIND=adiabaticwall ZERO_AXIS=x \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_y_5steps \
  BC_KIND=adiabaticwall ZERO_AXIS=y \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate physical-direction MPI gating with:

```bash
OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_x_np2_physical \
  BC_KIND=adiabaticwall ZERO_AXIS=x NP=2 TOPOLOGY=2,1,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/adiabaticwall_phasee_y_np2_physical \
  BC_KIND=adiabaticwall ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: all four commands print `status: pass` for statistics and field comparison; reconstructed `q5` errors stay below `6.3e-13`. `BC_KIND=adiabaticwall ZERO_AXIS=z` exits before running because CPU `noslip_adibatic` has no `ndir=5/6` implementation.

## Phase F Slip-Isothermal Wall `bctype=411` Slice

Validate `bctype=411,411` slip-nonslip isothermal wall behavior for y direction only. The CPU implementation `slipisotwall` implements `ndir=3/4`; x/z directions are therefore outside the CPU-supported scope and are intentionally rejected by the validation driver. The lower wall uses `x <= xslip` for the slip segment and the upper wall follows the CPU no-slip isothermal branch. After C19, the slip segment extrapolates the full velocity and projects it onto the physical tangent plane; optional lower-wall blowing is prescribed on the no-slip section along the physical normal. Species and turbulence models remain outside the GPU validation scope.

```bash
OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np1_5steps \
  BC_KIND=slipisotwall ZERO_AXIS=y \
  XSLIP=3.141592653589793d0 WALL_TEMP=273.15d0 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate physical-face MPI gating and transverse decompositions with:

```bash
OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np2_physical \
  BC_KIND=slipisotwall ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np2_xslab \
  BC_KIND=slipisotwall ZERO_AXIS=y NP=2 TOPOLOGY=2,1,1 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipisotwall_phasef_y_np2_zslab \
  BC_KIND=slipisotwall ZERO_AXIS=y NP=2 TOPOLOGY=1,1,2 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: all commands print `status: pass` for statistics and field comparison; reconstructed `q5` field errors stay below `6e-13`. `BC_KIND=slipisotwall ZERO_AXIS=x/z` exits before running because CPU `slipisotwall` has no `ndir=1/2/5/6` implementation.

## Phase G Slip-Adiabatic Wall `bctype=421` Slice

Validate `bctype=421,421` slip-adiabatic wall behavior for y direction only. The CPU implementation `slipadibwall` implements `ndir=3/4`; x/z directions are therefore outside the CPU-supported scope and are intentionally rejected by the validation driver. After C19, both y slip faces extrapolate the full velocity and project it onto the physical tangent plane. Optional y-min wall blowing is then added along the physical normal; y-max remains tangent-only. The historical commented `xslip` branch is not reactivated.

```bash
OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np1_5steps \
  BC_KIND=slipadibwall ZERO_AXIS=y \
  XSLIP=3.141592653589793d0 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Validate physical-face MPI gating and transverse decompositions with:

```bash
OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np2_physical \
  BC_KIND=slipadibwall ZERO_AXIS=y NP=2 TOPOLOGY=1,2,1 \
  MAXSTEP=5 FEQCHKPT=5 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np2_xslab \
  BC_KIND=slipadibwall ZERO_AXIS=y NP=2 TOPOLOGY=2,1,1 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh

OUT_DIR=tests/gpu_validation/out/slipadibwall_phaseg_y_np2_zslab \
  BC_KIND=slipadibwall ZERO_AXIS=y NP=2 TOPOLOGY=1,1,2 \
  MAXSTEP=3 FEQCHKPT=3 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
  STATS_ATOL=1e-9 FIELD_ATOL=1e-8 \
  tests/gpu_validation/run_xextrap_phaseb_compare.sh
```

Current expected result: all commands print `status: pass` for statistics and field comparison; reconstructed `q5` field errors stay below `6e-13`. `BC_KIND=slipadibwall ZERO_AXIS=x/z` exits before running because CPU `slipadibwall` has no `ndir=1/2/5/6` implementation.

## Phase H Wall-Family Regression Matrix

Use Phase H as the reusable wall-family regression gate after changing boundary, halo, filter, diffusion, or capability-gate code. It combines the currently supported Cartesian wall slices and also checks unsupported CPU-scope directions are still rejected.

```bash
OUT_DIR=tests/gpu_validation/out/wall_family_phaseh_matrix \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
  LFILTER=t DIFFTERM=t RUN_FIELD=t \
  tests/gpu_validation/run_wall_family_phaseh_matrix.sh
```

The default supported matrix covers:

| Boundary | Supported axes in Phase H | MPI coverage |
|---|---|---|
| `bctype=41` isothermal no-slip | x/y/z | physical-direction NP=2 slab for each axis |
| `bctype=42` adiabatic no-slip | x/y | physical-direction NP=2 slab for each supported axis |
| `bctype=411` slip-isothermal | y | physical y NP=2 plus transverse x/z NP=2 slabs |
| `bctype=421` slip-adiabatic | y | physical y NP=2 plus transverse x/z NP=2 slabs |

The default reject matrix checks `42-z`, `411-x`, `411-z`, `421-x`, and `421-z`. These cases should exit before running CPU/GPU solvers because the corresponding CPU boundary routines do not implement those directions. A rejected case passing is a validation failure, not progress.

Useful controls:

```bash
# Parse and list the default matrix without launching ASTR.
DRY_RUN=t tests/gpu_validation/run_wall_family_phaseh_matrix.sh

# Run a focused subset.
MATRIX='slipisotwall:y:2:1,2,1 slipadibwall:y:2:1,2,1' \
  tests/gpu_validation/run_wall_family_phaseh_matrix.sh

# Check only unsupported CPU-scope directions.
RUN_SUPPORTED=f RUN_REJECTS=t \
  tests/gpu_validation/run_wall_family_phaseh_matrix.sh

# Disable field comparison for a faster statistics-only smoke.
RUN_FIELD=f FEQCHKPT=99 tests/gpu_validation/run_wall_family_phaseh_matrix.sh
```

Current expected result: the full default matrix prints 11 supported pass lines and 5 reject pass lines in `matrix_summary.txt`. Each supported entry should print `status: pass` for both `flowstate_compare.txt` and `flowfield_compare.txt`. Field comparison uses the CPU complete-RK validation snapshot so both backends are sampled at the same phase. The current full run passed every supported entry at `1e-10` field tolerance and kept `42-z`, `411-x/z`, and `421-x/z` rejected with exit status 2. Phase H does not expand the physics contract beyond the already validated Cartesian slices; it prevents regressions and accidental scope creep.

## Phase I Lid-Driven-Cavity Gate

Use the LDC Phase I gate to keep the next case expansion explicit. The original `examples/Lid-Driven-Cavity` case is a 2D cavity with x/y non-periodic directions, isothermal walls, and a top-lid `bctype=0` UDF boundary. It is intentionally outside the current GPU contract.

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phasei_gate \
  tests/gpu_validation/run_ldcavity_phasei_gate.sh
```

Current expected result: the GPU run exits nonzero and `gate_summary.txt` reports `pass gpu-reject`. The current baseline rejects the original 2D case with `GPU first-stage supports 3D cases only`.

Useful probes:

```bash
# Confirm the CPU explicit LDC case still completes before using it as a future oracle.
OUT_DIR=tests/gpu_validation/out/ldcavity_phasei_gate_cpu_probe \
  RUN_CPU=t \
  tests/gpu_validation/run_ldcavity_phasei_gate.sh

# Force a 3D explicit probe to expose the next GPU gate after the 2D blocker.
OUT_DIR=tests/gpu_validation/out/ldcavity_phasei_gate_3d \
  GRID=32,32,32 \
  tests/gpu_validation/run_ldcavity_phasei_gate.sh
```

Current status: CPU `RUN_CPU=t` completes a one-step explicit LDC probe without `ieee_invalid`. The warning was traced to `gridcube(1.d0,1.d0,0.d0)` in Release/O2 builds: the old loop contained a non-taken `lz/real(ka,8)` expression when `ka==0`, which NVHPC could still evaluate speculatively and raise `0/0` invalid for 2-D zero-thickness grids. `src/gridgeneration.F90` now precomputes guarded `dx/dy/dz` values before the loop. The same file also keeps original 2-D LDC grid generation for `ka==0`, but uses a unit z length for forced 3-D LDC probes with `ka>0`. The latest original-case check is `OUT_DIR=tests/gpu_validation/out/ldcavity_phaseib_original_filter_reject RUN_CPU=f tests/gpu_validation/run_ldcavity_phasei_gate.sh`; the GPU run still rejects the original/default filtered case as expected. The GPU LDC boundary routine mirrors the CPU boundary order: x isothermal no-slip walls first, y lower isothermal no-slip next, and y upper moving lid last. Therefore the top-lid/side-wall corner velocity follows the CPU implementation and is overwritten to `(u,v,w)=(1,0,0)`.

Validate the narrow Phase I-A no-filter/no-diffusion LDC slice with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseia_compare \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: `flowstate_compare.txt` and `flowfield_compare.txt` both print `status: pass`. This script forces a 3-D LDC probe with `GRID=32,32,32`, `LFILTER=f`, `DIFFTERM=f`, and explicit `643e`. The latest 5-step check:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseia_script_5steps \
  MAXSTEP=5 FEQCHKPT=5 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

passed with max `flowstate.dat` difference `maxq5=2.8620661396416835e-11` and reconstructed field `q5` `L_inf=2.8421709430404007e-14`. This validates x/y physical boundary ownership plus z periodic halo for convection without diffusion.

Validate the Phase I-B no-filter diffusive LDC slice with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseib_diffusion_5steps \
  DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: `flowstate_compare.txt` and `flowfield_compare.txt` both print `status: pass`. The latest 5-step diffusive single-rank check passed with final `flowstate.dat` difference `maxq5=3.9619862945983186e-11` and reconstructed field `q5` `L_inf=8.5265128291212022e-14`.

Validate the current LDC multi-rank x/y physical halo route with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_phaseib_diffusion_np4_221 \
  DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 NP=4 TOPOLOGY=2,2,1 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: the run passes statistics and HDF5 field comparison. This covers x/y internal MPI halos plus true physical x/y faces, while z remains periodic. Phase I-C below supports filtering for the forced 3-D slice; the original 2-D LDC case remains outside the GPU contract.

Validate the current Phase I-C filter isolation with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_filter_only_1step_check \
  LFILTER=t DIFFTERM=f MAXSTEP=1 FEQCHKPT=1 \
  COMPARE_STATS=f COMPARE_FIELD=t \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: strict field comparison prints `status: pass`. The latest run had reconstructed `q5` `L_inf=1.9895196601282805e-13`, so the LDC multi-axis explicit filter path is correct in isolation.

Validate the combined `LFILTER=t,DIFFTERM=t` Phase I-C gate with:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_filter_diff_1step_fixed_clean \
  LFILTER=t DIFFTERM=t MAXSTEP=1 FEQCHKPT=1 \
  COMPARE_STATS=f COMPARE_FIELD=t FIELD_ATOL=1e-10 FIELD_RTOL=1e-10 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: strict field comparison prints `status: pass`; the latest reconstructed `q5` maximum was `2.2737367544323206e-13`. The previous failure at `(i,j,k)=(0,31,1)` was caused by GPU x/y physical diffusion RHS kernels applying all three `is:ie/js:je/ks:ke` restrictions at once. CPU `diffrsdcal6` applies direction-specific RHS ranges, so GPU `xyphysical` diffusion RHS and `x_xphysical` diffusion RHS now follow the same ownership rule.

Use the 5-step strict stats plus field gate for the short regression:

```bash
OUT_DIR=tests/gpu_validation/out/ldcavity_filter_diff_5step_fixed_clean \
  LFILTER=t DIFFTERM=t MAXSTEP=5 FEQCHKPT=5 \
  COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-10 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_ldcavity_phaseia_compare.sh
```

Current expected result: statistics and same-phase field comparisons both print `status: pass`; after the physical-filter closure correction, the latest five-step reconstructed `q5 L_inf` is `3.6948222259525210e-13`. Do not use the current 20-step small-grid LDC run as a correctness gate: CPU/GPU diagnostics both become abnormal or crash-prone by roughly steps 16-20, so that setup is an oracle-quality problem rather than a GPU-only failure.

## Phase J Rayleigh-Taylor Explicit Variant

The canonical Phase J regression driver is:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_matrix_driver_current \
  tests/gpu_validation/run_rti_phasej_matrix.sh
```

Current expected result: `matrix_summary.txt` contains 13 `pass` lines. The default matrix covers single-rank `LFILTER=f/t,DIFFTERM=f/t`, `NP=2` x/y/z slabs, `NP=4` combined slabs, `NP=8 TOPOLOGY=2,2,2`, and 20-step single-rank plus `NP=4 TOPOLOGY=2,2,1`.

Validate the RTI explicit validation variant with:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_5step \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

This driver starts from `examples/Rayleigh–Taylor-Instability/datin/input.rti`, but intentionally rewrites it to a forced 3-D explicit oracle: `GRID=32,64,32`, `643e`, `rk3`, `numq=5`, no species, no turbulence, x/z periodic, y fixed `bctype=31`, and RTI gravity source. It is not a validation of the original compact 2-D RTI input.

Current expected result: statistics and field comparisons both print `status: pass`; the latest single-rank 5-step `LFILTER=t,DIFFTERM=t` run had final `flowstate.dat maxq5=3.5704772471945034e-13` and reconstructed `q5 L_inf=7.9936057773011271e-15`. The single-rank 1-step and 5-step matrix also passed for all `LFILTER=f/t` and `DIFFTERM=f/t` combinations.

Validate the current two-rank RTI halo slices with:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_np2_121 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=2 TOPOLOGY=1,2,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_np2_211 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=2 TOPOLOGY=2,1,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_filter_diff_np2_112 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=2 TOPOLOGY=1,1,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

Current expected result: all three NP=2 entries print `status: pass` for both reports. The `1,2,1` case covers true y fixed physical faces plus internal y halo, while `2,1,1` and `1,1,2` cover transverse x/z periodic MPI halos. The latest maximum observed reconstructed `q5 L_inf` was `8.8817841970012523e-15`, and the maximum observed final `flowstate.dat maxq5` difference was `4.4408920985006262e-13`.

The current extended RTI multi-rank matrix also passes:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_221 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=2,2,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_212 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=2,1,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_122 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=1,2,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np8_222 \
  MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=8 TOPOLOGY=2,2,2 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

Current expected result: all four entries print `status: pass` for both reports. The latest 5-step extended matrix had maximum reconstructed `q5 L_inf=1.0658141036401503e-14` and maximum `flowstate.dat maxq5=4.6984638402136625e-13`.

Use the 20-step RTI checks as the current stronger short-run regression:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/sr20 \
  MAXSTEP=20 FEQCHKPT=20 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=1 TOPOLOGY=1,1,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh

OUT_DIR=tests/gpu_validation/out/rti_phasej_current_matrix/np4_221_20 \
  MAXSTEP=20 FEQCHKPT=20 LFILTER=t DIFFTERM=t GRID=32,64,32 \
  NP=4 TOPOLOGY=2,2,1 COMPARE_STATS=t COMPARE_FIELD=t \
  FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 STATS_ATOL=1e-10 STATS_RTOL=1e-10 \
  tests/gpu_validation/run_rti_phasej_compare.sh
```

Current expected result: both 20-step entries print `status: pass`. The latest single-rank run had reconstructed `q5 L_inf=1.0658141036401503e-14` and `flowstate.dat maxq5=4.9382720135326963e-13`; the latest `NP=4 TOPOLOGY=2,2,1` run had reconstructed `q5 L_inf=1.3322676295501878e-14` and `flowstate.dat maxq5=4.7073456244106637e-13`.

## Phase K Source Capability Regression

The canonical Phase K source regression driver is:

```bash
OUT_DIR=tests/gpu_validation/out/source_phasek_matrix_current \
  tests/gpu_validation/run_source_phasek_matrix.sh
```

Current expected result: `matrix_summary.txt` contains three `pass` lines:

- `tgv_nosource`: verifies the source dispatcher leaves no-source cases alone.
- `channel_source`: verifies the channel source path and `lihomo`-gated source dispatch.
- `rti_source`: verifies the RTI source path with `NP=2 TOPOLOGY=1,2,1`, including y fixed boundary plus internal y halo.

After changing the source capability layer, also rerun the full RTI matrix:

```bash
OUT_DIR=tests/gpu_validation/out/rti_phasej_matrix_after_capability \
  tests/gpu_validation/run_rti_phasej_matrix.sh
```

Current expected result: all 13 RTI entries pass. The source capability table currently covers only `GPU_SOURCE_NONE`, `GPU_SOURCE_CHANNEL`, and `GPU_SOURCE_RTI`; it is not chemistry, turbulence, or generic UDF source support.

## Phase S0-A Shock-Format Readiness

S0-A1 is the first completed shock-format plumbing gate. It uses a forced 3-D extruded Sod case to open the explicit upwind path without adding open boundaries, high-speed wall coupling, characteristic decomposition, sensors, diffusion, or filtering.

Validated S0-A1 contract:

- `flowtype='sod'` through the `sodini` initialization path
- controlled validation input, not `examples/sod/datin/input.sod` as-is
- `GRID=200,8,8`
- `deltat=5.d-4`, `maxstep=20`
- periodic x/y/z boundaries, with the run short enough to avoid wave interaction with the x-periodic boundary
- `conschm='543e'`, `difschm='643e'`
- `recon_schem=-1`, `lchardecomp=.false.`
- `diffterm=f`, `lfilter=f`
- CPU/GPU statistics tolerances `STATS_ATOL=1e-10`, `STATS_RTOL=1e-10`
- field tolerances `FIELD_ATOL=1e-10`, `FIELD_RTOL=1e-10`
- at minimum, report max differences for `q(:,:,:,1:5)`

Run the canonical gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a1_gate_200x8x8_20 \
  tests/gpu_validation/run_sod_phase_s0a1_compare.sh
```

Current status: pass. The controlled `sod` path generates a positive-volume 3-D grid, the GPU capability gate accepts only `NP=1`, and compact upwind plus characteristic decomposition remain rejected. The recorded 20-step run passed statistics and field checks at `1e-10`. Maximum statistic differences were `maxq2=4.3021142204224816e-15` and `maxq5=1.3322676295501878e-15`; maximum conservative-field differences were `q1=2.2204460492503131e-16`, `q2=8.8446089343311681e-17`, `q3=3.1150835073543049e-18`, `q4=3.1456319031046116e-18`, and `q5=1.7763568394002505e-15`.

This result validates first-order Steger-Warming flux-splitting plumbing only. It is not a high-order shock-accuracy claim.

S0-A2 adds the CPU-semantic WENO reconstruction with `recon_schem=1`, `lchardecomp=.false.`, and the same periodic Sod oracle. In the current CPU implementation this selection uses WENO7 on periodic interior interfaces; WENO5 is only a physical-boundary fallback inside `recons_exp`.

Run the canonical S0-A2 gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a2_gate_200x8x8_20 \
  tests/gpu_validation/run_sod_phase_s0a2_compare.sh
```

Current S0-A2 status: pass. The recorded 20-step run passed statistics and field checks at `1e-10`. Maximum statistic differences were `maxq1=4.7384318691001681e-13` and `maxq5=4.1477932199995848e-13`; maximum conservative-field differences were `q1=8.8817841970012523e-16`, `q2=6.5503158452884060e-17`, `q3=1.1111687016389874e-17`, `q4=1.1204301546017956e-17`, and `q5=1.3322676295501878e-15`. The GPU path reuses one haloed scalar `flux_work_d` across directions and conservative components. This is a porting-equivalence result; Sod exact-solution error and discontinuity resolution remain a separate accuracy gate.

Run the independent S0-A2 exact-solution gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a2_accuracy \
  tests/gpu_validation/run_sod_phase_s0a2_accuracy.sh
```

The accuracy driver uses `GRID=800,5,5`, `deltat=5.d-4`, and `maxstep=400`, giving `t=0.2` and `dx=0.0125`. Every 3-D dimension must be at least `hm=5`; the comparison driver rejects smaller dimensions before launching ASTR because the CPU program otherwise only warns and can continue into invalid halo accesses. The exact solver uses the standard ideal-gas Sod states, extracts the x profile from ASTR HDF5 output without assuming h5py axis order, and evaluates only the central Riemann problem so the periodic-boundary discontinuity is excluded.

The gate reports normalized `L1`, `L2`, and `Linf` errors for `ro`, `u1`, `p`, and `q1:q5` away from wave fronts; normalized primitive-variable over/undershoot; contact and shock 10%-90% density thickness; and 50% crossing-position error. The finite limits are `1e-3`, `6e-3`, `6e-2`, `2e-2`, 4 cells, 3 cells, and 1 cell, respectively. A missing 10%, 50%, or 90% density crossing is reported as an unresolved transition and fails even when numeric thresholds are disabled.

Current exact-solution status: pass for both CPU and GPU. The recorded calibration under `tests/gpu_validation/out/sod_s0a2_accuracy_calibration_800x5x5_400` had maximum smooth errors `L1=7.1864e-4`, `L2=4.0087e-3`, and `Linf=4.6338e-2`; maximum normalized bound violation `1.3889e-2`; contact thickness `2.9780` cells; shock thickness `2.1989` cells; and maximum position error `0.7527` cells. CPU/GPU reconstructed-field `Linf` was at most `q5=8.4377e-15`. Recorded step-400 times were `74.907 s` for NP=1 CPU and `8.102 s` for NP=1 GPU; these timings are supporting evidence from the accuracy run, not a controlled performance benchmark.

S0-A3 adds the CPU-semantic MP reconstruction with `recon_schem=3`. The controlled periodic path uses MP7 at every interface; CPU MP5 physical-boundary degradation is outside this gate. WENO7 and MP7 share the generic explicit-reconstruction kernels and the same scalar `flux_work_d`; the reconstruction selector is a kernel argument, and every positive-flux, negative-flux, and directional-RHS kernel remains explicitly synchronized.

Run the S0-A3 equivalence and exact-solution gates with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_s0a3_gate_200x8x8_20 \
  tests/gpu_validation/run_sod_phase_s0a3_compare.sh

OUT_DIR=tests/gpu_validation/out/sod_s0a3_accuracy_800x5x5_400 \
  tests/gpu_validation/run_sod_phase_s0a3_accuracy.sh
```

Current S0-A3 status: pass. The 20-step CPU/GPU comparison passed statistics and fields at `1e-10`; maximum statistic differences were `maxq1=4.6985e-13` and `maxq5=4.6629e-13`, and maximum conservative-field difference was `q5=1.3323e-15`. The 400-step accuracy run passed the unchanged S0-A2 finite thresholds. Maximum smooth errors were `L1=7.2439e-4`, `L2=4.1951e-3`, and `Linf=4.8797e-2`; maximum normalized bound violation was `8.5021e-3`; contact and shock thicknesses were `2.9496` and `1.6939` cells; and maximum position error was `0.7272` cells. MP7 resolved the shock more sharply and reduced overshoot relative to this WENO7 run, while its largest smooth-region errors were slightly higher. Same-run CPU/GPU field `Linf` was at most `q5=1.2879e-14`. Recorded step-400 times were `57.408 s` for NP=1 CPU and `7.277 s` for NP=1 GPU; treat these as supporting run evidence, not a controlled benchmark. A one-step S0-A3 Compute Sanitizer memcheck also completed with `ERROR SUMMARY: 0 errors` under `mpirun -np 1` and the `ob1/self/pt2pt` MPI settings.

S0-A4 isolates the Ducros/pressure-curvature sensor before enabling Roe characteristic reconstruction. The driver forces Shu-Osher onto a positive-volume 3-D periodic grid because the initial Sod velocity is zero and would make the Ducros compression factor an all-zero oracle. It uses `GRID=400,8,8`, `deltat=1.d-4`, `maxstep=1`, `recon_schem=3`, `lchardecomp=f`, `LFILTER=f`, and `DIFFTERM=f`.

Run the S0-A4 single-rank sensor gate with:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_sensor_s0a4_compare \
  tests/gpu_validation/run_shuosher_sensor_s0a4_compare.sh
```

Current S0-A4 status: pass. `compare_shock_sensor.py` checks every raw-sensor value and requires the byte mask to match exactly. The maximum raw-field difference was `1.1102230246251565e-16`; CPU/GPU raw maxima were both `6.9999992499998120e-1`, averages were `4.3855622929106558e-3` and `4.3855622929106566e-3`, and both masks contained 1944 shock nodes with zero mismatches. One-step statistics passed at `1e-10`, and the largest conservative-field difference was `q5=4.4408920985006262e-16`. A three-stage one-step Compute Sanitizer run reported `ERROR SUMMARY: 0 errors`. This is a validation-only capability selected by `ASTR_SHOCK_SENSOR_DUMP`: the mask is not yet consumed by MP7/Roe fluxes.

S0-A5 extends the raw sensor to the existing generic `hm`-layer HaloTransport for `NP=2`, `TOPOLOGY=2,1,1`. The driver sets `ASTR_SHUOSHER_SHOCK_X=0.d0`, shifting the test discontinuity beside the global x-slab interface while preserving the normal Shu-Osher initialization when the variable is absent. Each rank writes its local sensor dump with global offsets; `compare_shock_sensor.py` merges the rank files, verifies overlapping nodes, and compares the global CPU/GPU field and mask.

Run the S0-A5 sensor-halo gate with:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_sensor_s0a5_mpi_compare \
  tests/gpu_validation/run_shuosher_sensor_s0a5_mpi_compare.sh
```

Current S0-A5 status: pass. The merged global shape is `401,9,9`; raw `L_inf=1.1102230246251565e-16`, mask mismatches are zero, and both global masks contain 1944 shock nodes. The one-step multi-rank field comparison passed at `1e-10`, with `q5 L_inf=2.7089441800853820e-14`. The two-rank sanitizer command requires `OMPI_MCA_btl=self,tcp`, `OMPI_MCA_coll='^hcoll,ucc'`, `OMPI_MCA_opal_cuda_support=0`, and `UCX_MEMTYPE_CACHE=n`; with those MPI components disabled, both ranks report `ERROR SUMMARY: 0 errors`. This establishes raw-sensor halo correctness only for `2x1x1`; Roe characteristic reconstruction remains separate.

S0-A6 consumes the CPU-equivalent expanded mask in a single-rank selective Roe characteristic path. The controlled gate uses the same forced-3D periodic Shu-Osher configuration, but sets `lchardecomp=t`. Sensor-active interfaces project the five split Euler fluxes with the local Roe left eigenvectors, reconstruct each characteristic component with MP7, and project the interface flux back with the right eigenvectors. Other interfaces retain physical-space MP7. A reusable five-component `flux_characteristic_work_d` stores one direction at a time before the conservative flux difference.

Run the S0-A6 characteristic gate with:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a6_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a6_compare.sh
```

Current S0-A6 status: pass for `NP=1`. The one-step raw sensor and mask retain `L_inf=1.1102230246251565e-16`, zero mismatches, and 1944 shock nodes. The one-step conservative field maximum is `q5=4.4408920985006262e-16`; the maximum statistic difference is `4.9098503041022923e-12`. A three-step repetition passed with `q5 L_inf=1.4210854715202004e-14`. The prescribed x block remains `(512,1,1)`; a file-local NVHPC `-gpu=maxregcount:128` cap is required because the initial kernel used 255 registers/thread and could not launch at 512 threads. The characteristic kernel now caches each five-component Steger-Warming split for the original seven positive and seven negative MP7 stencil positions before Roe projection. This preserves the operator and adds no global workspace, kernel, or synchronization. On S0-A6, x/y/z characteristic-flux means are `1.051/1.236/1.440 ms` versus `2.026/3.116/3.948 ms` before caching; x-kernel achieved occupancy rises from about 14% to 22%. Compute Sanitizer passed the three-step GPU run with `ERROR SUMMARY: 0 errors`. This capability is limited to periodic Shu-Osher with `ASTR_SHOCK_SENSOR_DUMP`; S0-A7 through S0-A10 separately validate multi-rank characteristic reconstruction.

## Phase S0-B0 X Physical Reconstruction

Run the controlled x-physical explicit-MP7 gate with:

```bash
OUT_DIR=tests/gpu_validation/out/sod_phase_s0b0_xphysical \
  tests/gpu_validation/run_sod_phase_s0b0_xphysical_compare.sh
```

The gate uses forced-3D Sod, `bctype=50,50,1,1,1,1`, `conschm=543e`, `recon_schem=3`, and no characteristic decomposition, diffusion, or filtering. It is not an open-boundary case. The x interface/RHS kernels preserve CPU active ranges and `npdci` semantics: physical-rank interfaces use two-point average, SUW3, MP5, then MP7; the opposite MPI face retains the exchanged `hm` halo. The 20-step same-topology CPU/GPU gate passes for NP=1 and `NP=2 TOPOLOGY=2,1,1`; NP=2 full-field `q5 L_inf=5.8841820305133297e-15`, and both memcheck ranks report zero errors.

The corresponding selective-Roe physical-face gate uses forced-3D Shu-Osher:

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0b0_xphysical \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=3 FEQCHKPT=3 \
  tests/gpu_validation/run_shuosher_characteristic_s0b0_xphysical_compare.sh
```

It keeps `bctype=50,50`, MP7, `lchardecomp=t`, and no diffusion/filter. The
driver checks rank-local raw sensor and mask data for MPI runs, then statistics
and full fields. NP=1 and `NP=2 TOPOLOGY=2,1,1` pass at three steps; NP=2 has
raw `L_inf=1.1102230246251565e-16`, zero mask mismatches, and `q5
L_inf=7.1054273576010019e-15`. This gate also corrects the CPU reference:
`npdci=4` must clamp both physical sides in `ducrossensor`; leaving it untreated
reads undefined physical-face halos. The GPU sensor matches that rule and skips
the periodic x `ssf` fill on a physical x face. It is not inflow/outflow, NSCBC,
or sponge validation.

## Phase S0-B1 Classical Open Boundary

```bash
OUT_DIR=tests/gpu_validation/out/openshock_s0b1 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_openshock_s0b1_compare.sh
```

This gate is a stationary Mach-3 normal shock at x=0, with x-min `11,free` and
x-max `21,10.333333333333333`. The second outlet value is `p2/pinf`; setting
`pout=1` would impose upstream pressure at a subsonic outlet and invalidate the
stationary state. Ten-step NP=1 and `NP=2 TOPOLOGY=2,1,1` comparisons pass with
`q5 L_inf=8.8817841970012523e-16` and `9.9920072216264089e-16`, respectively.
NSCBC, sponge, diffusion, filtering, and characteristic Roe are excluded.

## Phase S0-B2 NSCBC Open Boundary

`run_openshock_s0b2_nscbc_compare.sh` prepares the restricted `12/22`
OpenShock input. The GPU implementation includes characteristic boundary RHS
terms, separate inlet-domain/outlet-plane Mach reductions, and the CPU
sixth-order explicit outlet y-filter. CPU `time_integration_rk` refreshes
halos with a full-rank pre-`boucon` `qswap` when `bctype=22` is present; this
defines the CPU filter input without changing the normal post-boundary swap.
Ten steps pass for NP=1 (`q5 L_inf=8.8817841970012523e-16`) and NP=2
`TOPOLOGY=2,1,1` (`q5 L_inf=9.9920072216264089e-16`). The NP=2 memcheck run,
with the documented `ob1/self,tcp/pt2pt` MPI settings, reports zero errors on
both ranks. Scope remains Cartesian x faces, periodic y/z, no species, no
sponge, no global filter, and physical-space MP7 only.

## Phase S0-B3 x-Max Sponge

```bash
OUT_DIR=tests/gpu_validation/out/openshock_s0b3_sponge_np2 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_openshock_s0b3_sponge_compare.sh
```

This gate retains the S0-B2 `12/22` NSCBC pair and sets only `spg_im=80` on
`GRID=400,8,8`. It is not a reference-state relaxation: after each RK update
the CPU-equivalent device path refreshes all applicable q halos, then smooths the
x-max active layer in conservative variables with the geometric coefficient
field and the six direct neighbors. The temporary values use `qwork_d`; no
additional full-field device allocation or stage-level D2H/H2D copy occurs.
Ten-step NP=1 and NP=2 `2x1x1` comparisons pass with final `q5
L_inf=8.8817841970012523e-15`; the one-step two-rank Compute Sanitizer run
reports zero errors on both ranks. Only x-max `layer` mode is enabled.

## Phase S0-B4 Combined Sponge MPI

```bash
OUT_DIR=tests/gpu_validation/out/openshock_s0b4_sponge_2x2x2 \
  NP=8 TOPOLOGY=2,2,2 GRID=400,16,16 MAXSTEP=10 FEQCHKPT=10 \
  tests/gpu_validation/run_openshock_s0b3_sponge_compare.sh
```

This is the combined x/y/z MPI oracle for the B3 x-max layer. It corrected a
CPU reference error: the six-neighbor stencil requires current halos in every
direction, so `spongefilter_layer` now calls full `dataswap(q)` before each
layer. CUDA Fortran uses the matching q-only three-direction halo exchange.
The ten-step gate passes with final `q5 L_inf=8.8817841970012523e-15`; an
eight-rank one-step Compute Sanitizer run reports zero errors for every rank.
This workstation test oversubscribes two GPUs and is not a scaling measurement.

## Phase S1-A0/A1 Explicit Flat Plate

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a_20steps \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The driver creates an isolated `64x64x8` Cartesian, z-extruded `bl` case and
a nonuniform `datin/inlet.prof`. It uses non-dimensional `Mach=0.3`, explicit
central `643e/643e`, RK3, diffusion, no global filter, x `11,prof/21`, y
`41/51`, and periodic z. The CPU-active `51` branch is an extrapolative
upper-boundary update; its characteristic implementation is commented out in
the CPU source and is not claimed here. GPU uploads the profile once after
`flowinit`, keeps it on device, computes the matching four BL statistics on
device, and compares the final full field plus `massflux`, `fbcx`,
`wallheatflux`, and `wrms`. The 20-step gate passes at `1e-10`; a valid
one-rank Compute Sanitizer run reports zero errors.

S1-A1 uses the same case with explicit MP7 convection (`543e/643e`). Select
it through `CONSCHM=543e`:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_20steps \
  CONSCHM=543e MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The CPU reference now accepts the single-rank double-physical-face encoding
`npdci=npdcj=4`: `recons_exp` applies the same two-point, SUW3, MP5, then MP7
sequence at both ends, and `convrsduwd` uses the physical `[0,dim]` stencil
range for y/z as it already did for x. CUDA uses matching x/y face degradation
and restricts every x/y/z explicit-upwind RHS update to CPU active ranges
`is:ie`, `js:je`, and `ks:ke`. The Steger-Warming split reads local and
exchanged-halo metric entries directly; physical `j=jm` no longer reads the
periodic `jacob(:,0,:)`. The 20-step A1 gate passes with final
`q5 L_inf=1.7763568394002505e-14`, maximum statistic difference
`9.6256336235001072e-14`, and one-rank memcheck reports zero errors. Scope is
Cartesian, profile inflow, explicit MP7, and the active extrapolative `51`
upper boundary; it is not a curvilinear, high-Mach, multi-axis-MPI, or
characteristic-farfield claim.

The same S1-A1 gate also passes in two x slabs. The profile and both y physical
faces remain local to each rank; only x uses the existing solution halo exchange:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np2_20steps \
  CONSCHM=543e NP=2 TOPOLOGY=2,1,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The final parallel field result is `q5 L_inf=1.7763568394002505e-14` and the
maximum statistic difference is `5.3623772089395061e-14`. Both ranks report
`ERROR SUMMARY: 0 errors` in the one-step MPI Compute Sanitizer gate.

The same controlled MP7 case passes in two y slabs. This is the physical-y
decomposition gate: the two ranks exchange solution and geometry halos at the
internal y interface, while only the global lower-y rank contributes `fbcx`,
`wallheatflux`, and wall area to the BL statistic reduction.

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np2_yslab_20steps \
  CONSCHM=543e NP=2 TOPOLOGY=1,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The 20-step result has `q5 L_inf=1.7763568394002505e-14` and maximum statistic
difference `5.4956039718945249e-14`; the two-rank memcheck reports
`ERROR SUMMARY: 0 errors` for both ranks. `REYNOLDS` is an optional driver
parameter for diagnostic runs; the validated physical case uses `REYNOLDS=1000`.
The periodic z direction also passes in two z slabs:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np2_zslab_20steps \
  CONSCHM=543e NP=2 TOPOLOGY=1,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The 20-step result has `q5 L_inf=1.7763568394002505e-14`, maximum statistic
difference `5.4733995114020217e-14`, and a zero-error two-rank memcheck.

The first combined topology is also validated under two-GPU oversubscription:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np4_2x2x1_20steps \
  CONSCHM=543e NP=4 TOPOLOGY=2,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

This `2x2x1` gate passes with `q5 L_inf=1.7763568394002505e-14`, maximum
statistic difference `4.9737991503207013e-14`, and four zero-error memcheck
reports. The x/z combined topology is also validated under the same two-GPU
oversubscription:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np4_2x1x2_20steps \
  CONSCHM=543e NP=4 TOPOLOGY=2,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The `2x1x2` gate passes with `q5 L_inf=1.7763568394002505e-14`, maximum
statistic difference `5.0071058410594561e-14`, and four zero-error memcheck
reports. The y/z combined topology is also validated:

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np4_1x2x2_20steps \
  CONSCHM=543e NP=4 TOPOLOGY=1,2,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The `1x2x2` gate has the same final `q5 L_inf=1.7763568394002505e-14`,
maximum statistic difference `5.0071058410594561e-14`, and four zero-error
memcheck reports. The full three-direction topology needs `KM=16`: the default
`KM=8` would make each z slab four points wide, below `hm=5` for MP7.

```bash
OUT_DIR=tests/gpu_validation/out/s1_flatplate_s1a1_np8_2x2x2_20steps \
  IM=64 JM=64 KM=16 CONSCHM=543e NP=8 TOPOLOGY=2,2,2 \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_flatplate_s1a_compare.sh
```

The `2x2x2` gate passes with final `q5 L_inf=1.7763568394002505e-14`, maximum
statistic difference `4.9737991503207013e-14`, and eight zero-error memcheck
reports. All four- and eight-rank cases are two-GPU-oversubscribed correctness
evidence only. Curvilinear geometry, high-Mach conditions, and characteristic
farfield behavior remain outside this contract.

## Phase S1-B0 HBL-Inspired Mach 3 Gate

`examples/Hypersonic_Boundary_Layer` is not directly GPU-runnable: its inputs
are two-dimensional, compact, and mostly dimensional. The S1-B0 driver creates
a separate non-dimensional, z-extruded explicit counterpart with the M3 x
extent, wall-normal clustering, `Mach=3`, `Re=100000`, and the M3 heated-wall
ratio `Twall/Tinf=568.89/226.65`:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b0_20steps \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

It uses explicit MP7 `543e/643e`, a deterministic heated profile, `11,prof`,
`21`, lower `41`, upper `51`, diffusion, no global filter, no characteristic
decomposition, and periodic z. The 20-step CPU/GPU gate passes with
`q5 L_inf=5.5511151231257827e-16`, maximum statistic difference
`7.8936857050848630e-14`, and a one-rank Compute Sanitizer report of zero
errors. This validates high-Mach parameters and the heated-wall explicit path,
not the original two-dimensional compact HBL case, characteristic farfield, or
SBLI behavior.

S1-B1 extends the same controlled HBL gate to individual MPI slabs:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b1_np2_x20 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b1_np2_y20 \
  NP=2 TOPOLOGY=1,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b1_np2_z20 \
  KM=16 NP=2 TOPOLOGY=1,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

All three runs pass with final `q5 L_inf=5.5511151231257827e-16`; their
maximum statistic differences are `6.3726801613483985e-14`,
`5.9729998724833422e-14`, and `7.8936857050848630e-14` for x/y/z,
respectively. The z slab requires `KM=16` because `KM=8` would leave only four
active z points per rank, below `hm=5`. A two-rank physical-y Compute Sanitizer
run reports zero errors on both ranks. Combined-axis and production-scaling
claims remain out of scope.

S1-B2 covers the three NP=4 two-axis combinations:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b2_np4_2x2x1_20 \
  NP=4 TOPOLOGY=2,2,1 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b2_np4_2x1x2_20 \
  KM=16 NP=4 TOPOLOGY=2,1,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b2_np4_1x2x2_20 \
  KM=16 NP=4 TOPOLOGY=1,2,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

All three runs pass with final `q5 L_inf=5.5511151231257827e-16`; maximum
statistic differences are `5.0182080713057076e-14`,
`6.3726801613483985e-14`, and `5.9729998724833422e-14`, respectively. The
`1x2x2` four-rank Compute Sanitizer run reports zero errors for every rank.
These are two-GPU-oversubscribed correctness tests. The full `2x2x2` HBL gate
remains separate.

S1-B3 closes the controlled HBL topology matrix with full x/y/z decomposition:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1b3_np8_2x2x2_20 \
  KM=16 NP=8 TOPOLOGY=2,2,2 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1b0_compare.sh
```

The `96x96x16` grid gives each rank `48x48x8` active points. The 20-step run
passes with `q5 L_inf=5.5511151231257827e-16`, maximum statistic difference
`5.0182080713057076e-14`, and zero-error Compute Sanitizer summaries from all
eight ranks. This is two-GPU-oversubscribed full-halo correctness evidence, not
a multi-GPU scaling result.

## Phase S1-C1 Mach 5 Sutherland Similarity Inlet

S1-C1 is the first HBL gate whose inlet comes from a coupled, isothermal-wall
compressible Blasius solution rather than an analytic exponential seed. The
generator uses `gamma=1.4`, `Pr=0.72`, Sutherland viscosity with
`Tref=226.65 K`, `Mach=5`, `Re=1.83052e6`, and
`Twall/Tinf=1176.64/226.65=5.191440547760865`. Its virtual leading-edge
station is `STATION_X=1.0`; the generated profile is imposed at the domain
inlet, whose x coordinate is `-1`.

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c1_m5_similarity_20 \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh
```

The `192x192x8` explicit MP7 case has a computed
`delta_99=9.5190551023136317e-3`. The 20-step CPU/GPU result passes with
`q5 L_inf=4.4408920985006262e-16` and maximum statistic difference
`3.5171865420124959e-13`.

The profile first line selects its density contract. `density=reconstruct` is
the backward-compatible default: ASTR sets `p=pinf` and reconstructs density
from temperature. `density=provided` preserves file density and computes
pressure from the ideal-gas equation of state; it rejects the input if that
pressure differs from `pinf` by more than `1e-10`. The similarity generator
writes the mathematically equivalent `rho=1/T` in both modes.

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c1_density_reconstruct \
  PROFILE_DENSITY_MODE=reconstruct MAXSTEP=2 \
  tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c1_density_provided \
  PROFILE_DENSITY_MODE=provided MAXSTEP=2 \
  tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh
```

Both modes pass their CPU/GPU comparisons. Comparing their CPU fields gives
maximum `T L_inf=6.2172489379008766e-15`; the difference is floating-point
round-off from evaluating `1/T` directly or through `pinf` and the equation of
state. This gate does not prove a complete physical flat-plate solution:
`blini` still copies the inlet profile over x at initialization, `51` is not a
characteristic farfield, and mesh/time/streamwise-development convergence and
external `Cf`/heat-transfer validation remain pending.

## Phase S1-C2 Mach 5 Similarity Field

C2 removes C1's x-uniform initialization by writing the existing CPU-readable
`datin/flowini3d.h5` and selecting `ninit=3`. The generator solves the C1
similarity ODE once, then maps it at every x with the local
`sqrt(2*x_s/Re)` scale. `VIRTUAL_LEADING_EDGE=-2` makes the `[-1,10]` domain
cover similarity stations `x_s=[1,12]`; HDF initialization remains CPU-owned,
followed by the normal one-time device upload.

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_similarity_field_20 \
  MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh
```

The `192x192x8` 20-step CPU/GPU run passes with
`q5 L_inf=7.7715611723760958e-16` and maximum statistic difference
`3.3406610810970960e-13`. This validates the HDF initial-field reader and GPU
resident-field handoff, not combined-axis multi-rank HDF I/O, mesh/time
convergence, characteristic farfield treatment, or external skin-friction/
heat-transfer comparison.

The same C2 HDF field is validated in NP=2 x/y/z slabs:

```bash
OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_field_np2_x20 \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_field_np2_y20 \
  NP=2 TOPOLOGY=1,2,1 MAXSTEP=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh

OUT_DIR=tests/gpu_validation/out/s1_hbl_s1c2_m5_field_np2_z20 \
  KM=16 NP=2 TOPOLOGY=1,1,2 MAXSTEP=20 \
  tests/gpu_validation/run_s1_hbl_s1c2_m5_similarity_field_compare.sh
```

The x/y runs have maximum statistic differences `1.9806378759312793e-13` and
`2.9809488211185453e-13`; z has `3.3406610810970960e-13` and field
`q5 L_inf=9.9920072216264089e-16`. `KM=16` is required for z because each
rank otherwise has only four active z points, below `hm=5`.

## Phase S1-C3 Mach 5 NSCBC Farfield

S1-C3 is the candidate upper `bctype=52` NSCBC farfield gate for the same
Mach-5 flat-plate family. CPU probing initially showed that
`bc:farfield_nscbc` could apply its transverse filter on the `jmax` boundary
using stale k-halo values before a current halo refresh. In that state an
initially constant upper physical boundary was filtered into a nonconstant
range such as `0.84375..1.078125`, which is a halo artifact rather than a valid
characteristic-farfield result.

Under the CPU bug decision gate, GPU support did not reproduce that artifact.
The CPU oracle now refreshes halos before `boucon` whenever either
`bctype=22` or `bctype=52` is present, and the GPU `52` path is validated
against that corrected oracle:

```bash
tests/gpu_validation/run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh

OUT_DIR=/tmp/astr_s1c3_52_20 MAXSTEP=20 FEQCHKPT=20 \
  tests/gpu_validation/run_s1_hbl_s1c3_m5_nscbc_farfield_compare.sh
```

Current expected result: both statistics and full-field comparisons print
`status: pass`. The latest default `192x192x8` two-step gate had reconstructed
`q5 L_inf=4.4408920985006262e-16`. The latest 20-step gate had reconstructed
`q5 L_inf=1.4432899320127035e-15` and maximum statistic difference
`3.4716673980028645e-13`.

## Phase S2-A0 Oblique-Shock HBL Compatibility

S2-A0 is the first deliberately narrow shock/boundary-layer compatibility gate.
It does not claim a resolved physical SBLI. It keeps the S1 Mach-5
Sutherland-viscosity flat-plate contract, initializes from `ninit=3`, and adds
an analytic oblique-shock overlay to the x-varying compressible Blasius HDF
field. The overlay updates `rho`, `T`, `u`, and `v` together using perfect-gas
oblique-shock ratios so CPU `readflowini3d` reconstructs a consistent pressure
from `rho*T`.

```bash
tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh

IM=64 JM=64 KM=8 MAXSTEP=2 FEQCHKPT=2 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_oblique_shock_smoke \
  tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh

OUT_DIR=tests/gpu_validation/out/s2_hbl_oblique_shock_mpirank_matrix \
  tests/gpu_validation/run_s2_hbl_oblique_shock_mpirank_matrix.sh

OUT_DIR=tests/gpu_validation/out/s2_hbl_inlet_sustained_shock \
  tests/gpu_validation/run_s2_hbl_inlet_sustained_shock_compare.sh

OUT_DIR=tests/gpu_validation/out/s2_hbl_inlet_sustained_shock_mpirank_matrix \
RUN_NP1=f RUN_NP2=t RUN_NP4=t RUN_NP8=t RUN_LONG=f RUN_STRESS=f \
PROFILE_OBLIQUE_SHOCK=t PROFILE_PRESSURE_MODE=provided \
SHOCK_X0=-1.0 SHOCK_Y0=0.18 PROFILE_SHOCK_Y_MIN=0.18 \
  tests/gpu_validation/run_s2_hbl_oblique_shock_mpirank_matrix.sh
```

Current expected result: statistics and full-field comparisons print
`status: pass`. The current `64x64x8` smoke gate passes at one and two steps
with reconstructed `q5 L_inf=8.8817841970012523e-16` for the two-step run. The
default `192x192x8` two-step gate also passes with reconstructed
`q5 L_inf=1.3322676295501878e-15` and maximum statistic difference
`1.0755840662568517e-12`. The same two-step default gate now passes NP=2 x/y/z
slabs, NP=4 xy/xz/yz planes, and `NP=8 TOPOLOGY=2,2,2`; z-decomposed cases use
`KM=16` so local `km>=hm`. All MPI runs retain `q5 L_inf=1.3322676295501878e-15`.
The S2-B0 long subset currently passes `MAXSTEP=20` for NP=1 and
`NP=8 TOPOLOGY=2,2,2`; both have reconstructed `q5 L_inf=3.3306690738754696e-15`,
with largest statistic difference `1.2061462939527701e-12`.
The optional stress subset also passes `MAXSTEP=100` for NP=1 and NP=8; both
have reconstructed `q5 L_inf=5.7731597280508140e-15`, with largest statistic
difference `1.2216894162975223e-12`. Run it through the matrix driver with
`RUN_STRESS=t`.

S2-B1 adds a sustained compressed-inlet variant through
`run_s2_hbl_inlet_sustained_shock_compare.sh`. This wrapper enables
`PROFILE_OBLIQUE_SHOCK=t`, writes `density=provided pressure=provided` in
`inlet.prof`, and places the analytic shock line at the inlet so the upper
profile continuously injects the compressed post-shock state. `inletprofile`
reads the optional fifth pressure column only when the first profile line
contains `pressure=provided`; for non-dimensional pressure-provided density
profiles, it rejects inputs whose pressure is inconsistent with `rho*T`. The
current `64x64x8` two-step smoke has `q5 L_inf=8.8817841970012523e-16`. The
default `192x192x8` NP=1 two-step gate has `q5 L_inf=1.3322676295501878e-15`
and maximum statistic difference `7.8381745538536052e-13`. The same two-step
`192x192x8` core MPI matrix passes NP=2 x/y/z slabs, NP=4 xy/xz/yz planes, and
`NP=8 TOPOLOGY=2,2,2`; z-decomposed cases use `KM=16`. All MPI gates retain
`q5 L_inf=1.3322676295501878e-15`. The maximum statistic differences are
`7.8204109854596027e-13`, `7.8292927696566039e-13`, and
`1.2243539515566226e-12` for NP=2 x/y/z; `7.8292927696566039e-13`,
`3.5793590313915047e-13`, and `3.5882408155885059e-13` for NP=4 xy/xz/yz; and
`3.5882408155885059e-13` for NP=8. This remains a boundary-forced compatibility
gate, not a full physical SBLI validation.

This gate exposed a GPU `bctype=21` outlet compatibility bug that was hidden by
smooth S1 fields: CPU extrapolates sound speed as
`extrapolate(sos(T1),sos(T2))`, while the GPU previously used
`sqrt(extrapolate(T))/Mach`. The GPU outlet now extrapolates sound speed
directly.

S2-C3 starts the simple-boundary shock-sensor-coupled selective-Roe HBL path.
It deliberately avoids new NSCBC/farfield decisions: the generated case uses
`11,prof/21/41/51`, `543e/643e`, `lchardecomp=t`, `lfilter=f`, and
`diffterm=f`, with a pressure-provided sustained compressed inlet profile.

```bash
IM=64 JM=64 KM=8 MAXSTEP=1 FEQCHKPT=1 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_smoke_64_default \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh

IM=64 JM=64 KM=8 MAXSTEP=3 FEQCHKPT=3 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_3step_64_field \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh

MAXSTEP=1 FEQCHKPT=1 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_np1_192_1step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh

IM=64 JM=64 KM=8 NP=2 TOPOLOGY=2,1,1 MAXSTEP=1 FEQCHKPT=1 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_np2_x_64_1step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh

IM=64 JM=64 KM=16 NP=2 TOPOLOGY=1,1,2 MAXSTEP=1 FEQCHKPT=1 \
OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_np2_z_64_1step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh
```

Current expected result: these commands print `status: pass` for statistics and
field comparison at `1e-10`. The `64x64x8` NP=1 one-step gate has final
`q5 L_inf=6.6613381477509392e-16`; the three-step gate has
`q5 L_inf=8.8817841970012523e-16`. The default `192x192x8` one-step gate has
`q5 L_inf=8.8817841970012523e-16`. The NP=2 x-slab one-step gate has
`q5 L_inf=6.6613381477509392e-16`; the NP=2 z-slab one-step gate uses `KM=16`
and has `q5 L_inf=8.8817841970012523e-16`.

S2-C3 now passes every NP=2 slab orientation. The repaired
`NP=2 TOPOLOGY=1,2,1` gate passes at `1e-10` for one and three steps, with
final `q5 L_inf=6.6613381477509392e-16` and `8.8817841970012523e-16`.
The fault was in the vector Steger-Warming characteristic split: it used
periodized Jacobian indices despite reading an exchanged physical/MPI state
halo. Direct halo geometry indexing matches the scalar split and CPU oracle.
All NP=4 two-axis gates (`2x2x1`, `2x1x2`, and `1x2x2`) now pass for one and
three steps, as does `NP=8 TOPOLOGY=2,2,2`; their final three-step `q5 L_inf`
is `8.8817841970012523e-16`. The z-decomposed gates use `KM=16`, preserving
local `km=8>=hm`. This is a simple-boundary CPU/GPU equivalence matrix only,
not diffusion/filter/NSCBC/farfield/sponge or physical-SBLI validation.

The same simple-boundary path has 20- and 100-step strict field/statistics
passes for NP=1 (`64x64x8`) and NP=8 `2x2x2` (`64x64x16`). The final 100-step
`q5 L_inf` values are `3.3306690738754696e-15` and
`4.8849813083506888e-15`, respectively. Run the full three-direction long
gate with:

```bash
NP=8 TOPOLOGY=2,2,2 IM=64 JM=64 KM=16 MAXSTEP=100 FEQCHKPT=100 \
  OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_np8_100step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh
```

The separate C3-S wrapper enables only the existing x-max `layer` sponge. It
keeps C3's simple boundaries and uses `SPONGE_IM=16` by default; NP=1 and
`NP=8 TOPOLOGY=2,2,2` pass 1, 20, and 100-step strict comparisons. The
100-step `q5 L_inf` values are `7.5495165674510645e-15` and
`8.6597395920762210e-15`. Run the NP=8 gate with:

```bash
NP=8 TOPOLOGY=2,2,2 IM=64 JM=64 KM=16 SPONGE_IM=16 MAXSTEP=100 FEQCHKPT=100 \
  OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_sponge_np8_100step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_sponge_compare.sh
```
Full physical-y sensor dumps also include CPU halo-history-dependent boundary
planes, so the default S2-C3 wrapper keeps `COMPARE_SENSOR=f`; enable it only
when explicitly debugging sensor internals.

S2-C4 is the restricted characteristic-inflow/NSCBC-farfield extension of C3.
It uses `12/21/41/52`, `543e/643e`, MP7 Roe reconstruction,
`lchardecomp=t`, `lfilter=f`, `diffterm=f`, and no sponge. Run the complete
three-direction correctness gate with:

```bash
NP=8 TOPOLOGY=2,2,2 MAXSTEP=20 FEQCHKPT=20 \
  OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c4_np8_20step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c4_compare.sh
```

The current `192x192x8` gate passes CPU/GPU statistics and full HDF5 fields at
`1e-10`; the NP=8 20-step field maxima are `q1=2.7045032879868813e-13` and
`q5=1.9406698470447736e-13`. The output bridge deliberately performs host
`qswap -> boucon -> qswap` after syncing device state, because CPU NSCBC
transverse filters require current host halos. A long-step y-slab audit also
fixed the GPU upper-y normal-Mach reduction: only ranks owning the true upper
physical face may contribute, matching CPU `farfield_nscbc`. The repaired
NP=8 `2x2x2` 100-step gate has `q5 L_inf=9.3258734068513149e-15` and maximum
statistic difference `5.9863225487788441e-12`.

S2-C5 is the restricted C4 plus x-max layer-sponge extension. It sets only
`SPONGE_IM=16`, retains the C4 `12/21/41/52` boundary contract, and supports
only NP=1 and NP=8 `2x2x2`. The sponge retains the existing device-resident
two-kernel `qwork_d` ping-pong with explicit synchronization and operates on
the CPU active ranges. Both 100-step field/statistics comparisons pass at
`1e-10`: final `q5 L_inf` is `8.8817841970012523e-15` for NP=1 and
`9.3258734068513149e-15` for NP=8. Run the NP=8 gate with:

```bash
NP=8 TOPOLOGY=2,2,2 SPONGE_IM=16 MAXSTEP=100 FEQCHKPT=100 \
  OUT_DIR=tests/gpu_validation/out/s2_hbl_selective_roe_s2c5_np8_100step \
  tests/gpu_validation/run_s2_hbl_selective_roe_s2c5_sponge_compare.sh
```

This is a restricted two-GPU-oversubscribed correctness gate, not general
NSCBC, full SBLI, y/z/circular sponge, NP=2/4 sponge, diffusion/filter, or
performance validation.

S0-A7 extends the same selected Roe path to `NP=2 TOPOLOGY=2,1,1`. The driver places the jump at the global x-slab interface using `ASTR_SHUOSHER_SHOCK_X=0.d0`; raw `ssf` is exchanged through the generic `hm` transport before each rank expands its local mask and evaluates characteristic interfaces.

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a7_mpi_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a7_mpi_compare.sh
```

Current S0-A7 status: pass for x slabs. The three-step comparison has global raw sensor `L_inf=1.1102230246251565e-16`, zero mask mismatches, `q5 L_inf=9.1482377229112899e-14`, and maximum statistic difference `4.9098503041022923e-12`. The two-rank memcheck configuration used for S0-A5 reports zero errors on both S0-A7 ranks. This does not validate y/z characteristic interfaces or general multi-rank Roe support.

S0-A8 validates y slabs with `NP=2 TOPOLOGY=1,2,1` and `GRID=400,16,8`, so every local y domain satisfies `jm>=hm`. CPU raw `ssf` overlaps consistently at the duplicated y activity face, but its expanded `lshock` is locally owned; use rankwise CPU/GPU comparison instead of a globally merged mask.

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a8_y_mpi_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a8_y_mpi_compare.sh
```

Current S0-A8 status: pass. The three-step gate has rankwise raw `L_inf=1.1102230246251565e-16`, zero local mask mismatches, `q5 L_inf=8.8817841970012523e-16`, and maximum statistic difference `4.4906300900038332e-12`. Both Compute Sanitizer ranks report zero errors. Do not use `GRID=400,8,8` for this topology because local `jm=4<hm=5` causes halo pack ranges to include an undefined local halo value.

S0-A9 completes the corresponding z-slab gate with `NP=2 TOPOLOGY=1,1,2` and `GRID=400,8,16`, keeping local `km>=hm`. It uses the same rankwise raw/mask validation contract as S0-A8.

```bash
OUT_DIR=tests/gpu_validation/out/shuosher_characteristic_s0a9_z_mpi_compare \
  tests/gpu_validation/run_shuosher_characteristic_s0a9_z_mpi_compare.sh
```

Current S0-A9 status: pass. The three-step gate has zero local mask mismatches, `q5 L_inf=8.8817841970012523e-16`, and both Compute Sanitizer ranks report zero errors. The single-axis `NP=2` gates do not validate combined `2x2x2` topology.

The one-step S0-A2 path also passed Compute Sanitizer memcheck with `ERROR SUMMARY: 0 errors` using `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self`, and `OMPI_MCA_osc=pt2pt` to prevent OpenMPI UCX initialization from generating CUDA-context false positives. Post-S0-A2 regressions passed for S0-A1, filtered/diffusive TGV, all three Phase K source entries, and all 13 Phase J RTI entries.

The larger `256^3 NP=1/NP=2` Nsight profile can be generated with:

```bash
OUT_DIR=tests/gpu_validation/out/nsys_tgv_256_np1_np2 \
  tests/gpu_validation/run_tgv_256_nsys_profile.sh
```

Useful controls:

```bash
GRID=256,256,256 MAXSTEP=10 FEQCHKPT=9999 LFILTER=t DIFFTERM=t \
  NP2_TOPOLOGY=2,1,1 \
  tests/gpu_validation/run_tgv_256_nsys_profile.sh
```

The driver prepares GPU-only `NP=1` and `NP=2` case copies, profiles both with Nsight Systems, and compares their `flowstate.dat` statistics. The reusable driver has a recorded pass in `documents/GPU_VALIDATION_MATRIX.md`; rerun it when profiling, halo exchange, residency, or build configuration changes.

## GPU Multi-Rank X-Slab Validation

Validate the current two-rank, two-GPU TGV x-slab path:

```bash
MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_stats_compare_postgit \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_field_compare_postgit \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The post-git run had max statistic differences `kenergy=2.9254376698872875e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=8.5597761517730575e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=5.8832938520936295e-12`.

Validate the two-rank, two-GPU y-slab path:

```bash
MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_yslab_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_yslab_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The y-slab stats run had max differences `kenergy=2.9226621123257246e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=8.5489341300482025e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.1106675275368616e-12`.

Validate the two-rank, two-GPU z-slab path:

```bash
MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=1,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_zslab_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=1,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_zslab_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The z-slab stats run had max differences `kenergy=2.9240498911065060e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=5.7571135358980285e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.2243543652584776e-12`.

Validate the four-rank x-y combined topology on a two-GPU workstation. This is an oversubscription correctness test, not a performance test:

```bash
NP=4 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x2x1_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=4 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,1 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x2x1_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9254376698872875e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.5654148784233257e-12`. The GPU log should show ranks `0/2` sharing GPU 0 and ranks `1/3` sharing GPU 1.

Validate the four-rank x-z combined topology on a two-GPU workstation. This is also an oversubscription correctness test, not a performance test:

```bash
NP=4 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x1x2_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=4 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=2,1,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_2x1x2_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9254376698872875e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=5.9685589803848416e-12`. The GPU log should show ranks `0/2` sharing GPU 0 and ranks `1/3` sharing GPU 1.

Validate the four-rank y-z combined topology on a two-GPU workstation. This is also an oversubscription correctness test, not a performance test:

```bash
NP=4 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_1x2x2_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=4 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=1,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank4_1x2x2_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9226621123257246e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=5.9685589803848416e-12`. The GPU log should show ranks `0/2` sharing GPU 0 and ranks `1/3` sharing GPU 1.

Validate the eight-rank full x-y-z combined topology on a two-GPU workstation. This is also an oversubscription correctness test, not a performance test:

```bash
NP=8 MAXSTEP=2 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank8_2x2x2_stats_compare \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=8 MAXSTEP=1 FEQCHKPT=1 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank8_2x2x2_field_compare \
  tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```

Current expected result: both commands print `status: pass`. The stats run had max differences `kenergy=2.9240498911065060e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`; the one-step full-field comparison had reconstructed `q5` `L_inf=6.4517280407017097e-12`. The GPU log should show ranks `0/2/4/6` sharing GPU 0 and ranks `1/3/5/7` sharing GPU 1.

Validate higher-rank full x-y-z topology combinations on a two-GPU workstation. These are native-statistics smoke tests only: they use `MAXSTEP=1`, disable checkpoint field output with `FEQCHKPT=99`, and are meant to exercise rank topology and face-halo composition under oversubscription. They are not field-output comparisons and not performance tests.

```bash
NP=16 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=4,2,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank16_4x2x2_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=16 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,4,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank16_2x4x2_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=16 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=2,2,4 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank16_2x2x4_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh

NP=32 MAXSTEP=1 FEQCHKPT=99 LFILTER=t DIFFTERM=t TOPOLOGY=4,4,2 \
  OUT_DIR=tests/gpu_validation/out/tgv_mpirank32_4x4x2_stats_smoke \
  tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh
```

Current expected result: all four commands print `status: pass`. The `4x2x2`, `2x4x2`, and `2x2x4` smoke runs had max differences `kenergy=1.1227130336521896e-14`, `enstophy=8.3266726846886741e-15`, and `dissipation=4.2392304944183223e-17`. The `4x4x2` smoke run had max differences `kenergy=1.1241008124329710e-14`, `enstophy=8.2711615334574162e-15`, and `dissipation=4.2392304944183223e-17`.

For a two-rank no-checkpoint transfer audit, prepare a GPU-only case with `feqchkpt` greater than `maxstep` and profile MPI plus CUDA:

```bash
OUT_DIR=tests/gpu_validation/out/tgv_mpirank2_nsys_nochk_postgit
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

python3 tests/gpu_validation/prepare_tgv_case.py \
  --src-case examples/Taylor_Green_Vortex \
  --dst-case "$OUT_DIR/gpu" \
  --use-gpu t \
  --maxstep 1 \
  --feqchkpt 99 \
  --lfilter t \
  --diffterm t \
  --scheme 643e

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY=2,1,1 \
    nsys profile --trace=cuda,mpi --stats=true --force-overwrite=true \
    -o ../nsys_mpirank2_nochk \
    mpirun -np 2 /home/dell/workspace/astr_gpu/build_gpu_probe/bin/astr run datin/input.tgv
)

nsys stats --force-export true \
  --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  --format csv \
  --output "$OUT_DIR/nsys_mem" \
  "$OUT_DIR/nsys_mpirank2_nochk.nsys-rep"
```

The post-git audit reported `Device-to-Host` count `116`, total `380.607 MB`, max `4.793 MB`, and `Host-to-Device` count `188`, total `867.120 MB`, max `104.333 MB`. For this `2x1x1` case, one local interior scalar field is about `8.39 MB`; therefore the observed maximum D2H is halo-buffer scale, not full-field scale. The largest H2D events are initial resident array setup and are not evidence of per-kernel or per-step full-field D2H. ASTR may still create the initial CPU-owned `outdat/flowfield.h5`; this audit is scoped to checkpoint-disabled GPU compute-loop transfers, not to removing every file output path.

## Multi-Rank Matrix Driver

Run the core multi-rank stats and field matrix:

```bash
OUT_DIR=tests/gpu_validation/out/tgv_mpirank_matrix \
  tests/gpu_validation/run_tgv_mpirank_matrix.sh
```

By default this covers `2x1x1`, `1x2x1`, `1x1x2`, `2x2x1`, `2x1x2`, `1x2x2`, and `2x2x2` for both native statistics and one-step field comparison. Higher-rank oversubscription smoke tests are opt-in:

```bash
RUN_SMOKE=t OUT_DIR=tests/gpu_validation/out/tgv_mpirank_matrix_smoke \
  tests/gpu_validation/run_tgv_mpirank_matrix.sh
```

Override `STATS_MATRIX`, `FIELD_MATRIX`, or `SMOKE_MATRIX` with entries of the form `NP:i,j,k` to run a narrower set.

## GPU Gradcal Dvel Compare

Validate the first-stage GPU `gradcal` path for TGV velocity gradients:

```bash
MAXSTEP=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/tgv_gpu_gradcal_643e_diff_filter_10 \
  tests/gpu_validation/run_tgv_stats_compare.sh

python3 tests/gpu_validation/compare_gpu_gradcal.py \
  --gpu tests/gpu_validation/out/tgv_gpu_gradcal_643e_diff_filter_10/gpu \
  --report tests/gpu_validation/out/tgv_gpu_gradcal_643e_diff_filter_10/gpu_gradcal_compare.txt \
  --atol 1e-10 --rtol 1e-10
```

The GPU run used to write `gpu_gradcal_dvel_compare.dat` after CPU `gradcal()` and GPU `gradcal_dvel_kernel` both ran on the same refreshed state. That bridge validation passed and remains useful as a regression fixture. The current GPU-resident path no longer runs CPU `gradcal()` every step; `dvel_d` is computed on device for GPU statistics.

## S2-C6 Constant NSCBC Farfield Target

The physical opt-in slice for Cartesian upper-y `bctype=52` uses an
incoming-wave-only target state while retaining the existing transverse filter
interface. It is restricted to the nonreacting five-equation path. The target
pressure is derived from `rho` and `T`; do not provide an independent target
pressure.

Run the default nontrivial target gate:

```bash
OUT_DIR=tests/gpu_validation/out/s2_c6_np1 \
tests/gpu_validation/run_s2_hbl_nscbc52_incoming_only_compare.sh

NP=2 TOPOLOGY=1,2,1 MAXSTEP=2 FEQCHKPT=2 \
OUT_DIR=tests/gpu_validation/out/s2_c6_np2_y \
tests/gpu_validation/run_s2_hbl_nscbc52_incoming_only_compare.sh
```

The driver exports `ASTR_NSCBC_FARFIELD_MODE=incoming_only` plus global
`ASTR_NSCBC_FARFIELD_RHO/U/V/W/T` values. Set the mode to `compatibility`, or
leave it unset, to preserve the historical C4/C5 behavior.
## Curvilinear periodic TGV gate

`generate_curvilinear_tgv_grid.py` constructs the static, smooth periodic mapping

```
x = xi + a sin(xi) sin(eta)
y = eta + a sin(eta) sin(zeta)
z = zeta + a sin(zeta) sin(xi)
```

with `a=0.15` by default. Its analytic Jacobian is positive on the `32^3`
smoke grid (`0.79103125 <= J <= 1.24271875`), and all three cross derivatives
are nonzero. `run_curvilinear_tgv_compare.sh` reads this HDF5 grid through the
normal `lreadgrid=t` path and uses `643e` for both convection and diffusion.

Run the strict no-filter field gate with:

```bash
MAXSTEP=1 FEQCHKPT=1 LFILTER=f DIFFTERM=f \
  OUT_DIR=tests/gpu_validation/out/curvilinear_tgv_convective_smoke \
  tests/gpu_validation/run_curvilinear_tgv_compare.sh
```

This passed with reconstructed `q5 L_inf=8.5265128291212022e-14`.

For the resident numerical gate with diffusion and the explicit 10th-order
filter, run the complete-RK same-phase field comparison:

```bash
MAXSTEP=10 FEQCHKPT=10 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/curvilinear_tgv_same_phase_diff_filter_10 \
  tests/gpu_validation/run_curvilinear_tgv_compare.sh
```

The driver sets `ASTR_VALIDATION_RK_SNAPSHOT` only for the CPU run. At the
final complete RK state before the matching checkpoint, CPU writes a separate
HDF5 snapshot after the same temporary `boucon/qswap` output preparation that
GPU applies to its host copy, then restores its arrays. The production
checkpoint phase remains unchanged. The default field comparison uses this
same-phase CPU snapshot instead of CPU's next-first-RK checkpoint. The
10-step `LFILTER=t DIFFTERM=t` gate passes with maximum statistic difference
`4.7573056605187958e-14` and reconstructed `q5 L_inf=2.8421709430404007e-13`.

Set `NP` and `TOPOLOGY=i,j,k` to exercise the same gate with a forced MPI
topology. The two-rank curve-grid slab matrix below passes at the same field
tolerance; each local direction retains more than the five-node halo width.

```bash
NP=2 TOPOLOGY=2,1,1 MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/curvilinear_tgv_np2_x_filter_5 \
  tests/gpu_validation/run_curvilinear_tgv_compare.sh

NP=2 TOPOLOGY=1,2,1 MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/curvilinear_tgv_np2_y_filter_5 \
  tests/gpu_validation/run_curvilinear_tgv_compare.sh

NP=2 TOPOLOGY=1,1,2 MAXSTEP=5 FEQCHKPT=5 LFILTER=t DIFFTERM=t \
  OUT_DIR=tests/gpu_validation/out/curvilinear_tgv_np2_z_filter_5 \
  tests/gpu_validation/run_curvilinear_tgv_compare.sh
```

The largest statistic difference in this matrix is `4.9599213625128868e-14`;
the largest reconstructed field error is `q5 L_inf=3.1263880373444408e-13`.
The gate specifically guards that `jacob_d` and `dxi_d` use direct local/halo
indices at MPI interfaces, rather than treating a local subdomain endpoint as
a periodic duplicate plane.

## Curvilinear free-stream preservation gate

`run_curvilinear_freestream_compare.sh` uses the same periodic mapping with a
uniform HDF5 initial field, nonzero velocity in all three physical directions,
and explicit `643e` convection. Diffusion, filtering, sources, and physical
boundaries are disabled so any field drift exposes a geometric-metric or
metric-halo inconsistency. The CPU reference uses the opt-in complete-RK
snapshot, and the checker separately requires both CPU and GPU fields to
remain at the prescribed analytic uniform state.

Run the complete topology matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_freestream_matrix \
  tests/gpu_validation/run_curvilinear_freestream_matrix.sh
```

The default `32^3`, `a=0.15` matrix covers NP=1, all three NP=2 slabs, all
three NP=4 two-axis decompositions, and NP=8 `2x2x2`. The NP=1 ten-step gate
and all five-step MPI gates pass. Across the matrix, CPU/GPU primitive-field
differences remain at about `4.7e-16` or below; uniform velocity drift is at
about `4.7e-15` or below, and pressure drift is `4.3e-14`. NP=4/8 runs share
two GPUs and are correctness evidence only. This gate does not validate an
independent analytic `dxi/jacob` oracle, curved physical boundaries, moving
grids, or production scaling.

## Curvilinear analytic metric gate

`run_curvilinear_metric_convergence.sh` enables the validation-only
`ASTR_GEOMETRY_DUMP` hook and compares ASTR's `jacob` and all nine `dxi`
components with the analytic derivatives of the periodic mapping. The oracle
uses ASTR's integer computational coordinates: physical mapping columns are
scaled by `2*pi/im`, `2*pi/jm`, and `2*pi/km` before inversion. Comparing
against unscaled continuous coordinates is incorrect.

Run the three-level default gate with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_metric_convergence \
  tests/gpu_validation/run_curvilinear_metric_convergence.sh
```

The `16^3`, `24^3`, and `32^3` maximum `jacob` errors are
`4.0224641635611125e-7`, `1.0639206314555505e-8`, and
`8.0351376247067563e-10`. Maximum error across all nine `dxi` components
decreases from `6.6675760468815071e-5` through `9.6589979468042486e-6` to
`2.3469382739449429e-6`. Every numerical Jacobian is positive, all checked
arrays are finite, and the largest sixth-order discrete metric-identity
residual is `5.3733059668381600e-16`.

## CURVE-C13 six-face curvilinear symmetry gate

`run_curvilinear_symmetry_compare.sh` accepts `AXIS=x`, `y`, or `z` and
generates an analytic `J=1` mapping whose selected physical-face pair is
wavy. CPU and GPU extrapolate all three velocity components and apply
`u_t=u^*-(u^* dot n)n` with the CPU-generated discrete face normal. The GPU
uploads all six `bnorm_i0/im`, `bnorm_j0/jm`, and `bnorm_k0/km` planes once
and keeps them resident.

Run the complete topology matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_symmetry_six_face_matrix \
  tests/gpu_validation/run_curvilinear_symmetry_six_face_matrix.sh
```

The 12 entries cover x/y/z NP=1, physical-axis NP=2 slabs, NP=4 plane
decompositions, and NP=8 `2x2x2`. All one-step full-field comparisons pass
at `1e-10`. The maximum conservative-field error is
`q5 L_inf=8.5265128291212022e-14`, and the maximum statistic difference is
`kenergy=4.6171400036598698e-14`.

For NP=1, `ASTR_GEOMETRY_DUMP` records CPU's actual `dxi` fields. The
invariant checker reconstructs the same discrete unit normals and requires
`|u dot n|_inf <= 1e-12`; the observed maximum is
`7.2316284904783146e-17`. It separately checks the analytic surface normal:
x/y residuals are about `6.84e-8` and the z residual is about `3.46e-11`,
which retain the boundary metric truncation instead of hiding it in the
projection check. Coordinate-normal velocity remains nonzero, so the test
would reject a Cartesian component clamp.

CPU review identified two deterministic defects before GPU expansion:
`bnorm_jm` used the lower `j=0` metric plane, and the y-face plus k-min
symmetry branches discarded or omitted geometric projection. Both CPU
defects were corrected after approval. Targeted x/y/z NP=1 Compute Sanitizer
runs report `ERROR SUMMARY: 0 errors`; the Cartesian Phase C matrix and C12
one-step NSCBC/sponge case also pass after the correction.

This gate closes static single-block six-face geometry projection for
`bctype=60`. It does not generalize no-slip, NSCBC, farfield, inflow/outflow,
or sponge conditions to arbitrary faces.

## CURVE-C14 physical HBL refinement gate

CURVE-C14 tests physical consistency of the C7 static nonorthogonal Mach 5
laminar boundary layer. The analyzer applies the existing sixth-order
computational derivative to the coordinate and flow fields and reconstructs
the physical gradient with the inverse mapping. With local streamwise tangent
`t` and wall normal `n`, it evaluates

```text
Cf = 2 mu/Re * t . [grad(u) + grad(u)^T] . n
qw = mu/[Re Pr (gamma-1) M^2] * grad(T) . n
```

`analyze_curvilinear_hbl_physics.py` also compares velocity and temperature
profiles at `x=0,3,7` with the Blasius similarity solution used by
`prepare_s1_flatplate_case.py`. The refinement summary is generated by
`summarize_curvilinear_hbl_refinement.py`.

Run the complete dual-wall-temperature matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_physical_refinement \
  tests/gpu_validation/run_curvilinear_hbl_physical_refinement.sh
```

The matrix uses `96x96x8`, `144x144x8`, and `192x192x8` grids. Each grid is
advanced to `t=2e-4` with `deltat=1e-5` for 20 steps and `5e-6` for 40 steps.
It runs a near-recovery hot wall (`Tw/Tinf=5.191440547760865`) and a cold wall
(`Tw/Tinf=3`). All 12 CPU/GPU pairs pass field, statistics, wall-invariant,
and positive-J checks. The largest field error is
`T=1.4166445794216997e-13`, the largest statistic error is
`massflux=7.616129948928574e-14`, the maximum wall residual is
`1.7763568394002505e-15`, and the minimum numerical Jacobian is
`2.529663846823165e-7`.

At the fine time step, required medium-to-fine spatial deltas are smaller than
coarse-to-medium deltas. Cold-wall ratios are `0.4964` for `Cf` and `0.5943`
for `qw`; hot-wall `Cf` is `0.4697`; all velocity and temperature profile
ratios lie in `0.4355-0.4560`. Fine-grid time-step sensitivities are below the
corresponding spatial deltas. On the fine grid, Blasius relative errors are
below `4.5e-4` for `Cf`, below `1.17e-3` for cold-wall `qw`, and below
`3.1e-5` for the sampled profiles.

Hot-wall `qw` is reported but is not an acceptance quantity. Its reference
changes sign very close to the wall because the chosen temperature is near
the Mach 5 recovery temperature, making a relative norm ill-conditioned
below the first resolved wall interval. Cold-wall `qw` supplies the required
thermal gate. CURVE-C14 is matched-time transient grid/time consistency; it
does not establish a steady DNS solution, experimental agreement, or physical
shock-boundary-layer interaction fidelity.

## CURVE-C15 performance and residency closure

CURVE-C15 uses the C10 static nonorthogonal Mach 5 viscous selective-Roe path
at `256x256x256`. An initial Nsight Systems trace found that
`gpu_s1_bl_scalars()` copied `65536 x 6` real(8) block partials, or
`3,145,728 B`, from device to host for each statistics evaluation. This was
not a full flow-field transfer, but it violated the 64 KiB RK residency gate.

The statistics path now performs a deterministic two-stage reduction. The
existing first kernel forms block partials. A second kernel launches six
256-thread blocks, one for each diagnostic, reduces all block partials on the
device, and returns only six real(8) values. Both kernels are followed by the
required explicit synchronization. The CPU MPI reduction and diagnostic
formulas are unchanged.

Run the post-change residency trace with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c10_256_residency \
  GRID=256,256,256 MAXSTEP=2 \
  tests/gpu_validation/run_curvilinear_hbl_c10_256_residency.sh
```

The interval beginning at the first
`characteristic_upwind_rhs_x_xyphysical_global_kernel` contains 221 kernels.
It has two 176-byte H2D operations, two 48-byte D2H operations, and no transfer
at or above 64 KiB. The second reduction kernel averages about `77.6 us` per
invocation. A targeted `64x64x8` NP=1 Compute Sanitizer run reports
`ERROR SUMMARY: 0 errors`, and the complete C14 dual-wall CPU/GPU matrix
passes unchanged with maximum field/statistics errors
`1.4166445794216997e-13` and `7.616129948928574e-14`.

Run the repeated benchmark with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark \
  GRID=256,256,256 MAXSTEP=5 REPEATS=3 \
  tests/gpu_validation/run_curvilinear_hbl_c10_256_benchmark.sh
```

The controller advances six RK steps because it executes `nstep=0..maxstep`.
Initialization and full-field output are included. NP=1 wall times are
`57.297/57.517/57.712 s`; NP=2 `2x1x1` wall times are
`36.320/36.487/36.548 s`. Median two-GPU speedup relative to one GPU is
`1.5764x`, with `78.82%` parallel efficiency. Peak device-level memory is
`9876 MiB` for NP=1 and `5573 MiB` for NP=2; both configurations reach 100%
sampled GPU utilization. `nvitop` separately observed about `9.68 GiB` for
the NP=1 process and about `5.24 GiB` per rank for NP=2, with the ranks bound
to different GPUs.

This gate establishes current-machine RK residency and two-GPU x-slab scaling
for the tested short C10 case. It is not a CPU speedup, long-time throughput,
communication-overlap, scaling beyond two GPUs, or production SBLI result.
GPU HDF5 remains outside scope; output transfers are excluded from the RK
residency interval but included in end-to-end benchmark time.

## CURVE-C16 six-face isothermal no-slip gate

CURVE-C16 extends the zero-blowing C6 `bctype=41` comparison harness to x-, y-, and z-wavy
physical wall pairs. `WALL_AXIS=x/y/z` selects the matching analytic `J=1`
mapping, homogeneous directions, boundary tuple, and HDF5 face axis. The two
faces normal to the selected computational direction are isothermal no-slip;
the other four faces are periodic.

Run the complete matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_wall41_six_face_matrix \
  tests/gpu_validation/run_curvilinear_wall41_six_face_matrix.sh
```

For each axis, the matrix runs NP=1 convection isolation, NP=1 five-step
filter plus diffusion, a five-step NP=2 slab decomposed in the physical
direction, an NP=4 plane, and NP=8 `2x2x2`. The 15 entries pass. The maximum
full-field difference is `q5=6.2527760746888816e-13`, the maximum statistic
difference is `kenergy=4.9640846988552312e-14`, and the maximum wall-invariant
error is `1.1368683772161603e-13`. Wall velocity and momentum residuals are
zero. Targeted y/z NP=1 Compute Sanitizer runs report `ERROR SUMMARY: 0
errors` when OpenMPI uses the established `ob1/self/pt2pt` isolation.

This gate establishes CPU/GPU numerical equivalence and halo composition for
the tested static single-block six-face isothermal no-slip family. The zero
wall-velocity vector is independent of face orientation. Pressure is still
extrapolated along the CPU computational line, so C16 does not establish a
geometric-normal pressure Neumann condition, six-face adiabatic or slip wall
support, general characteristic boundaries, or multi-GPU scaling beyond the
two physical devices. C16 itself is zero-blowing evidence; the later C19 gate
validates the shared CPU/GPU physical-normal `wallbs.dat` route on y-min.

## CURVE-C17 six-face zero-extrapolation gate

Run the x/y/z NP=1/2/4/8 matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_zeroextrap_six_face_matrix \
  tests/gpu_validation/run_curvilinear_zeroextrap_six_face_matrix.sh
```

The invariant checker follows `src/bc.F90:zeroextrap` rather than imposing a
direction-independent rule. X faces directly extrapolate velocity, pressure,
and temperature, then reconstruct density from the EOS. Y/z faces directly
extrapolate velocity, pressure, and density, then reconstruct temperature.
The CPU oracle is `outdat/rk_complete_snapshot.h5`, matching the GPU
checkpoint preparation phase.

All 15 entries pass. Maximum field, statistic, and boundary residuals are
`3.9790393202565610e-13`, `4.9960036108132044e-14`, and
`7.6170181273482740e-12`. Targeted y/z NP=1 filter-plus-diffusion memchecks
report zero errors with OpenMPI `ob1/self/pt2pt`.

## CURVE-C18 x/y adiabatic no-slip gate

Run the supported matrix and z-direction reject with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_wall42_matrix \
  tests/gpu_validation/run_curvilinear_wall42_matrix.sh
```

For the no-`wallbs.dat` subset, CPU `noslip_adibatic` defines only x/y faces. The boundary velocity is zero,
pressure and temperature are extrapolated along the computational line, and
density is reconstructed through the nondimensional ideal-gas EOS. The ten
x/y NP=1/2/4/8 entries pass, while z exits with status 2 before solver launch.
Maximum field, statistic, and wall residuals are
`3.6948222259525210e-13`, `4.9071857688431919e-14`, and
`5.1585402616183273e-12`. Targeted x/y memchecks report zero errors.
CPU's optional y-min blowing route is not part of this passing gate.

## CURVE-C19 y-face slip-wall and wall-blowing gate

CURVE-C19 validates the approved geometric contract for y-wavy
`bctype=41/42/411/421`. Slip sections extrapolate all three velocity
components and apply `u_t=u*-(u* dot n)n` using the CPU-generated discrete
unit normal. The `411` no-slip section remains zero velocity. The `421` lower
section combines the projected tangent velocity with prescribed normal wall
velocity, while its upper section is tangent-only. CPU has no x/z `411/421`
branches, so those directions remain outside this gate.

`wallbs.dat` retains the CPU scaling
`vwall=A uinf fx gz (1+r)`, but the resulting signed scalar now acts along the
physical normal. Positive amplitude follows the stored inward normal. The 10%
perturbation uses global i/k indices
and a fixed integer hash, which makes the forcing topology-independent. Exact
forcing checks use unique periodic nodes; duplicate periodic endpoint planes
are still checked for geometric alignment but are excluded from the value
oracle because they represent the same physical degree of freedom.

Run one case or the full matrix with:

```bash
BC_KIND=421 WALL_BLOWING=t NP=8 TOPOLOGY=2,2,2 \
  OUT_DIR=/tmp/astr_c19_421_np8 \
  tests/gpu_validation/run_curvilinear_wall_c19_compare.sh

OUT_DIR=/tmp/astr_c19_matrix \
  tests/gpu_validation/run_curvilinear_wall_c19_matrix.sh
```

The default matrix passes 15 entries spanning positive blowing, negative
suction, NP=1, NP=2 `1x2x1`, NP=4 `2x2x1`, and NP=8 `2x2x2`. Maximum
CPU/GPU field and statistic differences
are `8.5265128291212022e-14` and `4.6865289426989420e-14`; the maximum
same-backend topology difference is `5.6843418860808015e-14`, and the maximum
NP=1 physical-normal residual is `6.0281640790194047e-17`. Separate five-step
filter-plus-diffusion checks pass for zero-blowing `411` and blowing `421`,
with maximum `q5` errors `3.4106051316484809e-13` and
`4.2632564145606011e-13`. The blowing `421` memcheck reports zero errors.
A three-step no-checkpoint Nsight Systems trace contains 336 kernels after the
first filter kernel and no H2D/D2H transfer at or above 64 KiB; maximum H2D
and D2H sizes are 128 and 1024 bytes.

## CURVE-C6 x-wavy isothermal wall gate

CURVE-C6 uses the static mapping

```text
x = xi + a sin(eta) sin(zeta), y = eta, z = zeta
```

with `a=0.15`, analytic `J=1`, `bctype=41/41` on the two x faces, and
periodic y/z. The scope is the nonreacting five-equation explicit path with
`643e` convection/diffusion and the explicit 10th-order filter. It does not
include species, turbulence, chemistry, immersed boundaries, moving grids,
or compact schemes.

CPU `noslip(ndir=1/2)` sets `(u,v,w)=(0,0,0)`, sets `T=Tw`, extrapolates
pressure as `(4 p_1-p_2)/3` along the computational i line, evaluates density
from the ideal-gas EOS, and rebuilds `q`. The zero velocity vector does not
require a geometric wall normal. The pressure extrapolation is not an
explicit geometric-normal Neumann condition, so this gate checks CPU/GPU
compatibility rather than defining a general curved-wall pressure model.

The CPU audit found that `filterq_explicit10` previously called a centered
10th-order stencil at every physical node. Physical halos are not defined by
`bctype=41`, so the result depended on stale or undefined values. CPU and GPU
now use the same closure on each physical side:

```text
face       : unchanged
point 1-2  : one-sided sixth order
point 3    : centered sixth order
point 4    : centered eighth order
point >= 5 : centered 10th order
```

The mirrored rule applies at the upper face. Periodic and MPI interfaces keep
the centered 10th-order formula

```text
F_i = 193/256 f_i
    + 105/512 (f_(i-1) + f_(i+1))
    - 15/128  (f_(i-2) + f_(i+2))
    + 45/1024 (f_(i-3) + f_(i+3))
    - 5/512   (f_(i-4) + f_(i+4))
    + 1/1024  (f_(i-5) + f_(i+5)).
```

The first-stage RHS diagnostic also found that the GPU x-physical y/z
convection kernels could update x-wall nodes on a curved grid. CPU
`convrsdcal6` applies every directional contribution only inside
`is:ie,js:je,ks:ke`; the GPU kernels now use the same active box.

Run the complete acceptance matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_wall41_matrix \
  tests/gpu_validation/run_curvilinear_wall41_matrix.sh
```

The matrix covers NP=1 one-step no-filter/no-diffusion, NP=1 and NP=2
`2x1x1` five-step filter+diffusion, and NP=4 `2x2x1` one-step
filter+diffusion. `check_wall41_invariants.py` checks both CPU and GPU x faces
for zero velocity, prescribed temperature, ideal-gas EOS consistency, zero
wall momentum, and the no-slip total-energy reduction. The current matrix
passes with maximum reconstructed `q5 L_inf=5.1159076974727213e-13`, maximum
statistic difference below `5e-14`, wall-temperature error
`5.6843418860808015e-14`, and EOS error `4.4408920985006262e-16`.

Run NP=1 Compute Sanitizer from a prepared C6 GPU case with:

```bash
cd tests/gpu_validation/out/curvilinear_wall41_matrix/np1_filter_diff/gpu
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
  mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  ../../../../../build_gpu_probe/bin/astr run datin/input.tgv
```

The targeted run reports `ERROR SUMMARY: 0 errors`. A separate three-step
profile with `feqchkpt=99` confirms that the no-checkpoint RK window has no
full-field H2D/D2H. After the first RHS kernel begins, maximum H2D and D2H
operations are `128 B` and `1024 B`. Temporary first-stage phase/RHS dump
hooks used to isolate the curved-wall residual were removed after diagnosis.

Phase H and Phase I-C field drivers use the CPU complete-RK validation
snapshot for same-phase comparisons. Comparing the CPU output-preparation
checkpoint directly with the GPU complete-RK HDF5 field is not a valid
numerical error test.

CURVE-C6 proves CPU/GPU numerical equivalence and wall-state algebraic
invariants for the tested static x-wavy single block. It does not validate
curved-wall shear stress, heat flux, wall units, or other physical wall
diagnostics.

## CURVE-C7 three-dimensional curvilinear Mach 5 BL gate

CURVE-C7 starts from the validated S1-C1/C2 Mach 5 single-species boundary
layer and uses a static, boundary-preserving nonorthogonal mapping:

```text
x = X + ax sin(pi s) y (1-y)
y = Y + ay sin(2 pi s) y (1-y)
z = Z
```

Here `s` is the normalized streamwise coordinate. The perturbation vanishes
on all physical x/y faces, so the existing profile inlet, extrapolation,
isothermal lower wall, and farfield upper target remain compatible. The
initial similarity field is evaluated at every physical `(x,y)` point instead
of copying one wall-normal profile across a warped station.

The CPU audit found that the three-dimensional `fbcxbl` and `whfbl`
statistics used an i-j quadrilateral even though the lower wall is an i-k
face. GPU statistics also approximated volume as Cartesian
`abs(dx*dy*dz)`. With explicit user approval, CPU now evaluates the true 3-D
wall quadrilateral and GPU uses the same quadrilateral plus CPU-compatible
six-term hexahedral volume.

Run the acceptance matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c7_matrix \
  tests/gpu_validation/run_curvilinear_hbl_c7_matrix.sh
```

The matrix covers NP=1 20 steps, NP=2 `2x1x1`, `1x2x1`, and `1x1x2` for
five steps, and NP=4 `2x2x1` for one step. All use `numq=5`, Sutherland
viscosity, explicit MP7 convection, sixth-order explicit diffusion,
`bctype=11/21,41/51`, periodic z, `lfilter=f`, and `lchardecomp=f`.

The current matrix passes same-phase fields, native statistics, and lower-wall
invariants. Maximum primitive error is `T=4.1744385725905886e-14`, maximum
conservative error is `q5=4.4408920985006262e-16`, and the NP=1 maximum
statistic errors are `massflux=4.6074255521943996e-14`,
`fbcx=4.5129915429709122e-18`, and
`wallheatflux=2.3420460991581404e-19`. Lower-wall temperature and EOS errors
are `8.8817841970012523e-16` and `2.2204460492503131e-16`; velocity,
momentum, and energy-reconstruction residuals are zero.

The independent grid check reports analytic
`J=[2.2772279043379687e-6,5.0695588091900692e-4]`, maximum cross derivative
`3.1461989215822009e-2`, and zero physical-face coordinate error. Because
ASTR uses one-sided physical-boundary metric closure, the metric oracle
reports both the full domain and a three-layer interior gate. Interior
`jacob L_inf=1.1397685692384613e-11` and inverse-metric
`L_inf=6.9346652153967625e-6`; the larger full-domain values remain visible in
`metric_check.txt`.

Targeted NP=1 Compute Sanitizer reports `ERROR SUMMARY: 0 errors`. A
three-step `feqchkpt=99` Nsight Systems profile has no transfer at or above
64 KiB in the RK window. The remaining H2D transfers are three 176-byte
operations and the D2H transfers are three 6144-byte statistic reductions,
not full fields. CURVE-C0-C6 and zero-warp S1-C1/C2 regressions pass on the
same sources.

This gate proves CPU/GPU numerical equivalence for the controlled static
curvilinear slice. It does not validate boundary-layer grid convergence,
wall-heat-flux physical accuracy, filtering on this case, characteristic
farfield behavior, shock-boundary-layer interaction, moving or multi-block
grids, or production scaling.

## CURVE-C8 filtered curvilinear Mach 5 BL gate

CURVE-C8 enables the explicit 10th-order filter on the C7 grid and boundary
contract. It keeps `543e` MP7 convection, `643e` diffusion, Sutherland
viscosity, `bctype=11/21,41/51`, periodic z, `lchardecomp=f`, full-`q`
ping-pong storage, and the C6 physical `0-6-6-6-8-10` closure. Sponge,
species, turbulence, chemistry, compact schemes, and characteristic paths
remain outside the capability gate.

Run the acceptance matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c8_filter_matrix \
  tests/gpu_validation/run_curvilinear_hbl_c8_filter_matrix.sh
```

The matrix contains six cases: NP=1 one-step filter isolation with diffusion
disabled; NP=1 20 steps with diffusion; NP=2 `2x1x1`, `1x2x1`, and `1x1x2`
for five steps; and NP=4 `2x2x1` for one step. The z slab uses `KM=16` so
each local active extent remains at least `hm=5`.

The isolation gate passes with maximum primitive error
`T=1.7541523789077473e-14`. The 20-step gate passes with
`T=8.9714902173909650e-12`, `u1=6.5788485770212901e-12`, and maximum
statistic error `wallheatflux=3.8150392986446471e-12`. All field and
statistic comparisons remain below `1e-10`; lower-wall velocity, momentum,
and energy residuals are zero, and the largest reported wall temperature
error is `1.7763568394002505e-15`.

The filter-isolation failure used during development was not a filter
coefficient or boundary-closure defect. CPU `filterq` updates conservative
`q` but leaves the pre-filter primitive variables in place through flux
assembly. CPU Steger-Warming supersonic branches use filtered `q`, including
`q1`, while subsonic split coefficients use the stage `rho`, `vel`, and
`tmp`. GPU now follows the same distinction. `commvar_gpu::mach_d` is copied
once after reference-variable setup so device sound speed is evaluated as
`sqrt(tmp_d)/mach_d` without an RK host bridge.

Run NP=1 Compute Sanitizer from a prepared filter-isolation GPU case:

```bash
cd tests/gpu_validation/out/curvilinear_hbl_c8_filter_matrix/np1_filter_isolation/gpu
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
  mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  ../../../../../build_gpu_probe/bin/astr run datin/input.flatplate
```

The targeted run reports `ERROR SUMMARY: 0 errors`. For the residency audit,
prepare the same GPU case with `maxstep=3` and `feqchkpt=99`, then profile
`cuda,mpi` with Nsight Systems. In the interval beginning at the first
`explicit_upwind_rhs_x_xyphysical_global_kernel`, the current trace contains
three H2D copies of 176 bytes and three D2H statistic copies of 6144 bytes.
There is no full-field transfer in the RK interval.

CURVE-C0-C7, the C6 filter matrix, the five-step Phase I-C LDC filter plus
diffusion gate, and zero-warp S1-C1/C2 regressions pass on the same CPU/GPU
binaries. CURVE-C8 establishes numerical equivalence and compute-loop
residency for this controlled filtered slice only. It does not establish
mesh/time convergence, wall heat-flux accuracy, NSCBC, shock sensing,
selective Roe, SBLI, moving or multi-block grids, GPU HDF5, or production
scaling.

## CURVE-C9 curvilinear selective-Roe gate

CURVE-C9 combines the static C7 nonorthogonal Mach 5 grid with the S2-C3
Ducros sensor and selective Roe-characteristic MP7 path. The inlet profile
sustains an oblique pressure, density, and velocity discontinuity. The test
uses `543e/643e`, `recon=3`, `lchardecomp=t`, `lfilter=f`, `diffterm=f`, no
sponge, `bctype=11/21,41/51`, periodic z, and no species or turbulence.

The CPU-oracle audit found a deterministic boundary-index defect before the
acceptance run. CPU `ducrossensor` treated `npdcj=4` and `npdck=4` as neither
physical side, while x correctly treated `npdci=4` as both sides. Raw
pressure-curvature and expanded-mask stencils could therefore read invalid
y/z physical halos. After explicit user approval, CPU and GPU now clamp both
physical sides consistently. The Ducros formula, threshold, and fixed-`hm`
sensor exchange are unchanged.

Run the acceptance matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c9_selective_roe_matrix \
  tests/gpu_validation/run_curvilinear_hbl_c9_selective_roe_matrix.sh
```

The matrix covers NP=1 one-step raw-sensor isolation, NP=1 20 steps, NP=2
`2x1x1`, `1x2x1`, and `1x1x2` for three steps, and NP=4 `2x2x1` for three
steps. The z slab uses `KM=16` to preserve the local halo extent. Raw-sensor
`L_inf` is `9.9973866084364236e-16`; the CPU and GPU masks have zero
mismatches and both contain 7659 shock nodes. The maximum field error is
`T=1.4921397450962104e-13`, and the maximum statistic error is
`massflux=4.9649173661237000e-13`. Lower-wall invariants remain below
`9e-16`. The numerical Jacobian is finite and positive in
`[2.2722591030002285e-6,5.0507585440825813e-4]`.

NP=1 Compute Sanitizer reports `ERROR SUMMARY: 0 errors`. The no-checkpoint
three-step Nsight trace is exported to SQLite and checked with:

```bash
python3 tests/gpu_validation/analyze_nsys_rk_residency.py \
  --sqlite tests/gpu_validation/out/curvilinear_hbl_c9_goal/residency/c9_np1_nochk.sqlite \
  --start-kernel characteristic_upwind_rhs_x_xyphysical_global_kernel \
  --large-transfer-bytes 65536 \
  --report tests/gpu_validation/out/curvilinear_hbl_c9_goal/residency/residency_report.txt
```

After the first characteristic x-RHS kernel, the trace contains three
176-byte H2D operations and three 6144-byte D2H statistic reductions. No
transfer is at or above 64 KiB. The sensor and selective-Roe kernels retain
the project rule of an explicit synchronization after every kernel.

CURVE-C0-C8, the ten-case Cartesian S2-C3 matrix, S0-A4-A10, and zero-warp
S1-C1/C2 regressions pass on the same sources. CURVE-C9 is a controlled
numerical-equivalence and residency gate. It does not establish physical
SBLI fidelity, mesh/time convergence, shock-position accuracy, NSCBC or
sponge behavior, filter/diffusion coupling, species, turbulence, chemistry,
IBM, moving or multi-block grids, GPU HDF5, or production scaling.

## CURVE-C10 viscous curvilinear selective-Roe gate

CURVE-C10 combines the C9 Ducros/selective Roe-characteristic MP7 path with
the C7/C8 sixth-order explicit diffusion and Sutherland viscosity path. It
retains `lfilter=f`, no sponge, `bctype=11/21,41/51`, periodic z, no species,
and no turbulence. The dedicated
`gpu_s2_hbl_selective_roe_diffusion_supported()` predicate does not broaden
the existing inviscid C9 predicate.

Run the default topology matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c10_viscous_selective_roe_matrix \
  tests/gpu_validation/run_curvilinear_hbl_c10_viscous_selective_roe_matrix.sh
```

The default matrix covers NP=1 one-step sensor isolation and 20 steps, all
three NP=2 slabs, all three NP=4 plane decompositions, and NP=8 `2x2x2`.
Set `RUN_STRESS=t` to add NP=1 and NP=8 100-step runs. Z-decomposed cases use
`KM=16` so every local active extent remains at least `hm`.

All gates pass at `1e-10`. The one-step raw-sensor `L_inf` is
`9.9973866084364236e-16`; CPU and GPU masks have zero mismatches and both
contain 7659 marked nodes. Across the executed matrix and separate 100-step
runs, the largest field error is `u1=7.6039174956576971e-13` and the largest
statistic error is `massflux=5.0892623448817176e-13`. Lower-wall invariants
remain below `1.8e-15`, and the numerical Jacobian remains finite and positive
in `[2.2722591030002285e-6,5.0507585440825813e-4]`.

NP=1 Compute Sanitizer reports `ERROR SUMMARY: 0 errors`. A dedicated
no-checkpoint three-step Nsight trace contains 297 kernels after the first
`characteristic_upwind_rhs_x_xyphysical_global_kernel`. H2D is limited to
three 176-byte operations and D2H to three 6144-byte statistic reductions;
there is no transfer at or above 64 KiB. CURVE-C8 and C9 single-rank
regressions pass on the same GPU binary. C10 therefore establishes numerical
equivalence, halo composition, and compute-loop residency. It does not
establish physical SBLI fidelity, mesh/time convergence, shock-position
accuracy, filter/NSCBC/sponge coupling, arbitrary curved physical faces,
moving or multi-block grids, GPU HDF5, or production scaling.

## CURVE-C11 full-domain filter audit

The first CURVE-C11 probe tested scheme A: retain the CPU full-domain explicit
10th-order filter while enabling the C9/C10 Ducros and selective
Roe-characteristic MP7 path. The isolation case used the sharp sustained
oblique-shock input on `64x64x8`, NP=1, `lfilter=t`, and `diffterm=f`.

The CPU reference failed before CPU/GPU comparison. After the first complete
RK step, `crashcheck` stopped at step 1. The same-phase HDF snapshot contained
non-finite values in 22529 density entries, 22160 entries in each velocity
component, 22412 pressure entries, and 22104 temperature entries out of 38025
grid points. The failed run was isolated under
`/tmp/astr_curve_c11_filter_isolation`; it is diagnostic evidence rather than
a retained validation artifact.

This result rejects scheme A on the current discontinuous input. Applying a
global centered filter across the discontinuity is the likely source of
nonphysical overshoot, and the CPU filtered-`q`/pre-filter-primitive timing may
amplify it. The first corrupting operator stage has not been instrumented, so
the evidence does not establish a coefficient-level filter bug. No C11 GPU
capability or dedicated driver is retained. The approved policy closes this
negative gate and keeps shock-containing selective-Roe paths at `lfilter=f`.
A future sensor-aware or positivity-preserving filter requires a separate
numerical-policy decision and validation campaign.

## CURVE-C12 characteristic-boundary and sponge gate

CURVE-C12 combines C10's static nonorthogonal Mach 5 grid, sixth-order
diffusion, Sutherland viscosity, Ducros sensor, and selective Roe MP7 with the
Cartesian S2-C4 boundary contract `12/21,41/52` and the restricted S2-C5
x-max `layer` sponge. The new diffusion capability predicates admit no sponge
or only positive `spg_im`. Filtering, y/z/circular sponge, species,
turbulence, and other boundary combinations remain rejected.

Run the default topology matrix with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_hbl_c12_nscbc_sponge_matrix \
  tests/gpu_validation/run_curvilinear_hbl_c12_nscbc_sponge_matrix.sh
```

The matrix covers NP=1 NSCBC and sponge isolation, NP=1 20 steps, all NP=2
slabs, all NP=4 plane decompositions, and NP=8 `2x2x2`. Set `RUN_STRESS=t`
to append NP=1 and NP=8 100-step sponge runs. All comparisons pass at
`1e-10`. The one-step raw-sensor `L_inf` is
`9.9973866084364236e-16`; masks have zero mismatches with 7659 marked nodes.
Across the 100-step gates, the largest field error is
`u1=7.6039174956576971e-13`, and the largest statistic error is
`massflux=4.9964921089440395e-11`. Lower-wall invariants remain below
`1.8e-15`, and the Jacobian remains finite and positive in
`[2.2722591030002285e-6,5.0507585440825813e-4]`.

The first C12 isolation run exposed a GPU routing defect. The new diffusion
predicates entered the C4/C5 main loop but were absent from
`gpu_prepare_rkfirst_stats()`, so statistics boundary preparation modified q
without the required snapshot and restore. After explicit approval, the
snapshot route includes both predicates. C10 and Cartesian C4/C5 single-rank
regressions pass after the correction.

NP=1 Compute Sanitizer reports `ERROR SUMMARY: 0 errors`. A no-checkpoint
three-step Nsight trace is checked with:

```bash
python3 tests/gpu_validation/analyze_nsys_rk_residency.py \
  --input /tmp/astr_curve_c12_nsys_rank0.sqlite \
  --start-kernel characteristic_upwind_rhs_x_xyphysical_global_kernel \
  --large-transfer-bytes 65536 \
  --report /tmp/astr_curve_c12_residency_report.txt
```

After the selected kernel, the trace contains 503 kernels and no H2D/D2H
transfer at or above 64 KiB. Maximum H2D and D2H sizes are 176 B and 6144 B.
C12 therefore establishes numerical equivalence, tested fixed-`hm` halo
composition, and compute-loop residency for this restricted curved
NSCBC/sponge slice. It does not establish physical SBLI fidelity, mesh/time
convergence, arbitrary curved-face characteristic conditions, y/z sponge,
moving or multi-block grids, GPU HDF5, or production scaling.

## CURVE-C20 open-boundary inventory

Validate the corrected complete-state `bctype=11` inlet with an initial field
that deliberately differs from `inlet.prof`:

```bash
OUT_DIR=/tmp/astr_profile_inflow11_state \
  tests/gpu_validation/run_profile_inflow11_state_compare.sh
```

Run the matching single-rank memory check from the generated GPU case with
OpenMPI's UCX/CUDA-aware probes isolated:

```bash
cd /tmp/astr_profile_inflow11_state/gpu
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
  mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  /home/dell/workspace/astr_gpu/build_gpu_probe/bin/astr \
  run datin/input.flatplate
```

The gate checks the CPU complete-RK snapshot and GPU output independently
against the x-min profile, excluding y-endpoints that are owned by the later
wall/farfield boundary passes. The current maximum target error is
`4.4408920985006262e-15`, the EOS residual is
`2.2204460492503131e-16`, and CPU/GPU reconstructed `q5 L_inf` is
`7.1054273576010019e-15`. The matching Compute Sanitizer run reports
`ERROR SUMMARY: 0 errors`. Without the documented MPI component isolation,
OpenMPI/UCX CUDA pointer probes produce initialization-layer false positives.

For a mixed subsonic/supersonic profile inlet, enable pressure extrapolation
only on locally subsonic points with:

```bash
OUT_DIR=/tmp/astr_profile_inflow11_mach_pressure \
  tests/gpu_validation/run_profile_inflow11_mach_pressure_compare.sh
```

The runner sets `ASTR_PROFILE_INFLOW_MODE=mach_pressure`. The inward normal
velocity is evaluated by projecting the prescribed profile velocity onto the
geometric x-min boundary normal. Locally supersonic points retain the complete
profile state. Locally subsonic points prescribe velocity, temperature, and
species, extrapolate pressure from the first two interior planes, and recover
density from the equation of state. The default `complete_state` mode is kept
for existing `bctype=11 + prof` cases.

Run the required unsupported-geometry checks with:

```bash
OUT_DIR=/tmp/astr_c20_open_rejects \
  tests/gpu_validation/run_curvilinear_open_boundary_rejects.sh
```

The three entries require dedicated pre-launch rejection of a curved x-max
`21` face, any `22` condition on a nonorthogonal grid, and a curved upper-y
`51` face. Axis-aligned stretched faces remain admissible. Cartesian
OpenShock `12/22`, C7 `11/21,41/51`, and C12 `12/21,41/52` remain positive
regressions. C20 keeps `12/52` case-specific and does not infer missing CPU
directions or standardize the deferred NSCBC relaxation policy.

## CURVE-C21 aggregate closure

Run the complete static single-block curvilinear release gate with:

```bash
OUT_DIR=tests/gpu_validation/out/curvilinear_c21_aggregate \
  tests/gpu_validation/run_curvilinear_c21_aggregate.sh
```

The driver rebuilds the existing top-level CMake CPU/GPU build trees, runs
all C0-C20 supported matrices, checks the C11 closed-negative policy without
rerunning its known non-finite CPU shock/filter probe, runs the required
curved-open-boundary rejects, repeats representative x/y/z memchecks, and
executes the C15 `256^3` residency and three-repeat timing gates. Set
`START_STAGE` and `STOP_AFTER` only for diagnosis; a release closure requires
the uninterrupted full run. `RUN_PERFORMANCE=f` intentionally fails C15 and
cannot produce a passing aggregate closure.

The driver writes `c21_stage_summary.tsv`, per-stage logs, and a generated
`c21_aggregate_report.md`. The report is produced by:

```bash
python3 tests/gpu_validation/summarize_curvilinear_c21.py \
  --input tests/gpu_validation/out/curvilinear_c21_aggregate \
  --output tests/gpu_validation/out/curvilinear_c21_aggregate/c21_aggregate_report.md
```

The 2026-09-05 run passed all 23 stages. Across 136 field reports, 128
statistic reports, and 226 boundary-invariant reports, the aggregate maxima
are `q5=6.2527760746888816e-13`,
`massflux=2.5093260802577788e-11`, and required boundary residual
`7.6170181273482740e-12`. All 70 explicit finite-field checks pass, the
minimum checked numerical Jacobian is `2.5296638468231652e-7`, and x/y/z
Compute Sanitizer runs each report zero errors.

The `256^3` no-checkpoint interval contains 221 kernels and no transfer at or
above 64 KiB. H2D is `2 x 176 B` and D2H is `2 x 48 B`. Three-repeat GPU
NP=1/NP=2 x-slab medians are `58.490/38.989 s`, giving `1.5002x` speedup and
`75.01%` two-GPU efficiency on the current machine. C21 closes only the
tested static single-block, nonreacting, explicit-scheme scope. It does not
establish arbitrary curved open boundaries, moving/multi-block support, GPU
HDF5, physical SBLI fidelity, or scaling beyond two GPUs.

## CURVE-C22 curved upper-y non-reflecting GCBC

`ASTR_NSCBC_FARFIELD_MODE=nonreflecting` enables a restricted inviscid
five-equation `bctype=52` condition on a static curved upper eta face. The CPU
and GPU paths use the local eta normal and balance only incoming
characteristics against the metric-plus-transverse source. The mode does not
use a target farfield state, empirical relaxation length, upper-face Mach
reduction, or the legacy x/z boundary-plane filters. The established
bctype=52 full-RK snapshot remains active because it preserves statistics and
integration phase semantics independently of filtering.

Run the algebra, uniform, acoustic, topology, safety, and runtime-work gates
with new output directories:

```bash
tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh
OUT_DIR=/tmp/curve_c22_uniform \
  tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh
OUT_DIR=/tmp/curve_c22_acoustic \
  tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh
CASE=uniform OUT_DIR=/tmp/curve_c22_uniform_matrix \
  tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh
CASE=acoustic OUT_DIR=/tmp/curve_c22_acoustic_matrix \
  tests/gpu_validation/run_curvilinear_nscbc52_matrix.sh
OUT_DIR=/tmp/curve_c22_memcheck \
  tests/gpu_validation/run_curvilinear_nscbc52_memcheck.sh
OUT_DIR=/tmp/curve_c22_profile \
  tests/gpu_validation/run_curvilinear_nscbc52_profile.sh
```

The three-grid acoustic sequence `64x48x64`, `80x60x80`, and `96x72x96`
gives CPU non-reflecting coefficients `0.01620258`, `0.00662146`, and
`0.00393555`; matched compatibility values are `0.17063313`, `0.11807481`,
and `0.10561771`. The maximum CPU/GPU reflection and final-field differences
are `1.88e-12` and `2.21e-13`. All eight NP=1/2/4/8 topologies pass owner and
field checks; these oversubscribed runs are correctness evidence, not scaling
measurements. NP=1 and NP=2 y-slab memchecks report zero errors. A two-step
Nsight Systems audit observes six non-reflecting RHS launches and zero legacy
Mach/x-filter/z-filter launches.

This gate does not validate viscous characteristic source terms, curved
target-relaxation modes, other open faces, moving or multiblock grids,
chemistry, or production SBLI farfield fidelity.

## CURVE-C23 curved upper-y viscous-source-coupled GCBC

CURVE-C23 extends only the established CURVE-C22 upper-eta `bctype=52`
non-reflecting mode to the approved viscous slice. The solver captures the
five-component upper-face RHS immediately before sixth-order explicit
diffusion, forms the discrete viscous contribution from the post-diffusion
RHS difference, projects that contribution with the same local eta
characteristic basis, and removes only its incoming characteristic content.
The CPU uses a face-only snapshot and the GPU uses a device-resident face
buffer. Each new GPU capture and correction kernel is followed by an explicit
synchronization.

Run the algebra, uniform, viscous acoustic, topology, safety, and residency
gates with new output directories:

```bash
tests/gpu_validation/run_curvilinear_nscbc52_policy_probe.sh
OUT_DIR=/tmp/curve_c23_uniform \
  tests/gpu_validation/run_curvilinear_nscbc52_viscous_uniform_compare.sh
OUT_DIR=/tmp/curve_c23_acoustic \
  tests/gpu_validation/run_curvilinear_nscbc52_viscous_acoustic_compare.sh
OUT_DIR=/tmp/curve_c23_matrix \
  tests/gpu_validation/run_curvilinear_nscbc52_viscous_matrix.sh
OUT_DIR=/tmp/curve_c23_memcheck \
  tests/gpu_validation/run_curvilinear_nscbc52_viscous_memcheck.sh
OUT_DIR=/tmp/curve_c23_profile \
  tests/gpu_validation/run_curvilinear_nscbc52_viscous_profile.sh
```

The uniform ten-step NP=1 gate preserves the prescribed state to about
`1e-15` and gives a maximum CPU/GPU field difference of
`7.99e-15`. The Mach 5 curved HBL NP=1 20-step gate gives maximum field and
statistic differences `T=5.06262e-14` and
`massflux=5.83977e-14`; the lower-wall residual remains below
`1.78e-15`, and the numerical Jacobian stays positive in
`[2.2722591030002285e-6,5.0507585440825813e-4]`. All eight NP=1/2/4/8
topologies pass. NP=1 and NP=2 y-slab Compute Sanitizer runs report zero
errors.

The viscous acoustic refinement sequence `64x48x64`, `80x60x80`, and
`96x72x96` gives CPU reflection coefficients `0.01499173359`,
`0.005955882611`, and `0.003584736637`. The trend is strictly decreasing;
the maximum CPU/GPU reflection and final-field differences are
`3.24e-13` and `2.13e-13`.

A two-step NP=1 trace contains six inviscid non-reflecting, six pre-diffusion
capture, and six viscous correction launches. After the first non-reflecting
kernel, no H2D or D2H transfer is at or above 64 KiB; the maximum H2D/D2H
sizes are 320/48 B. The NP=2 y-slab owner trace observes one physical
upper-y owner among two ranks.

This gate proves CPU/GPU numerical equivalence, owner routing, memory safety,
and compute-loop residency for the tested static single-block, nonreacting,
five-equation upper-eta slice. It does not establish a general complete
viscous open boundary, other open faces, moving or multiblock grids,
chemistry, or production SBLI physical fidelity.

## TGV 256 single-GPU performance gate

Build the GPU binary from the repository top-level `CMakeLists.txt`, then run
one process warm-up followed by five measured processes:

```bash
OUT_DIR=/tmp/astr_tgv_perf \
  tests/gpu_validation/run_tgv_256_performance_benchmark.sh
```

The driver fixes the default case at FP64 `256^3`, explicit `643e/643e`,
tenth-order filtering, diffusion, and NP=1. It disables checkpoint output in
the measured loop, discards the first complete RK timing in each process, and
records per-process timing, device memory, and utilization. Override `GPU_ID`
to select a device.

Collect the matching Nsight Systems trace and residency report with:

```bash
PROFILE_TOOL=nsys OUT_DIR=/tmp/astr_tgv_nsys \
  tests/gpu_validation/run_tgv_256_performance_profile.sh
```

The residency audit permits a transfer at or above 64 KiB only when a D2H
copy immediately follows one of the three named TGV statistic partial-reduction
kernels. H2D transfers are never exempted. Raw, allowed, and forbidden counts
are reported separately.

Collect one full Nsight Compute instance of the current diffusion-flux hotspot
with:

```bash
PROFILE_TOOL=ncu MAXSTEP=0 OUT_DIR=/tmp/astr_tgv_ncu \
  tests/gpu_validation/run_tgv_256_performance_profile.sh
```

The NCU path embeds available Fortran source into the report and also emits
`ncu_details.txt` plus the source-correlated `ncu_source.csv`. Inspect an
existing report directly from the CLI with:

```bash
ncu --import /tmp/astr_tgv_ncu/diffusion_flux_full.ncu-rep \
  --page details --print-details all
ncu --import /tmp/astr_tgv_ncu/diffusion_flux_full.ncu-rep \
  --page source --print-source cuda,sass --csv --print-units base
```

Use `ncu --list-sets` and `ncu --list-sections` to enumerate the locally
installed metric groups. `NCU_SET`, `NCU_KERNEL`, and `NCU_LAUNCH_SKIP`
select the collection scope. NCU replay duration is diagnostic and must not
replace the five-repeat complete-RK benchmark.

The same commands are the A800 rerun entry points. Configure a fresh CMake
build on that system instead of copying a workstation build tree or cache.
The accepted workstation evidence and exact correctness commands are recorded
in `documents/ASTR_TGV_PERFORMANCE_OPTIMIZATION_REPORT.md`.

Set `SYNC_MODE=selective` to evaluate the opt-in TGV-only synchronization
path with either entry point. The program rejects this mode for non-TGV or
non-periodic cases. The default is `SYNC_MODE=explicit`. On the RTX 4000 Ada
workstation, selective mode reduces profiled RK `cudaDeviceSynchronize` calls
from 275 to 15 but changes the five-run median from `0.682327627` to
`0.683028141 s/RK`, so it is not the performance default. NP=1 and NP=2
x-slab ten-step fields match explicit mode exactly, all CPU/GPU field and
statistic reports pass at `1e-10`, and the representative selective memcheck
reports zero errors. The residency report now records synchronization count
and cumulative API duration in addition to transfer evidence.

The retained arithmetic optimization now imports the existing CPU
`constdef::num1d60` parameter in both `solver_gpu` and `gradcal_gpu`. All GPU
sixth-order central derivatives multiply their outer difference by this
compile-time constant. WENO reconstruction and unrelated divisions are not
changed. The final `256^3` five-run median is `0.580044424 s/RK` with `2.027%`
spread, compared with the first-round frozen baseline of `0.622173751 s/RK`
and the original `0.762855769 s/RK` baseline. The reductions are `6.771%` and
`23.964%`, respectively. Peak sampled memory is `9738 MiB`.

Collect the 12-kernel NCU matrix with:

```bash
OUT_DIR=/tmp/astr_tgv_ncu_matrix \
  tests/gpu_validation/run_tgv_256_ncu_hotspot_matrix.sh
```

The matrix covers convection x/y/z, diffusion flux, diffusion RHS x/y/z,
filter x/y/z, primitive conversion, and `gradcal`. It records SM/DRAM
throughput, registers, occupancy, instructions, excessive L2 sectors,
available branch-uniformity data, normalized not-issued warp stalls, and the
three highest sampled source lines. `--target-processes-filter regex:^astr$`
prevents NCU from treating the MPI launcher as the profile target.

The retained candidate removes all `MUFU.RCP64H` sites from the profiled
sixth-order lines. NCU reports convection x decreasing from `28.184` to
`17.581 ms` and from 383.47 to 252.50 million instructions. `gradcal`
decreases from `35.463` to `17.972 ms` and from 671.53 to 403.93 million
instructions, while register use decreases from 88 to 72. The corresponding
normal-execution Nsight Systems trace reduces convection x/y/z by `28.771%`,
`gradcal` by `29.729%`, and total GPU-kernel time by `10.046%`.

NP=1 and NP=2 `2x1x1` ten-step field/statistic gates pass at `1e-10`; maximum
field errors are `q5=2.8421709430404007e-13` and
`q5=3.1263880373444408e-13`. A five-step `64^3` channel filter-plus-diffusion
statistics gate also passes. Its complete HDF comparison retains the known
x/z periodic output-phase defect. For the narrower physical-boundary
regression, use `compare_flowfield_h5.py --trim-boundary-axis 0
--trim-boundary-axis 2`; this retains all y-wall planes and gives
`q5 L_inf=4.618527782440651e-14` in the periodic interior. It is not a
full-checkpoint pass.

Run the representative memcheck with:

```bash
OUT_DIR=/tmp/astr_tgv_memcheck \
  tests/gpu_validation/run_tgv_gpu_memcheck.sh
```

The default case is FP64 `32^3`, one step, explicit synchronization,
filtering and diffusion enabled. The retained candidate reports
`ERROR SUMMARY: 0 errors`. Its `256^3` Nsight Systems residency audit finds
zero forbidden H2D/D2H transfers at or above 64 KiB; the 12 permitted 512 KiB
D2H operations are the existing TGV diagnostic reductions.

## GPU optimization candidate gate

Use the unified gate only after writing a bounded optimization hypothesis and
declaring the exact source paths allowed to change:

```bash
CANDIDATE_ID=conv-load-reuse-01 \
BASELINE_REF=<git-ref> \
BASELINE_TIMINGS=/absolute/path/to/baseline_timings.tsv \
TARGET_KERNELS='convection x/y/z' \
ALLOWED_PATHS='src_gpu/solver_gpu.cuf' \
HYPOTHESIS='Reuse primitive and metric loads without changing arithmetic order' \
GATE_SET=full \
OUT_DIR=/tmp/astr_candidate_conv_load_reuse_01 \
  tests/gpu_validation/run_gpu_optimization_candidate_gate.sh
```

`correctness` runs the top-level CPU/GPU builds and ten-step NP=1/NP=2 TGV
field/statistic gates. `performance` adds the five-repeat `256^3` benchmark,
comparison with the frozen baseline TSV, and the Nsight Systems residency
audit. `full` additionally runs the 12-kernel NCU matrix and Compute Sanitizer.
All modes enforce explicit synchronization and `atol=rtol=1e-10`. Use
`DRY_RUN=t` to inspect the command sequence without executing it.

The complete acceptance contract and prohibited optimization list are in
`documents/ASTR_GPU_OPTIMIZATION_CANDIDATE_PROTOCOL.md`. Passing this TGV gate
does not replace the regression matrix for a physical boundary, shock,
curvilinear, or other subsystem touched by the candidate.

## Phase P2 shock-path closure

The P2 runners use separate timing and shock-activity runs for periodic
Shu-Osher and SBLI:

```bash
CASE=shuosher OUT_DIR=/tmp/p2_shuosher \
  tests/gpu_validation/run_p2_shock_performance_benchmark.sh
CASE=sbli OUT_DIR=/tmp/p2_sbli \
  tests/gpu_validation/run_p2_shock_performance_benchmark.sh
```

The retained P2-1B path computes all five physical Steger-Warming split-flux
components once per stencil point in the nonshock branch. It keeps the single
sensor-coupled interface kernel, FP64, fixed direction blocks, and explicit
synchronization. The corrected `bctype=52` statistics path prepares the same
x-filter, halo, z-filter state observed by CPU `rkfirst`, while reusing
`qsave_d` to restore the complete pre-statistics RK state before integration.
The final ten-step SBLI online `massflux` maximum difference is
`5.0293103015519591e-13` at the unchanged `1e-10` gate.

Profile one or two physical GPUs with the same driver:

```bash
CASE=shuosher PROFILE_TOOL=nsys NSYS_TRACE=cuda,mpi \
NP=2 TOPOLOGY=1,2,1 GPU_IDS=0,1 \
OUT_DIR=/tmp/p2_shuosher_np2_y \
  tests/gpu_validation/run_p2_shock_performance_profile.sh

python3 tests/gpu_validation/summarize_p2_sensor_halo_nsys.py \
  --input /tmp/p2_shuosher_np2_y/p2_shock_nsys.sqlite \
  --runtime-log /tmp/p2_shuosher_np2_y/nsys.log \
  --sensor-message-bytes 339240 \
  --report /tmp/p2_shuosher_np2_y/sensor_halo_timeline.md
```

`sensor-message-bytes` must be computed from the decomposed plane as
`hm * (local transverse points) * 8` for the one-component sensor. The
summarizer pairs raw and expanded sensor kernels on each GPU and counts only
same-process MPI events inside those windows. The validated NP=2 y-slab upper
bounds are `3.488%` for Shu-Osher `256x64x32` and `1.421%` for SBLI
`256x192x32`. The sensor pack/unpack kernels themselves are below `3%`; host
staging and blocking MPI remain a P3 concern.

P2-1C sequential split-array reuse was rejected. Under the same 10-RK-sample
five-run contract, it improved complete RK by only `1.975%` on Shu-Osher and
`1.208%` on SBLI, while SBLI y-kernel local
spilling increased from `15,059,748` to `18,007,902`. P2 is closed with P2-1B
as the only retained shock-path kernel optimization. Full commands, evidence
directories, and final five-repeat numbers are recorded in
`documents/ASTR_PHASE_P2_BASELINE_REPORT.md`.

### Phase P3 Transport Baselines

P3 is in progress. See `documents/ASTR_PHASE_P3_IMPLEMENTATION_PLAN.md` and
`documents/ASTR_PHASE_P3_BASELINE_REPORT.md`. The shared performance driver
`run_p2_shock_performance_benchmark.sh` now accepts `CASE=tgv|shuosher|sbli`,
`NP`, `TOPOLOGY`, and `GPU_IDS`. Performance runs require one distinct GPU per
rank. It records per-rank RK durations and summarizes their stepwise maximum;
the maximum is not a sum across ranks. Output directories cannot overwrite
existing timing TSV files. The new timing output requires a current executable.

Build and execute exact-value host transport tests through top-level CMake:

```bash
cmake -S . -B build_gpu_probe -DASTR_WITH_CUDA=ON
cmake --build build_gpu_probe --target halo_transport_test -j4
mpirun -np 1 build_gpu_probe/bin/halo_transport_test
mpirun -np 2 build_gpu_probe/bin/halo_transport_test
mpirun -np 3 build_gpu_probe/bin/halo_transport_test
```

These tests exercise paired messages, component counts 1/3/6, widths 5/6,
periodic duplicate peers and MPI_PROC_NULL. They complement, rather than replace,
the CPU/GPU CFD halo comparisons. The later device-aware section records the
separate A800 admission; these host tests do not provide that evidence.
The solver has an experimental pinned blocking selection. Paired nonblocking
was screened and removed; its evidence is retained, not its runtime selection.

Pinned-memory prerequisite probes (still not a solver transport selection):

```bash
cmake --build build_gpu_probe --target halo_transport_test halo_pinned_probe -j4
build_gpu_probe/bin/halo_pinned_probe
mpirun -np 2 build_gpu_probe/bin/halo_transport_test pinned
compute-sanitizer --tool memcheck --error-exitcode 99 build_gpu_probe/bin/halo_pinned_probe
env UCX_MEMTYPE_CACHE=n mpirun --mca pml ob1 --mca osc pt2pt \
  --mca btl self,tcp --mca coll '^hcoll,ucc' --mca opal_cuda_support 0 \
  -np 2 compute-sanitizer --tool memcheck --error-exitcode 99 \
  build_gpu_probe/bin/halo_transport_test pinned
```

Use the MPI installation linked by the build. The host-only MPI settings above
are diagnostic: default HPC-X/UCX triggers CUDA context API errors in MPI_Init
even for the pageable exact test. The pinned host-only NP=2 probe and standalone
copy probe pass memcheck with zero errors. Retain this distinction when reporting
results; do not reuse host-only settings to claim CUDA-aware coverage or compare
performance against a different MPI configuration.

The solver defaults to `ASTR_GPU_HALO_TRANSPORT=pageable`; `pinned` explicitly
requests registration before the first solver halo exchange. Each rank reports
its selected state and registered host bytes. A registration-admission failure
on any rank causes collective cleanup and pageable fallback before communication.
Different rank settings or unknown backend names are errors, not silent fallback.
This candidate has not yet passed the complete P3 acceptance matrix.

The archived `pinned-nonblocking` experiment posts two receives and two sends,
then Waitall. Its largest reproduced incremental gain over pinned blocking was
2.425%, below the independent 3% gate, so current executables reject this option.
`out/p3_nonblocking_frozen/astr` and its source snapshot retain the experiment;
all nine timing groups, controls and traces are documented in the P3 report.
No compute/communication overlap was demonstrated by this experiment.

Production registration lifecycle and collective rollback can be tested with:

```bash
cmake --build build_gpu_probe --target halo_transport_setup_test -j4
env ASTR_GPU_HALO_TRANSPORT=pinned mpirun -np 2 \
  build_gpu_probe/bin/halo_transport_setup_test normal
env ASTR_GPU_HALO_TRANSPORT=pinned mpirun -np 2 \
  build_gpu_probe/bin/halo_transport_setup_test overflow
```

The overflow test exceeds the fixed registry capacity on rank zero only. Both
ranks must fall back, and all previously registered allocations must accept a
new registration after cleanup. Run this test under the same diagnostic
host-only MPI memcheck settings above when checking for zero CUDA API errors.

MPI progress/independent-work prerequisite (not an ASTR performance benchmark):

```bash
cmake --build build_gpu_probe --target halo_overlap_probe -j4
env CUDA_VISIBLE_DEVICES=0,1 mpirun --mca coll_hcoll_enable 0 -np 2 \
  build_gpu_probe/bin/halo_overlap_probe poll 1981470 8388608 30 8
python3 tests/gpu_validation/summarize_mpi_kernel_overlap.py \
  tests/gpu_validation/out/p3_overlap_probe_nsys/poll.sqlite
```

Arguments are mode (`blocking`, `wait`, `poll`), message elements, work elements,
kernel passes and retained iterations. The probe runs one extra warmup iteration,
requires one visible GPU per rank and retains explicit synchronization after
every kernel. Use all three modes with identical parameters. Exact payload and
linear derivative checks are mandatory. The SQLite analyzer reports same-process
MPI API/kernel intersections, deduplicating request rows and unioning kernel
intervals. It does not measure concurrent communication bytes or establish an
ASTR RK speedup. Full commands, results and evidence limits are in the P3 report.
Use `start_wait_calls_overlapping_kernel` to distinguish request-record events
from MPI_OTHER_EVENTS calls such as tests after requests have become null.

Experimental solver overlap is selected with
`ASTR_GPU_HALO_TRANSPORT=pinned-overlap`. It passed the initial nine-case timing
screen with an explicit Shu-Osher x noise/repeat audit, but final admission still
requires the remaining validation matrix. No default backend has changed.
Only fully periodic, distributed stored diffusion with a nonempty radius-three
core activates the split; all other cases retain blocking communication.
Check both the per-rank request log and `periodic_diffusion_active` records.
Use `OMPI_MCA_sharedfp=individual` consistently for solver runs, as in the P3
benchmark driver, to avoid the observed OMPIO shared-file-pointer initialization
stall. The transport callback itself can be checked independently:

```bash
env ASTR_GPU_HALO_TRANSPORT=pinned mpirun -np 3 \
  build_gpu_probe/bin/halo_transport_test pinned work
env OMPI_MCA_sharedfp=individual ASTR_GPU_HALO_TRANSPORT=pinned-overlap \
  NP=2 TOPOLOGY=2,1,1 MAXSTEP=10 FEQCHKPT=10 \
  OUT_DIR="$PWD/tests/gpu_validation/out/p3_overlap_tgv_x_new" \
  bash tests/gpu_validation/run_tgv_mpirank2_field_compare.sh
```
# P3 CUDA-Aware Diagnostic Admission

`halo_cuda_aware_probe` is an excluded diagnostic target, not an ASTR backend.
Configure with the repository's top-level GPU CMake build, then build with
`cmake --build build_gpu_probe --target halo_cuda_aware_probe`.
Run `mpiexec -np 2 build_gpu_probe/bin/halo_cuda_aware_probe 256 256`
for representative large device-buffer messages. Optional third argument
`early-bind` binds the device before MPI_Init for diagnosis only. Default
arguments are 3 and 2. Exact payload checks include widths 5/6, components
1/3/5/6/9, periodic peers and MPI_PROC_NULL. Failures abort rather than continue.

The local RTX 4000 Ada HPC-X stack does not qualify CUDA IPC; its explicit
no-IPC `cuda_copy` path is retained for local correctness checks. The installed
MPIX query cannot qualify a backend alone. The solver now provides a fail-closed
`device-aware` option, but IPC production evidence comes from A800 job `460370`
and solver jobs `460439/460441`, not from this historical local probe. See the
device-aware section below for the current contract and evidence boundary.

## P3 Retained-Backend Matrix Checkpoint

`out/p3_final_phase_matrix_summary.json` records 45 passing ten-step CPU/GPU
comparisons: pageable/pinned/pinned-overlap, TGV/Shu-Osher/SBLI, NP=1/2 x/y/z/8.
Fields and statistics use existing 1e-10 tolerances; sensor checks retain
1e-12. SBLI must set `IM=64 JM=64 KM=16 SAME_PHASE_FIELD=t COMPARE_SENSOR=t`
for this matrix. Setting GRID does not override that driver's dimensions.
Use MAXSTEP=10 FEQCHKPT=10 and OMPI_MCA_sharedfp=individual for all three cases.
TGV uses 128^3 with filter and diffusion; Shu-Osher uses GRID=400,16,16 and the
existing s0a6/s0a7/s0a8/s0a9/s0a10 topology drivers. CPU TGV references can be
reused only with identical inputs, topology and executable; the accepted run
records this provenance in reference.json. NP=8 is correctness only.

`out/p3_final_memcheck_summary.json` records nine small NP=2 solver checks and
18 zero-error rank logs. Exact commands are stored in each
`out/p3_final_memcheck_<backend>_<case>/command.json`. These checks use host-only
MPI, not CUDA-aware MPI. The source and CPU/GPU executables are frozen in
`out/p3_final_frozen/`. This checkpoint precedes the outstanding MPI_PROC_NULL
receive-upload guards and is not final P3 admission.

The later endpoint-cleanup executable is frozen in `out/p3_endpoint_frozen/`.
Its 18 receive-upload guards pass the supplemental source test
`test_halo_endpoint_upload_contract.py`. All 45 CPU/GPU comparisons pass again
in `out/p3_endpoint_v2_*`, with nine additional x/y/z physical wall41
filter/diffusion checks in `out/p3_endpoint_wall41_*`. These wall checks use
32^3, MAXSTEP=2, NP=2 in the wall-normal direction, and explicitly set both
field/statistics tolerances to 1e-10. The updated memcheck summary is
`out/p3_endpoint_memcheck_summary.json`: 18 runs and 36 zero-error rank logs
under host-only MPI. Shu-Osher x reference reuse must preserve
ASTR_SHUOSHER_SHOCK_X=0.d0, not merely the input-file contents.

Production GPU pack/unpack tests with deliberately different shared-interface
values and final-source performance reconciliation remain open; host transport
payload tests alone do not close the averaging contract.

### Production Halo Contract Executable

Build the excluded `halo_exchange_contract_test` target through the top-level
GPU CMake build. It recompiles production modules in a separate module directory
without the ASTR main program; no pack/unpack kernel is copied into the test.
Run with NP=2 and NP=3, setting ASTR_GPU_HALO_TRANSPORT to each retained mode.
The fixture seeds different exact integer values on every rank, including
conflicting shared planes and all untouched halo regions. It checks all three
directions with periodic duplicate peers and physical chains, hm+1 averaging,
hm filter exchange and q/qwork ownership, and 1/3/6-component field exchange.
NP=3 places a rank between two physical endpoints. A full-array exact comparison
also detects writes to unused components, transverse edge/corner halos and
MPI_PROC_NULL faces. This is a communication-contract test, not CFD performance.

The six NP=2/3 backend runs pass exactly in `out/p3_production_halo_*.log`.
NP=3 host-only memcheck passes for all three modes, with nine zero-error rank
logs in `out/p3_production_halo_memcheck_*`. The executable and sources are
frozen in `out/p3_production_halo_frozen/`. This closes the averaging test gap
noted in the earlier checkpoint; final performance reconciliation stays open.
The new final-source TGV x timing groups are in `out/p3_final_performance_*_tgv_x`.
The overlap group has spread 8.020% and must be retained as inconclusive in full.

Final-source Shu-Osher/SBLI timing summaries are
`out/p3_final_performance_shu_sbli_summary.json` and
`out/p3_final_performance_shu_z_repeat1_summary.json`. The original Shu z pinned
group has spread 26.623%; retain it unchanged and use the separately identified
complete reverse-order repeat when assessing stable groups. All three backends
were rerun, not just one selected process. Maximum per-device memory growth is
3.436%. These cases report overlap inactive and do not prove overlap performance.

The existing NSYS SQLite request records lack the Testall completion flag.
Request-associated API/kernel intersections are not, by themselves, proof of
unfinished communication during useful work. Completion-state/progress evidence
must accompany them before the overlap candidate receives final admission.

### Profiling Actual MPI Completion Flags

An optional PMPI interposer now closes that request-state evidence gap:

```bash
cmake -S . -B build_gpu_probe -DASTR_BUILD_MPI_COMPLETION_TRACE=ON
cmake --build build_gpu_probe --target mpi_completion_trace mpi_completion_probe -j4
nsys profile --trace=nvtx --sample=none --cpuctxsw=none -o completion_probe \
  mpiexec -np 2 env LD_PRELOAD="$PWD/build_gpu_probe/lib/libmpi_completion_trace.so" \
  "$PWD/build_gpu_probe/bin/mpi_completion_probe"
```

Build through the configured top-level NVHPC GPU project. The interposer uses
the same MPI Fortran ABI as ASTR, and NVTX3 headers from that NVHPC installation.
Apply LD_PRELOAD to rank processes, not the launcher. Never preload it for
formal performance benchmarks. No additional Testall calls are introduced.
Marks distinguish actual incomplete/completed flags for active four-request
calls; all-null request polls are excluded. The controlled probe passes both
without and with the interposer, and its final all-null poll emits no mark.

For a prepared solver case, use `nsys profile --trace=cuda,nvtx` around
`mpiexec -np 2 env LD_PRELOAD=<absolute-library-path> <absolute-astr-path> run <input>`.
Keep the case's usual CUDA_VISIBLE_DEVICES, ASTR_FORCE_MPI_TOPOLOGY,
ASTR_GPU_SYNC_MODE=explicit, ASTR_GPU_HALO_TRANSPORT=pinned-overlap and
OMPI_MCA_sharedfp=individual settings. Export the report as SQLite, then run:

```bash
python3 tests/gpu_validation/summarize_mpi_kernel_overlap.py \
  tests/gpu_validation/out/p3_completion_tgv_y_nsys/trace.sqlite --completion
```

The final-source TGV 256^3 NP=2 y-slab trace records 4237 pending and 24 completed
marks during same-process diffusion kernels. This is request-state overlap
evidence, not bandwidth or performance evidence. See the P3 baseline report for
the controlled probe, exact solver hash and interpretation limits.

### P3 Local Acceptance

Local acceptance is complete: default pageable and optional pinned blocking
are retained; optional pinned-overlap is admitted only for the implemented
fully periodic stored-diffusion core. Standalone paired nonblocking is rejected;
CUDA-aware remains deferred on the tested stack. No numerical scheme, halo
width, interface average or explicit kernel synchronization changes.

The final performance audit is `out/p3_final_performance_complete_audit.json`:
34 full groups and 204 process logs including warmups, with all inconclusive
groups preserved. Final TGV y/z pass. TGV x overlap's original and repeat1
groups are inconclusive; its complete repeat2 pinned/overlap pair passes.
Valid overlap increments over paired pinned are x/y/z: 2.616/3.085/2.944%.
Pinned gains across nine formal cases are 2.829%--23.728%; maximum per-device
memory growth is 3.436%. Input/controller hashes match within each case/topology.

Builds, 68 Python tests and 96 bash syntax checks pass. The existing final-source
CFD and zero-error host-only MPI sanitizer matrices remain applicable because
solver sources are unchanged. `out/p3_completion_field_control/` adds TGV 64^3,
NP=2 y, MAXSTEP=2 comparisons against frozen pre-P3 GPU L0. Endpoint pageable,
endpoint overlap, real NSYS/preloaded overlap and the rebuilt solver all have
zero difference in all eleven compared fields. The comparator's `--cpu` input
is a GPU reference in these additional tests. Rebuilt completion probes pass
with and without preload. See the final P3 report for binary hashes and scope;
do not relabel frozen-binary timings as new rebuilt-binary measurements.
# Physical-face Euler RHS gate

`run_boundary_rhs_gate.py --out <new-directory>` runs the top-level CMake CPU/GPU
executables with `astr test bcrh`, then a GPU Compute Sanitizer memcheck.
The manufactured affine-density state has an independently evaluated exact Euler
flux divergence. All physical points, including faces/corners, are checked for
both physical-space and Roe MP7 paths. The driver requires unique PASS markers,
finite errors below `1e-10`, successful exit codes and a clean sanitizer summary.
Binary hashes, commands, timings and logs are retained. Unit checks:

```bash
python3 -m unittest discover -s tests/gpu_validation -p test_boundary_rhs_gate.py
```

Only the NP1 sanitizer subprocess selects `pml=ob1`, `osc=pt2pt`,
`btl=self,vader,tcp` to avoid UCX's CUDA context probes during MPI initialization.
CUDA API errors are not suppressed. This probe validates no MPI transport,
nonuniform geometry halo, viscous RHS, RK staging or SBLI physical result.
The optional CPU `physical_halo_rhs` path is not yet enabled in production.

The same probe also checks CPU `diffrsdcal6(physical_boundary_rhs=.true.)` using
quadratic velocities and constant temperature. The cubic viscous-work flux has
known second-order boundary truncation errors, which are evaluated analytically;
both the continuous-PDE error and the discrete residual are reported. That
three-physical-direction check uses the CPU viscous routine. A separate CUDA
check invokes the actual xy-physical diffusion flux/RHS kernels for a z-uniform
periodic extrusion, with staged periodic flux halos. Both compare to analytic
discrete results; neither validates production MPI transport or RK integration.

## OpenSBLI restart gate

`run_opensbli_restart_equivalence.sh` checks restart behavior with CPU NP=1 and
GPU NP=2. For each backend it compares a continuous four-step result with a
two-step checkpoint followed by two restarted steps, requires field agreement
within `1e-10`, verifies monotonic unique `flowstate.dat` steps, and confirms
that inconsistent auxiliary/HDF checkpoint steps fail closed.

```bash
OUT_DIR=/tmp/astr_opensbli_restart_gate \
  tests/gpu_validation/run_opensbli_restart_equivalence.sh
```

The restart path rebuilds the prescribed pressure-inlet ghost state from
`datin/inlet.prof`; it does not infer that persistent state from the evolved
inlet plane. Evidence directories must be new and are never overwritten.

## Four-A800 TGV campaign

`run_zhongke_a800_tgv_campaign.sbatch` is the TGV-only submission driver. It
runs the `256^3` CPU/GPU wall-time baseline, the `512^3` NP=1/2/4 strong-scaling
matrix, the NP=4 pageable control, one NP=4 NSYS trace, and a segmented NP=4
production trajectory to approximately `t=20`.

The initial production time step is not hard-coded. A `512^3` preflight reads
ASTR's own CFL=1 time step and selects a target initial CFL of `0.50`. The
production driver writes a checkpoint every 2,000 steps, retries a failed
segment once, and refuses to advance past an invalid checkpoint. The final
statistics are compared with
`documents/reference_data/tgv/spectral_Re1600_512.gdiag`; plots are emitted as
EPS and JPEG without a title.

Login-node dry-run, which does not execute ASTR or allocate a GPU:

```bash
ASTR_CAMPAIGN_DRY_RUN=1 \
  tests/gpu_validation/run_zhongke_a800_tgv_campaign.sbatch
```

The platform audit in `documents/ASTR_A800_TGV_CAMPAIGN_PLAN.md` was approved,
and the formal campaign was submitted as Slurm job `451398` on 2026-09-08. It
started on 2026-09-10 but failed in the first compute-node preflight because
`libsz.so.2` was unavailable on `gpu01`; no TGV timing or production result was
created. The job now records compute-node `ldd` output and rejects unresolved
libraries before launching ASTR. It also fixes the single-node host-staged MPI
backend to `ob1` with `self,vader,tcp`, avoiding the UCX locked-memory path that
blocked the Fang precursor preflight. These checks detect environment failures;
the user-space HDF5/SZIP dependency must still be made self-contained before
resubmission.

## GPU time-series profile inflow gate

`run_dynamic_inflow_compare.sh` exercises the resident GPU
time-series inlet path for `bctype=11,turbinf=intp`. It generates deterministic
four-dimensional HDF5 slice data, advances across one slice replacement, and
compares the complete CPU/GPU RK state without trimming physical boundaries.

```bash
OUT_DIR=/tmp/astr_dynamic_inflow_np1 \
  tests/gpu_validation/run_dynamic_inflow_compare.sh
```

Dynamic-inflow checkpoints deliberately do not replay host `boucon` after the
device-to-host copy. The CPU validation oracle uses the same unprepared RK
phase. This prevents the CPU and GPU time-series caches from independently
advancing the shared `ninflowslice` counter during a GPU checkpoint. Other
boundary cases retain the existing temporary checkpoint preparation.

The `32x32x8`, three-step NP=1 central gate passes at `atol=rtol=1e-10`; the
largest primitive and reconstructed conservative differences are `5.33e-15`
and `1.42e-14`. The explicit MP7 gate also passes for NP=2 `2x1x1`, `1x2x1`,
and `1x1x2`, plus NP=4 `4x1x1` and `2x2x1`. The z-slab uses `KM=16`, keeping
the local z extent at least `hm=5`; NP=4 locally oversubscribes the two GPUs
and therefore supplies correctness rather than scaling evidence.

The D2 matrix adds a non-polynomial temporal sequence and covers a slice node,
one crossed slice, multiple crossed slices in one solver step, and a filtered
curvilinear NP=4 `2x2x1` case. Unfiltered continuous/restart comparisons are
strict. For filtering, the upstream primitive-field checkpoint is used as one
shared restart source for CPU and GPU; this checks the same reconstructed phase
without claiming that the upstream checkpoint serializes the complete q/halo
state.

`run_dynamic_inflow_d2_stage_compare.sh` writes validation-only q snapshots at
`pre_rhs` and `post_update` for all three RK stages. The NP=1 case compares six
files, while filtered CURVE NP=4 compares 24 rank files. Both pass with a
maximum absolute difference of `1.4210854715202004e-14`. These snapshots add
full-field device-to-host copies only when `ASTR_VALIDATION_RHS_PREFIX` is set;
the production path is unchanged. Physical turbulence statistics and
production performance remain D3/D4 acceptance gates.

`prepare_turbulent_inflow_from_slices.py` closes the file-contract gap between
an ASTR precursor and `turbinf=intp`. ASTR precursor `writeslice` files contain
absolute `u1/u2/u3/p/t`, while the inlet reader expects `ro/u1/u2/u3/t`
fluctuations relative to `inlet.prof`. The converter reconstructs density from
the perfect-gas EOS when `ro` is absent, forms temporal/spanwise mean primitive
profiles, writes a pressure-consistent `inlet.prof`, and writes the residual
fluctuations as a new uniformly timed series. It rejects a broken periodic
endpoint, nonuniform times, nonfinite values, and nonpositive density or
temperature before creating output.

```bash
python3 tests/gpu_validation/prepare_turbulent_inflow_from_slices.py \
  --source-dir PRECURSOR/islice \
  --output-dir PREPARED_INFLOW \
  --grid PRECURSOR/datin/grid.h5 \
  --mach MACH --reynolds RE --reference-temperature TREF \
  --target-re-theta TARGET_RE_THETA --target-re-tau TARGET_RE_TAU \
  --relative-re-tolerance RE_TOL \
  --mass-flow-relative-drift-max MASSFLOW_TOL \
  --correlation-lag LAG --correlation-abs-max CORR_TOL
```

The report contains `Re_theta`, `Re_tau`, mass-flow drift, a velocity temporal
autocorrelation, Favre means and stresses, and the exact gate decisions. The
tooling tests use manufactured periodic data; they do not constitute a
statistically converged turbulent precursor.

The GPU main loop now treats slice output and checkpoint output as independent
events. If either is due, at most one full flow synchronization is performed.
GPU slices call `writeslice(...,include_derivatives=.false.)`: they contain the
absolute primitive fields required by the converter but deliberately omit
`dudx...dwdz/dtdy`, because those derivative arrays are not synchronized to the
host output phase. CPU calls keep the legacy derivative-complete format.

A local end-to-end smoke used an `80x48x40`, NP=2 `2x1x1` GPU precursor for five
steps with `feqslice=1`. It produced five HDF5 files at steps 1--5, each with
`u1/u2/u3/p/t/time/nstep` and exact spanwise periodic endpoints. The converter
then produced `inlet.prof` and five fluctuation slices; a separate NP=2
`turbinf=intp` run loaded the first four slices and completed two GPU steps.
This is a file/runtime round-trip gate only. The five-sample report remains
`diagnostic_only` and does not satisfy D3 turbulence statistics.

## Compact GPU production statistics

The compact path accumulates z-line raw moments on the GPU at completed steps
that satisfy `nstep > 0`, `lavg`, and `mod(nstep,feqavg) == 0`. For a physical
line measure `ds`, the density-weighted mean and Favre stress are reconstructed
offline as

```text
rho_mean       = integral(sum_n rho ds) / (Nsample * integral(ds))
u_tilde        = integral(sum_n rho*u ds) / integral(sum_n rho ds)
u_i''u_j''     = integral(sum_n rho*u_i*u_j ds) / integral(sum_n rho ds)
                 - u_i_tilde*u_j_tilde
```

Wall pressure, geometry-projected streamwise traction, and normal conductive
heat flux use the same physical z-segment integration. The independent CPU
oracle is implemented by `compact_statistics_host_reference.py`; the production
CUDA kernels and checkpoint sidecar are in `production_statistics_gpu.cuf`.

Run the numerical matrix with new output directories:

```bash
tests/gpu_validation/run_compact_statistics_matrix.sh --case all
```

The accepted matrix covers static and dynamic NP=1, z-slab NP=2, filtered CURVE
dynamic NP=4 `2x2x1`, three-direction NP=8 `2x2x2`, and same-topology restart.
Every raw moment, derived mean/stress, wall field, measure, and metadata check
passed at `atol=rtol=1e-10`; the largest observed absolute difference was
`1.02e-14`. NP=8 is a communication and assembly correctness test, not a scaling
measurement.

Safety, residency, and complete-step overhead are separate gates:

```bash
tests/gpu_validation/run_compact_statistics_profile.sh \
  --mode memcheck --result-dir /tmp/compact_memcheck
tests/gpu_validation/run_compact_statistics_profile.sh \
  --mode residency --result-dir /tmp/compact_residency
tests/gpu_validation/run_compact_statistics_profile.sh \
  --mode performance --result-dir /tmp/compact_performance
```

The 2026-09-10 local acceptance on RTX 4000 Ada reported zero leaked bytes,
zero memory errors, and zero race hazards. Nsight Systems found no H2D/D2H
transfer larger than 64 KiB after the first statistics kernel. Five paired runs
gave median complete-step times of `75.388 ms` with statistics disabled and
`75.808 ms` with statistics enabled, an overhead of `0.557%`. This is local
software/performance evidence; D3 turbulence convergence and A800 production
performance remain separate requirements.

## Mixed-precision periodic upwind workspace

The experimental MP1 path keeps the authoritative state and RHS in FP64 while
storing the periodic physical-space `543e` scalar flux workspace in FP32.
It is restricted to WENO7 or MP7 with homogeneous x/y/z boundaries,
`lchardecomp=f`, `lfilter=f`, and `diffterm=f`.

Run the three-way CPU FP64, GPU FP64, and GPU mixed comparison with a new output
directory:

```bash
OUT_DIR=/tmp/astr_mp1_weno7 MAXSTEP=20 FEQCHKPT=20 RECON_SCHEM=1 \
  tests/gpu_validation/run_tgv_upwind_mixed_precision_compare.sh
```

The evidence-based provisional limits are `2e-6` absolute for every field and
`5e-11` absolute for TGV statistics, both with zero relative tolerance. MP7 is
selected with `RECON_SCHEM=3`; `NP` and `TOPOLOGY` expose MPI correctness gates.

Memory safety and paired complete-RK timing are separate:

```bash
OUT_DIR=/tmp/astr_mp1_memcheck \
  tests/gpu_validation/run_tgv_upwind_mixed_precision_memcheck.sh
OUT_DIR=/tmp/astr_mp1_benchmark GRID=128,128,128 \
  tests/gpu_validation/run_tgv_upwind_mixed_precision_benchmark.sh
```

The benchmark requires at least five repeats, alternates FP64/mixed ordering,
and imposes no speedup threshold. On RTX 4000 Ada, `128^3` WENO7 halves the
workspace from `21,484,952` to `10,742,476` bytes and reduces sampled peak
memory by `10 MiB`, but mixed complete-RK time is `0.363%` slower. The mode is
therefore viable for memory study but is not a production default.

## Periodic upwind flux-pair fusion

The optional `ASTR_GPU_FLUX_PAIR_MODE=fused` route combines the positive and
negative physical-space `543e` WENO7/MP7 flux kernels. The default remains
`split`; physical-boundary, characteristic-space, shock, filter, and diffusion
routes are unchanged.

Run same-precision field and statistics comparisons for FP64 and the FP32
workspace mode:

```bash
OUT_DIR=/tmp/astr_flux_pair_compare \
  tests/gpu_validation/run_tgv_upwind_flux_pair_compare.sh
```

Run the interleaved complete-RK benchmark with at least five repeats:

```bash
OUT_DIR=/tmp/astr_flux_pair_benchmark GRID=128,128,128 REPEATS=5 \
  tests/gpu_validation/run_tgv_upwind_flux_pair_benchmark.sh
```

On RTX 4000 Ada, the FP64 `128^3` WENO7 medians are `1.219539164 s/RK` for
split and `0.845358029 s/RK` for fused, a `30.682%` reduction (`1.44263x`).
This is local opt-in evidence; A800 NP=1/2/4 and long/restart gates remain.

The local `256^3` FP64 result uses five interleaved repeats and two retained RK
samples per repeat. Median split/fused times are `8.678503483/6.052904717 s`,
with `1.223%/1.136%` spread. This is a `30.254%` reduction or `1.43378x`
speedup. The FP64 split/fused field gate remains `1e-12`. Mixed workspace uses
its own `MIXED_FIELD_ATOL=1e-8` gate because sparse FP32 quantization can make
the two algebraically equivalent kernel organizations differ by one rounding
interval; both must still pass the separate MP1 FP64-reference gate.

## MP2 derivative and viscous-flux workspaces

MP2 keeps `ASTR_GPU_PRECISION_MODE=fp64` as the production default. Experimental
runs set `ASTR_GPU_PRECISION_MODE=mixed_workspace` and exactly one of:

```text
ASTR_GPU_MIXED_CANDIDATE=derivative
ASTR_GPU_MIXED_CANDIDATE=viscous_flux
```

The derivative candidate changes diagnostic `dvel/dtmp` storage only; solver
diffusion still reconstructs gradients from FP64 primitives. The viscous-flux
candidate stores final `sigma/qflux` in FP32 while retaining FP64 constitutive
arithmetic and RHS accumulation. Its pack/unpack kernels convert through the
existing FP64 halo buffers, so MPI payload type, tags, counts, and ordering do
not change. Every kernel launch retains an explicit synchronization.

Run the candidate-specific local gates with:

```bash
tests/gpu_validation/run_mp2_derivative_tgv_compare.sh
tests/gpu_validation/run_mp2_derivative_hbl_compare.sh
tests/gpu_validation/run_mp2_derivative_curve_compare.sh
tests/gpu_validation/run_mp2_derivative_mpi_matrix.sh
tests/gpu_validation/run_mp2_derivative_memcheck.sh
tests/gpu_validation/run_mp2_derivative_benchmark.sh

tests/gpu_validation/run_mp2_viscous_flux_tgv_compare.sh
tests/gpu_validation/run_mp2_viscous_flux_hbl_compare.sh
tests/gpu_validation/run_mp2_viscous_flux_curve_compare.sh
tests/gpu_validation/run_mp2_viscous_flux_mpi_matrix.sh
tests/gpu_validation/run_mp2_viscous_flux_memcheck.sh
tests/gpu_validation/run_mp2_viscous_flux_benchmark.sh
```

The TGV viscous-flux gate fixes absolute tolerances at `1e-9` for fields and
`1e-10` for statistics. HBL and CURVE gates use `1e-8` because they include
physical-boundary and derived wall/profile quantities. The 2026-09-12 local
evidence is under `tests/gpu_validation/out/mp2_*_20260912*`.

Both candidates pass NP=1/2/4, Cartesian HBL, CURVE-C23, and invalid-access
memcheck gates. At `64^3`, each selected workspace uses exactly half its FP64
bytes. Five interleaved measurements show no local whole-step speedup:
derivative is `2.120%` slower and viscous flux is `7.789%` slower. Their status
is `local-pass-not-promoted`; A800 and long/restart validation remain pending.

## MP3 characteristic-flux workspace

MP3 adds the mutually exclusive `characteristic_flux` candidate for the
all-periodic Shu-Osher selective-Roe path, the bounded S0-B0 x-physical
zero-extrapolation path, and the exact Cartesian viscous S2-C3 HBL capability
reported by `gpu_s2_hbl_selective_roe_diffusion_supported()`. Roe averages,
characteristic matrices, MP7 reconstruction, the Ducros sensor and mask, RHS
accumulation, and RK state remain FP64. Only the final five-component interface-flux
workspace is stored in FP32 and promoted to FP64 when differenced into the RHS.
Physical boundaries and diffusion outside these exact capabilities, filtering,
CURVE, species, chemistry, and non-RK3 paths are rejected by the runtime
eligibility gate.

Run the frozen S0-A6 comparison and S0-A7 through S0-A10 MPI matrix with new
output directories:

```bash
tests/gpu_validation/run_mp3_characteristic_flux_compare.sh
TOLERANCE_FILE=<absolute-path-to-mp3_tolerances.env> \
  tests/gpu_validation/run_mp3_characteristic_flux_mpi_matrix.sh
```

Run memory safety and the five-repeat interleaved benchmark separately:

```bash
tests/gpu_validation/run_mp3_characteristic_flux_memcheck.sh
GRID=400,16,16 MAXSTEP=5 REPEATS=5 \
  tests/gpu_validation/run_mp3_characteristic_flux_benchmark.sh
```

Run the x-physical three-way comparison, frozen NP=1/2 matrix, and two-rank
memcheck with new output directories:

```bash
tests/gpu_validation/run_mp3_characteristic_flux_xphysical_compare.sh
TOLERANCE_FILE=<absolute-path-to-mp3-xphysical-tolerances.env> \
  tests/gpu_validation/run_mp3_characteristic_flux_xphysical_matrix.sh
tests/gpu_validation/run_mp3_characteristic_flux_xphysical_memcheck.sh
```

The 2026-09-12 local S0-A6 to S0-A10 matrix passes with frozen field and
statistics absolute tolerances of `2e-6`, exact GPU-FP64/candidate sensor values,
and zero mask mismatches. Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.
At `400x16x16`, five interleaved runs give FP64/candidate median complete-RK
times of `0.023973636/0.025151473 s`, so the candidate is `4.913%` slower.
The selected workspace falls exactly from `11,984,760` to `5,992,380` bytes;
sampled peak device memory falls from `656` to `650 MiB`. Peak sampled GPU
utilization is `95%/88%`. The candidate is `local-pass-not-promoted`: it
provides a bounded workspace-memory option but no local whole-step speedup,
and it does not change the FP64 production default.

The 2026-09-12 MP3-XP1 and MP3-XP2 gates use the complete `400x8x8`, three-step
S0-B0 field, including both x physical planes. NP=1 and NP=2 `2x1x1` pass the
frozen `2e-6` field/statistics gates. Both decompositions give maximum field
and statistics differences of `1.5973888878306752e-7` and
`1.0126266403176487e-7`; GPU FP64 and MP3 raw sensors are bitwise identical and
mask mismatches are zero. The NP=2 Compute Sanitizer log contains two
`ERROR SUMMARY: 0 errors` records. The classification is
`x-physical-local-pass-not-promoted`; `11/21`, NSCBC, walls, diffusion,
filtering, CURVE, long-time shock motion, and physical SBLI remain outside this
gate.

Run the MP3-HBL1 three-way calibration, frozen NP=1/three-slab matrix, and
candidate-only x/y-slab sanitizer gates with new output directories:

```bash
tests/gpu_validation/run_mp3_characteristic_flux_hbl_compare.sh
TOLERANCE_FILE=<absolute-path-to-mp3-hbl-tolerances.env> \
  tests/gpu_validation/run_mp3_characteristic_flux_hbl_matrix.sh
tests/gpu_validation/run_mp3_characteristic_flux_hbl_memcheck.sh
```

The 2026-09-12 MP3-HBL1 gate uses a `192x192x8`, two-step Cartesian Mach-5
S2-C3 slice with `543e/643e`, MP7 characteristic reconstruction, diffusion
enabled, filtering disabled, and `bctype=11/21/41/51/1/1`. Calibration froze
`CANDIDATE_FIELD_ATOL=5.0e-07`, statistics and sensor absolute tolerances at
`1.0e-12`, and all relative tolerances at zero. NP=1 and NP=2 x/y/z slabs pass.
The maximum GPU-FP64/candidate field and statistic differences are
`2.0437756598212786e-08` and `1.6875389974302379e-14`; raw-sensor differences
are zero and every shock-mask mismatch count is zero. Candidate x/y-slab
Compute Sanitizer runs each report two `ERROR SUMMARY: 0 errors` records. The
five-component workspace is exactly 50% of its FP64 allocation in every tested
topology. This is classified as `hbl-cartesian-local-pass-not-promoted`.
Long-time HBL physics, physical SBLI, NSCBC and `bctype=52`, sponge, filtering,
CURVE, chemistry, A800 timing, and production speedup remain outside the gate.

## Fixed air5 chemistry C0-C3 source and integrator gate

Run the fixed-mechanism, thermodynamic, source/Jacobian, build-contract,
independent Radau, and ROS-2 tests from the repository root:

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_air5_mechanism_generator.py \
  tests/gpu_validation/test_chemistry_thermo.py \
  tests/gpu_validation/test_chemistry_source.py \
  tests/gpu_validation/test_air5_cmake_contract.py \
  tests/gpu_validation/test_air5_radau_reference.py \
  tests/gpu_validation/test_chemistry_ros2.py
```

The CPU-only C0-C1 result is `46 passed, 18 subtests passed`. The gate covers the
fixed five-species/two-temperature FP64 model, analytic Jacobian, scaled
six-by-six LU, KPP ROS-2 2(1), coupled and component-only source modes,
representative in-domain trajectories, strict out-of-domain rejection, and
comparison with an independent JSON-driven SciPy Radau implementation. GNU 15,
NVHPC 26.1, and the NVHPC CUDA-enabled top-level builds pass.

Build and run the C2 CUDA source probe through the top-level CMake project:

```bash
cmake -S . -B /tmp/astr_c2_build \
  -DASTR_WITH_CUDA=ON -DASTR_WITH_AIR5_CHEMISTRY=ON
cmake --build /tmp/astr_c2_build --target chemistry_source_gpu_probe -j2
ASTR_CHEMISTRY_GPU_PROBE_EXE=/tmp/astr_c2_build/bin/chemistry_source_gpu_probe \
  python3 -m pytest -q tests/gpu_validation/test_chemistry_source_gpu.py
compute-sanitizer --tool memcheck --leak-check full \
  /tmp/astr_c2_build/bin/chemistry_source_gpu_probe
```

The combined 2026-09-14 result is `52 passed, 18 subtests passed`. The GPU
probe runs both `count=1` and fourteen-state batches for coupled,
chemical-only, and VT-only modes. It covers cold, near-equilibrium, strongly
reacting, valid pressure-endpoint, out-of-domain temperature/pressure,
negative-species, and NaN states. Maximum normalized gate ratios are
`1.8764e-2` for the six source components, `1.4003e-2` for the twelve reaction
progress rates, and `4.8773e-4` for the analytic Jacobian. Total-energy source
and `q5` change are exactly zero, all CPU/GPU status codes match, and Compute
Sanitizer reports zero errors and zero leaked bytes.

C2 is an integrator-independent CUDA source evaluator. It does not claim GPU
`numq=11` CFD coupling, species transport or halo exchange, reacting boundary
conditions, or reacting-flow physical validation.

Build and run the C3 CUDA ROS-2 probe through the same top-level project:

```bash
cmake -S . -B /tmp/astr_c3_build \
  -DASTR_WITH_CUDA=ON -DASTR_WITH_AIR5_CHEMISTRY=ON
cmake --build /tmp/astr_c3_build --target \
  chemistry_source_gpu_probe chemistry_ros2_gpu_probe -j2
ASTR_CHEMISTRY_GPU_PROBE_EXE=/tmp/astr_c3_build/bin/chemistry_source_gpu_probe \
ASTR_CHEMISTRY_ROS2_GPU_PROBE_EXE=/tmp/astr_c3_build/bin/chemistry_ros2_gpu_probe \
  python3 -m pytest -q tests/gpu_validation/test_air5_cmake_contract.py \
  tests/gpu_validation/test_air5_mechanism_generator.py \
  tests/gpu_validation/test_air5_radau_reference.py \
  tests/gpu_validation/test_chemistry_thermo.py \
  tests/gpu_validation/test_chemistry_source.py \
  tests/gpu_validation/test_chemistry_ros2.py \
  tests/gpu_validation/test_chemistry_source_gpu.py \
  tests/gpu_validation/test_chemistry_ros2_gpu.py
compute-sanitizer --tool memcheck --leak-check full \
  /tmp/astr_c3_build/bin/chemistry_ros2_gpu_probe
```

The combined C3 result is `58 passed, 18 subtests passed`. The four-state,
three-mode trajectory matrix and `count=1` path match the CPU ROS-2 oracle;
adaptive accept/reject and RHS/Jacobian counts are identical. The maximum final
state gate ratio is `2.7558e-4`, while mass, nitrogen, and oxygen drifts remain
below `1.6e-15`. Invalid controls, invalid states, and attempt exhaustion return
the CPU status and initial state. Direct GPU LU tests cover pivoting, scaled,
and singular matrices. Compute Sanitizer reports zero errors and zero leaked
bytes.

For local profiling only, run:

```bash
/tmp/astr_c3_build/bin/chemistry_ros2_gpu_probe profile
ncu --set basic --kernel-name regex:air5_ros2_batch_kernel --launch-count 1 \
  /tmp/astr_c3_build/bin/chemistry_ros2_gpu_probe profile
```

On the local RTX 4000 Ada, the 8192-state mixed batch uses 156 registers per
thread, reaches 12.4% achieved occupancy, and shows substantial local-memory
traffic. Accepted and rejected substep ranges are `1..15` and `0..2`. This is
a bottleneck diagnosis on a 0.44-wave grid, not a production throughput result.
C3 remains isolated from the CFD state. C4 owns resident `numq=11`, species/`Ev`
transport, and halo exchange; C5 owns Strang coupling.

Phase C4 connects the fixed-air5 state to the nonreacting CFD transport loop.
Build through the top-level project and run the complete unit/probe matrix:

```bash
cmake -S . -B /tmp/astr_c4_build \
  -DASTR_WITH_CUDA=ON -DASTR_WITH_AIR5_CHEMISTRY=ON
cmake --build /tmp/astr_c4_build --target \
  astr chemistry_source_gpu_probe chemistry_ros2_gpu_probe \
  chemistry_flow_gpu_probe halo_exchange_contract_test -j2
ASTR_CHEMISTRY_GPU_PROBE_EXE=/tmp/astr_c4_build/bin/chemistry_source_gpu_probe \
ASTR_CHEMISTRY_ROS2_GPU_PROBE_EXE=/tmp/astr_c4_build/bin/chemistry_ros2_gpu_probe \
ASTR_CHEMISTRY_FLOW_GPU_PROBE_EXE=/tmp/astr_c4_build/bin/chemistry_flow_gpu_probe \
  python3 -m pytest -q \
  tests/gpu_validation/test_air5_c4_conservation.py \
  tests/gpu_validation/test_air5_c4_halo_contract.py \
  tests/gpu_validation/test_air5_c4_species_variance.py \
  tests/gpu_validation/test_air5_cmake_contract.py \
  tests/gpu_validation/test_air5_mechanism_generator.py \
  tests/gpu_validation/test_air5_radau_reference.py \
  tests/gpu_validation/test_chemistry_flow.py \
  tests/gpu_validation/test_chemistry_flow_gpu.py \
  tests/gpu_validation/test_chemistry_thermo.py \
  tests/gpu_validation/test_chemistry_source.py \
  tests/gpu_validation/test_chemistry_ros2.py \
  tests/gpu_validation/test_chemistry_source_gpu.py \
  tests/gpu_validation/test_chemistry_ros2_gpu.py \
  tests/gpu_validation/test_chemistry_state_layout.py
```

The C4 result is `80 passed, 18 subtests passed`. The fixed state is
`q(1)=rho`, `q(2:4)=rho*u`, `q(5)=rho*E`, `q(6:10)=rho*Ys`, and
`q(11)=Ev`. In an air5-enabled build, runtime `lcomb=t` requires exactly five
species, `turbmode=none`, and `lfilter=f`, then resolves
`num_modequ=1,numq=11`. Invalid combinations fail before changing the layout;
`lcomb=f` preserves the established nonreacting layout.

Run the independent diffusion-direction gate with a nonuniform composition:

```bash
INITIAL_CONDITION=species-wave GRID=16,16,16 MAXSTEP=0 DELTAT=1.d-6 \
  BUILD_DIR=/tmp/astr_c4_build \
  OUT_DIR=/tmp/air5_c4_species_wave_np1 \
  tests/gpu_validation/run_air5_c4_transport_compare.sh
```

The `air5wave` case has zero velocity, uniform pressure and temperature, and a
three-axis N2/O2 sinusoidal perturbation. After one RK step, the density-weighted
N2 variance decreases by `3.3721e-6` while its mean changes by at most
`1.46e-16`. This gate caught a shared CPU/GPU sign error that field equivalence
and global conservation could not detect: because
`Js=-rho*Ds*grad(Ys)`, species RHS assembly must subtract `div(Js)`. The same
gate passes for NP=2 x/y/z slabs and NP=8 `2x2x2`.

Uniform-composition TGV passes NP=1, all NP=2 slabs, and NP=8 `2x2x2`; the
largest CPU/GPU field errors are `4.44e-15` and `3.55e-15`. The NP=1 ten-step
maximum relative mass, energy, and elemental drift is `7.37e-14`. The
production halo executable passes exactly for NP=2 and NP=3 with `pageable`,
`pinned`, and `pinned-overlap` backends.

For memcheck, isolate the CUDA Fortran executable from OpenMPI CUDA buffer
instrumentation:

```bash
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 \
UCX_MEMTYPE_CACHE=n mpirun -np 1 \
  compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
  /tmp/astr_c4_build/bin/chemistry_flow_gpu_probe
```

The isolated run reports `0 errors` and `0 bytes leaked`. Without this MPI
isolation, Compute Sanitizer can report invalid accesses from OpenMPI's CUDA
buffer-detection path rather than from an ASTR kernel.

C4 cases do not call the ROS-2 chemistry kernel from the CFD time loop. The
first C5 gate is a separate `air5reactor` flowtype, so the C4 TGV and
`air5wave` cases remain frozen-chemistry references.

## Fixed air5 chemistry C5 embedded-reactor gate

Build the top-level CUDA and air5 executable, then run the NP=1 gate:

```bash
BUILD_DIR=/tmp/astr_c5_build \
OUT_DIR=/tmp/air5_c5_embedded_reactor_np1 \
  tests/gpu_validation/run_air5_c5_embedded_reactor_compare.sh
```

The driver uses a periodic `6x6x6` grid, `lfilter=f`, sixth-order explicit
convection/diffusion, `dt=2e-10 s`, and one complete Strang step. It compares
active CPU/GPU states before chemistry, after each chemistry half step, and
after transport. The dedicated checker requires a uniform field, unchanged
`q(1:5)` across chemistry, a numerically zero spatial update, a nonzero
chemistry update, species-mass closure, and N/O elemental closure.

Run the three NP=2 slab gates with local dimensions no smaller than six:

```bash
GRID=12,6,6 MPI_NP=2 TOPOLOGY=2,1,1 OUT_DIR=/tmp/air5_c5_np2_x \
  tests/gpu_validation/run_air5_c5_embedded_reactor_compare.sh
GRID=6,12,6 MPI_NP=2 TOPOLOGY=1,2,1 OUT_DIR=/tmp/air5_c5_np2_y \
  tests/gpu_validation/run_air5_c5_embedded_reactor_compare.sh
GRID=6,6,12 MPI_NP=2 TOPOLOGY=1,1,2 OUT_DIR=/tmp/air5_c5_np2_z \
  tests/gpu_validation/run_air5_c5_embedded_reactor_compare.sh
```

NP=1 and all three NP=2 slabs pass. The maximum CPU/GPU phase error is
`1.36e-12`; transport drift is `2.27e-13`; species-mass closure is
`2.08e-17`; and N/O relative drift is `7.77e-16`. Each half step reports
maximum accept/reject/RHS/Jacobian counts of `56/3/118/59`. The C5 coupling
kernel is compiled separately from the C4 directional transport file because
ROS-2 requires more registers than the 128-register cap used to keep the
prescribed 512-thread transport kernels launchable.

With OpenMPI CUDA detection disabled, the NP=1 full executable reports
`ERROR SUMMARY: 0 errors` and `LEAK SUMMARY: 0 bytes leaked`. This gate does
not validate nonuniform reacting transport, reacting boundaries or restart,
post-shock relaxation, high-temperature TGV, or reacting SBLI.

## Fixed air5 chemistry C5 frozen-transport gate

The item-2 driver runs three periodic quasi-one-dimensional cases in one
invocation: a translating N2/O2 composition wave, an N2/O2 diffusion layer on
a fixed N/O/NO background, and a vibrational-energy pulse. Chemistry remains
frozen. The default `dt=5e-7 s` is subject to a hard `current CFL < 1` gate.

Run NP=1 and the three NP=2 slab layouts as follows:

```bash
BUILD_DIR=/tmp/astr_c5_cuda_build \
OUT_DIR=/tmp/air5_c5_frozen_np1 \
  tests/gpu_validation/run_air5_c5_frozen_transport_compare.sh

GRID=48,6,6 MPI_NP=2 TOPOLOGY=2,1,1 \
BUILD_DIR=/tmp/astr_c5_cuda_build OUT_DIR=/tmp/air5_c5_frozen_np2_x \
  tests/gpu_validation/run_air5_c5_frozen_transport_compare.sh

GRID=48,12,6 MPI_NP=2 TOPOLOGY=1,2,1 \
BUILD_DIR=/tmp/astr_c5_cuda_build OUT_DIR=/tmp/air5_c5_frozen_np2_y \
  tests/gpu_validation/run_air5_c5_frozen_transport_compare.sh

GRID=48,6,12 MPI_NP=2 TOPOLOGY=1,1,2 \
BUILD_DIR=/tmp/astr_c5_cuda_build OUT_DIR=/tmp/air5_c5_frozen_np2_z \
  tests/gpu_validation/run_air5_c5_frozen_transport_compare.sh
```

The checker assembles the global periodic x line across x ranks and
independently reconstructs temperature, vibrational temperature, transport
properties, correction flux, complete `q5/Ev` diffusion flux, and the
sixth-order semi-discrete RHS from the versioned mechanism JSON. Only the
advection case has a continuum analytic Fourier-translation comparison. The
diffusion-layer and `Ev`-pulse gates use the independent semi-discrete oracle,
global conservation, invariants, extrusion consistency, and variance decay.

All four layouts pass at observed CFL `0.6066073-0.6672681`. Across 12
CPU/GPU reports the maximum absolute field difference is `1.1642e-10`.
The maximum conservation drift is `1.8176e-13`; the Fourier translation
scaled error is `9.0452e-11`. The diffusion layer has correction-velocity
peak `3.5471e-3`, activates all five species RHS, and reduces N2 variance by
about `2.1298e-5`. The `Ev` pulse reduces variance by `3.1685e-5` while the
`q5-Ev` invariant scaled error remains below `2.0474e-14`.

Prepare a diffusion-layer case and run the full executable under memcheck with
OpenMPI CUDA buffer detection isolated:

```bash
python3 tests/gpu_validation/prepare_air5_c4_case.py \
  --destination /tmp/air5_c5_frozen_memcheck --grid 48,6,6 \
  --maxstep 0 --deltat 5.d-7 --diffterm t --use-gpu t \
  --initial-condition diffusion-layer
cd /tmp/air5_c5_frozen_memcheck
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 \
UCX_MEMTYPE_CACHE=n ASTR_FORCE_MPI_TOPOLOGY=1,1,1 mpirun -np 1 \
  compute-sanitizer --tool memcheck --error-exitcode 99 \
  /tmp/astr_c5_cuda_build/bin/astr run datin/input.air5_c4
```

The isolated run reports `ERROR SUMMARY: 0 errors`. This invocation does not
enable full leak checking, so it provides no leak-byte conclusion. C5 item 2
does not validate chemistry acting on nonuniform transport, reacting boundary
conditions, restart, post-shock relaxation, high-temperature TGV, or SBLI.

The complete C0-through-C5-item-2 chemistry unit, probe, and static-contract
matrix reports `92 passed, 18 subtests passed`. The existing `air5wave`
frozen-chemistry regression also remains passing after the C5 integration,
with maximum CPU/GPU absolute field difference `2.91e-11`.

## Fixed air5 chemistry C5 post-shock reference

Generate the independent steady one-dimensional reference profile with:

```bash
python3 tests/gpu_validation/generate_air5_postshock_profile.py \
  --mechanism chemMech/air5_kimjo12.json \
  --output /tmp/air5_postshock_profile.dat \
  --points 513 --length 0.02 --source-mode coupled --rtol 1e-10
```

The reference first applies a frozen normal-shock jump to a `T1=500 K`,
`p1=5 kPa`, `M1=8`, `Tv1=500 K` air state. It then integrates the steady
finite-rate equations while holding mass flux, momentum flux, and total
enthalpy constant. Density, speed, pressure, and translational temperature
are reconstructed from these fluxes at every Radau trial state. This is not a
constant-density zero-dimensional reactor mapped from time to distance.

The input composition contains positive N/O/NO seeds totaling 4 ppm so that
the independent finite-difference Radau Jacobian can evaluate reverse
reactions without clipping a negative trial state. N2/O2 are adjusted so the
mass fractions sum to one. The generated columns are:

```text
x rho u p T Tv Y_N2 Y_O2 Y_N Y_O Y_NO Ev q5
```

Run the reference contracts with:

```bash
PYTHONPATH=tests/gpu_validation pytest -q \
  tests/gpu_validation/test_air5_postshock_reference.py
```

The reference and executable contract result is now `11 passed`. The ASTR
path adds a dedicated `air5postshock` profile initializer, complete
`q(1:11)` x boundaries, and CPU/GPU sixth-order physical-x convection and
diffusion closure. Run all three source modes with diffusion using:

```bash
DIFFTERM=t OUT_DIR=tests/gpu_validation/out/air5_c5_postshock_diffusion_np1 \
  tests/gpu_validation/run_air5_c5_postshock_compare.sh
```

Repeat with `MPI_NP=2` and `TOPOLOGY=2,1,1`, `1,2,1`, and `1,1,2` for the
three slab gates. All layouts pass. The coupled maximum CPU/GPU same-phase
difference is `1.804437488e-9`; maximum independent internal errors are
`6.3979e-5 m/s` for velocity, `7.845685e-4 K` for temperature,
`4.88424e-4 K` for vibrational temperature, and `7.29805e-8` for mass
fraction. The maximum extrusion error is `8.731e-10`. Isolated NP=1 Compute
Sanitizer reports zero errors. These results close C5 items 3 and 4 for the
dedicated Cartesian postshock contract.

## Fixed air5 chemistry C5 reacting TGV

The `air5tgv` initializer uses `rho=0.05 kg/m3`, `T=6000 K`, `Tv=1000 K`,
a `100 m/s` TGV velocity amplitude, and fixed positive five-species mass
fractions. It applies the standard TGV pressure perturbation and reconstructs
the local temperature from `p/(rho*Rmix)` so the primitive and conservative
initial states remain consistent.

Run the NP=1 gate with:

```bash
OUT_DIR=tests/gpu_validation/out/air5_c5_reacting_tgv_np1 \
  tests/gpu_validation/run_air5_c5_reacting_tgv_compare.sh
```

The same driver passes `MPI_NP=2` for all three slabs and `MPI_NP=8
TOPOLOGY=2,2,2`. Across those layouts, the maximum CPU/GPU same-phase
difference is `4.3201e-12`; the maximum N/O drift is `1.086e-15`; species
mass closure is within `6.25e-17`; and the observed CFL is `1.86e-4`.
Compute Sanitizer reports zero errors. A three-step Nsight Systems trace has
no H2D or D2H transfer of at least 64 KiB after the first chemistry kernel;
maximum loop H2D/D2H transfers are `320 B/72 B`. This closes C5 item 5 as a
3-D coupling, conservation, memory-safety, and residency gate. It is not a
chemistry-model physical validation.

The current C0-through-C5-item-5 chemistry unit, probe, reference, and static
contract matrix reports `124 passed, 18 subtests passed`. MPI field runs,
Compute Sanitizer, and Nsight Systems remain separate gates.

C5 item 6 A0 fixes a Cartesian noncatalytic isothermal wall with `Tv=Tw`, zero
wall-normal species diffusion, complete-state inflow/farfield, extrapolated
supersonic outflow, and periodic z. The `31x31x7` NP=1, NP=2 x/y/z slab, NP=8
`2x2x2`, and Compute Sanitizer gates pass for the first complete step.

`air5_hbl_diagnostics.py` provides the first C5-6A1 diagnostic layer. It reads
an NP=1 complete `q(1:11)` snapshot, reconstructs `u/T/Tv/Ys`, and writes wall
`Cf`, translational/vibrational/species heat-flux components and selected
profiles to NPZ plus a text report. Heat flux is positive from the lower wall
into the fluid. For a selected nonzero step, set both variables before running:

```bash
ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
ASTR_VALIDATION_RHS_STEP=1000 \
  mpirun -np 1 build/bin/astr run datin/input.air5_c4
```

The default selected step remains zero. The request predicate includes the
step gate, so the GPU does not copy the field to the host on unselected steps.

The first two-step A1 attempt exposed a fail-closed numerical blocker shared by
CPU and GPU. After RK transport, the outflow-adjacent NO partial density reached
about `-2.89249e-21 kg/m3`, while the preceding chemistry state was positive.
The matched `diffterm=f` run isolated the failure to explicit centered species
diffusion.

The approved repair rewrites the existing sixth-order derivative and physical
boundary closures exactly as shared face-flux differences. At each SSPRK3 stage,
one scalar ratio per cell bounds all negative species-diffusion contributions
against the positive no-diffusion baseline. A face uses the minimum ratio of its
two adjacent cells, and that ratio is applied to momentum, total-energy,
species, and vibrational-energy diffusion together. The GPU exchanges only this
one FP64 scalar halo. No clipping, renormalization, artificial trace floor, or
low-order fallback is used.

After the diffusion repair, the same case exposed a second independent blocker
at step 185: centered convection made the no-diffusion baseline of trace NO
negative. The approved species-only conservative FCT keeps the original
sixth-order density, momentum, total-energy and vibrational-energy RHS. It uses
the high-order density face flux to form a closed upwind species baseline, then
limits the five-species anti-diffusive correction with one shared cell ratio and
the adjacent minimum at each face. It does not clip, renormalize, or impose a
trace floor. The diffusion and convection limiters execute separately and reuse
the same haloed FP64 scalar workspace.

The triggering `31x31x7`, `dt=1e-10 s` case now passes through step 185 for
NP=1, and the one-step NP=2 `1x2x1` halo gate passes. CPU and GPU report identical
`240/240/240` limited-point counts and minimum ratios for both limiter stages.
At step 185, the true RK update range has maximum CPU/GPU absolute difference
`3.5763e-7` under `atol=1e-9, rtol=1e-10`; minimum species density is
`8.4485e-20 kg/m3`, and relative species closure is about `1e-14`. The focused
memcheck reports zero errors and zero leaked bytes. The diffusion reconstruction
and positivity contracts are covered by `test_air5_diffusion_limiter.py`; the
species FCT identity, positivity and closure contracts are covered by
`test_air5_species_convection_limiter.py`. The focused air5 suite reports
`112 passed, 9 subtests passed`. These gates remove the observed numerical
positivity blockers, but do not close long-time HBL convergence or independent
physical validation.

The HBL driver now keeps two acceptance modes explicit. Its default short-step
mode remains the elementwise `atol=1e-9, rtol=1e-10` comparison and the
`2e-10` extrusion-scaled tolerance. A `31x31x7`, `dt=1e-9 s`, NP=1 20-step
run passes that strict mode. CPU/GPU extrusion errors are `6.7096e-12` and
`7.1109e-12`; the largest physical-scaled difference across five same-phase
snapshots is `4.3196e-12`.

The sustained-run mode must be requested explicitly:

```bash
MAXSTEP=150 VALIDATION_STEP=150 DELTAT=1.d-9 TIMEOUT_SECONDS=1200 \
EXTRUSION_SCALED_TOL=1e-7 SAME_PHASE_SCALED_TOL=1e-7 \
OUT_DIR=tests/gpu_validation/out/air5_c5_hbl_long_time_step150 \
  tests/gpu_validation/run_air5_c5_hbl_compare.sh
```

Here the same-phase scaled error is
`max(abs(q_gpu-q_cpu)/max(abs(q_cpu),1))`. Existing step-150 snapshots pass
with a maximum of `8.8715e-8`; CPU/GPU extrusion errors are `4.4148e-8` and
`5.7158e-8`. Positivity, species closure, element conservation, chemistry
activity, and inlet/farfield/wall/outflow contracts remain separate gates and
also pass. The same snapshots still fail the default elementwise comparison,
so the sustained mode does not weaken the strict mode. The step-150 margin is
only about 11 percent. This is bounded sustained-integration evidence, not
mesh/time-step convergence, restart qualification, independent two-temperature
flat-plate validation, or a production-duration claim.

Long-time extrusion checks use a separate, explicit contract:

```bash
EXTRUSION_GATE=long-mean \
SPANWISE_MOMENTUM_RELATIVE_TOL=1e-10 \
EXTRUSION_SCALED_TOL=1e-7 \
  tests/gpu_validation/run_air5_c5_hbl_compare.sh
```

This keeps pointwise extrusion checks for every primary conservative component
except spanwise momentum. It gates spanwise momentum with
`max_xy(abs(mean_z(rho*w)))/(rho_inf*u_inf)` and still reports the raw local
maximum and RMS. Existing step-9000 and step-12000 snapshots pass with primary
extrusion errors `9.62e-9/7.35e-9` and mean spanwise-momentum ratios
`3.95e-18/1.98e-18`.

The step-12000 solution does not pass the independent profile gate. Its
`u/T/Tv/Cf/total-wall-heat` errors are
`1.15%/4.41%/5.35%/4.64%/11.24%`. Halving the time step and refining the
wall-normal grid to `31x255x7` produce much smaller same-time changes, while a
matched filter-off continuation increases the `Cf` and heat-flux errors to
`21.75%` and `83.40%`. Do not promote this case as physically closed.

Checkpoint-only limiter diagnostics now report active admissibility constraints
and species. CPU and GPU one-step continuations from the same step-12000
checkpoint agree exactly: all `3384/2816/3192` limited points across the three
RK stages are species constrained. NO accounts for most active points, atomic N
sets the minimum ratio near `0.0411`, and density, vibrational, and translational
constraints do not activate. This supports testing a hierarchical limiter next;
it does not authorize changing the production numerical method without review.

The fixed-gamma five-equation Steger-Warming, Roe, and MP7 path remains
inadmissible for the five-species two-temperature state. A separate
multispecies nonequilibrium shock-flux and reconstruction gate is required
before C5-6B SBLI.

The later C5-6B matrix retains NASA NPARC/WIND Hypersonic Ramp Study 1 Run E as
a public Mach-7 five-species finite-rate code-to-code benchmark. The pinned
input and small surface/profile files live in
`documents/reference_data/nasa_hypramp_mach7/`. Verify their hashes and data
contract with:

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_nasa_hypramp_reference.py
```

NASA reports no analytical or experimental comparison for this study, so its
curves must not be promoted to experimental truth or used to replace the
independent C5-6A1 FP64 two-temperature flat-plate reference.

The independent reference starts with the A1-R0 equation contract in
`air5_hbl_reference.py`. It uses eight parabolized equations in the fixed order
mass, streamwise momentum, independent O2/N/O/NO species, total energy, and
vibrational energy. Dominant N2 closes total mass, avoiding cancellation when
recovering trace NO. The implementation
independently evaluates the streamwise conservative flux, wall-normal
convective/diffusive flux and coupled chemistry/V-T source in FP64. Run its
contract tests with:

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_air5_hbl_reference.py
```

Passing A1-R0 fixes signs and state ownership only. It does not qualify the
planned streamwise marcher or the A1 two-temperature flat-plate physics gate.

The authoritative source-off, single-temperature A1-R1 reference is the
independent frozen-air5 Dorodnitsyn similarity solution in
`air5_hbl_similarity.py`. It reads the same JSON mechanism but does not call
ASTR Fortran or CUDA. The global x-y mass-streamfunction scaffold is in
`air5_hbl_collocation.py`; the finite-rate A1-R2 reference uses the positive
implicit station path in `air5_hbl_marcher.py`. Run the authoritative reference,
global-discrete consistency, and station checks with:

```bash
python3 -m pytest -q \
  tests/gpu_validation/test_air5_hbl_similarity.py \
  tests/gpu_validation/test_air5_hbl_collocation.py \
  tests/gpu_validation/test_air5_hbl_marcher.py
```

The BVP passes its boundary, FP64 solver-residual, wall-flux scaling, and
three-grid discrete mass/momentum/energy convergence checks, closing only the
A1-R1 frozen single-temperature reference. The global scaffold preserves
uniform flow below `1e-12`. It uses a three-point streamwise and five-point
wall-normal explicit derivative, strictly eliminates prescribed inlet/wall/edge
Dirichlet values, and matches the pointwise FP64 frozen-flux oracle within
`2e-15` relative tolerance. The `3x13`, `3x17`, and `3x25` nonlinear systems
all close below `2.20e-14`. Their streamwise-velocity L2 errors against the BVP
are `5.149%`, `4.999%`, and `0.880%`; temperature errors are `7.086%`, `3.729%`,
and `0.996%`. This closes the source-off single-temperature prerequisite, not
R2 chemistry/V-T. Long-time ASTR HBL, MPI, restart, sanitizer, and residency
gates remain open. The legacy explicit and generic least-squares station
candidates remain diagnostic only. Do not relax a residual or add state clipping
to convert either route into a pass. At the source-off checkpoint, the focused air5 suite reported
`137 passed, 9 subtests passed`.

The first frozen-composition V-T-only extension adds independent `T` and `Tv`
unknowns, total- and vibrational-energy fluxes, and the Millikan-White/Park
relaxation source. The vectorized two-temperature flux and direct V-T source
match the general pointwise FP64 oracle within `4e-15` and `2e-15` relative
tolerance. A source-free Ev predictor is followed by source-strength homotopy
at `0/0.1/0.3/0.6/1.0`; the final stage always uses the complete source. The
`3x13` nonuniform flat plate closes to `2.6058e-14` in 396 total function
evaluations, remains positive, preserves all prescribed boundaries, and has a
maximum `|T-Tv|` of `178.52 K`. A separate `3x17` diagnostic closes to
`5.2607e-14` after 1017 evaluations, but its maximum temperature separation is
`262.11 K`. These are a single-grid regression and a second-grid algebraic
closure, not by itself a grid-converged finite-rate physical reference.

The approved finite-rate station path retains all eight conservative residuals,
eliminates fixed wall and edge degrees of freedom, and represents the four
independent species with nonnegative ratios to N2. A 40-color block-banded
Jacobian, sparse Newton step, and feasible line search preserve temperature and
composition bounds without clipping, renormalization, or a trace-species floor.
The noncatalytic wall is eliminated against the actual one-sided derivative so
the discrete wall-normal species gradient vanishes.

The opt-in streamwise and wall-normal convergence gates are:

```bash
ASTR_RUN_EXPENSIVE_REFERENCE=1 python3 -m pytest -q \
  tests/gpu_validation/test_air5_hbl_marcher.py

ASTR_RUN_LONG_REFERENCE=1 python3 -m pytest -q \
  tests/gpu_validation/test_air5_hbl_marcher.py::test_positive_implicit_coupled_htr_profile_survives_accumulated_march
```

The 1/2/4-step comparison reduces the fine-level profile difference below
`0.25` of the preceding difference. On 17/33/65 wall-clustered grids, the final
two levels differ by `1.08e-4` in `Cf` and `9.48e-4` in total wall heat flux.
The 33-point accumulated march over ten `1e-5 m` stations keeps the scaled
residual below `7.42e-11`, closes mass fractions within `2.22e-16`, and reaches
maximum `|T-Tv|=170.5 K`. These checks close the independent A1-R2 numerical
baseline. ASTR CPU/GPU profile and wall-quantity comparison, restart, long-time,
MPI, sanitizer, and residency gates remain open for A1-R3/C5-6A1.

## P4-0 trusted local performance baseline

P4-0 provides an opt-in, fail-closed benchmark lifecycle for three-dimensional,
fully periodic GPU TGV. It skips only generated-grid and startup-flowfield HDF5
writes. The default runtime, checkpoint, restart, statistics, and later output
paths are unchanged. Local timing numbers are screening evidence only and must
not be reported as A800 performance.

Configure and build both binaries from the top-level project:

```bash
ROOT=/home/dell/workspace/astr_gpu
cmake -S "$ROOT" -B "$ROOT/build_gpu_p4" \
  -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build_gpu_p4" --target astr benchmark_runtime_probe -j2

cmake -S "$ROOT" -B "$ROOT/build_cpu_p4" \
  -DCMAKE_Fortran_COMPILER=nvfortran -DASTR_WITH_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build_cpu_p4" --target astr benchmark_runtime_probe -j2

bash "$ROOT/tests/gpu_validation/run_benchmark_runtime_contract.sh" \
  "$ROOT/build_gpu_p4/bin/benchmark_runtime_probe" \
  "$ROOT/build_cpu_p4/bin/benchmark_runtime_probe"
```

Run the default-output CPU/GPU field gates before performance measurements:

```bash
ROOT=/home/dell/workspace/astr_gpu
for steps in 1 10 100; do
  OUT_DIR="$ROOT/tests/gpu_validation/out/p4_0_np1_${steps}step" \
  MAXSTEP="$steps" FEQCHKPT="$steps" ATOL=1e-10 RTOL=1e-10 \
  CPU_EXE="$ROOT/build_cpu_p4/bin/astr" \
  GPU_EXE="$ROOT/build_gpu_p4/bin/astr" \
    bash "$ROOT/tests/gpu_validation/run_tgv_field_compare.sh"
done

for topology in 2,1,1 1,2,1 1,1,2; do
  label="${topology//,/_}"
  OUT_DIR="$ROOT/tests/gpu_validation/out/p4_0_np2_${label}" \
  MAXSTEP=10 FEQCHKPT=10 MPI_NP=2 TOPOLOGY="$topology" \
  ATOL=1e-10 RTOL=1e-10 FILTER_WORKSPACE=full \
  CPU_EXE="$ROOT/build_cpu_p4/bin/astr" \
  GPU_EXE="$ROOT/build_gpu_p4/bin/astr" \
    bash "$ROOT/tests/gpu_validation/run_tgv_mpirank2_field_compare.sh"
done
```

Run one process warm-up plus five independent retained processes for NP=1 and
all two-rank slabs. Phase timing stays off in the complete-RK baseline:

```bash
ROOT=/home/dell/workspace/astr_gpu
BASE="$ROOT/tests/gpu_validation/out/p4_0_256_baseline"
GPU_EXE="$ROOT/build_gpu_p4/bin/astr"

OUT_DIR="$BASE" LABEL=np1 GRID=256,256,256 MAXSTEP=20 \
DISCARD_STEPS=1 REPEATS=5 GPU_EXE="$GPU_EXE" GPU_IDS=0 \
NP=1 TOPOLOGY=1,1,1 SYNC_MODE=explicit PHASE_TIMING=0 \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
  bash "$ROOT/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"

for topology in 2,1,1 1,2,1 1,1,2; do
  label="np2_${topology//,/_}"
  OUT_DIR="$BASE" LABEL="$label" GRID=256,256,256 MAXSTEP=20 \
  DISCARD_STEPS=1 REPEATS=5 GPU_EXE="$GPU_EXE" GPU_IDS=0,1 \
  NP=2 TOPOLOGY="$topology" SYNC_MODE=explicit PHASE_TIMING=0 \
  HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
    bash "$ROOT/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
done
```

Collect phase attribution separately because the additional synchronization
and logging perturb complete-RK timing:

```bash
ROOT=/home/dell/workspace/astr_gpu
PHASE_BASE="$ROOT/tests/gpu_validation/out/p4_0_256_phases"
GPU_EXE="$ROOT/build_gpu_p4/bin/astr"

OUT_DIR="$PHASE_BASE" LABEL=np1_phase GRID=256,256,256 MAXSTEP=20 \
DISCARD_STEPS=1 REPEATS=5 GPU_EXE="$GPU_EXE" GPU_IDS=0 NP=1 \
TOPOLOGY=1,1,1 SYNC_MODE=explicit PHASE_TIMING=1 \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
  bash "$ROOT/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"

OUT_DIR="$PHASE_BASE" LABEL=np2_x_phase GRID=256,256,256 MAXSTEP=20 \
DISCARD_STEPS=1 REPEATS=5 GPU_EXE="$GPU_EXE" GPU_IDS=0,1 NP=2 \
TOPOLOGY=2,1,1 SYNC_MODE=explicit PHASE_TIMING=1 \
HALO_TRANSPORT=pageable FILTER_WORKSPACE=full \
  bash "$ROOT/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
```

Capture the matching NP=2 x-slab timeline. HCOLL is disabled for this local
profile only because the workstation lacks the requested HCOLL transport:

```bash
ROOT=/home/dell/workspace/astr_gpu
NSYS_OUT="$ROOT/tests/gpu_validation/out/p4_0_nsys"
mkdir -p "$NSYS_OUT"
python3 "$ROOT/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT/examples/Taylor_Green_Vortex" \
  --dst-case "$NSYS_OUT/case" --use-gpu t --grid 256,256,256 \
  --maxstep 2 --feqchkpt 9999 --lfilter t --diffterm t \
  --lreadgrid f --scheme 643e
[[ ! -e "$NSYS_OUT/case/datin/grid.h5" ]] || \
  unlink "$NSYS_OUT/case/datin/grid.h5"
(
  cd "$NSYS_OUT/case"
  CUDA_VISIBLE_DEVICES=0,1 OMPI_MCA_coll_hcoll_enable=0 \
  ASTR_FORCE_MPI_TOPOLOGY=2,1,1 \
  ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  ASTR_GPU_PHASE_TIMING=1 ASTR_GPU_SYNC_MODE=explicit \
  ASTR_GPU_HALO_TRANSPORT=pageable ASTR_GPU_FILTER_WORKSPACE=full \
    nsys profile --trace=cuda,mpi,nvtx --sample=none --cpuctxsw=none \
      --force-overwrite=true -o ../np2_xslab \
      mpirun -np 2 "$ROOT/build_gpu_p4/bin/astr" \
      run datin/input.tgv > ../np2_xslab.log 2>&1
)

find "$NSYS_OUT/case" -type f \
  \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print
nsys stats --report cuda_api_trace,mpi_event_trace,cuda_gpu_trace \
  --format csv --output "$NSYS_OUT/np2_xslab_trace" \
  "$NSYS_OUT/np2_xslab.nsys-rep"
```

The frozen local results and interpretation boundary are recorded in
`documents/ASTR_PHASE_P4_0_BASELINE_REPORT.md`.

## P4 per-axis pipeline contexts

The `pinned-pipeline` backend maintains one transport context per active MPI
axis. Solution halos post all active-axis MPI transactions before waiting and
unpack in x/y/z order. Filter directions remain ordered because each direction
consumes the previous ping-pong result. Fused diffusion visits every active
axis but overlaps the independent interior RHS only with the first transaction.

Build and run the focused multi-context transport contract:

```bash
ROOT=/home/dell/workspace/astr_gpu
cmake --build "$ROOT/build_gpu_p4" --target halo_transport_setup_test -j4
env ASTR_GPU_HALO_TRANSPORT=pinned-pipeline \
  mpirun -np 2 "$ROOT/build_gpu_p4/bin/halo_transport_setup_test" contexts
```

Run the two- and three-axis field gates with the full filter and diffusion path:

```bash
ROOT=/home/dell/workspace/astr_gpu
for spec in '4 2,2,1' '8 2,2,2'; do
  read -r np topology <<<"$spec"
  label="np${np}_${topology//,/_}"
  OUT_DIR="$ROOT/tests/gpu_validation/out/p4_context_${label}_100step" \
  MAXSTEP=100 FEQCHKPT=100 MPI_NP="$np" TOPOLOGY="$topology" \
  LFILTER=t DIFFTERM=t FILTER_WORKSPACE=full \
  SYNC_MODE=explicit HALO_TRANSPORT=pinned-pipeline \
  CPU_EXE="$ROOT/build_cpu_p4/bin/astr" \
  GPU_EXE="$ROOT/build_gpu_p4/bin/astr" \
    bash "$ROOT/tests/gpu_validation/run_tgv_mpirank2_field_compare.sh"
done
```

Current expected result: both reports pass at `1e-10`; maximum reconstructed
`q5` errors are `7.1054e-13` for NP=4 `2x2x1` and `7.6739e-13` for NP=8
`2x2x2`. These ranks share two local GPUs and therefore qualify correctness,
not scaling.

The NP=4 `2x2x1` full-solver Compute Sanitizer gate uses a `32^3`, one-step
case with the established host-only OpenMPI sanitizer isolation. All four ranks
must report zero leaked bytes and zero errors under full memcheck, followed by
zero hazards under racecheck. The retained evidence is under
`build_gpu_p4/validation/multiaxis_memcheck_np4`.

The latest valid one-rank-per-GPU local performance comparison is NP=2 x-slab,
`256^3`, ten retained RK samples per process and five processes per path. The
medians are `0.529076911 s/RK` for pinned explicit, `0.506394356 s/RK` for
pipeline explicit, and `0.504894878 s/RK` for pipeline dependency. Explicit
pipeline is `4.287%` faster than pinned; dependency adds only `0.296%`, so it
remains opt-in. Multi-axis performance requires at least four physical GPUs.

## P4 device-aware MPI admission

`ASTR_GPU_HALO_TRANSPORT=device-aware` is a fail-closed optional backend. It
binds each process to its rank-local CUDA device before `MPI_Init`, requires a
positive collective `MPIX_Query_cuda_support` result, and sends the existing
packed FP64 device buffers directly through MPI. The pageable, pinned,
pinned-overlap, and pinned-pipeline paths remain available.

The direct path covers solution, full and scalar filter, FP64 diffusion,
mixed-storage diffusion, shock sensor, sponge, species, and generic-field
halos. Widths, tags `21001:21006`, endpoint rules, and unpack semantics are
unchanged. Build and run the source contracts with:

```bash
ROOT=/home/dell/workspace/astr_gpu
cmake --build "$ROOT/build_gpu_probe" \
  --target astr halo_transport_test halo_transport_setup_test -j2
python3 -m unittest \
  "$ROOT/tests/gpu_validation/test_device_aware_halo_contract.py" -v
```

On the local RTX 4000 Ada workstation, UCX CUDA IPC is not qualified. The
explicit `UCX_TLS=self,sm,cuda_copy` fallback passes five-step `128^3` TGV
pinned/device-aware comparisons for x/y/z slabs and three corresponding
Compute Sanitizer runs. All four TGV diagnostics are bitwise identical and each
sanitizer run reports zero errors. This is a local no-IPC correctness result.

On the Zhongke A800 stack, payload job `460370` passed exact data, protocol,
and sanitizer gates with `cuda_ipc/cuda`. Reproduce the solver admission under
the approved work root with:

```bash
ROOT=/data/user/hd56000/weiph/astr_gpu_cuda_aware_qualification_c801eeb
cd "$ROOT"
sbatch tests/gpu_validation/run_zhongke_a800_device_aware_solver_admission.sbatch
sbatch --ntasks=4 --ntasks-per-node=4 --gres=gpu:4 \
  --export=ALL,MPI_NP=4 \
  tests/gpu_validation/run_zhongke_a800_device_aware_solver_admission.sbatch
```

Jobs `460439` and `460441` completed in 27 s and 32 s. They cover NP=2 slabs
and NP=4 planes with full filter/diffusion, five steps, explicit synchronization,
and no field HDF5. Pinned/device-aware time, kinetic energy, enstrophy, and
dissipation are bitwise identical. Every topology records `cuda_ipc/cuda`, and
the NP=2/4 solver sanitizer reports zero errors. This admits the backend for
single-node periodic TGV correctness on the recorded A800 MPI/UCX stack. It
does not promote device-aware MPI to the default before repeated performance,
non-periodic case, and multi-node gates pass.

The non-reacting non-periodic matrix is driven by
`run_device_aware_nonreacting_matrix.sh`. It reuses the established CPU/GPU
comparison harnesses and injects `device-aware` only into the GPU subprocess.
The default NP=2 matrix contains 20 cases:

- three Cartesian zero-extrapolation cases with the decomposed axis normal to
  the physical faces;
- three CURVE `bctype=41` cases with x-, y-, and z-wavy physical walls;
- three Cartesian HBL x/y/z slabs with the Ducros sensor and selective Roe
  characteristic reconstruction enabled;
- eleven supported Phase H `41/42/411/421` wall-family entries.

Each case requires passing statistics and complete-RK field comparisons. The
shock cases additionally require an exact byte mask and a raw-sensor
comparison. Every GPU log must select `device-aware`, finish without NaN or
IEEE exceptions, and, when `REQUIRE_CUDA_IPC=t`, record `cuda_ipc/cuda`.

The local RTX/HPC-X stack only qualifies the no-IPC path, so the functional
matrix was run with `UCX_TLS=self,sm,cuda_copy` and
`REQUIRE_CUDA_IPC=f`. All 20 cases passed. The maximum statistics difference
was `7.8159700933611020e-13`, the maximum conservative `q5` difference was
`2.8421709430404007e-13`, the maximum raw-sensor difference was
`1.1102230246251565e-15`, and all three shock masks had zero mismatches. This
does not qualify local CUDA IPC.

Run the production A800 gate only after the source and build directories point
to the same revision:

```bash
ROOT=/data/user/hd56000/weiph/astr_gpu
cd "$ROOT"
sbatch tests/gpu_validation/run_zhongke_a800_device_aware_nonreacting_admission.sbatch
```

The A800 wrapper is fail-closed: it requires the earlier payload qualification,
two Slurm-visible GPUs, resolved executable dependencies, strict floating-point
compiler flags, all 20 numerical reports, and `cuda_ipc/cuda` in every GPU log.
Until that job passes, non-periodic production admission remains pending.

## Fixed air5 numq11 selective shock-capturing gate

The first fixed-air5 shock path is selected only by `conschm=643e`,
`recon_schem=3`, and `lchardecomp=f`. Smooth interfaces retain the sixth-order
explicit central flux. Ducros-marked interfaces use a frozen-composition
spectral radius, local Lax--Friedrichs splitting, and componentwise MP7 for all
11 conservative fluxes. The species fluxes are closed so that their sum equals
the density flux. The full MP7 path requires at least four halo layers.

Air5 MP5/MP7 use the scale-independent inclusive interval shortcut
`min(center,mp) <= linear <= max(center,mp)`, then the existing full MP bounds.
There is no absolute shortcut tolerance or product susceptible to underflow.
Run `python3 -m pytest -q tests/gpu_validation/test_air5_mp_reconstruction.py`
to compile the production CPU pure helpers with gfortran and check recorded
tiny-flux data, scale homogeneity, smooth polynomials, and CPU/GPU helper
algebra. Device execution is checked separately by the CPU/GPU case drivers.

Run the bounded periodic gate with:

```bash
ROOT=/home/dell/workspace/astr_gpu
cd "$ROOT"
OUT_DIR="$ROOT/tests/gpu_validation/out/air5_numq11_shock_tube" \
  BUILD_DIR="$ROOT/build_gpu_probe" \
  tests/gpu_validation/run_air5_numq11_shock_tube_compare.sh
```

The recorded `64x8x8`, five-step run passed all 12 matching CPU/GPU snapshots.
The maximum absolute difference was `2.3283e-10`, and the maximum difference
scaled by the conservative-variable magnitude was `6.0743e-14`. Density,
pressure, temperature, and all species densities remained positive. Species
mass closure was at most `1.1103e-15`, the global conservation scaled error was
`7.7415e-14`, and the shock mask was nonempty. The three existing frozen AIR5
transport cases remained passing after this path was added.

The same short gate also passed with all three NP=2 slabs: `2x1x1`, `1x2x1`,
and `1x1x2`. Each topology produced 24 matching CPU/GPU snapshots. The y/z
slabs retained a maximum conservative-variable-scaled difference of
`6.4531e-14`. Compute Sanitizer reported zero errors and zero leaked bytes on
both ranks for x/y/z slabs.

A one-step `32x8x8` GPU run under Compute Sanitizer reported zero errors and
zero leaked bytes when OpenMPI CUDA probing was isolated with
`OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self`, and `OMPI_MCA_osc=pt2pt`. The initial
CUDA 700 failure exposed a real allocation defect: the species convection
positivity limiter used `diffusion_ratio_d` even when `diffterm=f`, while the
array was allocated only for diffusion. It is now allocated for every fixed
AIR5 transport run.

This is a periodic numerical-core gate. A 100-step `64x8x8` run passed for
NP=1 and NP=2 `2x1x1`; the latter carries the wave system across both the MPI
interface and global periodic seam. Maximum CPU/GPU conservative-variable-scaled
differences were `8.0078e-13` and `9.0442e-13`. The NP=1 program-integrated
mass, total-energy and elemental drift was at most `9.2245e-14`; the independent
snapshot checks were at most `2.5918e-13` for NP=1 and `2.7976e-13` for NP=2.
The shared-plane sensor decision is paired on both adjacent interfaces so the
periodic/MPI flux difference remains conservative when the shock mask reaches a
partition seam.

This periodic gate does not admit physical boundaries, normal-shock relaxation,
or finite-rate SBLI. Its static pressure discontinuity has zero velocity
divergence at the first RK stage, so the Ducros-pressure sensor begins marking
at the second stage. The coupled normal-shock gate below resolves startup with
a compressive steady-jump initial condition rather than pressure pre-marking.

## Fixed air5 coupled normal-shock gate

The first open-boundary reacting-shock gate uses `air5normalshock`. Its input
states are generated by the independent FP64 reference as a Mach-8 frozen
Rankine--Hugoniot jump at `T1=500 K` and `p1=5 kPa`. The x-min boundary fixes
all 11 upstream conservative variables, the x-max boundary extrapolates the
last active state into ghost layers, and y/z remain periodic. The nonzero
compressive velocity activates the Ducros mask in the first RK stage.

Run the bounded NP=1 gate with:

```bash
ROOT=/home/dell/workspace/astr_gpu
cd "$ROOT"
GRID=16,6,6 NP=1 TOPOLOGY=1,1,1 \
  OUT_DIR="$ROOT/tests/gpu_validation/out/air5_c5_normal_shock_smoke" \
  tests/gpu_validation/run_air5_c5_normal_shock_compare.sh
```

For the x-slab MPI gate use `NP=2 TOPOLOGY=2,1,1`. Both recorded one-step runs
at `dt=1e-8 s` pass. The maximum CPU/GPU difference over 10 snapshots for NP=1
and 20 snapshots for NP=2 is `2.3283e-10`. The initial mass, momentum and
total-energy flux relative spread is `1.2849e-16`, species mass closure is at
most `1.3878e-16`, and 396 first-stage interfaces are marked. The measured
frozen upstream/downstream Mach numbers are `8.0000/0.39289`.

The diffusion-enabled matrix also passes NP=1 and NP=2 x/y/z slabs. Its maximum
CPU/GPU absolute difference is `1.8190e-10`, with species mass closure no larger
than `1.3878e-16`. The isolated NP=1 full leak-check reports zero Compute
Sanitizer errors and zero leaked bytes. A continuous three-step run and a run
restarted from the step-1 checkpoint agree at step 2 with maximum
physical-scaled error `5.1285e-14`. That gate exposed a restart lifecycle defect:
the normal-shock constant boundary states were configured only on fresh
initialization. Restart now rebuilds the same boundary states from the pinned
input without changing the flow field.

The long-time captured-shock driver is experimental and currently fails its
physical gate. The completed `32x6x6`, `dt=8e-8 s`, 201-update run preserves
positive states but develops transverse variation and a migrating shock.
The extrapolating subsonic outlet does not impose the reference back pressure.
Do not treat another unchanged long run as the next acceptance step. See
[`ASTR_AIR5_NORMAL_SHOCK_LONG_RUN_DIAGNOSIS.md`](../../documents/ASTR_AIR5_NORMAL_SHOCK_LONG_RUN_DIAGNOSIS.md)
for the failed legacy evidence and subsequently approved pressure outlet.

The opt-in pressure metadata is produced by
`generate_air5_normal_shock_states.py --outlet-reference-length 0.01`.
Set `OUTLET_REFERENCE_LENGTH=0.01` in the normal-shock comparison, restart,
or memcheck driver to exercise it; leaving that variable unset preserves those
drivers' original extrapolation tests. The CMake target
`air5_pressure_outlet_probe` checks the CPU acoustic correction, preserved
outgoing variables, and rejected invalid states.

The captured-shock long driver defaults to the pressure outlet and 601 updates
(`MAXSTEP=600`). Use `OUTLET_REFERENCE_LENGTH=0` for the old extrapolation mode.
The physical checker now also requires `--previous-checkpoint` and
`--current-checkpoint`, passed from the case's `bakup/` and `outdat/`.
Its stationarity threshold is `1e-3` maximum field-scaled primitive change per
reference transit time. The transit uses the fixed initial post-shock length
and the reference velocity integral, not the current moving shock location.

The reproduction command is:

```bash
ROOT=/home/dell/workspace/astr_gpu
cd "$ROOT"
OUT_DIR="$ROOT/tests/gpu_validation/out/air5_c5_captured_normal_shock" \
  tests/gpu_validation/run_air5_c5_captured_normal_shock.sh
```

It writes validation snapshots only at the initial and final requested steps,
keeps periodic checkpoints, locates the captured shock from the largest
pressure jump, excludes the shock layer, and compares downstream mass,
momentum, and total-energy fluxes plus `rho/u/T/Tv/Ys` against the independent
steady FP64 relaxation ODE. A separate `8x6x6`, `dt=1e-7 s`, three-step probe
remains positive and conservative but fails this steady-profile gate as
expected: `3e-7 s` is far shorter than one downstream flow-through time and the
line has only three comparison nodes. Thus diffusion, all NP=2 slabs, restart,
and memory safety are closed; stationary long-time relaxation, resolution/time
step sensitivity, and wall-bounded reacting SBLI remain open.

If a run times out after a periodic checkpoint without a failed numerical gate,
increase `MAXSTEP` and
resume the same evidence directory without regenerating the initial state:

```bash
OUT_DIR="$ROOT/tests/gpu_validation/out/air5_c5_captured_normal_shock" \
  RESUME=1 MAXSTEP=800 \
  tests/gpu_validation/run_air5_c5_captured_normal_shock.sh
```

The resume contract requires the original `GRID`, `DELTAT`, `NP`,
`TOPOLOGY`, outlet mode, and state-file content; the driver records and checks
them before reading the checkpoint. Legacy contracts without boundary metadata
are rejected rather than implicitly converted to the new physical problem.

### Air5 Long-Time Transverse Diagnostic

`run_air5_long_time_diagnostic.py` is a separate, user-approved diagnostic,
not a relaxed version of the captured-shock physical gate. It copies verified
fresh-control inputs and a frozen executable, runs 625 updates at `dt=8e-8 s`
with NP=1 and explicit GPU synchronization, and records warning-only extrusion
plus physical-scale RMS statistics. Invalid accepted states and solver failures
still stop. Output directories must be new; there is no automatic restart.

```bash
python3 tests/gpu_validation/run_air5_long_time_diagnostic.py \
  --baseline tests/gpu_validation/out/air5_mp_interval_20260923/fresh_control \
  --executable-manifest tests/gpu_validation/out/air5_transverse_diagnosis_20260923/quarter_dt40_uninterrupted19/provenance.json \
  --output tests/gpu_validation/out/air5_long_time_diagnostic_NEW
python3 -m unittest discover -s tests/gpu_validation -p test_air5_long_time_diagnostic.py
```

The first baseline supplies input provenance; the optional executable manifest
supplies an already verified rebuild's provenance. Generated `grid.h5` is not
reused (`lreadgrid=f`). Historical local artifacts are required for this frozen
experiment; this is not a standalone case generator for arbitrary checkouts.
`statistics.jsonl` samples checkpoints every five updates; final `50 us` is
read from `post_chemistry.step00000624.rk02`, not the step-620 HDF.
The full definitions and interpretation are in
`documents/ASTR_AIR5_NORMAL_SHOCK_LONG_RUN_DIAGNOSIS.md`.

The current run is maintained by user service
`astr-air5-longdiag-20260923-r2.service` with `Restart=no`, 12-hour driver cap,
and user linger enabled. Inspect without interfering:

```bash
systemctl --user status astr-air5-longdiag-20260923-r2.service --no-pager
tail -n 2 tests/gpu_validation/out/air5_long_time_diagnostic_20260923/statistics.jsonl
cat tests/gpu_validation/out/air5_long_time_diagnostic_20260923/status.json
```

### Air5 C5-6B2 Oblique-Shock Boundary Data

As of 2026-09-24, transverse-error diagnosis is deferred by user decision,
not marked passed. B2 boundary engineering may proceed while the existing
long-time and physical gates remain open. No hard invalid-state gate is waived.

`generate_air5_oblique_shock_states.py` prepares frozen oblique-shock states
for the prescribed top boundary. It reuses the fixed-air5 normal jump, preserves
tangential velocity, species and specific vibrational energy, and restores
the full kinetic energy in q11. It is not a Python flow time integrator.

```bash
python3 -m pytest -q tests/gpu_validation/test_air5_oblique_shock_states.py
python3 tests/gpu_validation/generate_air5_oblique_shock_states.py \
  --mechanism chemMech/air5_kimjo12.json \
  --output tests/gpu_validation/out/air5_c5_b2_preparation_20260924/oblique_m8_beta30.json \
  --mach 8 --temperature 500 --tv 500 --pressure 5000 \
  --shock-angle-deg 30 --top-x 0.004 --top-y 0.004
```

This example is a jump-data check, not a selected production SBLI case.
The JSON is not yet consumed by ASTR. Twenty-one tests check normal q11 flux,
normal-shock limit, rotated inflow, thermodynamic domain, top breakpoint and
frozen-state invariants. `--shock-angle-deg` is relative to the upstream
velocity; optional `--inflow-angle-deg` is measured above +x. The shock must
descend toward the wall. The exact top breakpoint belongs to the downstream
state. Pressure/temperature units are Pa/K; lengths are metres; q uses SI.
`--mass-fractions` uses N2,O2,N,O,NO order and is never clipped/renormalized.
The reported straight-line wall intersection is geometric, not a viscous
shock-impingement prediction. No NSCBC, full-SBLI or production-performance
pass is implied.

### Air5 C5-6B2 Runtime Startup

`air5sbli` retains the original HBL inlet, noncatalytic isothermal wall and
outflow. It replaces only the top condition with complete frozen-jump q11
states on physical and ghost nodes. `air5hbl` retains its existing behavior.
`prepare_air5_sbli_case.py` derives the jump from the actual mapped profile
edge and writes the four-record `datin/air5_incident_shock.dat` runtime file.
The solver reads this ASCII file, not the diagnostic JSON.

The startup gate uses beta=20 degrees relative to the actual incoming
velocity, domain `(80,8,2)*ref_len`, top break `20*ref_len`, wall temperature
2925 K, uniform-x initial HBL profile, coupled chemistry, diffusion and
selective LLF/MP7, no filter. The top downstream normal inflow is subsonic:
this is prescribed external forcing, not a nonreflecting characteristic BC.

```bash
python3 -m pytest -q tests/gpu_validation/test_air5_oblique_shock_states.py \
  tests/gpu_validation/test_air5_sbli_case.py
NP=1 OUT_DIR="$PWD/tests/gpu_validation/out/sbli_np1_new" \
  bash tests/gpu_validation/run_air5_sbli_startup.sh
NP=2 OUT_DIR="$PWD/tests/gpu_validation/out/sbli_np2_new" \
  bash tests/gpu_validation/run_air5_sbli_startup.sh
```

The runner refuses an existing output directory. Defaults: grid upper indices
`31,31,7`, `DELTAT=1.d-11`, `MAXSTEP=2` (iterations 0 through 2), x-slab
by default. Set `TOPOLOGY=1,2,1` for y; z requires e.g. `GRID=31,31,15`
and `TOPOLOGY=1,1,2` to retain a valid local stencil.
Root-CMake must build the executable first. The checker validates
accepted-state positivity, model bounds, mass closure, top physical/ghost
states, wall velocity/temperature, sensor activation and CFL. It compares
post-chemistry, pre-RHS, post-update and post-transport phases, not unmatched
startup phases or differently timed output files.

2026-09-24: NP1/NP2 CPU/GPU 18/36 snapshots passed, max scaled difference
8.50872e-16. Global one/two-GPU same-phase fields also passed, as did the
original HBL short regression. These tiny-time tests establish runtime
integration only. Larger-dt testing subsequently exposed an air5-specific
GPU sensor-gradient dispatch omission: the xy physical stencil was not selected.
After fixing the GPU route, the NP2 x/20-update case at dt=5e-10 passes field
comparison (max scaled 1.23e-12) and raw sensor comparison (max abs 2.02e-14),
with identical masks. Raw sensor and exact mask comparison are now mandatory.
NP2 y/z, NP2 restart and NP1 memcheck also pass. Source isolation and developed
SBLI physical validation remain separate work.

```bash
python3 tests/gpu_validation/run_air5_sbli_preflight.py --mode restart \
  --output tests/gpu_validation/out/sbli_restart_new
python3 tests/gpu_validation/run_air5_sbli_preflight.py --mode memcheck \
  --output tests/gpu_validation/out/sbli_memcheck_new
```

`run_air5_sbli_long.py` supervises ASTR on two GPUs with immutable executable,
input manifest, low-frequency checkpoints and atomic `status.json`. It does
not integrate the flow, repair invalid states or automatically restart failures.
The 2026-09-24 attempt targeted 10001 updates at dt=5e-10 (5.0005 us), but
stopped after checkpoint 700 (0.35 us). A bounded restart replay identifies
step 741/RK2 wall pressure 1.020294 MPa, exceeding the fixed 1 MPa domain.
The model bound must not be bypassed. This is an unsuccessful long-run gate.
CPU restart from the same checkpoint reproduces the same step/stage/wall-node
pressure failure (local peak-pressure relative difference 3.72e-10). This does
not establish whole-field long-time equivalence or distinguish physical
compression from a startup/grid overshoot.

For failure localization, the preflight driver supports `--mode failure-replay
--baseline <long-output>/gpu`, preserving the source checkpoint in a new output
directory and recording a diagnostic failure rather than calling it a pass.
`--use-gpu f` selects the matched CPU replay. This bounded diagnostic is tied
to checkpoint 700 and observation step 741, not a generic restart launcher.

`run_air5_sbli_domain_replay.py` supports bounded dt comparisons from the same
NP2 x-slab checkpoint (`--baseline`, `--dt`, `--updates`, new `--output`). It
copies the executable/checkpoint, keeps explicit sync and physical inputs,
and records a domain failure as `domain-failure-not-pass`. The replay clock
is `checkpoint_time + (step-checkpoint_step)*dt`, not `step*dt`. Its 300 s
timeout and model-domain stop are unchanged by diagnostic intent.
The 2026-09-24 dt/2 and dt/4 runs still failed near 0.37125 us. Wall pressure
extrapolation was the first node to exceed the bound; an ideal frozen
reflection estimate also exceeds 1 MPa. See
`documents/ASTR_AIR5_SBLI_PRESSURE_DOMAIN_DIAGNOSIS.md` before scheduling a
longer run. No wall-format or model-range change has been authorized here.

### Mach-4 Low-Pressure Precursor

The approved replacement case uses frozen Mach 4, T=Tv=1500 K, p=20 kPa,
Y(N2,O2)=(0.767,0.233), and a noncatalytic no-slip wall at T=Tv=3000 K.
`prepare_air5_mach4_case.py` prepares a new `air5hbl` directory without incident
forcing. The analytic quartic profile is only a seed. ASTR must develop the
plate before an inlet/initial field is extracted for SBLI; no Python flow
integrator is used. The old profile and failed case remain separate.

An optional `datin/air5_hbl_domain.dat` specifies the SI Cartesian dimensions:
header `air5_hbl_domain_v1`, followed by Lx Ly Lz. Omission preserves existing
air5hbl/air5sbli defaults. Both mesh generation and incident-top checks read
the same configuration. Invalid headers and nonpositive/nonfinite lengths fail.

```bash
python3 tests/gpu_validation/run_air5_sbli_long.py \
  --case mach4-precursor --grid 63,63,7 --updates 1001 --dt 2e-9 \
  --checkpoint 100 --timeout 3600 --output tests/gpu_validation/out/mach4_pilot_new
```

This is a bounded 0.207-flow-through-time pilot, not a developed-state claim.
The precursor checkpoint monitor additionally rejects T<1000 K without
changing the mechanism bounds. New short CPU/GPU evidence is in
`out/air5_mach4_precursor_gate_20260924/`; legacy regression is in
`out/air5_sbli_domain_interface_regression_20260924/`.
See `documents/ASTR_AIR5_MACH4_SBLI_PLAN.md` for the complete sequence.

`run_air5_sbli_long.py --baseline <archived-case>/gpu --executable
<archived-case>/astr` continues in a new output directory. Grid is inherited;
mechanism and executable identity are checked. The restart clock is stored
time plus subsequent updates times dt. Precursor checkpoints also preserve
fixed-station profiles and full fields for time-history diagnostics.
`check_air5_precursor_evolution.py --output <run-directory>` computes the
density-weighted displacement thickness, wall pressure, friction and two-mode
heat conduction without integrating a flow solution. It uses the existing
seven-point wall-derivative diagnostic and reports changes, not an automatic
steady or physical pass. Checkpoint evolution and last RK snapshots have
distinct phases; do not equate the last checkpoint with the final update.
Wall diagnostics retain both seven-point and three-point reconstructions to
expose stencil sensitivity; neither is selected to manufacture a physical pass.
The Mach4 y64/y128 comparison at identical t=2 us remains unresolved for wall
quantities (three-point Cf about 18.6%, heat flux about 9.5--10% relative to the
fine grid). The no-shock 10 us extension is bounded but not stationary. Preserve
these distinctions before extracting an inlet or enabling incident forcing.

`wall_response` also reports signed wall shear, friction velocity, wall kinematic
viscosity and first-node y+. The nodal wall is j=0: y1=dy, not dy/2. These are
instantaneous diagnostics, not time-averaged turbulence wall units.
`run_air5_inlet_sensitivity.py --baseline <y128-run>/gpu --output <new-directory>`
runs three inlet-only thickness factors (1.0, 0.9, 1.1) from one immutable
checkpoint, using the archived executable, sequentially on two GPUs.
The existing runner rejects thickness changes without a Mach4 restart baseline.
`check_air5_inlet_sensitivity.py --output <matrix-directory>` checks final-stage
states and compares checkpoint1500 at 3 us for the current step1000 baseline.
This checker intentionally targets this experiment, not arbitrary restart steps.
The initial interior field is unchanged; this tests a bounded inlet step response,
not three independently equilibrated flows. Physical gates remain open; see
`documents/ASTR_AIR5_MACH4_RESOLUTION_INLET_AUDIT.md`.

`run_air5_mach4_resolution.py --baseline <y128-run> --pilot <y512-short-run>
--output <new-directory>` runs y256/y512 at dt=2 ns and y512 at dt=1 ns,
comparing full-RK checkpoints at t=2 us. It checks each final update's 18 state
snapshots before proceeding. `--review-only` on a completed matrix runs no ASTR
jobs and writes extended near-inlet/interior diagnostics. Fixed executable,
mechanism, inlet and physical-domain identities are required. Grid interpolation
is only for profile comparison, not for restarting or constructing a reference
flow. Physical gates remain open despite bounded runs and y+ below one.

`run_air5_convection_limiter_probe.py --baseline <y512-dt2-run>/gpu
--reference <old-single-step-replay>/gpu --output <new-directory>` tests the
optional `ASTR_AIR5_CONVECTION_LIMITER=species_budget` path. The immutable
baseline must be checkpoint1000 at 2 us on topology 2,1,1. It first compares the
default `full_state` replay with archived snapshots, then runs candidate CPU/GPU
one-step and GPU two-half-step replays. State gates run after each case.
The candidate retains all-variable shared face coefficients and thermal bounds;
only the redundant artificial-face species constraint is removed because the
summed negative-face budget enforces species nonnegativity in exact arithmetic.
It does not alter diffusion, filtering, physical boundaries, or source integration.
`check_air5_mach4_error_replay.py --root <new-directory>` reproduces offline
operator diagnostics. Neither script establishes long-time physical convergence.

The pre-fix 2026-09-24 follow-up CPU two-half-step replay fails at step1001/RK3,
local `(1,84,2)`, although the GPU replay completed. The candidate remains opt-in
and is not production accepted. The one-step comparison did not cover this
failure; do not reinterpret its pass as a complete CPU/GPU or positivity gate.

For a read-only CPU diffusion-budget probe, add
`--backend cpu --diffusion-probe-node 0,1,36,0 --snapshot-step 1000` to
`run_air5_sbli_domain_replay.py`. The node argument is rank and local i,j,k,
not global coordinates. `--snapshot-step-secondary 1001` captures the second
update. The probe records the actual budget, signed face increments, active
constraints and shared coefficients without changing any flux or state.
It requires the existing RHS validation mode. Parse the log with
`check_air5_diffusion_probe.py --log <cpu/run.log> --report <probe.json>`.
The parser validates recorded sums; it is not a flow integrator.

CPU AIR5 RHS validation additionally writes `pre_updatefvar` after the RK update
and physical boundary application, before primitive conversion. These are
unaccepted states and can contain negative species on a failed run. Never count
them as accepted `post_update` states or silently ignore missing final snapshots.

The finite-precision transport fix reserves a component-scaled FP64 arithmetic
error budget before selecting conservative face coefficients. It does not clip
the updated species, change the EOS, or disable the negative-state checks.
Both `full_state` (default) and `species_budget` use this reserve. No bitwise
identity with the pre-fix executable is implied.

Build and run the actual Fortran algebra check through the root CMake project:

```bash
cmake --build tests/gpu_validation/out/c4_cuda_build_4 --target air5_transport_roundoff_probe -j 4
tests/gpu_validation/out/c4_cuda_build_4/bin/air5_transport_roundoff_probe
python3 tests/gpu_validation/run_air5_transport_roundoff_replay.py \
  --baseline tests/gpu_validation/out/air5_mach4_resolution_20260924/y512_dt2ns/gpu \
  --output tests/gpu_validation/out/air5_transport_roundoff_20260924
```

Use a fresh output path for every execution. The replay starts with the previously
failing CPU two-half-step path, then checks GPU two-half-step and CPU/GPU full-step
states. It stops on run failure, negative accepted species, model-domain failure,
or phase mismatch. `roundoff_summary.json` distinguishes failure from bounded
replay success; it is not a long-time convergence or inlet-accuracy acceptance.

To attribute the remaining inlet-adjacent velocity difference from saved ASTR
SSPRK3 snapshots, without integrating another flow model:

```bash
python3 tests/gpu_validation/check_air5_inlet_velocity_budget.py \
  --root tests/gpu_validation/out/air5_transport_roundoff_20260924 \
  --probe-root tests/gpu_validation/out/air5_inlet_stress_20260924 \
  --report tests/gpu_validation/out/air5_inlet_stress_20260924/inlet_velocity_budget.json
```

The optional probe root must contain CPU `cpu_dt2` (one 2 ns step) and `cpu_dt1`
(two 1 ns steps) replays with `--diffusion-probe-node 0,1,35,1`. Capture step1000
and additionally step1001 for the two-step replay. The checker requires identical
checkpoint/executable contracts and bitwise-identical snapshot sets with and
without the probe. It closes density/momentum RK budgets and converts their
differences to velocity with an exact endpoint identity, including density change.
The face-based split separates direct stress limiting from changes in unlimited
stress evaluated on the actual trajectories. It is not a prediction of a flow
advanced with stress limiting disabled. No long-time physical pass is implied.

### Energy-constrained layered AIR5 diffusion candidate

`ASTR_AIR5_DIFFUSION_LIMITER=full_state|layered` defaults to `full_state` and
is independent of the convection-limiter option. `layered` first applies a shared
species budget to all species and their reconstructed enthalpy/vibrational-energy
fluxes. It then applies a shared thermal-admissibility coefficient to the mixed
face flux. Species limiting alone leaves stress unchanged; an active thermal
constraint can still reduce the complete mixed flux. No state clipping is used.

```bash
python3 tests/gpu_validation/run_air5_transport_roundoff_replay.py \
  --baseline tests/gpu_validation/out/air5_mach4_resolution_20260924/y512_dt2ns/gpu \
  --output tests/gpu_validation/out/air5_layered_diffusion_new \
  --diffusion-limiter layered

python3 tests/gpu_validation/check_air5_inlet_velocity_budget.py \
  --root tests/gpu_validation/out/air5_layered_diffusion_new \
  --report tests/gpu_validation/out/air5_layered_diffusion_new/inlet_velocity_budget.json

ASTR_AIR5_DIFFUSION_LIMITER=layered GRID=48,6,6 MPI_NP=2 TOPOLOGY=2,1,1 \
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/air5_layered_frozen_new \
  bash tests/gpu_validation/run_air5_c5_frozen_transport_compare.sh

ASTR_AIR5_DIFFUSION_LIMITER=layered DELTAT=1.d-11 \
OUT_DIR=/home/dell/workspace/astr_gpu/tests/gpu_validation/out/air5_layered_memcheck_new \
  bash tests/gpu_validation/run_air5_c5_normal_shock_memcheck.sh
```

Use absolute `OUT_DIR` paths for shell drivers, since the executable runs inside
the prepared case directory. The Python replay resolves its output path itself.
The old single-node `--diffusion-probe-node` describes full-state diffusion only
and is rejected with `layered`. Use the RK snapshot budget without `--probe-root`.
Layered validation writes `diffusion_species_ratio` and `diffusion_energy_ratio`.
The existing `diffusion_ratio` snapshot and periodic limiter log refer to the
species coefficient in layered mode, not the final energy coefficient.
The replay checker requires complete coefficient snapshot sets, coefficients in
[0,1], CPU/GPU coefficient agreement and valid accepted states. Neither algebra
tests nor a replay in which every energy coefficient equals one establish the
active-energy fallback or long-time physical gate. The candidate remains opt-in.

The follow-up actual-ASTR stress gate covers active energy and species limits
on an x-slab MPI interface, including CPU NP1/NP2 and GPU NP2 comparisons:

```bash
python3 tests/gpu_validation/run_air5_layered_stress_gate.py \
  --baseline tests/gpu_validation/out/air5_layered_frozen_20260924/ev-pulse/gpu \
  --output tests/gpu_validation/out/air5_layered_stress_new
```

The baseline must be the 48x6x6 periodic frozen `air5evpulse` case. The driver
copies it into fresh directories and prepares constant-pressure states with
either a localized vibrational-temperature pulse or exact-zero trace species.
The default `ref_len=1e-4 m, dt=5e-9 s` exercises the limiter; it is not a physical
or diffusion-accuracy test. Missing limiter activation is a failed coverage gate,
even if the solver completes. Invalid states, interface mismatch, conservation
failure or CPU/GPU mismatch also stop the driver. No flow integrator is in Python.

For bounded Mach4 multi-step comparisons, run `run_air5_sbli_domain_replay.py`
twice from the identical checkpoint/executable using `--convection-limiter
species_budget --diffusion-limiter layered`. Use matched durations, not matched
step numbers. For the step1000 checkpoint, 2 ns x 100 updates and 1 ns x 200
updates require secondary snapshots at steps1099 and1199 respectively, in addition
to the common `--snapshot-step 1000`. Store them as `gpu_dt2` and `gpu_dt1`:

```bash
python3 tests/gpu_validation/check_air5_mach4_error_replay.py \
  --root tests/gpu_validation/out/air5_layered_multistep_20260924 --matched-window
```

`--window-cases gpu_dt1 gpu_dt05 --report <fresh-json-path>` compares a second
matched pair without overwriting the first report. A 0.5 ns x 400 update run uses
secondary step1399. The checker compares final `post_chemistry` snapshots and
rejects incompatible provenance or clocks. It reports timestep differences,
not an accuracy or long-time physical pass.

For late inlet localization, the replay driver accepts a CPU-only read-only
`--convection-probe-node rank,i,j,k` with selected snapshot steps. It records the
actual negative species budgets, six high-minus-low face increments and shared
coefficients without modifying the accepted flux. It also supports layered
diffusion; the separate full-state diffusion probe does not.

```bash
python3 tests/gpu_validation/check_air5_convection_probe.py \
  --case tests/gpu_validation/out/air5_layered_inlet_localization_20260924/cpu_dt2 \
  --report /tmp/air5_convection_probe.json
python3 tests/gpu_validation/check_air5_inlet_velocity_budget.py \
  --coarse-case tests/gpu_validation/out/air5_layered_inlet_localization_20260924/cpu_dt2 \
  --fine-case tests/gpu_validation/out/air5_layered_inlet_localization_20260924/cpu_dt1 \
  --report /tmp/air5_late_velocity_budget.json
```

The pair checker requires one full step versus two half steps, a common checkpoint,
binary, backend and limiter configuration, matching physical endpoints, and all
intervening RK snapshots. The probe checker verifies the face budget and its
identity with the recorded limiter RHS, not an independently evolved solution.
Neither checker grants temporal convergence or physical validation. A positive
unlimited trial at one node does not justify disabling globally shared limits.

### Sequential Consistent-Species Convection Candidate

`ASTR_AIR5_CONVECTION_LIMITER=consistent_species` enables an opt-in CPU/GPU
candidate. `full_state` remains the default; `species_budget` remains available.
The first shared-face coefficient limits fluid fluxes with independent species
held at their LF flux. The second limits a zero-sum species correction at fixed
mass, momentum, total-energy and vibrational-energy flux. N2 closes the mass flux
in both layers. Species nonnegativity and both thermal constraints are retained;
there is no species clipping or concentration floor. Invalid low/intermediate
states abort the run. This is not a pressure-equilibrium or physical validation
claim.

The candidate allocates one additional scalar halo field. Validation snapshots
`convection_fluid_ratio` and `convection_species_ratio` distinguish the layers;
`convection_ratio` aliases the second. The old single-pass convection probe is
rejected for this mode. The existing endpoint RHS budget checker still applies.

```bash
python3 tests/gpu_validation/run_air5_sbli_domain_replay.py \
  --baseline tests/gpu_validation/out/air5_layered_multistep_20260924/gpu_dt2/gpu \
  --output /tmp/air5_consistent_gpu_one_step \
  --dt 2e-9 --updates 1 --snapshot-step 1090 --backend gpu \
  --convection-limiter consistent_species --diffusion-limiter layered
python3 tests/gpu_validation/run_air5_layered_stress_gate.py \
  --baseline tests/gpu_validation/out/air5_layered_frozen_20260924/ev-pulse/gpu \
  --output /tmp/air5_consistent_stress \
  --convection-limiter consistent_species
```

Output directories must be new. The stress driver checks periodic conservation,
accepted states, shared halo coefficients, CPU NP1/NP2 and GPU NP2 fields, and
activation of both sequential convection layers near an MPI interface. The replay
driver accepts `--timeout-seconds` (default 300) for explicitly bounded windows.
Exit success alone is not acceptance: inspect `result.json`, run `state_gate`,
and compare same-phase fields. The late checkpoint is step1090 at t=2.18 us,
not the original step1000 checkpoint at t=2 us. Use the latter for the matched
100/200/400-update window.

Evidence and limitations are recorded in
[`ASTR_AIR5_MACH4_RESOLUTION_INLET_AUDIT.md`](../../documents/ASTR_AIR5_MACH4_RESOLUTION_INLET_AUDIT.md).
Local algebra tests are in `test_air5_consistent_convection.py`; Python does not
advance the flow solution.

The same stress driver now accepts `--fixture contact low_n2`. These are frozen,
inviscid periodic ASTR runs at p=100000 Pa, T=4000 K, Tv=1500 K, u=100 m/s,
one RK3 step of 5 ns, and grid upper bounds 48,6,6. `contact` places a localized
NO mass fraction of 1e-12 in a smooth N2/O2 composition wave. `--contact-trace 0`
removes that pulse as a matched control. `low_n2` tests N2 mass fractions between
1e-12 and 2e-12; these are input states, never positivity floors.

`--require-contact-equilibrium` additionally requires the final constant p/T/u
to satisfy the same tolerances used for the initial state. The current
`consistent_species` candidate with nonzero trace NO **fails this check**, despite
passing positivity and conservation. Removing the pulse or using `full_state`
preserves the constant state to roundoff in this fixture. This is a known
method-compatibility limitation, not a CUDA discrepancy. Do not promote the
candidate or resume long SBLI on the basis of the diagnostic driver's exit alone.

`check_air5_consistent_residual.py --coarse <replay> --fine <replay> --report <json>`
locates component/temperature differences and exactly decomposes temperature and
pressure changes from the conservative endpoints. `--rk-budget` additionally
requires one full step versus two half steps with every intermediate snapshot,
and attributes all 11 conservative increments to the recorded operators.
Long-window endpoints alone do not determine cumulative operator contributions.

The first-stage contact identity in `uniform_contact_drift` compares recorded
raw/limited convection RHS with the actual ASTR update. Diagnostic tests are in
`test_air5_consistent_residual.py`. No Python flow time integrator is introduced.

`check_air5_consistent_residual.py --contact-case <case> --np 2 --dt 5e-9
--report <json>` reuses recorded first-stage raw/limited RHS. It reports pointwise
EOS counterfactuals: energy-only temperature or pressure restoration, and removal
of O2 co-limiting with dependent N2 closure. These arrays are never written to
the solver and are not conservative flux constructions or general positivity
proofs. The recorded contact case shows that energy-only temperature restoration
increases the pressure error to 12.78 Pa. Removing O2 co-limiting reduces the
local discrepancy sharply, identifying a species-coupling issue for a future
flux-level design, not closing the pressure-equilibrium gate.

The CMake target `air5_transport_roundoff_probe` additionally exercises the CPU
`air5_close_species_flux` helper: Euclidean projection onto supplied five-species
flux bounds and a total-mass-flux equality. It tests signed fluxes, 1000 feasible
boxes and optimality conditions, cyclic species permutations, reversed face
orientation, scale changes, trace budgets and failure returns. Run with:

```sh
cmake --build build_cpu_probe --target air5_transport_roundoff_probe -j 2
build_cpu_probe/bin/air5_transport_roundoff_probe
```

This helper is not wired into the flow solver. Caller-provided intervals are not
yet proven complete-RK positivity budgets, and thermal compatibility is not part
of this projection. Probe success does not close the known contact-pressure
failure or qualify a new CPU/GPU convection path.

### ROS-2 small-update roundoff diagnostic

```sh
cmake -S . -B build_gpu_probe
cmake --build build_gpu_probe --target chemistry_ros2_roundoff_probe -j 4
build_gpu_probe/bin/chemistry_ros2_roundoff_probe 20000 5e-10 2000
python3 -m pytest -q tests/gpu_validation/test_chemistry_ros2.py
```

Arguments are update count, fixed step (seconds), and initial T=Tv (kelvin).
This actual ASTR CPU chemistry probe compares original additions, grouped
increments, and persistent six-component Kahan compensation. The production
integrator is unchanged; optional `update_terms` exposes successful ROS-2
increments before addition to the state and returns zeros on failure.

`DRIFT` columns are path, maximum relative mass/N/O drift, maximum absolute
mass-fraction-sum error, maximum scaled ROS-2 error norm, final minimum partial
density, and first update exceeding the 128-epsilon transport composition gate
(zero means no crossing). `STATE` reports six final chemical variables and T/Tv.
`WIDE_DIGITS` records the actual host diagnostic mantissa: NVHPC uses 53 bits,
whereas gfortran can use an extended accumulator. A reported zero is not an
exact-arithmetic conservation claim.

The probe stops on a failed chemistry/positivity/thermodynamic/error-norm gate.
Composition-gate crossings are measured, not passed to a transport calculation:
no flow RHS, spatial boundary, adaptive rejection, or restart is exercised.
Compensation is committed only after trial-state checks. Passing this diagnostic
does not close the outstanding coupled-flow long-window failure.

### CPU/GPU adaptive compensation candidate

```sh
cmake -S . -B build_gpu_probe
cmake --build build_gpu_probe --target chemistry_ros2_compensation_gpu_probe -j 4
build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe 20000
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 86 \
  build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe 100
```

The optional `compensation(6)` argument of CPU/GPU ROS-2 fixed-step and adaptive
routines holds Kahan correction terms for five partial densities and Ev. Source
and thermodynamic evaluations still use the FP64 high state, not a normalized
or clipped state. Existing production callers omit this argument. The fixed-step
routine commits correction only after candidate admissibility; adaptive advance
works on local trial copies and commits caller-owned correction only after the
entire requested interval succeeds. An exhausted attempt budget restores both
the state and incoming correction even after accepted internal substeps.

The probe compares CPU/GPU repeated adaptive calls for 1500/2000/3000 K air,
ordinary versus compensated updates, then tests exact within-backend agreement
of two calls versus one call with the same substep sequence. It also requires
successful rejection recovery, forced rejection rollback, accepted-prefix
rollback, and invalid-timestep rollback. Short sanitizer runs cover these paths
but do not replace the 20000-call numerical test. The CPU pytest suite compares
the compensated trajectory to the existing independent Radau reference.

No spatial transport, full-field compensation allocation, MPI carry exchange,
filter lifecycle or checkpoint format is implemented here. Default numerical
mode remains unchanged; performance of the changed device interface is not
qualified by this correctness probe.

The extended probe also exports two stiff 20 ns endpoints at initial
T/Tv=6000/1000 K and 4000/4000 K, both at 20 kPa with mixed air5 composition.
Run the existing independent zero-dimensional reference on the emitted inputs:

```sh
set -o pipefail
build_gpu_probe/bin/chemistry_ros2_compensation_gpu_probe 100 | tee /tmp/ros2-compensation.log
python3 tests/gpu_validation/check_ros2_compensation_radau.py \
  --log /tmp/ros2-compensation.log --report /tmp/ros2-compensation-radau.json
```

This compares both CPU/GPU and ordinary/compensated paths: mass-fraction
rtol=1e-6 and atol=1e-10, T/Tv rtol=1e-7, mass/N/O relative drift <=1e-14,
composition closure <=128 epsilon, and q5 EOS roundtrip <=32 epsilon.
Ratios in the JSON are errors divided by their acceptance limits, not relative
errors. EOS roundtrip checks algebraic compatibility, not an independent spatial
energy-conservation experiment. The probe additionally checks bitwise rollback
with NaN/Inf duration/step/tolerance, compensation, and species-state inputs.
The `100` argument reduces repeated low-temperature calls only; it does not
shorten the stiff 20 ns comparisons. It is not a replacement for the previous
20000-call accumulation gate.

### Prepared long-window matrix (not launchable yet)

`air5_long_window_plan.json` is a declarative, disabled preparation record, not
input to the replay driver. It fixes the step1090 Mach4 checkpoint, NP2 x-slab,
and matched endpoint windows at 2.38, 4.18 and 12.18 microseconds. The production
chemistry callers do not yet pass compensation, and spatial RK carry propagation
is not implemented. Do not treat a replay of the current executable as a
compensated full-flow acceptance run. The blocking lifecycle requirements and
output/comparison policy are in `documents/ASTR_AIR5_MACH4_SBLI_PLAN.md`.
## AIR5 Full-Flow Compensation Lifecycle (2026-09-26)

`ASTR_AIR5_COMPENSATION=on` enables persistent FP64 low parts for the AIR5
HBL/SBLI candidate. The default is `off`; filtering, sponge layers, immersed
bodies, automatic crash-fix and non-RK3 integration are rejected in this candidate. This is a
roundoff-control path, not mixed precision or a species normalization.

The represented state is `q - carry`. Chemistry advances six low parts
transactionally. SSP-RK carries all eleven conservative components and keeps
the existing Jacobian-weighted `qsave` interface unchanged. Prescribed or
reconstructed physical boundaries start with zero carry. MPI and periodic
duplicate nodes combine both low parts and the rounding residual of their
existing high-part average. Each completed step canonicalizes duplicate nodes
before it can become a global checkpoint, independently of output frequency.

Version-1 HDF5 checkpoints contain `acq01..11`, `acc01..11` and
`air5_compensation_version`. A restart restores conservative high parts after
the ordinary primitive reconstruction. An old checkpoint requires explicit
`ASTR_AIR5_COMPENSATION_RESTART=initialize`; this starts a new experiment with
zero low parts, not a continuation of unavailable historical low parts.

The replay driver accepts `--compensation on` and
`--compensation-restart initialize|restore`, and records both in its contract.
`check_air5_compensation_checkpoint.py REFERENCE CANDIDATE --exact` compares
all 22 fields bitwise for same-backend continuous/restart tests. Without
`--exact`, conservative fields use the existing `atol=1e-9, rtol=1e-10` gate.
Both modes enforce finite fields, nonnegative species, positive density and
sequential mass-fraction closure within `128*epsilon(FP64)`.

Build the arithmetic/MPI primitive probe through root CMake:

```sh
cmake --build build_gpu_probe --target chemistry_compensation_lifecycle_probe
mpirun --mca coll '^hcoll,ucc' -np 2 build_gpu_probe/bin/chemistry_compensation_lifecycle_probe
python3 -m pytest -q tests/gpu_validation/test_air5_compensation_checkpoint.py
```

Evidence roots:

- `out/air5_compensated_flow_20260926`: first integration, including the failed
  bitwise restart gate. Do not use this candidate for long windows.
- `out/air5_compensated_flow_canonical_20260926`: complete-step canonicalization
  candidate; executable SHA256 is recorded in every replay contract.
- The first `default_frozen` attempt used an older binary without the symmetric
  limiter and was rejected before integration. It is not a numerical comparison.
  `default_matched` uses the actual frozen symmetric-limiter baseline.

GPU face carry transport currently stages only physical face data on the host
using a private MPI communicator. It is a correctness candidate, not an accepted
communication optimization. Long-window admission remains separate from these
short lifecycle checks; see `air5_long_window_plan.json` and the project status.

### Mach4 Compensated Incident Startup

`prepare_air5_mach4_incident_restart.py` copies an existing compensated Mach4
HBL checkpoint without changing its high state or carry, verifies the saved top
edge against the inlet profile, and attaches the approved 25-degree frozen jump.
It refuses mismatched states or an existing destination. The precursor need not
be steady for this startup test, but the result must not be called developed SBLI.

Use `run_air5_sbli_domain_replay.py --compensation on
--compensation-restart restore` with the prepared seed. The driver copies the
incident metadata for `check_air5_sbli_startup.py`. Check CPU/GPU stage states,
sensor masks, top/wall invariants, the stricter 128-epsilon sequential composition
gate, and memcheck before extending the incident window. Preserve the no-shock
precursor separately. The older Mach6 startup preparer is not interchangeable.

Evidence: `out/air5_mach4_incident_20260926/`. The initial short matrix uses two
updates at 1 ns. The longer startup pair uses 20 updates at 1 ns and 40 at 0.5 ns,
both from the same step3000 checkpoint. Neither demonstrates an established
shock/boundary-layer interaction or physical convergence.

### AIR5 Explicit Filter Readiness

`run_air5_filter_readiness.py --executable <frozen-astr> --mpirun <launcher>
--output <new-directory>` checks the admitted filter path, not production restart
readiness. It fixes `symmetric_species` convection and `layered` diffusion,
prescribed top, compensation off, FP64, and one complete update at 1e-10 s.
The 32x32x16-node matrix covers full/scalar workspaces and topologies 1x1x1,
2x1x1, 1x2x1, and 1x1x2. Each case checks CPU/GPU phase fields, admissibility,
species closure, physical boundaries, and spanwise invariance. Saved fields are
also compared directly across backends, workspaces, and topologies, with
`abs(candidate-reference)/max(1,abs(reference)) <= 2e-10`.

Four startup rejection tests verify that CPU and GPU both reject characteristic
top plus filtering and compensation plus filtering before chemistry starts.
These are intentional unsupported combinations; do not remove their guards to
resume a compensated characteristic-top precursor.

Evidence: `out/air5_filter_readiness_20260928/result.json` and
`cross_comparison.json`: all eight paired cases and four rejection gates passed;
maximum cross-comparison scaled error was 3.1575e-14. This is a one-update gate,
not a long-window, filtered restart, or compensated-filter validation.

The separate `out/air5_filter_recheck_20260928_full` legacy full-state-limiter
attempt failed the CPU spanwise-invariance gate (1.8443e-8 versus 2e-10), before
GPU execution. It is not accepted or hidden by the passing current-limiter
matrix. Its uniform initial field also differs from historical matched-profile
tests. No tolerance was relaxed and no numerical source was changed.

Before filtering the paused production precursor, define how the filter acts
on the compensated representation `q - carry`, including species repair and
boundary lifecycle. Then validate characteristic-top coupling, compensated
restart, and a short matched filtered/unfiltered continuation before admitting
a long run. The current driver deliberately reports `production_ready=false`.

#### Compensated Characteristic-Top Filter Candidate

The new combination is gated by `ASTR_AIR5_FILTER_VALIDATION=on`. The earlier
rejection evidence above refers to its frozen executable, not the current
candidate. `run_air5_filter_readiness.py` now checks rejection without this
opt-in. No long precursor should be resumed solely because startup succeeds.

CPU and GPU normalize the compensated pair, temporarily swap physical q/carry,
filter the carry through the existing explicit filter and halo transport, swap
back, and filter q. Both passes use identical closures and axis order. The
temporary swap must never call the EOS, physical boundary reconstruction,
checkpointing, or compensated shared-node averaging. Only filter halo exchange
is permitted. This uses the selected full/scalar workspace without allocating
another complete 11-component field. Filter arithmetic itself remains FP64;
this is not a double-double filter implementation.

Species repair keeps the filtered low part of unchanged non-closure species.
Changed species are projected to the FP64 admissible target. The largest baseline
species is reconstructed from compensated density minus the other represented
species using TwoDiff residuals, then the pair is normalized. Represented
species negativity and invalid thermodynamics fail immediately. There is no
blanket carry reset. The characteristic top is not prescribed again: normal
endpoint preservation and transverse filtering precede the existing boundary,
primitive/halo refresh, and characteristic RHS path.

`run_air5_compensated_filter_gate.py` runs the coupled, viscous short-step matrix
on CPU/GPU, full/scalar, and single/x/y/z decompositions. This is a perturbation
near the characteristic top, not the stopped Mach4 precursor. The existing
`run_air5_characteristic_restart_gate.py` accepts `--filter` and
`--filter-workspace full|scalar` for filtered checkpoint-continuation tests at
1e-10 s. The lifecycle probe includes a low-part preservation/closure case;
the initial implementation with sequential Kahan additions failed that case
and was replaced by explicit subtraction residuals.

Candidate results (2026-09-28):

- `out/air5_compensated_filter_20260928/result.json`: 16 CPU/GPU cases,
  single/x/y/z decompositions and full/scalar, three updates at 1e-10 s.
  Maximum same-phase scaled difference: 1.44111e-15. Maximum checkpoint
  represented-state relative species closure: 3.05136e-16. Nonzero carry was
  exercised in every case. Initial reservoir T=Tv=1500 K.
- `out/air5_compensated_filter_restart_cpu_20260928/result.json` and
  `out/air5_compensated_filter_restart_gpu_segmented_20260928/result.json`:
  CPU NP=1 full and GPU NP=2 x-slab scalar continuations passed. All nine
  compared phase fields were identical; step, time, all 11 acq and 11 acc
  datasets were bitwise identical. Initial reservoir T=3000 K, Tv=1500 K;
  six updates at 1e-10 s. Four metadata/admission fault cases were rejected
  on each backend.
- The first GPU restart test stopped because the existing AIR5 monitor uses
  `status='new'` and refuses reusing `monitor/air5_probes.dat`. The driver now
  resumes into a separate directory, preserving the original monitor segment;
  the monitor implementation was not changed.
- Six filter-focused Python checks passed. The wider pair of historical
  contract suites had 31 passes and four source-string assertion failures;
  those same four failures were reproduced against Git HEAD, before these
  source changes. They are not recorded as a clean full-suite pass.
- `out/air5_compensated_filter_memcheck_20260928/validation/memcheck.*.log`:
  GPU NP=1 scalar, three coupled viscous updates, Compute Sanitizer memcheck
  reported zero errors. All nine phases at steps 0 and 2 matched the
  uninstrumented GPU scalar run exactly. This does not constitute racecheck
  coverage or a sanitizer pass for every MPI topology.

These results do not admit the paused Mach4 precursor for long filtered runs.
Filtered acoustic reflection, longer-window stability, and a matched short
continuation from that actual checkpoint remain separate gates. Keep the
validation opt-in and do not infer production readiness from the short matrix.

## Completed-Step Output Restart Extensions

`run_output_restart_validation.py --case channel --force feedback|fixed|frozen`
uses the bounded 16-cubed channel gate. `--case curve` uses the existing warped
static profile flatplate preset with explicit 543e convection, 643e diffusion,
and filtering. Restored curve cases delete the original grid/profile inputs,
requiring the frozen run resources to suffice. `--axis x|y|z` selects a slab;
`--filter-workspace scalar|full` selects the existing filter storage implementation.

`--legacy-statistics --backends cpu` includes all 44 CPU raw accumulated fields
and sampling identity (state role 6). This includes the user-approved missing
sgmam23 initialization fix, not a change in the statistics definition. CPU
NP=1/2 channel, CPU NP=1 curve, zero-sample NP=2 TGV restart and NP=2 y-channel
gates passed. With GPU this flag is restricted to `--case curve`: role 5 uses
17 components and original z-partition slots for plane/wall partial sums, not a
physical 3D statistics field. GPU x/y/z targeted same-topology gates passed.
Formal `--statistics` and legacy statistics require separate runs.

All these real-flow tests use 12 versus 5+7 completed steps and a 64 MiB
per-directory cap. Statistics compare periodic versus final-only checkpoints,
not fully disabled checkpoint output. New ordinary field/slice output, AIR5 SBLI,
repartitioned statistics and native rendering restart remain unaccepted.
See output redesign plan sections 10.13-10.14 for immutable evidence paths.

`test_checkpoint_export.py` checks the read-only basic-field exporter on an
asymmetric 9x7x5 synthetic bundle. The 19-test gate includes axis/slice checks,
strict source rejection, unit declaration checks, source-tree write prevention
and completion-publication fault injection. `run_checkpoint_export_validation.py`
uses existing 16-cubed CPU/GPU TGV/CURVE checkpoints and real ParaView XDMF3
readback with no rendering or new flow integration. The controlled reference
array cap is 2 MiB and each real test directory cap is 64 MiB.
Results: `out/or5_export_complete_20261001.xml` and
`out/or5_paraview_final_20261001/summary.json` (four sources, passed).
Use a ParaView build with file-reader support; the existing Catalyst-only
edition is insufficient. This passed with system ParaView 6.0.1 Xdmf3ReaderS,
not the old XDMF2 reader. Export and live GPU plane transfer are separate gates.

AIR5 role-7 cached fields are now included without EOS reconstruction:
density, velocity, pressure, T, Tv and Y_N2/Y_O2/Y_N/Y_O/Y_NO. Use `--units si`
with `run_checkpoint_export_validation.py`. The new
`out/or5_air5_export_20261001.xml` has 25 passed asymmetric-grid checks;
`out/or5_air5_paraview_20261001/summary.json` verifies four actual CPU/GPU
HBL/SBLI checkpoints (NP=1/2, x/y/z representatives) in ParaView 6.0.1.
All cached scalars, vectors, coordinates, planes and time match exactly, and
source checksums/mtimes remain unchanged. This closes the offline AIR5-export
limit, not independent live volume/slice scheduling or GPU plane transfer.

`test_output_fields.py` directly links root-production objects to test the
native bounded writer and actual GPU tile packer. Use completed AIR5 builds:

```bash
ASTR_FIELDS_BUILD="$PWD/build_cpu_probe" python3 -m pytest -q tests/gpu_validation/test_output_fields.py
ASTR_FIELDS_BUILD="$PWD/build_gpu_probe" ASTR_FIELDS_GPU=1 python3 -m pytest -q tests/gpu_validation/test_output_fields.py
```

The initial bounded-writer reports have 17 passed checks: six/twelve cached fields, NP=1/2,
x/y/z decomposition, volume and three planes, plus five rejection modes.
The 9x7x5 diagnostic uses warped coordinates and deliberate shared-node rank
differences to verify ownership. AIR5-named scalar encodings are not a physical
chemical state. Controlled host arrays are at most 2,160 bytes; GPU packing
workspace is at most 1,440/1,728 bytes for six/twelve fields. Each case is
below 4 MiB. Empty ranks issue collective HDF5 calls but do not pack/download.

Evidence: `out/or5_tiled_fields_cpu_metadata_20261001.xml`,
`out/or5_tiled_fields_gpu_metadata_20261001.xml`,
`out/or5_tiled_fields_readback_20261001/summary.json` (eight ParaView reads),
and `out/or5_tiled_fields_memcheck_hostcollectives_20261001/summary.json`
(NP=2 device packing, zero errors on both processes). The sanitizer invocation
disables UCC/HCOLL/CUDA collective plugins and GPU probing only to isolate
this kernel; it is not CUDA-aware MPI acceptance. The failed first API-probing
run and initial empty-rank/copy-count failures remain separate evidence.
Native diagnostic frame readback is available with
`pvpython --no-mpi run_checkpoint_export_validation.py --native-frame-check FRAME`,
where FRAME contains data.h5/data.xdmf and its reader_reference.npz.

The derivative-storage preparation in redesign plan 10.40 adds explicit
fourteen-slot field selections to the shared writer, while runtime derivative
requests remain rejected. Gradient names use velocity component first and
physical derivative direction second. Q_rs retains the full-strain rotation/
strain definition. Candidate `ASTR_DERIVED_FRAME_1` records, HDF5 field
inventories/definitions/units and single-segment/parent-chain layouts are
checked without interpreting encoded fixture values as physical derivatives.

Reports `out/or5_derived_layout_cpu_20261001.xml` (30 layout/rejection checks),
`out/or5_derived_basic_regression_20261001.xml` (51 basic regressions),
`out/or5_derived_index_20261001.xml` (96 index/chain checks), and
`out/or5_derived_writer_readback_20261001.xml` (two actual shared-geometry
writer/ParaView checks) pass. The last pair reads a synthetic volume and three
planes, with exact FP64 fields, vectors, coordinates and time. Reference arrays
are 67128/31508 bytes; directories are 344751/277004 bytes. Controlled writer
packing arrays stay within 2160 bytes. These are CPU/MPI file-format tests,
not numerical, GPU-kernel, AIR5 or production derivative admission.

```bash
ASTR_FIELDS_BUILD="$PWD/build_cpu_probe" python3 -m pytest -x -q tests/gpu_validation/test_output_fields.py -k derived
python3 -m pytest -x -q tests/gpu_validation/test_output_series_repair.py tests/gpu_validation/test_output_parent_series.py
```

The focused writer command now selects 32 checks including the two added
actual-reader cases. Choose new output/basetemp paths when retaining evidence;
do not replace historical reports. Real TGV gradients, private periodic halo,
same-phase CPU/GPU comparison, exact restart and GPU slice-only transfers still
require the proposed 10.39 gate before opening the runtime path.

Plan 10.41 now implements private CPU/GPU velocity-halo providers, using the
existing sixth-order operator and tiled selected-field downloads. Final probes
use 11³ manufactured nodes, NP=1/2 x/y/z, and an independent discrete-wavenumber
reference. Solver velocity and deliberately invalid solver halos remain bitwise
unchanged. All fourteen fields, selected subsets and slice-only callbacks are
covered, with no RK integration or runtime admission. Source/test snapshots,
metrics and controlled host arrays stay under 2 MiB; each test directory stays
under 4 MiB. Probe-only full velocity observation downloads are not included
in the production callback's selected-field transfer counter. Halo face traffic
is also separate from that counter, and full reusable halo capacities are
charged to the workspace budget.

`out/or5_private_snapshot_cpu_20261001.xml` has 13 passes and four GPU skips;
`out/or5_private_snapshot_gpu_20261001.xml` has 17 passes. Three opt-in memcheck
cases in `out/or5_private_snapshot_memcheck_20261001.xml` have six zero-error
process logs. Twelve original basic packing cases pass in
`out/or5_private_basic_gpu_20261001.xml`. These are not true TGV restart,
CUDA-aware MPI, wall/CURVE/AIR5, renderer or physical-validation gates.

```bash
ASTR_FIELDS_BUILD="$PWD/build_cpu_probe" python3 -m pytest -x -q tests/gpu_validation/test_output_fields.py -k private_
ASTR_FIELDS_BUILD="$PWD/build_gpu_probe" ASTR_FIELDS_GPU=1 python3 -m pytest -x -q tests/gpu_validation/test_output_fields.py -k private_
ASTR_FIELDS_BUILD="$PWD/build_gpu_probe" ASTR_FIELDS_GPU=1 ASTR_FIELDS_MEMCHECK=1 python3 -m pytest -x -q tests/gpu_validation/test_output_fields.py -k private_derivatives_memcheck
```

Plan 10.42 wires these callbacks into completed-step frames, workspace planning,
derived FRAME records and time indexes, but runtime derivative requests still
abort pending the approved TGV gate. The existing ASTROA02 flags are retained;
Q selection includes its divergence companion. Shared geometry does not acquire
derived fields. No environment or test bypass is added.
`out/or5_derived_switches_20261001.xml` has 67 parser/collective passes, including
16 canonical enabled/disabled layouts. Resource-query extraction is checked by
one CPU and one GPU NP=2 z manufactured test. The CPU/GPU basic-path reports
`out/or5_derived_wiring_basic_cpu_20261001/summary.json` and
`out/or5_derived_wiring_basic_gpu_20261001/summary.json` each pass the existing
16³ NP=2 z continuous/restarted state/statistics, output-switch, schedule and
ParaView frame/series gates. None executes the new derived runtime branch.
The 64 MiB runtime test budget is enforced per solver-case leaf, not for the
entire multi-case report directory. Complete derived-field lifecycle acceptance
and admission remain pending; no stage Git commit has been made.

These are writer/provider gates, not a solver model or complete live output.
The subsequent real TGV runtime gate now admits basic volume/slices, including
independent schedules, multi-plane aggregation and sealed frame publication.
Other runtime case families and derived fields remain rejected.

`run_output_archive_validation.py` runs the actual 16-cubed ASTR solver with
4096-byte field tiles and per-directory/per-controlled-buffer 64 MiB limits.
Set MPIEXEC to the MPI launcher matching the selected root build:

```bash
python3 tests/gpu_validation/run_output_archive_validation.py \
  --executable build_gpu_probe/bin/astr --mpiexec "$MPIEXEC" \
  --output tests/gpu_validation/out/tgv_archives_gpu_new \
  --backends gpu --ranks 1 2 --readback
```

The output directory must be new. `--axis y|z`, `--mode time`, `--statistics`,
`--initial-restart`, `--schedule-checks`, `--keep 1`, and `--budget-checks`
select the documented representatives. Initial-source preparation requires
keep=2; schedule overrides require steps mode and the step-5 source; reduced
device-budget checks require statistics off. Invalid combinations fail before
starting solver tests. `--readback` additionally needs /usr/bin/pvpython with
XDMF3; it reads copies outside sealed frames and does not render images.

Ten positive CPU/GPU NP=1/2 combinations cover x/y/z, step/time triggers,
initial/final output, exact same-topology 5+7/0+12 restart, selected accumulated
statistics and unchanged flow when output is toggled. Eight CPU/GPU NP=1/2
volume/grouped-plane products were read in ParaView 6.0.1 against the
authoritative same-phase checkpoint/cache values. keep=1 retires checkpoints
without removing either archive. Only slices do not create a full volume,
checkpoint or checkpoint geometry, and download 41,616 field bytes per frame
for three 17x17 planes (full volume: 235,824). A 512-byte GPU packing budget
still writes identical planes using at most 480 device/720 host tile bytes;
47 bytes reject before publication. Workspace counts do not claim third-party
MPI/HDF5 peak memory bounds.

Exact report directories and failed-checker evidence are in redesign plan
section 10.24. Runtime examples and format limitations are in
scripts/output/README.md and input.output.tgv.example. Each run still needs a
new output root and uses segment00000000; temporal indexes, reusable roots,
shared coordinates, other case families, derived fields and Catalyst restart
coupling remain pending. No long or remote job is part of this gate.

The following shared-coordinate increment removes the per-frame geometry copy:
each product has resources/data.h5 with coordinates/metadata only. Live HDF5
frames omit coordinates; XDMF uses explicit relative references and sealed
RESOURCES bind the shared file by bytes/CRC-64. Slice-only resources include
only the selected planes, not a volume/metric/halo field. Each new run still
creates its own resource; this is not existing-root or cross-segment reuse.

`test_output_archive_resources.py` validates real relocated volume/slice frame
bundles and rejects missing, mutated or symlink coordinates through the actual
Fortran validator. It never advances the solver or edits source artifacts.
Set ASTR_ARCHIVE_RESOURCE_SOURCE and ASTR_CHECKPOINT_BUNDLE_PROBE explicitly
when the documented local defaults differ. Build the probe through root CMake
in a BUILD_TESTING=ON build, then run the eight checks (each directory <2 MiB).
48 original CPU/GPU provider checks and seven real shared-coordinate runtime
representatives passed. Exact artifacts and build scope are in plan 10.25.

The next increment adds per-segment series.frames/series.xdmf, indexing only
published frames without additional field copies/downloads. `--readback` now
also reads every time in each volume/grouped-slice sequence through ParaView
6.0.1 and compares FP64 data, coordinates, vectors and time with sealed frames.
Final-frame comparisons still use authoritative checkpoint caches. CPU NP=1,
GPU NP=1/2 x, CPU NP=2 y time/statistics/0+12, and GPU NP=2 z retention/override/
budget representatives pass; source reports are in plan 10.26. Output roots
remain new and independent: this is not a cross-segment index.

`test_output_fields.py` now has 37 checks per CPU/GPU production-object build:
24 existing checks plus 13 valid-series/invalid-record checks. The metadata
probe does not advance a fluid. `test_output_series_filesystem.py` has 16
tests of the production C replacement helper, including symbolic/hard links,
directories and unsafe names. Individual replacements are atomic, but no
two-file transaction, crash repair or target-filesystem durability is certified.
Full sequence references use one frame at a time (<=2 MiB reference arrays);
whole readback case directories are checked against the approved 64 MiB limit.

The basic-archive gate now accepts `--case channel|curve|dynamic|air5hbl|air5sbli`
using the existing bounded case preparers and new-checkpoint admission rules.
Use `--legacy-statistics` for CPU raw moments, GPU CURVE compact statistics or
AIR5 mean44; GPU channel legacy statistics remain unsupported. Formal
`--statistics` is TGV-only. AIR5 schedule/budget rejection permutations are
not in this first archive matrix; their unsupported CLI combinations reject.
`--filter-workspace full` covers the existing alternative storage path.
Dynamic sources default to 12 frames, with a bounded 12:256-frame test range
matching the existing restart runner, not a production input limit.

Plan 10.27 lists 22 current real combinations: CPU/GPU NP=1/2 for channel,
static/dynamic CURVE and fixed AIR5 HBL/SBLI, plus a two-rank TGV regression.
All compare exact same-backend restart and output-toggle state, selected
statistics, final-frame/checkpoint correspondence and actual full-sequence
ParaView reads (264 time points). Maximum readback case size is 39,650,162
bytes; selected-plane GPU downloads remain 41,616 or 83,232 bytes for six or
twelve fields. No long or remote physical benchmark was run.

Repeated generation also exposed NVHPC 26.1 internal-read EOF affecting an
external source at NEWUNIT=-99. The series parser no longer deliberately
requests an extra item past the row. A metadata-only 50-generation regression
covers both unit parities. CPU/GPU each pass 15 targeted series checks;
24 existing basic packing checks are unchanged. Failed real-channel/index
diagnostics and short-source dynamic failure are retained, with paths in 10.27.

The publication/retention increment in plan 10.28 adds preflight protection for
LATEST and a common checkpoint failure context. `test_checkpoint_bundle.py`
has 26 affected publication/retention checks covering keep=1/2, owned/protected
directories, link rejection and injected EIO. The production C preflight and
replacement tests in `test_output_series_filesystem.py` now pass 26 checks.
The state probe additionally verifies NP=1/2 context reporting and clearing.

`test_output_publication_failure.py` uses the real 16-cubed solver, NP=2 x,
continuous 12 steps and step-10 failures at candidate creation, batch rename,
LATEST rename, old COMPLETE removal or old payload removal. Each CPU/GPU
case restores the correct complete point (5 or 10) in a new process and
compares final state/cache values exactly. A sixth case per backend checks
first-write failure after restart, correct source identification and immutable
source files. Build through root CMake, then run:

```bash
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" \
  ASTR_OUTPUT_MPIEXEC="$MPIEXEC" python3 -m pytest -x -q \
  tests/gpu_validation/test_output_publication_failure.py
```

The CUDA-root test runs both runtime backends (12 checks); CPU-only coverage
selects `-k cpu` and the CPU build (6 checks). Each test has a separate output
directory below 64 MiB; sharing every case under one capped directory was a
test setup error, preserved in the first CPU context XML. Current maximum is
13,425,020 bytes. Reference cases and test-only preload libraries are separate.
The preload is compiled temporarily, targets exact test paths and returns EIO;
it is not linked into production. It does not simulate power loss, disk full,
MPI process loss or external filesystem races. The 26 synthetic bundle checks
remain below 4 MiB each. Actual TGV/AIR5 archive regressions and readback
reports, root-build scope and remaining OR6 limits are listed in plan 10.28.

### Same-root archive segments and offline index repair

`test_output_archive_segments.py` checks own-root new generations against
12-step continuous versus 5+7 exact restart, old-file fingerprints, shared
coordinate reuse, keep=1 source protection and actual ParaView series readback.
TGV CPU/GPU NP=2 x, CPU CURVE NP=2 y and GPU AIR5 HBL NP=2 z pass; two TGV
changed-plane requests are rejected if the shared coordinate resource lacks
the new plane. All use existing 64 MiB directory/controlled-buffer limits and
2 MiB/frame reader references. This is not arbitrary historical forking or
repartition. The geometry-only probe is host-side, even in a CUDA root build.
Evidence and the three root-build variants are in redesign plan 10.29.

```bash
python3 -m pytest -x -q tests/gpu_validation/test_output_series_filesystem.py
ASTR_FIELDS_BUILD="$PWD/build_gpu_probe" python3 -m pytest -x -q \
  tests/gpu_validation/test_output_fields.py -k geometry
python3 -m pytest -x -q tests/gpu_validation/test_output_archive_segments.py
```

For the geometry probe, `ASTR_FIELDS_BUILD` selects the completed root build;
`ASTR_FIELDS_GPU=1` additionally enables GPU field packing in ordinary field
tests, but the geometry-only check still uses host coordinates. Set
`ASTR_OUTPUT_MPIEXEC` to the build's matching MPI executable when necessary.

`test_output_series_repair.py` uses 9x7x5 six-field volumes/twelve-field grouped
slices, below 4 MiB/test. It checks the offline production repair tool's default
read-only mode, exact layout references, CRC/type/unit/clock refusal, partial
candidate preservation, index failure and bounded catalogs. It never computes
a flow solution. `test_output_series_repair_runtime.py` copies immutable accepted
TGV CPU NP=2 x and AIR5 HBL GPU NP=2 z products from plan 10.28, repairs only the
copies and reads every time through ParaView 6.0.1. No solver is restarted.
Each actual-product test, including reader copies, remains below 64 MiB.

```bash
python3 -m pytest -x -q tests/gpu_validation/test_output_series_repair.py
ASTR_OUTPUT_REPAIR_REFERENCE_ROOT="$PWD/tests/gpu_validation/out" \
  python3 -m pytest -x -q tests/gpu_validation/test_output_series_repair_runtime.py
```

The latter skips with an explicit reference path if the immutable artifacts
are absent; a skipped reference is not a passed readback gate. Current evidence
`out/or5_series_repair_final_20261001.xml` has 44 passed, no skipped checks,
including 24 actual product timepoints. Source and all non-index copied files
remain unchanged. CRC scans full HDF5 files without loading flow arrays.
This is explicit stopped-segment repair, not a concurrent writer, fsync,
cross-segment, resource-RSS peak or production-scale I/O acceptance gate.
See plan 10.30 and scripts/output/README.md for the write opt-in and failures.

`run_output_air5_restart_validation.py` covers the bounded 16-cubed AIR5 HBL
completed-coupled-step restart. Role 7 stores q11, physical carry11 and all
primitive/species caches independently. Domain/profile/initial-field resources
are frozen and deleted from the restored case's original input directory.
The test compares 12 continuous versus 5+7 updates, periodic versus final-only
saves, nonzero carry, scalar statistics tails and source/top/compensation
contract rejection. The conserved diagnostic baseline is not connected and
its enabled mode is rejected before stepping, rather than reset on restore.

Latest evidence: `out/or3_air5_cpu_only_20261001/summary.json` (CPU-only,
CPU NP=1/2 x), `out/or3_air5_gpu_guard_20261001/summary.json` (GPU NP=1/2 x,
scalar), `out/or3_air5_gpu_y_full_20261001/summary.json` (GPU NP=2 y/full).
All passed at dt=1e-10 s with chemistry, viscosity, filtering, compensation and
characteristic top enabled. Per-case directory cap is 64 MiB; controlled
packing plus extras and GPU carry staging is charged against the 64 MiB host
budget, with no new device scratch allocation. This is same-backend exact
continuation, not a CPU/GPU bitwise equivalence or long-time physical gate.
AIR5 accumulated statistics are covered below; repartition remains separate. SBLI
resource/restart coverage added below supersedes the earlier HBL-only limit.
`out/or3_air5_provider_20261001.xml` has six passed MPI provider checks;
`out/or3_air5_boundary_interfaces_fixed_20261001.xml` has 30 passed checks.
The earlier interface report had stale source-string expectations, now updated
without relaxing numerical thresholds.

`--case sbli` freezes the additional incident-shock resource. The existing
AIR5 selective path is `643e`, `recon_schem=3`, `lchardecomp=f`; the value 3
is a selector, not a claim that every interface has third-order reconstruction.
The SBLI test defaults to this selector; HBL defaults to 5 (central path).
Generic `543e` is not admitted by AIR5 GPU transport. Evidence:
`out/or3_air5_sbli_selective_cpu_x_20261001/summary.json` and
`out/or3_air5_sbli_selective_gpu_x_20261001/summary.json`, CPU/GPU each NP=1/2 x.
Same-backend q/carry/cache, geometry, control and configuration match exactly.
Tau/source/compensation/top-mode/incident corruption and unregistered conserved
diagnostic modes are rejected without publishing a new COMPLETE. Selective
y/z/full restart combinations with mean statistics are covered below;
developed SBLI physics remains unverified here.
The earlier `or3_air5_sbli_*` x/y/z reports use selector 5; they must not be
presented as selective shock-path evidence. Details and bounded resource sizes
are in the output redesign plan, section 10.19.

`test_meanflow_air5_transport.py` links the tiny sampler against production
objects from a root CMake build with AIR5 enabled. It verifies the approved
AIR5 meanflow viscosity correction against the existing diffusive flux,
unchanged dimensional/nondimensional nonreacting branches, duplicate-sample
suppression and invalid AIR5 rejection. It does not model a new flow solver.
Run after completing the build, without rebuilding objects concurrently:

```bash
ASTR_MEANFLOW_BUILD="$PWD/build_cpu_probe" python3 -m pytest -q tests/gpu_validation/test_meanflow_air5_transport.py
```

Each host-only build variant has eight passed checks. To also invoke the actual
GPU accumulator on the same prepared states, use a completed CUDA/AIR5 build:

```bash
ASTR_MEANFLOW_BUILD="$PWD/build_gpu_probe" ASTR_MEANFLOW_GPU=1 python3 -m pytest -q tests/gpu_validation/test_meanflow_air5_transport.py
```

`out/or3_air5_mean_kernel_enabled_20261001.xml` has eight passed checks, including
NP=1/2 actual GPU accumulation, all 44 fields and sample metadata. GPU/CPU
same-state field differences use 1e-14*max(1,reference maximum absolute value).
This is not a full-flow CPU/GPU bitwise equality claim.

New output now supports raw44 AIR5 `lavg` for the fixed, dimensional, viscous
HBL/SBLI cases, preserving legacy first-stage sampling rather than redefining
it as completed-step sampling. `--mean-statistics --sample-interval 1|2|3`
checks continuous/restarted/final-only accumulated state exactly and confirms
statistics activation does not change flow. `--initial-restart` checks the
zero-sample step-0 batch; `--mode time` selects physical-time checkpointing.
There are no new species or Tv moments and no change to legacy GPU output.

Twelve bounded combinations passed at 16-cubed, dt=1e-10 s: HBL CPU/GPU NP=1,
selective SBLI CPU/GPU NP=1/2 x, NP=2 y/full, NP=2 z/prescribed/time/scalar,
and initial restart on CPU NP=1 and GPU NP=2 y/full/time. Mean sample counts
are 11/5/3 for intervals 1/2/3. Exact immutable report paths, transferred bytes
and controlled array budgets are in the output redesign plan, section 10.21.
The driver rejection matrix additionally checks changed statistics activation,
sample interval and insufficient GPU statistics allocation budget.

`out/or3_air5_mean_memcheck_hostmpi_20261001.log` reports zero errors for the
NP=1 production GPU sampler. MPI CUDA probing was disabled only for that
diagnostic run after the default HPC-X run reported MPI/UCX CUDA API errors.
This does not validate CUDA-aware transport or alter production MPI settings.

## Periodic Filter State And Repartition Gate

After explicit approval, CPU and GPU rebuild all physical-node primitives
after every filter only for the non-reacting, five-variable, all-periodic,
explicit spatial path. GPU covers first-stage preparation and later RK stages,
retaining explicit synchronization. AIR5, compact and physical-face paths are
unchanged. The former pre-filter interior/post-filter shared-face mixture
caused a topology-dependent result; see output redesign plan 10.32-10.33.

`test_output_repartition_runtime.py` contains the approved 16-cubed TGV matrix:
twelve CPU/GPU NP=1 <-> NP=2 x/y/z continuous-12 versus 5+7 comparisons at
2e-10; nine partition controls (CPU, GPU scalar/full storage); four completed
CPU/GPU field comparisons; two same-topology bitwise checks; two unvalidated
no-filter repartition rejections. The expanded suite also covers twelve NP=2
slab-direction changes, a GPU z->y full-workspace representative and a CPU
NP=4 rejection with valid local halo extents. The matrix no longer
skips by default; ASTR_OUTPUT_REPARTITION_CANDIDATE is obsolete.

```bash
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" python3 -m pytest -q tests/gpu_validation/test_output_repartition_runtime.py
```

The final CUDA build report `out/or4_cuda_admission_final_20261001.xml` records
20 passed checks (matrix excluding the nine earlier partition controls).
`out/or4_cpu_admission_final_20261001.xml` records seven passed CPU-only
checks (six bidirectional cases and no-filter rejection). Earlier repaired
CPU/GPU reports include all nine partition controls. Maximum repartition
state/cache difference is 2.2737367544323206e-13; statistics difference is
1.7763568394002505e-15. Complete-step CPU/GPU state difference is
2.5579538487363607e-13. Each case is below 64 MiB; GPU statistics compare
unique periodic nodes rather than padded upper endpoint planes.

The runtime admits changed topology only for generated-grid TGV, ninit=0,
643e/643e, filtering/diffusion enabled and no legacy mean/compact statistics.
Changed topology is limited to NP=1<->2 and the six ordered NP=2 slab-direction
changes. Other settings/families and larger rank counts remain exact-topology-only.
Saved or current render enablement rejects repartition, even when an explicit
output override disables the saved renderer.
`out/or4_filter_phase_exact_output_20261001/summary.json` confirms four
same-topology CPU/GPU NP=1/2 exact-restart/output-switch combinations. The
bounded scalar memcheck in `out/or4_filter_phase_memcheck_20261001/memcheck.log`
reports zero errors, with MPI CUDA probing disabled for that diagnostic only.
The original failed gate and guard-only reports are retained as history, not
current acceptance. No tolerance, filter coefficient, or spatial/RK operator
was changed. The first gate's reports remain immutable historical evidence,
not qualification of every later executable.

The approved direction increment (plan 10.37-10.38) is recorded in:

- `out/or4_direction_cuda_matrix_20261001.xml`: 21 passed checks, including all
  twelve directions, two exact restores, four rank-count regressions, two
  no-filter rejections and one larger-rank rejection.
- `out/or4_direction_cpu_fixed_20261001.xml`: eight CPU-only checks, with raw
  runs preserved in `out/or4_direction_cpu_fixed_files_20261001`.
- `out/or4_direction_gpu_full_20261001.xml`: GPU z->y full-workspace gate.
- `out/or4_direction_render_guard_20261001.xml`: eight same-topology rendering
  and saved/current render-history rejection checks.

All direction final fields, velocity vectors, coordinates and three indexed
planes match same-phase checkpoints exactly. CPU x->z and GPU z->y scalar/full
representatives also read their actual series through ParaView Xdmf3ReaderS.
Maximum state/cache and statistics errors are 2.5579538487363607e-13 and
1.7763568394002505e-15. Sample identities, duration, clocks and schedules are
exact; the largest aggregate test directory is 44572494 bytes, below 64 MiB.
These close only the bounded periodic TGV directions, not all OR4/OR6.

```bash
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" python3 -m pytest -x -q -o junit_family=xunit1 tests/gpu_validation/test_output_repartition_runtime.py -k 'slab_direction or keeps_exact or unapproved_larger or unvalidated_no_filter'
```

The focused command selects 18 checks; the recorded 21-check report additionally
contains four representative rank-count regressions and predates the separate
full-workspace test. When retaining raw evidence, choose a fresh `--basetemp` path;
pytest replaces that directory on a new run. Do not rebuild/relink an executable
between creating and restoring test checkpoints: the content fingerprint is
part of the continuation contract, including for source-equivalent relinks.

## Explicit Native Parent-Chain Indexes

`test_output_parent_series.py` covers SEGMENT-2 binding, explicit parent
cutoffs, unrelated branches, deep chains, whole-tree moves, external parent
roots and rejection of cycles, aliases, changed input/parents, unsupported
legacy lineage, changing layouts, duplicate clocks and exceeded catalogs.
SEGMENT-1 remains valid for single-segment repair, never inferred as a chain.

`test_output_archive_segments.py` additionally checks CPU/GPU 16-cubed NP=2
same-root and historical/external-root continuations, three-segment chains,
checkpoint-bound parent tampering refusal, CPU CURVE y and GPU AIR5 HBL z.
Each accepted merged time is read through ParaView Xdmf3ReaderS, comparing
saved FP64 scalars, velocity vectors, coordinates and actual clocks exactly.
Sources are fingerprinted before/after. New archive histories use ASTROA02;
exact schedule comparison excludes only the separately checked 48-byte segment
receipts, not clocks, intervals, sampling or accumulated state.

```bash
python3 -m pytest -x -q tests/gpu_validation/test_output_parent_series.py tests/gpu_validation/test_output_series_repair.py tests/gpu_validation/test_output_series_filesystem.py
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" python3 -m pytest -x -q tests/gpu_validation/test_output_archive_segments.py
```

Plan 10.34 records 119 current synthetic checks, CPU-only/native CUDA physical
reports and the extra three-chain/tamper/repartition report. Merged readback
covers 110 time points, with 28 additional single-segment points. Controlled
reference arrays remain below 2 MiB/frame and case directories below 64 MiB.
Only the approved periodic TGV topology change is admitted; these generic
archive checks do not certify CURVE/AIR5 repartition, new physics or production
I/O performance. The combiner is explicit/offline, with a fresh output directory
and immutable sources, not an automatic live index repair or durability gate.

## AIR5 GPU Conservation Baseline And Binary Tail Checks

`test_output_air5_conservation.py` exercises the existing GPU-only diagnostic
through native checkpoints. Its 29 checks cover continuous 12 versus 5+7 steps
(HBL NP=1 scalar and selective SBLI NP=2 z/full), 0+12 initialization (NP=1
and NP=2 y/full), exact baseline/history/future diagnostic rows, enable-switch
and CPU rejection, sealed-parser corruption tests, allocation/packing budget
rejection, simultaneous volume/slice output and Compute Sanitizer memcheck.
The protected step-five seed is from the same continuous reference; a
test-only publication hook creates PROTECT after its successful rename,
without a timing watcher or modifying the numerical state.

```bash
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" python3 -m pytest -x -q -o junit_family=xunit1 tests/gpu_validation/test_output_air5_conservation.py
python3 -m pytest -x -q tests/gpu_validation/test_checkpoint_stream_tail.py
```

`out/or3_air5_conservation_all_final_20261001.xml` records 29 passed tests.
The bounded memcheck uses explicit host-MPI components (ob1, pt2pt,
self/vader/tcp), disables optional MPI CUDA probing, and requires zero errors;
it does not qualify CUDA-aware MPI. The first default HPC-X diagnostic failed
with eleven UCX CUDA context API errors and is retained, not suppressed.
The sanitizer-only successful report is also in
`out/or3_air5_conservation_memcheck_final_20261001.xml`.

Native diagnostic scratch is 5632 bytes for NP=1 and 2816 bytes/rank for these
NP=2 slabs; its state member is 168 bytes and each evaluation downloads 88
bytes/rank. The largest recorded 5+7 continuation directory is 21964055 bytes;
the largest single-test root including runs/source copies is 66073804 bytes.
Missing-member rejection cases retain the copied original template grid
(51522584 bytes); it is budgeted, not a newly archived flow.
Joint archive output uses host/device tile buffers of 4080/3264 bytes;
physical volume/three-slice field downloads are 471648/83232 bytes per frame.
These are controlled arrays and payload counts, not RSS or third-party peaks.
All tests remain within the approved 64 MiB per-directory/resource bounds.

The baseline integrates stored q, at transport entry, and phase-1 samples are
after transport but before the second chemistry half-step. These are not
complete-step q-carry totals or physical open-boundary conservation gates.
Fixed-order reduction is enabled only by the new output path; legacy atomic
behavior is preserved. Each resumed diagnostic is exclusive and origin-named;
same-origin collisions fail, rather than silently overwrite old records.

An earlier exact-initialization gate failed because atomic addition reordered
unchanged FP64 values. A later corruption gate exposed Fortran partial-read
EOF accepting a 1-7 byte binary tail. Both failures were diagnosed without
loosening the gate. New binary readers use exact stream position/file-size
checks for control, archive history, AIR5 configuration, the diagnostic and
frozen inflow indexes. `out/or6_stream_tail_final_20261001.xml` records nine
NP=2 probe tests covering exact, truncated and short/long-tailed records.
This is I/O integrity validation, not a new numerical correction.

Affected runtime regressions are recorded in
`out/or6_stream_runtime_regression_20261001.xml` (four CPU/GPU TGV exact
restore/fault-recovery checks), `out/or3_dynamic_stream_regression_20261001/summary.json`
(two CPU/GPU dynamic-inlet cases) and
`out/or3_air5_disabled_conservation_final_20261001/summary.json` (CPU/GPU NP=2
y/full HBL with mean44, diagnostic disabled). The latter also requires exact,
canonical empty GPU diagnostic state; disabling the mean sampler preserves flow.

## Optional Catalyst Backend Admission

### Native Completed-Step TGV Render Continuation

`test_output_insitu_restart.py` uses the root CUDA/Catalyst executable and
the already approved TGV EGL preset; it does not change solver kernels,
statistics, the renderer precision patch or physical acceptance definitions.
Cases are 16-cubed, 12 continuous versus 5+7 steps, NP=1 step cadence and
NP=2 x time cadence. The implementation and executable paths can be overridden:

```bash
ASTR_OUTPUT_INSITU_EXE="$PWD/build_insitu_gpu/bin/astr" \
ASTR_OUTPUT_INSITU_BACKEND="/path/to/paraview/lib/catalyst" \
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_insitu_restart.py
```

`out/or5_insitu_native_fixed_binary_20261001.xml` records 33 passed checks.
The suite compares saved q/cache/statistics exactly, control/schedule bytes,
JPEG pixels and VTK pieces, frame-download counts, actual EGL/NVML GPU UUID,
nonblank 800x600 images and the existing 2e-10 analytic crossing gate. It
also checks final checkpoint 12->13 and 12->12 continuation, explicit render
enable/disable/frequency/initial overrides without extra statistical sampling,
render-only continuation, missing control and five sealed-parser corruptions.
The required native control is 144 bytes inactive or 313 bytes active.

Local observed host/device increments remain below 4 GiB/node and 2 GiB/physical
GPU with at least 1 GiB free. Forty-seven observer files/211 phase records show
max increments 1072877568/279449600 bytes and min free 18812960768 bytes.
The maximum test root is 56937734 bytes, below 64 MiB; the initial failed
render-only comparison accidentally dumped unnecessary oracle fields and
exceeded this aggregate cap. Its corrected no-render reference explicitly
disables that output, not the gate or saved products. Earlier failure reports
are retained. Payloads are 432344/228888 bytes per render frame/rank for
NP=1/NP=2 x; CPU extraction and pipeline copies remain, and no performance or
long-sequence memory claim follows from this short suite.

`out/or5_insitu_shared_fixed_binary_20261001.xml` has eight passed
non-Catalyst checks: CPU/GPU exact TGV/direction refusal, publication failure
recovery, AIR5 diagnostic/mean44 continuation and CURVE/AIR5 field/slice/parent
readback. Four root build variants and the schedule probe pass. Ordinary native
output does not load Catalyst. Only TGV same-topology rendering is admitted;
other renderer cases remain separate pending work. Bounded real TGV file-derived
fields are now covered by the separate gate below, not by this renderer suite.
The earlier 33/8-test reports are retained. A subsequent CUDA relink changed
the executable fingerprint, so the fixed-binary reports above use new references
and verify the saved executable identities against the current files. Build
before generating references; do not relink between seed and restoration.
The native executable contract is not bypassed for unchanged source code.

See [INSITU_ADMISSION.md](INSITU_ADMISSION.md) for the default-off build option,
`astr insitu-check` configuration, and MPI regression tests. This checks backend
loading and lifecycle only; it does not sample or render ASTR fields.

## Variable-Step Exact Native Continuation

`test_output_variable_clock.py` checks real 16-cubed periodic TGV with the
approved controller timestep values, FP64, sixth-order explicit operators,
filtering, viscosity and explicit GPU synchronization. No solver update or
controller reload cadence is modified. The Linux test preload library redirects
only a test-owned controller path to prepared complete input records. Startup
still reads dt=0.001, completed step 1 selects 0.002, and step 6 selects 0.0005.
The resulting 12 used steps are one at 0.001, five at 0.002 and six at 0.0005.

```bash
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_release_restart_cpu/bin/astr" \
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_variable_clock.py -k 'exact_continuation and cpu'

ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" \
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_variable_clock.py -k 'exact_continuation and not cpu'
```

Each backend covers NP=1 and NP=2 x/y/z. A test root contains five short runs:
continuous 12, seed 5, resume to 12, archives off, and slice-only continuation.
New-format checkpoint q/cache/statistics and continuation controls must match
exactly. Seed time is 0.009 and saved dt_next is 0.002, deliberately different
from restarted startup input; every actual used clock and reload identity is
checked. Final time is 0.014000000000000004 and clipped statistical duration is
0.011. Time archives use periods 0.002/0.003 with initial/final frames; all
resumed HDF fields match continuous same-phase frames. Actual ParaView reading
of resumed field/slice time indexes covers five frames per test, forty total.

`out/or6_variable_clock_cpu_20261001.xml` and
`out/or6_variable_clock_gpu_20261001.xml` each record four passed gates. The
same-named roots retain per-test inputs, logs and result.json including the
fixed executable SHA256. No rebuild is performed between seed and resume.
The maximum complete test root is 53795884 bytes, below 64 MiB. Host/device
controlled budgets remain 64 MiB with the existing 1 GiB device reserve;
maximum field tile allocations are 4032/2688 bytes. GPU slice-only downloads
83232 field bytes for two frames of three 17-by-17 planes. This excludes
checkpoint/statistics/halo transfers and does not represent RSS, third-party
peaks or production performance.

The first run stopped because NVHPC action='read' with default status opens
via fopen("r+")/O_RDWR; matching only O_RDONLY did not activate the replay.
A minimal Fortran reader, strace and GDB isolated this test-driver defect.
Only the preload matching changed, not the solver. Eight C API/mode checks
and that exact Fortran statement verify path scoping, source preservation
and read order. `out/or6_variable_clock_helpers_20261001.xml` records seventeen
helper/clock checks; `out/or6_variable_clock_fault_regression_20261001.xml`
records one existing publish-failure recovery with replay unset. These close
the bounded variable-clock gate, not OR6 as a whole or pending real derived
fields beyond the separate gate below, other boundaries, CURVE/AIR5 or renderer
combinations.

## Real Complete-Step Native Derivatives

`test_output_derived_runtime.py` implements the approved plan 10.39/10.44 gate:
16-cubed cells (17-cubed nodes), generated periodic Cartesian TGV, dt=1e-3,
dimensionless FP64, 643e/643e, viscosity/filter enabled and explicit sync.
The root binaries must be built before generating references and must not be
relinked between seed and restore. Default paths and local evidence:

```bash
ASTR_OUTPUT_CPU_EXE="$PWD/build_release_restart_cpu/bin/astr" \
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" \
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_derived_runtime.py
```

`out/or5_real_derived_runtime_20261001.xml` records 21 passes. Eight CPU/GPU
NP=1/2 x/y/z tests compare continuous 12, seed 5, resume to 12, archives off,
and slice-only continuation. Saved q/cache/statistics are exactly equal;
control and schedule payloads match while segment identities remain distinct.
The independent reference uses saved same-phase physical velocity, NumPy rolls
and sixth-order central weights, then combines the fourteen physical diagnostics.
It is not a continuous analytic derivative or ParaView default gradient.
Four CPU/GPU checks compare every published derived frame/plane; six selected
layout checks cover gradient-only, curl-only and Q/divergence on both backends.
Layout changes need explicit override; same-layout parents read as a two-segment
four-time sequence and incompatible parents are rejected. Two channel tests
prove that basic-output admission does not silently admit derived fields.

The eight final-state reports record maximum independent-reference error
3.9968028886505635e-15; all-frame CPU/GPU maximum is 1.0069852937610868e-14,
below the approved absolute 2e-10. ParaView XDMF3 actually reads 64 product-time
frames across this matrix. Each selected field/plane is also checked against
the checkpoint, including coordinates, units, definition and time metadata.
All-selected slice-only GPU continuation downloads 277440 field bytes for
two frames of three 17-by-17 planes, twenty columns each. Private device
velocity/halo storage and host-staged face communication still exist and are
not included in this field-download count. Maximum controlled host/device
allocations are 475520/613440 bytes; allocation preserves 1 GiB device free.
The complete per-test root maximum is 58240301 bytes, below 64 MiB. These
limits are not MPI/HDF5/ParaView transient peak or production-memory guarantees.

`out/or5_basic_cpu_regression_20261001/summary.json` and
`out/or5_basic_gpu_regression_20261001/summary.json` retain the unaffected basic
NP=2 z exact continuation, output-switch and actual single/series readback gates.
Their aggregate roots are 44674980/44686320 bytes. CPU-only, CPU/AIR5 and
CUDA/AIR5 no-Catalyst root builds pass. Device kernels are unchanged from
the manufactured-provider gate; plan 10.41's six zero-error memcheck logs are
reused, rather than relinking the accepted binaries or rerunning profiling.
The initial independent reference mistook geometry q0004 (Jacobian) for dxi;
the test stopped and its indexing was corrected to begin at q0005, without
changing solver numerics or lowering the gate. Other sizes, boundaries, CURVE,
AIR5 derivatives, rendering changes and OR6 as a whole remain unapproved here.
`out/or5_config_examples_20261001.xml` records 69 parser/collective checks,
including both committed TGV configuration examples and their exact selections.

### Derived Output Joint Lifecycles And Offline-Tool Installation

Plan 10.45 combines the already approved 16-cubed periodic TGV scopes, without
changing solver numerics or admitting another case family. The same 64 MiB
controlled host/device and aggregate per-test-directory budgets apply, with
at least 1 GiB free device memory before private allocations.

```bash
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_variable_clock.py -k derived_continuation
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_series_repair_runtime.py -k relocated_derived
```

`out/or6_derived_variable_clock_fixed_20261001.xml` records four passes:
CPU NP=2 x/z scalar, GPU NP=2 x scalar and z full, Q/divergence fields,
deterministic variable-step replay and independent physical-time schedules.
Continuous 12, seed 5, resume to 12, archives off and slice-only continuation
match q/caches/statistics exactly. Frame steps are fields 0/2/3/4/5/6/8/12,
slices 0/2/4/5/8/12; resumed lists are 6/8/12 and 8/12. Independent discrete
reference maximum is 3.4416913763379853e-15 and actual XDMF3 readback covers
twenty product-time frames. Slice-only GPU field downloads are 110976 bytes,
not a full-volume download. Including result JSON, the largest test tree is
55944616 bytes; controlled host/device maxima are 475104/612864 bytes.
`out/or6_variable_clock_helper_regression_20261001.xml` records two unaffected
basic-field CPU/GPU NP=2 z checks. The first derived attempt accidentally used
the basic tile-only resource checker; the corrected test reuses the approved
derived aggregate checker, not a larger budget.

The relocation test reuses immutable accepted NP=2 z donors from
`out/or5_real_derived_runtime_20261001` (override with
`ASTR_OUTPUT_DERIVED_REFERENCE_ROOT`). It requires the current executable hash
to match the donor's result, then copies and moves complete resource trees.
It rejects changing checkpoint interval without override, explicitly enables
override for keep=1 rotation and compares every new frame against the donor.
The protected step-5 restore point remains; step-10 checkpoint is retired,
but its field/slice frames survive. After moving the completed case again,
only damaged indexes are rebuilt; frames, coordinates and donor batches stay
unchanged. Actual single-segment and parent-chain readback covers twenty-four
product-time frames. All fourteen derived fields meet 2e-10, maximum
3.774758283725532e-15. `out/or6_derived_relocation_override_20261001.xml`
records two passes and report-inclusive test roots at most 45836162 bytes;
controlled host/device maxima are 474272/613440 bytes.
`out/or6_series_repair_affected_20261001.xml` records four unchanged basic
TGV/AIR5 index-repair regressions. No original donor is advanced or edited.

`test_checkpoint_export.py -k installed_output` installs the OutputTools
component from separately root-configured CPU/GPU builds to private prefixes.
Override build paths with `ASTR_OUTPUT_INSTALL_CPU_BUILD` and
`ASTR_OUTPUT_INSTALL_GPU_BUILD`; defaults are `build_release_restart_cpu` and
`build_gpu_probe`. It requires all three scripts, README and both examples,
byte-identical to source and under 4 MiB, then checks each script's CLI from an
unrelated working directory. Installing only this component must not install
the solver, example tree or Catalyst. Run root CMake configuration after
changing install metadata; no relink is needed for this gate.

```bash
python3 -m pytest -x -q tests/gpu_validation/test_checkpoint_export.py
```

`out/or6_output_install_export_20261001.xml` records all 27 passed checks,
including two install gates and 25 existing bounded export checks. The old
missing-component failure is preserved in `or6_output_install_before_20261001.xml`.
The accepted numerical binary hashes are unchanged after root reconfiguration.
These gates alone do not close all OR6, authorize wall/CURVE/AIR5 repartition or
establish production I/O performance or third-party transient memory peaks.

### Approved Channel Repartition Matrix And Generic Filter-Cache Fix

Plan 10.47 freezes 16-cubed bc41 channel, dt=1e-3, FP64, 643e/643e,
viscosity/tenth-order filtering and fixed forcing 1e-4. Each backend tests
NP=1<->2 x/z slabs and NP=2 x<->z, continuous 12 versus 5+7, with 2e-10
absolute tolerance for physical q/caches/driver and CPU legacy mean44.
Metadata, sample identity, clocks and schedules stay exact; same-topology
state remains exactly equal. No y repartition, feedback/frozen forcing,
CURVE/AIR5, rendering or new numerical coefficients are included.

`frozen_channel_initial` creates one global wall-compatible, nonconstant FP64
HDF5 file; `run_case(initial_resource=...)` feeds that identical file through
the original ninit=3 reader and verifies frozen restart resources. It does not
modify the rank-dependent chanini seed or claim a turbulent channel state.
The matrix's executable, layout and resources are generated together; old
donors from another executable hash cannot be reused for exact continuation.

The original CPU NP=1->2 x test failed at density 7.11e-8. Diagnosis subsequently
showed identical initial state, but one-step NP=1/2 difference 5.09e-7 without
any restart. The first-stage physical q is identical; cached pressure differs
by 2.23e-8. The old CPU post-filter primitive refresh only admitted periodic
boundaries, whereas qswap rebuilt interface caches from filtered q. The
partition therefore changed where caches used the current filtered state.
See `out/or4_channel_first_cpu_20261001.xml`,
`out/or4_channel_partition_regression_20261001.xml` and plan 10.48. These are
failed scientific gates, not acceptance evidence. The temporary CPU cache
snapshot instrumentation has been removed. `run_case(rhs_snapshot_step=0)`
remains available for bounded
existing first-stage q/conv/full diagnostics.

The user approved the generic fix, not only bc41: nonreacting numq=5,
num_species=0, explicit convection/diffusion refreshes every physical primitive
cache after filtering, before boundary/halo/gradient/RHS work. CPU and GPU
first-stage preparation/ordinary stages share this condition; scalar/full are
covered. Explicit synchronization, coefficients and boundary formulas remain
unchanged. AIR5 keeps its existing separate limiter/refresh ordering, and
compact paths are unchanged. Generate new donors with the matching rebuilt
executable; old binary fingerprints are not exact-continuation evidence.

```bash
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_repartition_runtime.py \
  -k channel
```

Twenty-eight channel checks passed across the following immutable records:

- `or4_channel_refresh_matrix_20261001.xml`: the first eight tests passed,
  covering four partition-independence and four complete-step CPU/GPU checks;
  the ninth sanitizer-launcher check failed. Do not report the whole XML passed.
- `or4_channel_refresh_memcheck_per_rank_20261001.xml`: two corrected memcheck
  gates passed, x/scalar and z/full, three steps; all four rank logs report zero
  errors. Sanitizer runs inside MPI, not around the launcher. Only this pinned
  diagnostic disables optional Open MPI CUDA/UCC probes; production is unchanged.
- `or4_channel_repartition_fixed_20261001.xml`: eighteen checks passed, including
  twelve cross-topology, two exact same-topology and four y-repartition rejections.

The one-step partition difference fell from 5.092547584265717e-7 to
7.105427357601002e-15. Complete-step CPU/GPU and repartition q/cache maximum
error is 5.3290705182007514e-14; CPU mean44 maximum is 1.4530598946294049e-12,
fixed-force error is zero and same-topology state is exactly equal. Sample
identity, clocks, schedules and source immutability are checked. GPU legacy
mean44 is not enabled or claimed. The cross-topology test root maximum is
38727327 bytes, below 64 MiB. Admission is now enabled only for the predicate
above; unsupported y repartition stays rejected. Plan 10.49 records the evidence.

`test_filtered_boundary_refresh_runtime.py` reuses existing physical-case
drivers with explicit synchronization, FP64, pinned halos and scalar/full,
absolute field/statistics tolerance 2e-10 with RTOL=0, and 64 MiB test roots.
It covers bc41 x/y/z, bc42 x/y, bc411/421 y, symmetry/zero extrapolation, LDC,
RTI, two CURVE walls and two warped-profile flatplates. Comparisons retain
boundary nodes and require finite state. Legacy CPU snapshots are taken at
complete RK; the RTI driver's old mixed-phase comparison is not used.
CURVE tests copy only required input text and generate their grid instead of
copying an unused large example grid. The source contract checks all three
CPU/GPU boundary-independent refresh predicates.

```bash
python3 -m pytest -x -q tests/gpu_validation/test_filtered_boundary_refresh_runtime.py
```

Fifteen physical short checks and the static contract passed:
`filter_refresh_physical_matrix_20261001.xml` contains the first eleven physical
passes and the static pass, then a file-budget failure from an unused old grid;
`filter_refresh_curved_matrix_20261001.xml` records both corrected CURVE walls
and both profile flatplates plus the static contract. Maximum CPU/GPU field
error is 3.979039320256561e-13. These are not long-time physical gates.

An additional nonreacting Ma5 bc52 flatplate with global lfilter=t is outside
the current GPU capability contract. Its transverse boundary filter is not
global-filter admission. `filter_refresh_capability_fixed_20261001.xml` records
the static contract and explicit rejection (initial output stays step/time=0).
Earlier coarse CPU state/metric and later GPU admission failures are retained
in `filter_refresh_nscbc52*_20261001.xml`; no positive field gate is claimed.
The first rejection-test assertion incorrectly disallowed initialized output;
it was corrected to check its phase, without changing solver output semantics.

`or4_filter_refresh_tgv_affected_20261001.xml` records nine affected periodic
TGV checks: four CPU/GPU complete-state comparisons, two exact restores, two
no-filter repartition rejections and one full-workspace slab-direction change.
Root CPU and CUDA/AIR5 builds passed. All current numerical records use the
program hashes in plan 10.49; no old-donor reuse is needed or authorized.

### OR6 Channel Relocation, Rotation, Index Repair And Failed Publication

Plan 10.50 extends only the already-approved bounded fixed-force channel
lifecycle. `test_relocated_channel_restart_rotation_and_index_recovery`
requires immutable exact-continuation donor reports from plan 10.49.
Set `ASTR_OUTPUT_CHANNEL_REFERENCE_ROOT` to their matrix root; the default is
`out/or4_channel_repartition_fixed_20261001`. Regenerate donors with the
channel matrix when the executable hash changes; missing/mismatched evidence
fails rather than silently using an older solver or skipping the test.

CPU NP=2 x/scalar and GPU NP=2 z/full copy the five-step source tree, move it
to a path containing spaces, and restore without the original initial-field
file. Checkpoint cadence changes from five to one steps: no override rejects,
explicit override remains exactly equal in q/caches/rank_extras, driver and
CPU mean44. Keep=1 retires steps 6:11 while protecting source step 5 and final
step 12. Long-lived fields/slices remain. Move the complete result again,
corrupt only its new-segment indexes, then repair while stopped. Every new
frame is equal to its continuous counterpart; ParaView actually reads both
segment and explicit parent-chain indexes. Repair reads zero field-array bytes.
Sources, resources, old frames and old checkpoints remain immutable.

```bash
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_series_repair_runtime.py::test_relocated_channel_restart_rotation_and_index_recovery
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_publication_failure.py::test_channel_first_save_failure_then_exact_recovery
```

`or6_channel_relocation_gpu_20261001.xml` contains both backend checks, not
only GPU; both passed. The earlier CPU-only corrected record is duplicate
acceptance, not a third distinct case. New fields are steps 6/8/10/12 and
slices 6/9/12. Each backend reads seven segment and fourteen parent-chain
product time frames. Parent chains include the seed's normal-end step-5
frame; do not equate that inventory to the uninterrupted reference's cadence.
Test roots are 42544862/35131224 bytes; estimated checkpoint-provider buffers
peak at 2234576 bytes, and mean44 buffers are checked separately. Archive
host/device tiles stay within 4096 bytes. GPU field/slice downloads total
943296/124848 bytes, excluding checkpoints, statistics, halos and third-party
allocations; this is not total transfer or production-peak accounting.

`or6_channel_publication_recovery_fixed_20261001.xml` records two passed
failures at the first resumed save, step 10: CPU batch_rename and GPU
latest_rename. The former leaves a sealed temporary candidate; the latter
leaves a complete new directory but no successful LATEST. Neither is reported
as success or advances to step 12. Original step 5 remains unchanged and is
identified in the error context; fresh recovery without injection equals
the continuous final state and CPU statistics exactly. Roots are
18374328/14668057 bytes. This is scoped EIO injection, not power-loss testing.

The first relocation negative test accidentally retained the seed's five-step
cadence, so it correctly did not reject; the corrected test changes to one
step. The first publication driver omitted the saved field/slice options and
was rejected before injection; matching options fixed the harness. Both failed
XML records are retained. No solver, tolerance or default was changed.
Plan 10.51 was subsequently approved; its separate acceptance is recorded
below. These channel lifecycle checks do not themselves enable CURVE or close
all OR4/OR6.

### OR4 Static-CURVE Periodic-z Repartitioned Continuation

Plan 10.51 approves the 16-cubed static warped profile flatplate, bl/prof,
ninit=0, nondimensional, warp_x=0.08/warp_y=0.04, dt=1e-5,
bctype=[11,21,41,51,1,1], 543e/643e, no turbulence model, MP7 physical-space
reconstruction, filtering and viscosity enabled. CPU scalar/GPU full each
cover NP=1<->2 z with unpartitioned x/y, continuous-12 versus 5+7 and exact
NP=2 z continuation. Statistics, derivatives and rendering stay disabled.
This is an extruded grid/profile; do not claim general 3D metric validation.

```bash
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_repartition_runtime.py -k curve
```

`or4_curve_matrix_20261001.xml` records sixteen passed checks: two one-step
partition-independence controls, four bidirectional restart comparisons,
two exact controls and eight source/target x/y refusal cases. The additional
`or4_curve_restore_memcheck_20261001.xml` runs GPU NP=1->2 z restore under
per-rank Compute Sanitizer; both rank logs report zero errors. Optional
MPI CUDA/UCC pointer probes are disabled only for this diagnostic invocation.
That historical acceptance totals seventeen checks. The command above now
selects the expanded 26-check CURVE matrix described below, not the old inventory.

Required physical q/caches and thirteen geometry fields must be finite and
within 2e-10. Shared nodes and defined periodic-z face halos are compared
against the continuous run in the same target layout, not old-layout extras.
Undefined x/y exterior and corner halos are excluded from physical matching;
all saved extras still must be finite. Same-topology state/geometry, clocks,
control and schedule contents are exact. Source resources, batches and frames
remain immutable. Original grid/profile inputs are absent on restore.
Every resumed frame is compared and ParaView 6.0.1 actually reads four volume
and three slice time points per positive case.

Cross-topology state error peaks at 1.4857751988228479e-18, target-halo error
at 1.4570025483958034e-18, with zero geometry error. One-step error peaks at
1.859184379663986e-18. Exact controls have zero differences including all
rank supplements. Largest matrix test root is 32504436 bytes; estimated
controlled state/geometry host buffers peak at 3031992/3583240 bytes. Archive
host/device tiles peak at 3672/2688 bytes. The initial description of a native
1 GiB reserve check was incorrect for basic fields: that check exists only
on the derivative path. Baseline free-memory observations are not a per-frame guard.
GPU resumed volume/slice downloads are 943296/124848 bytes, excluding other
providers, MPI and third-party allocations. None is a production peak claim.

`or4_curve_affected_20261001.xml` records four passed TGV/channel representative
repartition regressions. Both root builds passed; final binary hashes and
precise capability limits are in plan 10.52. Three preliminary probe XMLs
precede a narrower admission guard and are not counted as additional formal
cases. Plan 10.53 was subsequently approved; its x/x-z evidence is below.

### OR4 Static-CURVE x Slabs And x/z Changes

Approved plan 10.53 retains the same 16-cubed grid/configuration and 2e-10
absolute gate. CPU scalar/GPU full cover NP=1<->2 x and NP=2 x<->z, with
exact NP=2 x controls and unpartitioned y. The halo checker now includes
interior x communication faces plus shared nodes and periodic-z faces, for
both the eleven state fields and thirteen geometry fields in the target
layout. Undefined physical-face/corner padding is not a physical comparison,
but all stored state extras must remain finite. Initial grid/profile donors
remain immutable and can be absent in the resumed input directory.

The first GPU 1->2 x check failed in rank supplements, not physical fields.
Each rank had 1445 external x nodes with NaN u/v/w/p/T: GPU face conversion
ignored MPI_PROC_NULL and divided zero ghost q by zero density. CPU qswap
already guards these conversions. GPU x/y/z face converters now receive
neighbor/periodic flags; no-neighbor physical faces retain boundary-owned
caches. No q, CPU algorithm, boundary formula, averaging, clipping or sync
policy changed. The full-box converter is outside this fix.

```bash
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_repartition_runtime.py::test_curve_x_repartition \
  tests/gpu_validation/test_output_repartition_runtime.py::test_curve_repartition_keeps_exact_restore \
  tests/gpu_validation/test_output_repartition_runtime.py::test_curve_partition_independence \
  tests/gpu_validation/test_output_repartition_runtime.py::test_curve_nonperiodic_repartition_is_rejected
```

`or4_curve_x_halo_green_20261001.xml` records the GPU one-step control after
the corresponding red record. `or4_curve_x_matrix_fixed_20261001.xml` records
fifteen passes: CPU one-step control, eight cross-topology comparisons, two
exact controls and four y refusals. The additional x restore memcheck passes,
with two zero-error per-rank logs. These are seventeen distinct formal x gates;
preparatory controls and partial failed matrices are not counted again.
The earlier `or4_curve_x_matrix_20261001.xml` fails after three passes and
remains archived. Evidence retains its 20261001 filename; reporting finished
on 2026-10-02. All positive cases actually read back four volume and three
slice time points via ParaView, compare every field/time and keep source trees unchanged.

Maximum state/required-halo differences are
2.5457045834135963e-18/2.545281066939969e-18; physical and required extra
geometry differences are zero. Exact controls match all rank supplements.
Largest matrix root is 32504538 bytes; estimated state/geometry buffers peak
at 3031992/3583240 bytes, archive host/device tiles at 4032/2688 bytes.
Observed pre/post-run GPU free memory is 18735/18379 MiB, not an internal
minimum or third-party peak. These records did not enable native reserve
enforcement for basic archives; its later optional TGV gate is recorded below.

`or4_curve_x_affected_20261001.xml` records ten passes: two earlier CURVE z
directions, four TGV/channel repartitions, exact GPU TGV, physical x/z walls
and CURVE y CPU/GPU completed-state comparison including finite rank extras.
Physical-field error peaks at 2.5579538487363607e-13. y computation is not
y repartition admission; the four refusals still pass. Both root builds pass;
new hashes and the exact limits are in plan 10.54. y repartition, dynamic
inflow, AIR5, accumulated statistics, derivatives, rendering, larger ranks
and cross-backend restart remain outside this CURVE gate. The separate AIR5
scale-aware acceptance in plans 10.55-10.56 is recorded below, not certified by
these CURVE records.

## Fixed AIR5 HBL Periodic-z Repartition

Approved plans 10.55-10.56 use the existing fixed HBL preparer: 16-cubed cells,
domain 0.08/0.01/0.002 m, dt=1e-10 s, rho_ref=0.05 kg/m^3, T_ref=3000 K,
N2/O2=0.767/0.233, characteristic top, coupled chemistry with nonzero carry,
643e/643e, filter/viscosity enabled, CPU scalar and GPU full, explicit sync/FP64.
No new integrator, physical boundary formula or production case is introduced.
CPU needs an AIR5-enabled executable; the non-AIR5 CPU release build is not
a valid fallback for these tests. Root CMake builds must finish before testing.

```bash
cmake --build build_cpu_probe --target astr -j4
cmake --build build_gpu_probe --target astr -j4
ASTR_OUTPUT_AIR5_CPU_EXE="$PWD/build_cpu_probe/bin/astr" \
ASTR_OUTPUT_RUNTIME_EXE="$PWD/build_gpu_probe/bin/astr" \
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_repartition_runtime.py -k air5
```

`out/or4_air5_matrix_20261002.xml` records sixteen passes: two one-step
partition-independence controls, four CPU/GPU NP=1<->2 z continuations,
two same-layout NP=2 z exact controls and eight source/target x/y refusals.
`out/or4_air5_memcheck_20261002.xml` adds one GPU NP=1->2 z continuation with
two per-rank zero-error Compute Sanitizer logs. The sanitizer wraps each solver
process; optional MPI CUDA/UCC probes are disabled only for this memory gate.
MPI production transport settings are not changed.

The exact same-layout control compares full q/carry/cache/extras and geometry.
Cross-layout comparisons use the fixed pre-advance rho, sound, pressure and
temperature scales: a_ref=1100.3842893098433 m/s, p_ref=43244.48514856889 Pa.
q, carry and longdouble(q)-longdouble(carry) are checked separately, not q+carry.
Cache scales match each physical quantity; each geometry field uses its frozen
initial global maximum magnitude, with zero fields required to remain zero.
Dimensional and scaled errors are both retained in `air5_repartition.json`.
State and required-target-halo scaled maxima are
2.5849394912654093e-25/2.5849394142282115e-26; all thirteen geometry fields
and their defined halos have zero difference. All stored extras are finite.

An explicit `ASTR_CHECKPOINT_TEST_RESTORE_PROBE=1` reserializes the actual
host/device state before the first advance, without changing the output
schedule. A zero-advance restore and a normal advancing restore both preserve
all 34 physical q/carry/cache fields bitwise against the seed. The standalone
`outdat/restore_probe.h5` is test evidence, not a published restart bundle.
High and represented species densities stay nonnegative, and existing
128*epsilon closure checks, exact clocks/control/schedules, immutable source
resources and restore after removal of the three original inputs remain gates.

Each test root and controlled host estimate stays below 64 MiB; recorded maxima
are 65596431 and 9803688 bytes. No explicit new device workspace is introduced
by the AIR5 restore provider. This is not an estimate of compiler temporaries
or third-party peaks. Native basic-archive free-memory reserve enforcement
was not enabled in this AIR5 gate; the later TGV-only optional-guard acceptance
is recorded below, not provided by the separate derivative-path guard.

`or4_air5_affected_20261002.xml` records eight passes: TGV, channel and CURVE
cross-layout representatives plus the existing AIR5 HBL/SBLI same-layout
mean44/conservation diagnostic continuation. `or4_air5_provider_20261002.xml`
and `or4_air5_geometry_padding_20261002.xml` record six and seven provider/padding
regressions. All three root solver builds pass; final fingerprints are in
plan 10.56. Prepared/first CPU runs use earlier fingerprints and are not counted
again. The initial preparation failure selected a non-AIR5 build and stopped
before advancement, not a CPU numerical defect.

## Bounded Basic-Output GPU Free-Memory Reserve

Approved plans 10.57-10.58 add `&output device_reserve_bytes`: nonnegative
int64, default zero, broadcast using typed fields. A positive value queries
`cudaMemGetInfo` before basic frame packing allocation and after release.
Query failure or insufficient free memory aborts collectively before publishing
the failed frame. Existing derivative allocation keeps at least 1 GiB free.
These are device-wide point checks, not an OOM or intra-stage peak guarantee.
Checkpoint schema, field calculation and default output modes stay unchanged.

```bash
cmake --build build_gpu_probe --target astr -j4
python3 -m pytest -x -q -o junit_family=xunit1 \
  tests/gpu_validation/test_output_device_reserve.py
```

This test uses 16-cubed periodic TGV, dt=1e-3, GPU NP=1/2 x, FP64, explicit
sync, scalar filter storage, 643e/643e and viscosity/filter enabled. With
reserve=1073741824, continuous twelve steps, 5+7 exact continuation, default
reserve-zero and archive-off controls have bitwise equal q/caches/extras and
formal statistics; textual statistics and public frames also agree. Each
packing lifecycle has paired `ASTR_OUTPUT_DEVICE_RESERVE` measurements.
The impossible maximum int64 reserve is refused before allocation/publication;
an empty initialized `series.frames` header is allowed, but COMPLETE, frame
data and indexes referencing a failed frame are forbidden.

`out/or6_output_reserve_config_20261002.xml` records 76 parser and two-rank
broadcast passes. `or6_output_reserve_bounded_20261002.xml` records four real
state/refusal passes; each `reserve_validation.json` includes executable hash,
directory/controlled-buffer counts and native memory measurements.
`or6_output_reserve_memcheck_20261002.xml` adds one NP=2 basic-output gate with
two zero-error sanitizer logs. `or6_output_reserve_affected_20261002.xml` adds
five passes: NP=2 GPU derivatives and exact continuation, CPU/GPU derivative
comparison, CURVE x exact continuation and CPU/GPU AIR5 z exact continuation.
The last CURVE/AIR5 controls do not enable the new reserve and do not admit
reserve-enabled production runs for those cases.

The complete positive test directory, including all five comparison cases,
peaks at 53208457 bytes; controlled host estimate peaks at 3935984 bytes,
below 64 MiB. Host/device basic tiles peak at 4032/2688 bytes. Checkpoint
interval=99 saves each segment endpoint without reducing field/slice schedules.
Shared RSS, compiler temporaries, HDF5/MPI peaks and arbitrary third-party
device allocations are outside these controlled counts. Early test-harness
failures and the preparation round before the aggregate directory assertion are
retained but not counted again; plan 10.58 records their causes and final
CPU-only/AIR5 CPU/CUDA fingerprints. No production defaults or remote jobs
were changed; OR6 as a whole remains open.

AIR5 nonperiodic x/y or SBLI repartition, accumulated/conservation histories,
derivatives, render, other sizes/configurations and CPU/GPU migration remain
outside this gate. No Git writes or remote job actions were performed.

`out/or6_registered_reserve_20261003.xml` adds seven passes with the same binary:
GPU NP=2 channel x, static CURVE x, dynamic CURVE z, AIR5 HBL z and SBLI x
continuous/5+7/archive-off exact controls, plus impossible-reserve refusal for
CURVE and SBLI. Each positive test includes native before/after measurements;
all cases together in one test directory stay below 64 MiB (maximum 54211059
bytes). Basic host/device packing peaks at 4080/3264 bytes. Reproduce with
`python3 -m pytest -q -x tests/gpu_validation/test_output_device_reserve.py -k registered`.
This extends basic guard evidence, not repartition or derivative admission.

### OR5 Nonperiodic Complete-Step Derivatives

`test_output_boundary_derived_runtime.py` exercises registered 16-cubed channel
and static CURVE cases on CPU/GPU NP=1 and NP=2 x/z, and dynamic CURVE NP=2 z.
It reuses frozen cases, the 64 MiB per-test-directory limit, explicit GPU sync
and FP64. Continuous twelve steps, 5+7 restart and archive-off controls compare
the complete registered state exactly; dynamic inflow history is included.
The independent discrete reference uses the saved complete-step velocity,
643e physical-boundary closures and the full, untransposed grid metric.

`or5_boundary_derived_runtime_20261003.xml` records 18 passes, including three
CPU/GPU comparisons with actual ParaView time-series readback and one two-rank
Compute Sanitizer run (both logs zero errors). Maximum reference and CPU/GPU
differences are 1.281e-14 and 2.120e-12, below 2e-10. Maximum whole-test size
is 33899436 bytes; controlled host/device archive workspaces are 475520/613440
bytes. This is not truncation-error, long-time physics or production-I/O acceptance.

Manufactured checks use affine physical velocities on a non-affine grid with
nonzero cross metrics and all faces nonperiodic; even the boundary closure is
analytically exact for this quadratic computational-coordinate construction.
`or5_boundary_derivatives_manufactured_cpu_20261003.xml` has 14 passes;
`or5_boundary_derivatives_manufactured_gpu_20261003.xml` has 21 including
periodic controls and provider refusal. Source velocities and poisoned original
halos remain byte-for-byte unchanged. Whole probe directories stay below 4 MiB.
`or5_boundary_derived_tgv_regression_20261003.xml` adds 21 original TGV passes.
Plan 10.60 records current fingerprints and the corrected CPU tensor-flattening
preparation failure. No solver numerical algorithm, tolerance or remote job changed.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_boundary_derived_runtime.py
```

At this increment AIR5 derivatives were refused pending a dimensional contract;
the later OR5 AIR5 SI Derivatives section records their separate approved gate.
Neither derivative extension grants new repartition or rendering combinations.

### OR6 Dynamic Inflow And AIR5 Output Lifecycle

`test_output_boundary_lifecycle.py` reuses registered 16-cubed CPU/GPU NP=2
dynamic CURVE z, AIR5 HBL z and AIR5 SBLI x cases. Each shared fixture produces
a continuous twelve-step run and a five-step source with active statistics.
Positive tests move the source output tree, resume with explicit schedule
override and keep=1, move the finished output tree again, and repair deliberately
damaged indexes in stopped segments. Source checkpoints, immutable resources
and historical frames remain unchanged. State, statistics, dynamic inflow or
AIR5 compensation and GPU conservation histories match continuous controls
exactly. Real ParaView reads both repaired segments and explicit parent chains.
The six negative tests refuse changed scheduling without explicit override and
verify that no new COMPLETE is published and no previous payload is altered.

`out/or6_boundary_lifecycle_20261003.xml`: 12 passed in 184.15 seconds. The three
solver hashes are unchanged from plan 10.60; this adds no solver-code change.
Each positive test writes `lifecycle.json` with accounting and fingerprints.
Maximum test-directory size is 56135486 bytes; the separately stored shared
reference fixture is at most 33422858 bytes. Each directory is below 64 MiB;
the combined test-plus-reference disk footprint is not claimed below 64 MiB.
Basic field host/device packing blocks are at most 4080/3264 bytes, not full
process RSS or third-party peak allocation. These checks do not enable AIR5
derivatives, repartition, live index repair or production-capacity guarantees.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_boundary_lifecycle.py
```

The later `-k unregistered_history` subset adds twelve refusal checks for the
same nonzero-history sources: NP=2 to NP=1 and NP=2 x/z direction changes.
They require the exact rank/partition-mismatch diagnostic, no new COMPLETE,
and unchanged source fingerprints. They do not grant repartition support.
`out/or6_history_repartition_refusal_20261003.xml` records 12 passed in 94.64
seconds, with maximum refusal-directory size 135863 bytes (shared reference
fixtures are separate and each remains below 64 MiB). No solver change or
relaxation of a numerical acceptance threshold is involved.

### OR5 AIR5 SI Derivatives

`test_output_air5_derived_runtime.py` uses registered 16-cubed HBL NP=1/2 z and
SBLI NP=1/2 x. The independent discrete reference and CPU/GPU differences use
fixed gradient/curl/divergence and Q scales 110038.42893098433 s^-1 and
12108455841.599289 s^-2. Raw SI differences are also saved; neither test divides
by the observed field maximum or relaxes the approved normalized 2e-10 gate.
Continuous twelve steps, 5+7 continuation and archive-off controls preserve
the exact state, mean44, compensation and GPU conservation baseline.

`out/or5_air5_derived_runtime_20261003.xml`: 12 passes, two two-rank memchecks
with zero errors in all four logs, actual volume/slice ParaView readback.
Maximum normalized reference/CPU-GPU error: 6.034e-14/8.508e-12. Whole-test
directory maximum: 62573394 bytes; controlled host/device maximum:
476336/613456 bytes. The later narrowed direction gate adds six final-admission
positive regressions and four wrong-axis refusals (plan 10.64). Later metadata
changes are not retroactively attributed to the original executable hashes.

### OR5 Native Automatic Lineage

`test_output_native_lineage.py` links the production Fortran/C/HDF5 modules,
with manufactured 9x7x5-node 6/12-field fixtures and no flow integration.
It covers relative escaped references, explicit cutoffs, valid field/plane
overrides, invalid clocks/types/inventory/definitions, unbound resources,
symbolic/hard links and immutable parent payloads. No fake native file is
accepted merely because a test re-sealed it after injecting a defect.

`out/or5_native_lineage_integrity_final_20261003.xml`: 60 passed in 29.95 s,
including 24 native and 36 offline parent-chain regressions. The native writer
subset in `out/or5_native_writer_final_20261003.xml` adds 24 affected checks.
`test_output_archive_segments.py` additionally checks actual automatic ledgers
and compares native XML references against the independent stopped-source
catalog, then feeds rebased native XML to ParaView. Eight exact continuation,
two/three-segment and historical-cutoff checks and four valid override checks
passed. Overrides retain per-segment indexes and the parent edge, report
incompatibility, and do not alter advancement or statistics.

`out/or6_native_lineage_lifecycle_final_20261003.xml`: six dynamic CURVE/AIR5
HBL/SBLI CPU/GPU NP=2 relocated/retained/repaired native-reader gates passed
in 173.23 s. Each volume/slice lineage reads 8/6 frames. Whole-test directory
maximum: 56312943 bytes; shared reference directory separately: 33538398 bytes.
The matrix was rerun with frozen executable fingerprints after a preparation
round overlapped a rebuild. The executable contract check was not skipped.
Plan 10.65 records final CPU-only/AIR5 CPU/CUDA hashes, resource limits and
the separate, nontransactional index replacements. No Git or remote job action.

The final order correction compares plane sets, not HDF5-name-order positions;
the input permits unsorted selections and controls display order independently.
`out/or5_native_lineage_order_fixed_20261003.xml` has 61 passes in 28.55 s,
including a regression that first reproduced the wrong incompatibility report.
Final CPU-only/AIR5 CPU/CUDA binaries add four dynamic CURVE/HBL CPU/GPU
relocated exact controls and native-reader checks in 106.70 s:
`out/or6_native_lineage_order_fixed_runtime_20261003.xml`. Previous evidence
retains its original hashes rather than being relabeled with the newest binary.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_native_lineage.py \
  tests/gpu_validation/test_output_parent_series.py
```

### OR4 Static-CURVE Accumulated Statistics Migration

`test_output_curve_statistics_repartition.py` exercises the approved 16-cubed
extruded mapping, warp_x=0.08/warp_y=0.04, dt=1e-5, 543e/643e, FP64,
viscosity/filter and explicit synchronization. CPU mean44/GPU compact histories
cover 1<->2 x/z and two-rank x<->z, continuous12 versus5+7, plus exact same
topology controls. GPU x<->z adds a second exact restart after migration.
Counts, sample clocks and schedules must match; physical q/caches/history have
absolute tolerance 2e-10, never a tolerance on same-topology exact payloads.

GPU statistics metadata v2 stores inherited global moments in the same HDF5
file. `compact_total` independently folds baseline once plus local increments;
it excludes geometric lengths from cumulative addition. Source hashes remain
unchanged. Native first/last volume output measures the 1-GiB device reserve;
explicit staging bounds and whole test directories stay below 64 MiB. These
bounds are not HDF5/MPI/RSS or instantaneous third-party resource peaks.

`out/or4_curve_statistics_resource_matrix_20261003.xml`: 18 passes, 118.51 s;
CPU eight checks retain the final CPU hash. Final GPU-only ten checks including
two directional memcheck runs passed in 78.37 s:
`out/or4_curve_statistics_gpu_final_20261003.xml`. Earlier GPU hashes are not
relabeled after additional allocation/layout/finite-history guards. CPU/GPU
cumulative maxima: 2.2737367544323206e-13/7.105427357601002e-15;
state maxima: 2.5457045834135963e-18/1.4857751988228479e-18.
Largest directory: 33871910 bytes; controlled host bound: 3114696 bytes;
compact device arrays: 30600 bytes plus at most 4096 bytes output packing.
The refined packing bound has one additional x->z check in
`out/or4_curve_statistics_buffer_bound_20261003.xml`.

The shared-state probe adds `inherited_write/read`, proving grouped writes do
not overwrite root data and rejecting missing/wrong-shaped/FP32 fields.
`out/or4_inherited_state_contract_20261003.xml`: 11 passed in 22.21 s,
including the original statistics continuation and metadata rejections.
One final-GPU dynamic CURVE relocated exact continuation/reader regression
passed in 19.98 s (`out/or6_compact_legacy_regression_20261003.xml`); this is
not approval of dynamic repartition. The later plans 10.68-10.69 and separate
matrix below close only the approved bounded dynamic-inflow contract.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_curve_statistics_repartition.py
```

### OR4 Dynamic-CURVE Inflow And Accumulated History Migration

`test_output_dynamic_repartition.py` reuses the statistics gate with the
approved 16-cubed dynamic case, dt=6e-6 and twelve frozen nonpolynomial source
frames. Four cached frames are directly restored on physical nodes, with exact
window/slot metadata. The cursor must actually change during advancement.
CPU/GPU each cover four same-topology controls in total, twelve migrations,
two directional two-rank memchecks, and two unsupported-y refusals: 20 checks.
Both cross-axis migrations include a second exact restart for both backends.
Source fingerprints, sampling identity and schedule remain unchanged.

Evidence under `out/`, all dated 20261003:

| XML stem | Passed | Seconds |
| --- | --- | --- |
| or4_dynamic_cpu_exact | 2 | 11.84 |
| or4_dynamic_cpu_migration | 7 | 43.67 |
| or4_dynamic_gpu_exact | 2 | 14.79 |
| or4_dynamic_gpu_migration | 7 | 54.30 |
| or4_dynamic_memcheck | 2 | 19.93 |
| or4_dynamic_lifecycle_regression | 2 | 35.12 |
| or4_dynamic_final_admission | 4 | 31.91 |

CPU/GPU cumulative maxima: 2.2737367544323206e-13/7.105427357601002e-15;
q/cache maxima: 1.2636672781850093e-18/1.1049544796932348e-18;
inflow cache maximum: zero. Same-layout and second exact restarts are bitwise.
Largest test directory: 47291325 bytes; controlled host/device bounds:
3174808/80936 bytes. The original matrix's minimum native free-minus-planned
allocation is 19012646272 bytes; final regression minimum is 19012056448 bytes.
These selected checks are not instantaneous third-party memory peak certification.

The affected-regression XML retains two static migration passes followed by a
test-helper NameError, not a clean suite pass. The recursive HDF5 finite checker
was corrected, and both dynamic relocated lifecycle checks were rerun successfully.
After narrowing inherited-format admission to NP<=2, all three root builds passed
and four final affected checks above passed. Plan 10.69 records each executable
fingerprint separately; earlier matrices are not relabeled as newer executables.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_dynamic_repartition.py
```

### OR4 AIR5 HBL Mean44 And Conservation History Migration

`test_output_air5_statistics_repartition.py` exercises the approved plan 10.70
16-cubed dimensional HBL, CPU scalar/GPU full, dt=1e-10, viscosity/filter,
FP64 explicit synchronization, coupled chemistry and nonzero compensation.
Only periodic z may repartition; x/y remain complete. Fixed state/carry/cache
scales retain plan 10.55. Mean44 uses the fixed plan 10.70 dimensional scale
per component times actual sample count, not an observed near-zero denominator.
Conserved integrals use conserved scales times the fixed domain volume.
Both cross-layout normalized tolerances are 2e-10; exact controls remain bitwise.

Before opening migration, CPU/GPU each advance NP=1/2 three steps, sampling
once at step two. Pure restore serializes actual restored mean44 and global
conservation histories with the existing test-only restore probe, before any
RK update. All 34 state/cache/carry and 44 mean fields are bitwise preserved;
the scalar conservation baseline and metadata are bitwise unchanged. These
diagnostic files are not sealed checkpoints and are absent in normal launches.

Each backend covers continuous12 versus5+7, same-topology exactness and both
1<->2 directions. Each migration adds a second exact restart from step ten.
GPU diagnostics match every continued full-step sample against the continuous
reference, and phase zero must echo the saved baseline rather than count as
another sample. The source baseline must not be multiplied by rank count or
recomputed. Four wrong-axis refusals and four output-switch invariance checks
retain source hashes and require no published complete state after refusal.

Evidence under `out/`, dated 20261003:

| XML stem | Passed | Seconds |
| --- | --- | --- |
| or4_air5_history_partition | 2 | 19.24 |
| or4_air5_history_cpu_final | 3 | 105.58 |
| or4_air5_history_gpu_final | 4 | 63.39 |
| or4_air5_history_guards | 8 | 131.98 |
| or4_air5_history_affected | 4 | 56.62 |

The partition check predates migration-interface changes and retains its plan
10.69 executable hashes. Later checks use the plan 10.71 final hashes; no
computation or sampling kernel changed. GPU memcheck covers both ranks with
zero errors. CPU/GPU resumed-state normalized maxima: 2.585e-25/3.231e-26;
mean44 maxima: 5.736e-33/0. Conserved baseline versus independently initialized
reference: 8.596e-16; continued integral: 2.870e-34. Persisted source baseline
restore is bitwise, distinct from the partition-dependent initialization sum.

Largest restore test directory: 58035436 bytes; separately bounded immutable
reference fixture directory: 46791341 bytes. Controlled host/device maximum:
9808296/2165824 bytes. Minimum native first/last archive free-minus-planned
allocation: 18711442240 bytes. These are explicit buffers and discrete native
measurements, not total third-party RSS or instantaneous resource certification.

The first CPU restore XML retains an 82109750-byte directory-budget failure
after passing its numerical checks. Reference fixtures now remain immutable in
separately bounded directories and are reused across gates, rather than copied
into every restore test. Limits were not raised; final restore gates were rerun.
The HBL matrix does not admit SBLI migration, other axes, rendering or arbitrary
production sizes, and does not close OR0-OR6. Plan 10.72 requests the next scope.
The four affected regressions cover CPU dynamic exact restore, GPU dynamic x->z
migration, GPU HBL without mean/conservation history and the existing GPU SBLI
relocation/retention/native-reader lifecycle. Those SBLI checks were
same-topology only; the additional x migration gate is recorded below.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_air5_statistics_repartition.py
```

### OR4 Fixed AIR5 SBLI x History Migration

`test_output_air5_sbli_repartition.py` reuses the approved fixed 16-cubed SI
SBLI of plan 10.72. It checks CPU/GPU no-restart partition independence,
same-topology exact restore, NP=1<->2 x, another exact restart at step 10,
GPU dual-rank memcheck and CPU/GPU y/z rejection. Mean44 and the original GPU
global conservation baseline follow the state without resets or resampling.
The HBL z admission is unchanged and has two affected reverse-migration checks.

`out/or4_air5_sbli_partition_20261003.xml`,
`out/or4_air5_sbli_cpu_restore_20261003.xml`,
`out/or4_air5_sbli_gpu_restore_20261003.xml` and
`out/or4_air5_sbli_guard_20261003.xml` record 13 passes.
`out/or4_sbli_affected_hbl_20261003.xml` records two more passes.
CPU/GPU maximum fixed-scale state differences are 1.411e-12/7.087e-13;
mean44 differences are 1.625e-12/6.704e-13, below 2e-10.
Same-topology and pure-restore payloads are bitwise; both memcheck ranks
report zero errors. See plan 10.73 for exact fingerprints and budgets.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_air5_sbli_repartition.py
```

### OR6 Joint Completed-Step Products

`test_output_joint_products.py` combines native checkpoints, independent basic
volume/slices, formal velocity statistics and optional GPU EGL rendering in
one 16-cubed TGV run. CPU-only NP=1/2 uses no Catalyst; GPU NP=1/2 uses the
existing admitted preset. NP=1 uses step schedules, NP=2 physical-time schedules.
Continuous 12 versus 5+7 steps preserves authoritative state, statistics,
control/schedules and common archived fields bitwise. GPU JPEG pixels and VTP
geometry are also identical. Disabling volume/slices/rendering and changing
checkpoint frequency preserves flow and cumulative statistics.

`out/or6_joint_products_final_20261003.xml` records eight passes, 47.23 s.
Separate immutable reference/test roots stay below 64 MiB; the largest is
60134056 bytes. Source trees remain unchanged. Native resource lifecycle checks
are supplemented with the existing external NVML/process-tree sampler.
Its target sleep is 20 ms; query overhead is additional, and transient peaks
between observations are not guaranteed to be captured. Observed peak
differences relative to statistics-only GPU controls remain below the approved
4 GiB node RSS / 2 GiB physical-GPU limits, with more than 1 GiB free.
The sampler reports resource observations, not a production allocation proof.

The first attempt `out/or6_joint_products_20261003.xml` stopped because the
flow/mean-state test estimator does not cover formal velocity statistics.
The test now bounds those explicit arrays separately. Solver mathematics and
acceptance tolerances were not changed. Renderer admission remains separate:
these joint checks do not permit CPU rendering, CURVE/AIR5 rendering or
render-enabled topology changes.

```bash
python3 -m pytest -q -x -o junit_family=xunit1 \
  tests/gpu_validation/test_output_joint_products.py
```

`out/or6_final_affected_20261003.xml` adds five final-executable passes:
CPU batch-rename failure recovery, GPU LATEST-rename failure recovery,
NP=2 impossible reserve rejection, active-render repartition rejection and
render-control trailing-byte rejection. These supplement, rather than relabel,
the earlier immutable implementation-specific evidence. The renderer delivery
scope was subsequently resolved by user choice A in plan 10.77; the bounded
first-delivery closure and final executable evidence are recorded in 10.78.

Plan 10.76 updates the existing installation inventory to include the already
installed AIR5 derived example. `out/or6_install_inventory_final_20261003.xml`
records two CPU-only/CUDA installation passes, including byte-for-byte file
comparison and off-checkout CLI imports. The earlier inventory mismatch is
retained in `out/or6_install_inventory_first_20261003.xml` as a failure, not
relabeled as passed. `out/or6_installed_examples_parser_20261003.xml` records
three actual Fortran parser passes for all installed configuration examples,
with slice counts and both products' canonical derived-field indices checked.
No flow simulation or executable change is involved.

### OR0-OR6 Scope A Final Acceptance

User choice A closes the first delivery on the existing approved native matrix,
without removing CURVE/AIR5 state, history migration or derived output.
Rendering remains GPU Cartesian TGV, same topology only. CPU/CURVE/AIR5
rendering and render-enabled repartition are separate future goals and rejected.
Plan 10.77 records the implemented state/schema/defaults; 10.78 records closure.

After final root CPU-only/AIR5 CPU/CUDA/CUDA-Catalyst builds, the two CUDA
executables were relinked. Old evidence retains its original fingerprint.
The final executable supplement is:

| Artifact under `out/` | Passed tests |
| --- | --- |
| `or6_a_final_cuda_migration_20261003.xml` | 11, including static/dynamic CURVE and SBLI dual-rank memcheck |
| `or6_a_final_joint_derived_20261003.xml` | 12, joint TGV products and AIR5 exact derived restart/ParaView readback |
| `or6_a_final_fault_20261003.xml` | 5, failure recovery and reserve/render/corruption rejection |
| `or6_a_final_install_20261003.xml` | 2, installed tools/examples inventory |
| `or6_a_final_examples_20261003.xml` | 3, actual Fortran example parsing |

All 33 pass with no skips. Largest derived directory is 62848848 bytes, below
64 MiB. Sampled joint-render incremental RSS is at most 1113702400 bytes/node
and 277352448 bytes/physical GPU; observed free GPU memory is at least
18813878272 bytes. These fit the independently approved third-party limits;
the sampler does not guarantee capture of arbitrary transient peaks.
Final CUDA/AIR5 SHA256 is
`5348f91c03c9df90d0a5b0e75f5dc3c7054a7835b881da60434374c45d7b0f86`;
CUDA/Catalyst is
`fbc02832336131f59a3c0bc19a6121260cb77e125e88cc4fad0250c80f65a2ed`.
Bounded output correctness is not long-time physical or production-I/O
qualification. Legacy defaults are unchanged and no Git/remote jobs were run.
