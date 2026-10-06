# AP Adaptive Product Output

## Contract And Implementation Status

AP0.1--AP3.2 completed bounded local acceptance on 2026-10-06. Adaptive output
is disabled by default. This is an implementation and numerical acceptance
record, not a production validation or universal event-detection claim.

The first registered signal is Cartesian periodic nonreacting TGV kinetic energy:
`0.5 * mean((qx**2 + qy**2 + qz**2)/rho) / (roinf*uinf**2)`.
The equal-volume Cartesian nodes are owned exactly once, excluding each local
upper endpoint. This is the definition of `statistic:kenergycal`, evaluated from
authoritative complete-step conservative state rather than cached primitives.
GPU monitoring downloads two FP64 scalars (energy sum and invalid-state count),
16 bytes per rank per sample. Initial monitoring establishes history without
forcing statistical accumulation, a picture or a checkpoint. Subsequent samples
use actual simulation-time differences, not wall time or the product interval.

The registered runtime scope is internally generated 16-cubed/32-cubed,
periodic Cartesian nonreacting nondimensional TGV. Native CPU monitoring and
archives do not require Catalyst. Actual rendering retains the existing GPU
admission; no CPU rendering, wall, CURVE, AIR5, render repartition or 256-cubed
adaptive demonstration is admitted by this record.

## Implementation Map

| Responsibility | Source |
|---|---|
| Shared configuration, canonical IDs, events/windows, product clock and persistence | `src/adaptive_output.F90` |
| Optional trailing `&adaptive_output`, volume/slice binding, typed MPI agreement | `src/output_config.F90`, `adaptive_output_collective` |
| Native product binding and typed agreement | `src/insitu_run_config.F90`, `src/insitu_config_collective.F90` |
| Complete-step observation before statistics/render dispatch | `src/mainloop.F90`, `src/output_runtime.F90` |
| Owned-node CPU sum and GPU partial/final reductions | `src/output_runtime.F90`, `src_gpu/statistic_gpu.cuf` |
| Adaptive native archive dispatch | `src/output_archive.F90` |
| Render clocks and due-field union | `src/insitu_product_schedule.F90`, `src/insitu_session.F90` |
| Execute native decisions and report publication outcome, without another clock | `scripts/insitu/product_dispatch.py`, `tgv_pipeline.py`, `device_render_pipeline.py` |

The registry accepts up to eight independent events and eight windows. All
events currently consume the same kinetic-energy definition with individually
configured scales/thresholds/holds. One same-phase reduction serves every event.
Windows can independently use steps or simulation time, with `[start,end)`
semantics. Only linked, enabled products use the union of active conditions.

Each adaptive product retains its own steps/time mode and normal/dense interval.
Entry emits the current complete state once, subsequent targets use the most
recent successful actual emission, and exit does not force a frame. Approved
EIO/EACCES/ENOSPC image-pair failures instead wait from the failed attempt while
preserving the successful identity. Uncovered mean fields remain a distinct
outcome; numerical, MPI, geometry and encoding failures are fatal. Fixed
statistics/checkpoint clocks and nonadaptive output behavior are unchanged.

## State Version And Reset Matrix

The adaptive payload is explicitly versioned `ASTRAP01`; individual adaptive
clocks/bindings use `ASTRAC01`. Shared history is embedded in `control.bin`
(`ASTROC05` when enabled). Native archive clocks use `ASTROA03` inside
`archives.bin`; render clocks use `ASTRPF02` inside the existing `ASTRIR04`
`insitu_control.bin`. Disabled paths retain their fixed-format versions
`ASTROC04`/`ASTROA02`/`ASTRPF01`. No new restart file is required. Stable IDs,
configuration, monitor target, previous signal/time, active/hold state, window
phase, product level and success/attempt/next-target history are preserved.

| Explicit override change | Shared state | Dependent adaptive products | Unaffected products/statistics |
|---|---|---|---|
| Renderer/face transport only | Preserve | Preserve | Preserve |
| Monitor mode/interval or indicator | Reset sample history and all events | Reset origin to restored complete step | Preserve fixed clocks/statistics |
| One event scale | Reset that event's derivative history and latch | Reset only associated products | Preserve other events and clocks |
| One event thresholds/hold | Reset that event latch | Reset only associated products | Preserve other events and clocks |
| One window definition | Reset that window position | Reset only associated products | Preserve other windows and clocks |
| Product associations, dense/normal interval, enable switch | Preserve unrelated shared state | Reset changed product only | Preserve others |
| No semantic configuration change | Restore exactly | Restore exactly, including hold/missing attempt | Restore exactly |

Without `restart_output='override'`, semantic changes are rejected. Each reset
records its affected ID and reason. Restoring an absent/disabled adaptive
configuration is an explicit configuration change, never an implicit reset.
Existing AP-disabled fixed-only overrides retain their original registry-wide
reanchor rule. Selective dependency reset is exercised with AP enabled; it does
not silently revise the meaning of the older fixed-only override.

## Verification Stages

1. AP0: native synthetic sequences, malformed inputs, overflow, windows, event
   equality/hysteresis/hold, output entry/exit, missing attempts and exact restore.
2. AP1/AP2: 16-cubed TGV, `dt=1e-3`, CPU/GPU NP=1/2 x-slab, 12 versus 5+7
   complete steps; actual event entry and exit, isolation and selective override.
3. AP3: admitted 32-cubed GPU image/geometry publication and independent native
   volume/slice readback, budgets/costs, optional builds and fixed PF regressions.

New controlled buffers are limited to 64 MiB per rank and physical GPU; each test
directory is limited to 64 MiB. Rendering uses the separately approved 2 GiB/GPU,
4 GiB/node additional budget with at least 1 GiB device memory free. No remote
production job, wall/CURVE/AIR5 extension or 256-cubed demonstration is included.

## Frozen Inputs And Numerical Results

The native 16-cubed fixture uses `dt=1e-3`, `Re=1`, monitor every complete step,
`S_ref=T_ref=1`, `r_on=0.74`, `r_off=0.723`, hold `0.005`, and window
`[3,7)` in complete steps. Native volume intervals are 5/1 steps and slice
intervals 0.006/0.002 simulation time. Re=1 is a short scheduling diagnostic,
not a production DNS threshold. Event state is inactive at 0, active at 1--6,
and inactive at 7--12; checkpoint 5 lies inside the active hold. Event-linked
volume identities are 1,2,3,4,5,6,11.

CPU/GPU NP=1/2 x-slab maximum kinetic-energy difference is
`1.304512053934559e-15`; q/cache difference is `2.5579538487363607e-13`, and
rank extras difference is `2.8421709430404007e-13`, all below `2e-10`.
Same-backend 12 versus 5+7 steps restore state, statistics, monitor history and
control clocks exactly. On/off monitoring preserves the authoritative state,
statistical accumulation and checkpoint cadence. Twenty CPU/GPU override cases
cover thresholds, scales, monitor interval, hold, window, normal/dense interval,
association and product/global disable; unrelated fixed slices still occur at
8 and 12, rather than being globally reanchored.

Actual 32-cubed products use a separate immutable `r_off=0.73`, hold `0.0045`
fixture to keep event decisions away from thresholds. Q images use 6/2 steps;
instantaneous streamlines use 0.008/0.002 simulation time. Both associate with
the event. Compatible slice geometry uses 6/2 steps and the window. Pictures
occur at 1,3,5,11 (both views at the first three, Q only at 11), giving seven
JPEG/EPS pairs. Compatible geometry occurs at 3,5,11; strict entries write no
VTK geometry and report zero geometry host bytes. Native volume occurs at
3,5,10 and slices at 3,5,11. All persisted fields are finite.

The six primitive slice fields at step 11 independently match the same-step
checkpoint exactly. Compatible geometry interpolated velocity differs by at
most `4.440892098500626e-16`. Standard/direct entries, both face transports and
NP=1/2 pass the same product identity/readback checks. Two step-5 image-failure
cases preserve missing history across 5+7 restart, wait rather than retrying
every step, and leave no half image pair. Render threshold/period/transport
override checks preserve the unrelated time-clock product and statistics.

## Evidence Ledger

Receipts are under `tests/gpu_validation/out/`. Counts below are per receipt,
not a disjoint sum; safety/admission and early configuration checks overlap.

| Gate | Receipt | Accepted result |
|---|---|---|
| Native core/config/typed agreement | `insitu_ap_core_final_20261006.xml`, `insitu_ap_template_20261006.xml` | 38 + 1 passed; current 39-case AP configuration/core set |
| Wider affected parsers/clocks | `insitu_ap_core_v3_20261006.xml` | 218 passed; includes existing fixed config/collective/publication checks |
| Native physical runtime and selective override | `insitu_ap_runtime_final_20261006.xml`, `insitu_ap_runtime_tail_20261006.xml` | 26 passing cases retained from the first receipt, four remaining cases passed in the supplement; 30 native cases covered |
| Actual GPU products/readback/recovery | `insitu_ap_products_final_20261006.xml` | 17 passed, zero skipped |
| Admission, memcheck, scalar trace and cost | `insitu_ap_runtime_safety_final_20261006.xml` | Six passed; NP=1/2 memcheck gives three zero-error rank reports |
| Production source-copy/configure decoupling | `insitu_ap_decoupling_20261006.xml` | Five passed; tests absent, CPU/CUDA and optional AIR5 configure paths |
| AP-disabled fixed-product regression | `insitu_ap_fixed_regression_20261006.xml` | Six passed: three render entries, fixed-only override and missing-pair continuation |
| Original common-clock continuation | `insitu_ap_common_clock_regression_20261006.xml` | Two passed: NP=1 steps and NP=2 time, exact state/statistics/products |

The native main receipt contains one failed **test input**: disabled
`&adaptive_output /` was on one line, rejected by the existing native parser
before reaching restart validation. The group name must stand alone. The test
was corrected and both disable cases plus both admission cases passed in the
supplement; the original failed case is not counted as a pass. No tolerance or
parser rule was relaxed, and no long immutable run was repeated merely to
produce an all-green aggregate file.

Earlier failures retained locally include an observer called after rendering
(one-step stale events) and an archive poll before restored-step deduplication
(a pending publication with no write). The first was fixed by observing before
complete-step product dispatch; the second by deduplicating before polling.
The final native/product receipts above cover both fixes. An initial safety
fixture also left legacy sample exports active and copied an unused grid file;
it exceeded the directory budget. Only the new no-field-I/O fixture was
corrected, not the budget or solver output contract.

## Resource And Transfer Gates

The new adaptive registry/clocks/typed MPI buffers are bounded independently of
the field size; conservative source accounting is below 1 MiB/rank. The lazy
GPU reduction workspace is at most 1,040 bytes in the admitted 32-cubed scope
(hard implementation cap 2,064 bytes). No AP full-field history is allocated.
These are controlled-buffer bounds, not total process-memory measurements.

Native renderer observation across the final actual-product cases gives maximum
additional host memory `1,333,493,760` bytes/node, additional device memory
`371,986,432` bytes/physical GPU, and minimum device free memory
`18,791,596,032` bytes. All fit the separately approved renderer budgets.
Maximum actual-product case size is `58,955,769` bytes, below 64 MiB. Memcheck
case directories are 32,733/33,176 bytes; the trace case is 3,107,150 bytes.
Stage-boundary observations do not claim to capture every transient third-party
allocation; no new production capacity claim is made.

Two rank-local Nsight traces contain three partial/final reductions each (initial
state plus two steps). Each final reduction is followed by exactly one 16-byte
D2H, total 48 bytes/rank for monitoring. No field/checkpoint HDF5 or legacy
sample export occurs in that trace. Solver halo and other explicitly admitted
transfers remain distinct; this is not a total-zero-D2H assertion.

## Cost Records

`ASTR_INSITU_TIMING=1` adds separate monitor, event-clock/update, product-clock
and native archive inclusion timings. Values below sum each stage over its
samples and report the maximum rank total. Nested/inclusive stages must not be
added to their children. First-use allocation/render cost is retained.

| Stage | 16-cubed NP=2, monitoring only | 32-cubed NP=2, standard-device/pinned with products |
|---|---:|---:|
| Event clock | 0.000158354 s | 0.000545852 s |
| Monitor, 13 samples | 0.000832027 s | 0.001709185 s |
| Event update | 0.000050633 s | 0.000132508 s |
| Product clock | Not enabled | 0.000187170 s |
| Device consumer inclusive | Not enabled | 1.385652426 s |
| Native volume inclusive | Not enabled | 0.465903058 s |
| Native slices inclusive | Not enabled | 0.030262649 s |
| Complete advance/output window | 0.047349263 s | 2.623529342 s |

Monitoring-off 16-cubed comparison is 0.050016542 s. The shorter on value is
short-run variation, not evidence that monitoring accelerates the solver.
The 32-cubed log is
`insitu_ap_products_final_20261006/test_actual_adaptive_products_7/gpu_np2_adaptive_products/run.log`.
These one-window diagnostics identify cost, not production performance or the
physical adequacy of a chosen monitoring frequency.

## Builds And Reproduction

Root CMake builds passed for CPU/CUDA without Catalyst, CPU/CUDA compatible
Catalyst, and CUDA with strict resident render entries. CUDA-compatible and
strict device builds have tests enabled; no-Catalyst release builds have tests
disabled. Optional AIR5 configure checks are decoupling evidence only, not AP
AIR5 admission. The actual numerical/product checks use the first two builds:

| Build | SHA256 of `bin/astr` |
|---|---|
| `build_insitu_device_render` | `ea71eec4198c85f4d392c89f25bb35bb9ba2523b8a8b7fba62865528b0c35285` |
| `build_release_restart_cpu` | `0a33d57892467162599b321f26760dc0fc07b3cacff8d3122fee6692506cda16` |
| `build_release_restart` | `e576ef84eb1e0f2916debbd58de76b3410458ce270b6970b710824d422c3c62f` |
| `build_insitu_gpu` | `b8da9941ead4fe1373e666e426d75e677ac94f9f11e2d6173520ad2fb6ad7ad5` |
| `build_insitu_check` | `7bf6e815c264fd1b3ff296c5e4fd5e45b35fa608135af49846f58ea3e44d4e20` |

Local environment: two RTX 4000 Ada devices, NVIDIA driver 595.91.07, NVHPC
26.1/CUDA 13.1, GCC 13, HPC-X 2.25.1, HDF5 1.14.6, Catalyst 2.1.0 and the
isolated ParaView 6.1.1 device build. The existing dependency patches/runtime
paths documented by IS8-R remain prerequisites, not a stock-ParaView claim.
Tests were run on the local PF/AP revision in `feature/gpu_dev`, based on
`fe41f55c0219d8e21e1aa3bbb45bb1687d19ec85`, before the implementation commit.
This record includes the preceding PF implementation; executable hashes,
rather than the base commit alone, identify the tested AP revision.

Use the native example `scripts/output/input.output.tgv.adaptive.example` and
render template `scripts/insitu/presets/tgv32/adaptive.nml.in`; fill explicit
matching implementation/script paths. `USER_GUIDE.md` documents all fields,
constraints and override semantics. Test invocation/environment selection is in
`tests/gpu_validation/README.md`. All inputs, logs and current-revision receipts
are retained locally. No Git write or remote job operation was performed during
AP implementation and acceptance; subsequent source commits are separate.

## Deferred Scope

Other scalar indicators, production thresholds, larger grids, y/z or multi-axis
AP topology admission, walls/CURVE/AIR5 adaptive output, and render repartition
need separately scoped definitions and gates. Asynchronous analysis, independent
resources and online interaction remain unapproved candidates. Lower average
output frequency is not a substitute for memory-lifecycle correctness or a
matched performance comparison.
