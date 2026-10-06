# IS8-R Device-Resident Rendering Acceptance

Date: 2026-10-06. Branch: `feature/gpu_dev`.
Status: R0-R11 complete within the bounded scope below. Both entries pass their
independent non-performance gates; the five-round report and default-selection
checks are complete. This closes IS8-R, not general production admission.

## Scope

Both entries retain Catalyst and ParaView, common FP64 GPU sampling/extraction
and RK45 integration. `standard-device` supplies borrowed device arrays through
the patched VTK array/mapper path. `direct-device` supplies the same display
buffers to a narrow VTK mapper. Neither is a replacement renderer.

The bounded solver scope is internally generated periodic Cartesian FP64 TGV,
643e/643e, 32 cubed, NP=1 or NP=2 x/y/z. Products are the fixed velocity slice,
Q=0.25 surface, instantaneous and Reynolds/Favre mean streamlines; the constant
velocity trajectory is an independent crossing diagnostic. The separate
256-cubed NP=2 x-slab render-only preset uses Q=0 and instantaneous streamlines,
100 completed steps, every-step 1280x960 speed-colored JPEG/EPS, without new
checkpoint, volume, slice or VTK geometry outputs.

Rendering reads no 3-D field, coordinate, connectivity, color or accepted
trajectory arrays on the host. FP64 geometry is retained on CUDA; display
coordinates/indices/colors are FP32/UInt32/UInt8 and uploaded by D2D CUDA/GL
interop. Allowed host transfers remain image/color/depth composition, bounded
metadata and at most 2 KiB/rank/round of particle continuation state. Pinned
transport explicitly permits fixed halo/shared-node face staging. Explicit
checkpoint/statistics state transfer is a different operation. This is not
zero total D2H, GPU-only IceT composition, arbitrary-flow admission or DCU support.

## Evidence

Paths below are relative to `tests/gpu_validation/out/`.

| Gate | Receipt and result |
|---|---|
| Interop, GPU identity, arrays, projection/color/depth and CPU-access refusal | R0-R5 component records in `ASTR_INSITU_IS8_COMPONENT_PROGRESS.md`; independent dependency patches preserve the previous ParaView install |
| Two entries, two face transports, NP=1/NP=2 x/y/z | `insitu_is8_r6_20261006/numerical_v3.xml`, 16 passes; product maxabs 1.5543122344752192e-15, CPU/GPU state/cache maxabs 1.9895196601282805e-13 |
| Images and partition seams | Same directory `images.xml`, 16 passes, at most one pixel; `analytic_pixels.xml`, three modes on both physical GPUs |
| Native memory safety, transfers and resource refusal | `insitu_is8_r7_20261006/memory.xml`, 16 passes/28 zero-error rank reports; `native_audits_v2.xml`, 16 passes; `transfer_ledger.xml`, all eight rank traces checked |
| Exact same-entry continuation and lifecycle | `insitu_is8_r8_20261006/frozen_pipeline.xml`, 16 continuation cases pass; corrected six fault cases in `lifecycle_v2.xml`; six interrupted-save cases in `interrupted_v3.xml` |
| Optional CPU/CUDA/AIR5 builds and compatibility | `insitu_is8_r9_20261006/`, current root builds, disabled-Catalyst exact continuation, 87 configuration tests, four unavailable-entry refusals, three compact-entry regressions, one AIR5 statistics regression and installed one-frame startup |
| Current binary affected regressions | `insitu_is8_r10_32_regression_20261006/pytest_v2.xml`, 20 passes: full numerical matrix plus selected continuation and memory checks |
| 256-cubed preflight and transfer attribution | `insitu_is8_r10_256_preflight_20261006/report.json`, four preflights/three separate traces; `transfer_v2.xml`, six passing rank checks, no strict geometry or UVM D2H |
| 256-cubed five-round performance | `insitu_is8_r10_256_matrix_v2_20261006/report.json`, twenty complete runs; physical diagnostic maxabs 0, all images retained, budgets pass; interrupted first invocation excluded |
| Default, explicit entries and capability refusal | `insitu_is8_r11_20261006/config_launcher.xml`, 117 passes; `native_v2.xml`, four native continuation/compatible passes; `unbuilt.xml`, six explicit/default NP=1/2 unavailable-entry refusals |
| Independent default launcher | `insitu_is8_r11_20261006/standalone_default`, NP=2 pinned, frames 2/4, ten JPEG/EPS pairs, no VTP, balanced finalization and phase budgets; directory 78518807 bytes |

Failures caused by test setup, expected diagnostic values, script modification
between save/resume, unsupported OpenGL injection and interrupted invocation
remain in the evidence tree. Their causes and corrected receipts are recorded
in the component progress document; they are not counted as successful gates.

## Deployment Conditions

Root CMake defaults the optional dependency targets OFF. Device products need
CUDA, Catalyst and `ASTR_WITH_INSITU_DEVICE`; strict rendering additionally needs
`ASTR_WITH_INSITU_DEVICE_RENDERING`, and a matching private VTK/ParaView/Viskores
build. Reproducible patches and commands are in `scripts/insitu/patches/README.md`.
The dependency currently runs from its independent build tree, not a completed
installable prefix. Explicit runtime library discovery is required after ASTR
staging; CPU builds with Catalyst disabled link none of these new dependencies.

No silent rendering or transport fallback is allowed. Same-entry continuation
preserves q/caches/statistics/control and images exactly. Changing only the face
transport requires the existing explicit output override. Cross-entry restart
and render repartition remain rejected. Image publication failures are isolated
only for approved recoverable errno values, with collective missing-frame receipts;
initialization, draw, identity and unknown errors terminate collectively.

## Five-Round Timing

Two RTX 4000 Ada GPUs, 256 cubed, NP=2 x-slab, dt=1e-4, 100 completed steps,
FP64/643e with scalar tenth-order filter workspace. Each rendering case writes
200 JPEG and 200 EPS files. Native small diagnostics are matched; checkpoint,
volume, slice, geometry files and accumulated in-situ statistics are disabled.
Solver faces use pinned transport; postprocessing faces use device-aware MPI.
The four-case order rotates across five independent repetitions without profilers.

Times are the maximum of rank-local complete windows, excluding solver startup
but including first-frame lazy setup and image publication. Pure RK is separate;
nested stage times are not additive. Full raw values and stage distributions are
in the report, with min/median/max and sample standard deviation below, in seconds.

| Entry | Minimum | Median | Maximum | Sample SD | Pure RK median |
|---|---:|---:|---:|---:|---:|
| Off | 60.368274 | 60.580561 | 60.617399 | 0.100664 | 56.371491 |
| Compatible | 257.208569 | 257.746235 | 264.008394 | 2.867249 | 56.784510 |
| Standard device | 85.644225 | 85.719774 | 85.815857 | 0.072039 | 56.387421 |
| Direct device | 85.608615 | 85.666557 | 85.994778 | 0.160788 | 56.412208 |

Standard/direct median increments over off are 25.139213/25.085996 seconds,
versus 197.165674 for compatible. The compatible/standard complete-window ratio
is 3.0068; this is a visualization-entry comparison, not CPU/GPU solver speedup.
The 0.053217-second standard/direct median difference is not evidence of a
reliable winner. Kinetic energy, enstrophy and dissipation match exactly.

All native phase budgets pass. Standard/direct maximum additional node RSS is
815087616/808374272 bytes; per-GPU extra memory is at most 2036707328 bytes,
and free device memory stays above 12 GB. External 20 ms preflight sampling
also passes. These observations cannot guarantee interception of every transient
third-party allocation. The group occupies 23961800971 bytes before the binary
backup, and the largest case 1560568974 bytes, below 64 GiB/group and 2 GiB/case.

The frozen benchmark executable is retained as `benchmarked_astr`, SHA256
`882e1ac5b4ef40cdd5941d5151383c2eaa8e7537ddec2fcca15884786ede3728`.
The final R11 executable has SHA256
`ea9d7235326de6c91c45e40dfec4a234cfd51bd39c0ccc1cf574673f1b8f5a93`:
only the tested configuration/default selection changed after the timing matrix;
rendering scripts and computational kernels did not change. The performance
receipt retains its original binary identity, not the later binary's hash.

## Default And Next Work

In an enabled device-rendering configuration, omitting `rendering_pipeline`
selects `standard-device`. Host processing and disabled in-situ configurations
retain `compatible`; all build switches and in-situ work still default OFF.
`direct-device` and `compatible` remain explicit choices. Selecting a strict
entry without the matching build fails, even when selected by default, and
does not fall back. Use `device_render_pipeline.py` for the two strict entries
and `tgv_pipeline.py` for compatible.

Default save followed by explicit standard-device resume passes exact state,
statistics, control and image continuation. The independent launcher uses the
same selection rule and no validation-module imports. Its 32-cubed checkpoint
output is the existing restart fixture, not part of the render-only 256-cubed run.

Performance has no pass threshold and does not replace numerical, residency or
lifecycle evidence. Next are PF0-PF3 independent product clocks, then
AP0.1-AP3.2 variable frequency. CURVE/wall/AIR5 resident rendering, render
repartition, other hardware and asynchronous execution remain separate work.
