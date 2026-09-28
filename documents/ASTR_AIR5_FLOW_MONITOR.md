# AIR5 flow development monitoring

## Scope and sampling phase

The existing `readwrite::readmonc` / `commcal::monitorsearch` configuration and
ownership are reused. For AIR5, `readwrite::writemon(completed_time)` dispatches
to the compact CPU/GPU collector after a complete coupled update. The old
no-argument call does not sample AIR5 a second time. Non-AIR5 binary monitors,
their phase, and their existing restart behavior are unchanged.

Place integer global grid indices in `datin/monitor.dat`, after a header:

```text
# i j k
0 0 0
32 64 8
```

These configured probes are sampled every completed step, as for the original
per-step monitor cadence, without an environment switch. Output is the AIR5
extension `monitor/air5_probes.dat`: step, time, probe identifier, xyz,
rho, u/v/w, T/Tv, p, and five mass fractions. It is text, not the legacy binary
record, and does not contain the legacy gradient columns. Probe identifiers
and MPI ownership come from the original reader. Duplicate requested probes
remain separate identifiers. Physical-coordinate lookup is NOT supported by
the existing `monitorsearch`: its coordinate branch is empty. Use integer
indices. That pre-existing limitation is recorded, not silently repaired here.

Opt in with `ASTR_AIR5_FLOW_MONITOR_STRIDE=100` (positive integer).
This switch enables only the additional profiles and wall diagnostics;
they remain off by default. This diagnostic currently supports generated Cartesian
`air5hbl`/`air5sbli` grids, not imported or curved meshes. It records physical
states after the complete transport/chemistry update and shared-node processing,
with physical sample time `time + completed_step_dt`. Step labels retain ASTR's
zero-based update index. Monitoring does not modify the solution.

Existing `statistic::meanflowcal`, `lavg/feqavg`, and mean-field checkpoint
formats are not duplicated or redefined. They are NOT yet an admitted AIR5 GPU
full-field statistics path. In particular their viscosity/budget terms need
AIR5-specific physical properties; simply adding an AIR5 flowtype would not
make those terms valid. The small fixed-probe block analysis below supplements
these routines, rather than replacing full-field mean/restart statistics.

Three fixed x stations are nearest global grid indices to 1/4, 1/2 and 3/4
of the x index range. Every y point at k=floor(ka/2) is a fixed probe.
These are profiles on one fixed z plane, NOT spanwise-averaged sections.
The wall is sampled along x on that same plane. A three-dimensional flow
requires additional spanwise statistics before any turbulence claim.

CPU and GPU pack the same conservative states and coordinates. A unique MPI
owner is selected for duplicate interfaces. A count reduction requires exactly
one owner per sample. GPU transfers only the packed samples (15 FP64 values
per point), not a full field. The authoritative CPU AIR5 conversion and
transport functions evaluate this small diagnostic payload on rank zero.
This is deliberately not a full GPU postprocessing implementation.

## Files and definitions

`monitor/air5_profiles.dat` contains step, physical time, station and global
indices, coordinates, rho, u/v/w, T/Tv, p and all five mass fractions.
`monitor/air5_wall.dat` contains step, time, global x index, coordinates,
tau_xy, heat flux into the gas and wall pressure. Files use `status=new` to
avoid silently overwriting an earlier window. Restart in a separate case
directory; concatenate only compatible coordinates and nonoverlapping times.
The same no-overwrite rule applies to the AIR5 probe extension. This integration
does not claim same-directory AIR5 monitor rollback/resume support.

On a stationary planar noncatalytic wall:

    tau_xy = mu_w * d(u)/dy |_w
    q_into_gas = -k_TR,w * d(T)/dy |_w - k_V,w * d(Tv)/dy |_w

The wall value and first two interior values give a three-point one-sided
derivative, using their physical y coordinates. The temperature gradients are
not assumed equal. Species enthalpy diffusion is zero for the noncatalytic
zero-normal-species-flux wall. These are second-order diagnostic estimates,
NOT the solver's high-order wall flux, and require their own grid-sensitivity
check before quantitative heat-transfer comparisons. No boundary closure or
evolution operator is changed.

## Window statistics

`tests/gpu_validation/analyze_air5_flow_monitor.py` reads the small files:

    python tests/gpu_validation/analyze_air5_flow_monitor.py CASE/monitor \
      --window-samples 100 --output CASE/monitor_blocks.json

With configured probes only, the same command analyzes `air5_probes.dat`.
With profiles enabled too, the output additionally contains a
`configured_probes` section. Each stream uses its own last two complete
windows; probe and profile cadences need not be identical. Explicit time
ranges accompany the configured-probe windows.

For each fixed probe and equal-duration window, it evaluates the ordinary
mean and RMS using centered deviations. For velocities it separately evaluates
Favre mean `sum(rho*u)/sum(rho)` and RMS
`sqrt(sum(rho*(u-u_tilde)^2)/sum(rho))`. Reynolds and Favre quantities are
explicitly separate. Equal time spacing is required; variable-step sampling
must use a separately implemented time-weighted analysis, not silently reuse
these arithmetic means. Two complete windows are required.

The output includes preceding/recent block means, RMS, signed mean changes,
and `integral rho*u dy` per unit span at each fixed x plane. This flux is not
a three-dimensional cross-sectional integral. Wall negative-shear intervals
are tracked independently at every sample by zero-crossing interpolation;
multiple bubbles stay separate and intervals truncated by the domain boundary
have unknown lengths. No separation is reported when there is no negative shear.

Block stability is a diagnostic, not an automatic admission gate. Startup
drift is not turbulence. Confidence intervals require temporal correlation and
effective-sample assessment, which this first version does not implement.
The monitors cannot reconstruct unsaved wall/probe histories from old global
maximum records. Existing long jobs are not modified by enabling this feature
in a separately launched validation case.

## Verification on 2026-09-28

Root CMake GPU and CPU-only builds passed. CPU NP1 versus GPU NP2 x-slab
two-update tests passed the physical-state gates. Maximum scaled differences
were 2.46854e-15 for profiles and 1.01431e-14 for wall diagnostics.
The monitored GPU phase snapshots were bitwise equal to the earlier
unmonitored two-update baseline. Four-update x-slab Compute Sanitizer reported
zero errors on both ranks. Four-update y/z-slab profiles differed from x-slab
by at most 4.23517e-22 scaled; wall diagnostics were identical.

Evidence under `tests/gpu_validation/out/`:
`air5_monitor_startup_20260928`, `air5_monitor_memcheck_20260928`,
`air5_monitor_yslab_20260928`, `air5_monitor_zslab_20260928`.
The memcheck case also produced `monitor_blocks.json` with two two-sample
windows as an I/O and analysis smoke test, not physical statistics.
Four focused Python tests (analysis plus campaign preparation) passed.
Existing long-run input and executable processes were not altered.

## Existing monitor integration gate

After routing through `writemon`, both CPU-only and CUDA root-CMake builds
passed. The two-update integration test uses the original `monitor.dat` parser
with corner, shared-interface, and repeated probe entries. CPU NP1, GPU NP2
x-slab, GPU NP2 y-slab, and GPU probe-only mode passed. The maximum probe
CPU/GPU scaled difference was `2.3067425476713993e-16`. CPU and x-slab profiles
and wall diagnostics were unchanged against the prior immutable baseline;
y-slab profile difference was `6.834257009336975e-26`, wall difference zero.
Four analysis unit tests passed. These are implementation gates, not developed
flow or statistical convergence evidence.

Evidence: `tests/gpu_validation/out/air5_monitor_integrated_20260928/result.json`.
Reproduction driver: `tests/gpu_validation/test_air5_flow_monitor.py` with
`--baseline`, `--output`, and `--executable`. No long job was restarted and no
Git staging or commit was performed.
