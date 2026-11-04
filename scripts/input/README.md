# Named Input Migration

The adaptive-timestep development work reads a named main input (`input.case`)
and a separate `input.run`. Positional main inputs and `controller` are not
runtime fallbacks. Field and checkpoint output remain in `input.output`.

Convert into an existing, separate case directory:

```bash
python scripts/input/convert_legacy_input.py \
  examples/Taylor_Green_Vortex/datin/input.tgv \
  --controller examples/Taylor_Green_Vortex/datin/controller \
  --output /path/to/new-case/datin/input.case \
  --run-output /path/to/new-case/datin/input.run
```

The tool creates new files and does not overwrite existing files. It does not
copy or modify a mesh, inlet profile, initial field, chemical mechanism, output
configuration or checkpoint. Use `--comb` only when converting the positional
layout consumed by a COMB build; this retains `odetype` and `chemfile`.

The converted runtime input explicitly selects **fixed** stepping with the old
`deltat`. Its `max_steps` is the old `maxstep + 1`, matching the old inclusive
loop for a trajectory starting at step zero. This is not an automatic-timestep
comparison. To opt into automatic stepping, edit the timestep group to select
`mode='auto'`; the declared defaults are target acoustic/convective CFL 0.5
and growth factor 1.1. No viscous or chemical step limit is added in this phase.

## Main Input

Groups can be reordered. Character values must be quoted. Required groups are
`case_options`, `mesh`, `physics`, `numerics` and `boundary`; optional groups are
`initialization`, `sponge` and `species_transport`.

- Provide `flowtype`, `ia/ja/ka`, the reference state, `conschm/difschm/rkscheme`
  and all six `bctype` entries explicitly. Grid sizes retain the original index
  span convention, not a new node-count convention.
- `physics.nondimen=t` requires positive finite `ref_tem`, `reynolds` and `mach`.
  With `nondimen=f`, provide `ref_tem/ref_vel/ref_len/ref_den` instead.
- `bctype=41/411` requires positive `twall` on each selected face. The original
  `xslip`, `pout`, `turbinf`, `xrhjump` and `angshk` are still shared scalars,
  not independent per-face settings. Conflicting positional assignments are
  reported for manual review.
- Nonreacting species use `physics.num_species` and the corresponding array
  `species_transport.schmidt`. A disabled model does not require its unused
  boundary or immersed-body data.
- Optional flags default to CPU execution, nondimensional reference mode,
  diffusion enabled, no chemistry/filter/FFT/grid-file/immersed body/restart,
  no turbulence model, no sponge, and initialization method zero. Periodic
  direction flags default false; set them explicitly for a periodic problem.
- `input.run` is read beside the selected main input. From a case directory,
  `astr run` selects `datin/input.case`; an explicit path remains supported.

Numerics reconstruction defaults are `recon_schem=3`, no characteristic
decomposition, `bfacmpld=0.3` and `shkcrt=0.05`. An enabled filter still requires
`alfa_filter`; enabled FFT filtering requires `kcutoff`. These input defaults
do not expand the solver's numerical-method or boundary capability.

## Current Input Interface

The maintained solver accepts named inputs only. CPU/GPU explicit and compact
TGV, qualified wall/flatplate, CURVE, AIR5 and native-FP32 representatives have
local short-run evidence in `documents/ASTR_ADAPTIVE_TIMESTEP_PLAN.md`.
The physical/numerical entries below come from named groups, not environment
overrides. Invalid or ambiguous old files must be reviewed; conversion does not
invent reference values, reinterpret units or extend a case's numerical support.

## Maintained Launchers And Historical Tools

- `examples/GPU_Quickstart` and `scripts/insitu` use named templates directly.
- Current TGV/compact, M12, high-order boundary, native-precision, completed-step
  restart and selected-product validation launchers explicitly convert their
  editable positional fixtures at the last edit boundary through
  `tests/gpu_validation/named_fixture.py`. The solver never reads those fixtures
  or their `controller`. Runtime model settings are removed from the child
  environment after being written into the case input; execution settings stay.
- `prepare_hcw_laminar_sbli_case.py` emits named case/run files and identifies
  them in its manifest. Its positional files and `runtime.env` remain conversion
  records for existing fixture editing, not alternative runtime configuration.
- Historical scripts that restore `outdat/flowfield.h5` plus `auxiliary.txt`,
  use the retired paired-batch protocol, or compare the removed legacy field
  writer are frozen-version tools. They are not launchers for the current solver.
  Preserve their matching old program and inputs when reproducing old evidence;
  changing input syntax alone cannot restore their removed I/O protocols.
  Current restart checks use `run_output_restart_validation.py` or
  `run_output_air5_restart_validation.py`; variable-time in-situ statistics use
  `run_adaptive_insitu_gate.py`.

Similar launcher migrations are verified through representative cases and
static call-site checks, not separate CFD runs for every historical script.
Private old diagnostic binary headers are not the new checkpoint/archive ABI;
their 32-bit limits do not apply to the maintained 64-bit step identities, and
their upgrade is not claimed here.

## Physical And Numerical Options

Old physical/numerical environment variables are ignored with one startup
warning per defined name on the main rank. They cannot override `input.case`.
The converter can import an explicit file containing literal assignments:

```bash
python scripts/input/convert_legacy_input.py old/input.dat \
  --environment old/runtime.env --output new/datin/input.case
```

This reads text; it does not execute the file or read the current shell as a
source of physical defaults. Resolve conditional shell code and `$VARIABLE`
expressions manually before conversion. Hardware, library, communication,
thread-layout and diagnostic settings remain outside the physical input.

The converter and this guide are installed by the root CMake `InputTools`
component under `<prefix>/share/astr/input`. It can be installed without the
solver binary using `cmake --install <build> --prefix <prefix> --component InputTools`.

## Repository Examples

Named copies of historical examples use `<old-name>.case` and share the
`input.run` beside them. Original positional files remain conversion fixtures,
not a second runtime parser. The Quickstart generator uses its named templates
directly and does not copy its old positional files to a prepared run.

The LDC template's fourth nondimensional reference token and the disabled
immersed-body tail in the backstep template were not consumed by the original
CPU reader; their named copies omit only those unused values. HBL's shared
runtime file retains `1.d-9`, the first timestep actually read by the original
controller, not its later Mach-specific suggested alternatives. Edit that
explicit fixed value for a new study rather than assuming a filename chooses it.

Five dimensional templates lack `ref_den`: Perfectly_Stirred_Reactor,
Steady_H2flame, flame2d, hitflame and supersonic_backstep_dimension. Their physical
references have not been guessed or changed. The user approved retaining them
as non-runnable templates pending manual correction, outside this phase's
launch acceptance rather than silently assigning a density.
COMB examples require a COMB-capable build and their original mechanism resources;
AIR5 is not a replacement for that interface. A named example parsing correctly
does not extend numerical, output or GPU support. In particular, AIR5 TGV has
not acquired the new output lifecycle by converting its input.

| Group | Fields |
| --- | --- |
| `boundary_options` | `profile_inflow_mode`, `wall_blowing_mode`, `nscbc_farfield_mode`, `high_order_wall_mode`, `conservative_boundary_file` |
| `boundary_options` | `nscbc_farfield_rho/u/v/w/t`, `nscbc_farfield_shock_x`, `nscbc_farfield_shock_angle_deg` (the slash-separated targets denote five separate field names) |
| `transport_options` | `perfect_gas_prandtl`, `sutherland_temperature_k`, `channel_force_mode`, `channel_force_fixed` |
| `initial_conditions` | `shuosher_shock_x` |
| `precision_options` | `gpu_precision_mode`, `gpu_mixed_candidate`, `gpu_mp_positivity` |
| `chemistry` | `source_mode`, `chemistry_tolerance`, `filter_mode`, `convection_limiter`, `diffusion_limiter`, `top_mode`, `top_tau` |
| `chemistry` | `compensation`, `compensation_restart`, `step_history`, `low_activity_threshold`, `source_evaluation` |

Numeric fields are real namelist values, not quoted strings. The other fields
use quoted strings with their existing mode names. Omitted fields retain the
original model defaults: notably, an unspecified farfield target component is
taken from the initialized freestream, not replaced by zero. `low_activity_threshold`
is numeric; zero replaces the old textual `ASTR_AIR5_LOW_ACTIVITY=off` setting.
The precision group controls existing runtime numerical choices; it does not
change the build-time `fp32` or `fp64` `real_kind`.

For example, the following explicitly selects a constant channel driving term:

```fortran
&transport_options
  channel_force_mode = 'fixed'
  channel_force_fixed = 1.d-4
/
```

Existing `ASTR_*` diagnostic log tags may remain in output. They are not a
second input interface. The same parsed options feed CPU, GPU and checkpoint
identity consumers; enabling a mode still requires its existing solver support.
