# Periodic KEEP Taylor--Green Vortex

This self-contained FP64 input uses the built-in TGV initialization, a periodic
16 x 16 x 16 interval mesh, Re=1600, Ma=0.1 and four RK3 steps of dt=1e-3.
It needs no grid, initial-field or profile file. Main filtering, statistics and
crashfix are off; checkpoints are saved at steps 2 and 4. It is a short input
example, not a resolved TGV or a performance benchmark.

Build through the repository root CMakeLists.txt. The executable path below is
the FP64 GPU build; select your actual build and MPI installation paths. Work in
a fresh run directory so this example does not replace another run's output.

```sh
ROOT=/absolute/path/to/astr_gpu
EXE="$ROOT/build_gpu/bin/astr"
MPIEXEC=/absolute/path/to/mpiexec
RUN=/absolute/path/to/new_keep_tgv_run
mkdir "$RUN"
cp -r "$ROOT/examples/KEEP_Taylor_Green_Vortex/datin" "$RUN/"
mkdir -p "$RUN/outdat/new"
cd "$RUN"
export ASTR_GPU_SYNC_MODE=explicit
export ASTR_GPU_HALO_TRANSPORT=pinned
export ASTR_FORCE_MPI_TOPOLOGY=2,1,1
"$MPIEXEC" -np 2 "$EXE" run datin/input.case > run.log 2>&1
```

For CPU execution set `use_gpu=.false.` in the copied input and use the CPU or
GPU-enabled FP64 executable. CPU execution does not require a GPU.

## Independent Numerical Choices

Edit the copied `&numerics` group before starting a new run:

| Selection | Input | Meaning |
| --- | --- | --- |
| Explicit KEEP | `convection_form='keep', conschm='643e'` | Original arithmetic-average sixth-order split convection |
| Compact KEEP | `convection_form='keep', conschm='643c'` | Sixth-order compact split convection, with global line coupling across MPI ranks |
| Original viscosity | `viscous_form='standard', difschm='643e'` or `'643c'` | Existing nodal viscous discretization; independent of KEEP convection |
| ME6-Opti viscosity | `viscous_form='me6_opti', difschm='643e'` | Selected ME6 G2 half-point viscous operator; not another KEEP convection option |
| Explicit filter | `lfilter=.true., filter_scheme='explicit10'` | Existing tenth-order explicit filter |
| Compact filter | `lfilter=.true., filter_scheme='compact10'` | Existing tenth-order compact filter, independent of the convection choice |

The additional `datin/input.mp5_me6.case` selects original MP5 physical-space
convection (`543e`, recon=3, no characteristic decomposition), ME6 viscosity
and global `643c` nodal auxiliaries on the same periodic TGV. Use the same
copied `input.run` and `input.output` and replace only the launch input:

```sh
"$MPIEXEC" -np 2 "$EXE" run datin/input.mp5_me6.case > run.log 2>&1
```

Choose a fresh run directory, as for the KEEP example. For CPU calculation,
set `use_gpu=.false.` in that file. The MP5/ME6 example has native CPU/GPU
NP=2 x and exact 2+2 continuation checks. Its corrected dispatch identity
contains `mp5v1`; older unversioned batches cannot be used for exact continuation
with the corrected implementation. This is not another KEEP formula or a
strong-shock/long-time validation. The default `input.case` remains KEEP.

Do not equate the arithmetic-average KEEP choice with a shock-capturing or
strict entropy-stable method. This example contains no physical boundary and
does not validate the incomplete open-boundary combinations. Their separate
compatibility, incoming-only and nonreflecting status is recorded in
`documents/ASTR_GPU_KEEP_INTEGRATION_PROGRESS.md`.

## Restart

The checkpoint directory includes the numerical identity, geometry and current
RK-complete state. For an exact step-2 continuation, keep the precision,
numerical options and MPI topology unchanged, and set:

```fortran
&output
  directory = 'outdat/continued'
  restore_directory = 'outdat/new/checkpoints/step000000000002'
  exact_restart = .true.
  host_budget_bytes = -1
  device_budget_bytes = -1
  buffer_bytes = 67108864
/
```

Create the empty `outdat/continued` output directory before launching this
continuation. The solver creates its checkpoint and resource subdirectories.

Keep `max_steps=4` for the two remaining steps. A changed numerical method is
not an exact continuation: explicitly select `exact_restart=.false.` for the
existing non-exact restart path. Do not change `ninit` to select a checkpoint;
`restore_directory` selects the saved state.

For a non-exact method switch involving KEEP or ME6 with a different effective operator
identity, statistics start a new segment at the restored complete step; old
accumulators and endpoints are not combined with the new method. Same-method
continuation retains the existing history policy. CPU legacy time-weighted
means and CPU/GPU in-situ time statistics have separate representative checks;
GPU TGV does not thereby gain the CPU legacy-statistics interface. This example
keeps statistics off. See the current KEEP progress record for tested scopes.
