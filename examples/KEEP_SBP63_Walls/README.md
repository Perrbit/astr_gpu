# Explicit KEEP With Weak Isothermal Walls

This FP64 example selects `high_order_wall_mode='keep_sbp63'`: arithmetic-average
KEEP convection with internal D6 and six SBP6/3 rows at each y wall, standard
Sutherland viscosity with a boundary gradient lift, and weak inviscid/viscous
SAT wall fluxes. The previous strong bc41 state overwrite is not applied in
this mode. Existing low-order w24/w34 defaults are unchanged.

The default mesh has 16 intervals (17 nodes) per direction, uniform on
`[0,2*pi] x [0,pi] x [0,2*pi]`, with periodic x/z. Both walls are stationary,
Tw=1; Re=3000, Ma=0.3, reference temperature 273.15 K. The prescribed smooth
initial field has rho=T=1 and zero wall velocity. The channel force is zero.
Four RK3 steps of dt=1e-5 end at t=4e-5. Filtering, crashfix and time statistics
are off; exact checkpoints are saved at complete steps 2 and 4, with no volume
or slice sequence. This is a small numerical example, not developed channel DNS.

## Prepare And Run

Use a fresh run directory. Python requires NumPy and h5py only for preparation;
the solver has no Python or validation-tree dependency. Build ASTR through the
repository root CMakeLists.txt. Substitute your actual executable and MPI paths.

```sh
ROOT=/absolute/path/to/astr_gpu
RUN=/absolute/path/to/new_keep_sbp63_run
EXE="$ROOT/build_gpu/bin/astr"
MPIEXEC=/absolute/path/to/mpiexec
python "$ROOT/examples/KEEP_SBP63_Walls/prepare.py" --backend gpu --output "$RUN"
cd "$RUN"
export ASTR_FORCE_MPI_TOPOLOGY=1,2,1
export ASTR_GPU_SYNC_MODE=explicit
export ASTR_GPU_HALO_TRANSPORT=pinned
"$MPIEXEC" -np 2 "$EXE" run datin/input.case > run.log 2>&1
```

For CPU use `--backend cpu` and the CPU or GPU-enabled FP64 executable. To
exercise the approved boundary-band MPI cut, prepare `--intervals 20` and use
NP=4, topology `1,4,1`: the shared global node 5 must retain its actual SBP row
and norm rather than acquire another physical-boundary SAT. The local four-rank
test shared two GPUs and was a correctness check, not a four-GPU speed result.

## Restart And Interpretation

For two remaining steps, keep max_steps=4 and the same precision, method and
MPI topology. In the copied input.output add an absolute source path:

```fortran
restore_directory='/absolute/path/to/source/outdat/new/checkpoints/step000000000002'
```

Use a different empty output directory. Same-method, same-topology 2+2
continuation has been checked bitwise. Its numerical identity includes
`s63sat-v1`; low-order wall checkpoints are not exact continuations of this
operator. An explicitly requested non-exact method switch rebuilds derived
state and starts a new statistics segment under the existing KEEP rules.

SBP6/3 means internal sixth-order and boundary third-order first-derivative rows.
The nodal gradient/divergence composition has a generic second-order near-wall
viscous term. A separate steady manufactured test at 16/32/64 intervals shows
decreasing errors and approximately second-order finer-pair global maximum
errors for normal/spanwise velocity, not sixth-order Navier--Stokes accuracy.
Its test-only source is not selected by this physical example. Wall velocity and
temperature are allowed to differ from the imposed targets and must be reported
and checked for convergence. The numerical wall heat includes the SAT penalty;
the existing raw-gradient wall diagnostic is not that numerical energy flux.
Short CPU/GPU and MPI checks do not establish long-window stability. Compact
high-order walls, ME6 wall SAT, main filtering, open ends and general curved
walls are not implemented by this option.
