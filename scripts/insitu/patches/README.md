# ParaView Streamline Coordinate Precision

The user approved this local ParaView 6.1.1 dependency patch on 2026-09-29.
It changes four `vtkPoints` allocations to double storage in the serial
streamline thread output, merged output, and parallel tail creation/reception,
plus the parallel per-segment integration seed array from float to double.
RK45, interpolation, step sizes and ASTR fields are unchanged. Streamline point
coordinates require twice the storage of the original float coordinates; resource
measurements must be repeated for the complete patched pipeline.

Apply to a clean ParaView 6.1.1 source tree with GNU patch, without fuzz:

```bash
patch --dry-run --fuzz=0 -p1 -d "$PV_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-streamline-fp64.patch"
patch --fuzz=0 -p1 -d "$PV_SOURCE" -i "$ASTR_ROOT/scripts/insitu/patches/paraview-6.1.1-streamline-fp64.patch"
cmake --build "$PV_BUILD" --target FiltersFlowPaths FiltersParallelFlowPaths -j 2
cmake --install "$PV_BUILD/VTK/Filters/FlowPaths"
cmake --install "$PV_BUILD/VTK/Filters/ParallelFlowPaths"
```

`PV_BUILD` must already be configured with the intended install prefix and the
same compiler/MPI/ABI as the Catalyst installation. The two install commands
replace only the corresponding module installations, not a system ParaView.
Do not install while another process is using those libraries. For an already
patched tree, `patch --dry-run --reverse --fuzz=0 ...` checks patch presence.

Validate with the ASTR paired driver using `--streamlines --extracts` and the
matching Catalyst implementation. The constant-field oracle must pass 2e-10
endpoint/straightness error and MPI crossing. Double file storage alone does not
prove intermediate integration or communication remained double precision.
The original failed run is retained as `insitu_streamlines_20260929`.

This is a local dependency deviation, not an upstream VTK release guarantee.
Reassess the patch rather than force-applying it to later ParaView versions.
