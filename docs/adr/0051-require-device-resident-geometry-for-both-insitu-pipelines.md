# Require device-resident geometry for both in situ pipelines

On 2026-10-06 the user selected the same image-only residency requirement for
the standard device-array and device-geometry direct pipelines: neither may
download volume fields or extracted geometry for host preparation or rendering.
The objective is to remove geometry readback and host preparation costs, so an
official device-array input followed by host geometry rendering preparation
cannot satisfy the standard pipeline gate.

Images and bounded control states may be read back within the scope and budgets
frozen in plan section 5.3. Existing face-staging transport permission is
unchanged and is not a geometry-download exception. Existing host-compatible
routes remain separate and cannot be silently substituted to pass either new
pipeline. This records an accepted design constraint, not an implemented or
hardware-validated capability; implementation steps IS8-R0 through R11 are
defined in the in situ plan section 5.3 and remain unimplemented.

The user also selected shared extraction: both pipelines consume the same
device-resident geometry produced by the approved sampling, Q, contour and
streamline numerical implementations. They differ at the rendering bridge,
not in the diagnostic definitions. Existing host geometry containers are not
the shared boundary and cannot be reused to satisfy the residency requirement.

Minimum reproducible patches to independently installed, version-pinned
ParaView/VTK dependencies are allowed for the standard pipeline. This retains
the standard data model and rendering route and requires a bounded prototype;
it neither proves feasibility nor permits substituting the dedicated direct
mapper to claim acceptance of the standard route.

Allowed image readback includes rank-local color and depth buffers for host
MPI composition in ParaView/IceT. These transfers and their host buffers are
measured and budgeted separately; the allowance does not extend to host
geometry processing or imply zero device-to-host traffic. Device-side image
composition is not a prerequisite of the first implementation.

Bounded current particle states for the existing RK45 ownership and MPI
continuation schedule are an explicit additional allowance, including their
current coordinates. They cannot be accumulated into host trajectories;
accepted point sequences, connectivity and coloring arrays remain device
resident. Per-round state size and cumulative traffic are reported separately.

Numerical extraction remains FP64; device-generated FP32 display copies are
allowed but cannot change numerical geometry, integration or statistics.
After the separate correctness, residency, lifecycle and resource gates pass
and the five-round performance report is complete, the standard route becomes
the default for device processing. Measured speedup is not a promotion gate;
any slowdown must be disclosed, while direct and compatible routes remain
explicit choices. The local performance-group storage limit is 64 GiB.
