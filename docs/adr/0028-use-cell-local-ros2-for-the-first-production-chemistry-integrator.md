# Use cell-local ROS2 for the first production chemistry integrator

Status: accepted and implemented for the CPU FP64 zero-dimensional baseline.
The CUDA chemistry kernel and CFD coupling remain future work.

The first CPU and CUDA Fortran production chemistry integrator will use an
adaptive second-order L-stable Rosenbrock method. Its local unknown contains
five species densities and one vibrational modal-energy value. Every ROS2
stage solves a fixed six-by-six linear system while density, momentum, and
complete total energy remain fixed and the thermodynamic state is recovered
consistently.

The method is specifically the KPP ROS-2 2(1) formula with
`gamma=1+1/sqrt(2)`, `a21=1/gamma`, `c21=-2/gamma`,
`m1=3/(2*gamma)`, and `m2=e1=e2=1/(2*gamma)`. With the project Jacobian
layout `J(i,j)=dS_i/dy_j`, the implementation uses `J` without a transpose.
Each attempted substep factors one six-by-six matrix and applies two stage
backsolves.

Adaptive control uses a six-component weighted RMS norm with caller-provided
relative and componentwise absolute tolerances. The safety factor is 0.9 and
the step-size factor is limited to `[0.2,5]`. An inadmissible stage,
inadmissible candidate, failed thermodynamic recovery, or failed linear solve
rejects the substep without clipping or composition normalization. The solver
reports its proposed next step, accepted and rejected substeps, and RHS and
Jacobian evaluation counts.

The source evaluator and integrator expose coupled, chemical-only, and
VT-only modes. Coupled remains the default. The two component modes are
validation controls for isolating finite-rate chemistry and V-T relaxation;
they are not separate production time integrators.

The first GPU implementation assigns one grid cell to one CUDA thread. That
thread owns the local adaptive substeps, stage vectors, error estimate, and
six-by-six solve. This is a software work assignment, not a permanent mapping
between one cell and one physical CUDA core. A chemistry-half-step kernel may
therefore contain different substep counts across threads in a warp.

**Consequences**

Backward Euler is not the production baseline because its first-order local
accuracy would limit the intended Strang-integrated path. Explicit RK
subcycling is not the production baseline because high-temperature reaction
and V-T timescales can require prohibitively small steps. Candidate states
that violate species positivity, modal-energy admissibility, temperature
bounds, or linear-solve checks must reject and retry the substep rather than
be clipped.

The private matrix and vectors can create high register pressure, local-memory
spill, and adaptive warp divergence. Nsight Compute register, occupancy, and
local-memory evidence plus per-cell substep histograms are required before
optimization. The correctness baseline permits independent per-cell rejection.
If the measured `p95/p50` substep-count ratio exceeds 2 or effective thread
utilization falls below about 70 percent, predicted-step or stiffness bucketing
is evaluated first. Warp-synchronous rejection is a later option only within
already coherent buckets, so CUDA warp width does not enter the chemistry-core
contract. RODAS3 remains a future optional integrator and is not part of Phase
C1.
