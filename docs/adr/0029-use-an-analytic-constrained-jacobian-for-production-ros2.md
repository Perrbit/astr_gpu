# Use an analytic constrained Jacobian for production ROS2

The production CPU and CUDA Fortran ROS2 chemistry integrators will use an
analytic Jacobian of the coupled five-species and vibrational-energy source.
The derivative is taken at fixed density, momentum, and complete total energy,
so it must include the translational- and vibrational-temperature changes
induced by every species-density and modal-energy perturbation.

A separate CPU-only, scale-aware centered finite-difference Jacobian will
recompute the complete thermodynamic inversion for each perturbation. It will
serve as an elementwise derivative oracle across cold, hot, dissociated, and
near-equilibrium states. It will not be called from the production CPU or GPU
time loop.

**Consequences**

A frozen-temperature reaction-rate derivative is not an acceptable ROS2
Jacobian for this constant-energy local system. The analytic implementation is
more complex, but avoids the repeated source evaluations, perturbation tuning,
and high device cost of a numerical production Jacobian. CPU/GPU agreement is
not sufficient evidence because both production backends may share the same
analytic error; the independent finite-difference comparison is mandatory.
