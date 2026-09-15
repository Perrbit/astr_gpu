# Separate GPU equivalence from independent chemistry validation

Chemistry validation will use two distinct references. The CPU Fortran FP64
ROS2 production backend is the GPU equivalence oracle. An independently
implemented Python/SciPy zero-dimensional driver using high-accuracy Radau,
with BDF available for cross-checking, is the numerical and trajectory oracle.

Both references consume the same versioned mechanism data, whose generated
artifacts are protected by a test-time content hash, but they do not share the
production integration algorithm. Cantera is
limited to partial single-temperature thermodynamics, reference-energy, rate,
and equilibrium checks. Historical `zeroDimChem.zip` trajectories are retained
only as trend and scale references.

**Consequences**

CPU/GPU agreement is necessary evidence for a correct port but is not evidence
that the shared chemistry model is correct. The independent driver is a test
dependency and never enters the production ASTR executable. Acceptance reports
must state which oracle was used and must keep migration tolerances separate
from physical and stiff-integration tolerances.
