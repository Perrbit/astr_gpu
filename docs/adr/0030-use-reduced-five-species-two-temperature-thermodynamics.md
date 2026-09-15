# Use reduced five-species two-temperature thermodynamics

The first ASTR chemistry backend will use an ideal mixture of N2, O2, N, O,
and NO with fixed translational-rotational degrees of freedom, fixed species
formation energies, and harmonic-oscillator vibrational energy for the three
diatomic species. Atomic translational heat capacities are `3/2*Rs`; diatomic
translational-rotational heat capacities are `5/2*Rs`.

Under the complete-total-energy convention, translational temperature is
recovered algebraically after subtracting kinetic, vibrational, and formation
energy. The common vibrational temperature is obtained by a bounded inversion
of mixture vibrational energy at the current composition.

This model is selected because it matches the frozen Kim-Jo/Park neutral-air
scope and yields a compact, differentiable CPU/GPU thermodynamic closure for
the first chemistry backend.

**Consequences**

NASA variable-heat-capacity polynomials, electronic excitation, rotational
nonequilibrium, ions, and electrons are excluded. The model must not be
described as complete high-temperature-air thermodynamics. The admissible
temperature range will be established by comparison with independent
thermodynamic data, and the fitted reverse-reaction equilibrium constants must
be checked for consistency with the selected formation-energy convention.
