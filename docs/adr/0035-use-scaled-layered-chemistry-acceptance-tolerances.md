# Use scaled layered chemistry acceptance tolerances

Chemistry acceptance will use separate, scale-aware tolerances for
thermodynamic inversion, analytic derivatives, CPU/GPU porting equivalence,
independent integration, and conservation. A single absolute tolerance is
invalid because reaction rates and species concentrations span many orders of
magnitude.

Thermodynamic round trips use a normalized `1e-12` bound. The scaled analytic
Jacobian uses `rtol=1e-6, atol=1e-8` against centered finite differences at
strictly positive interior states. Exact-zero species cannot be perturbed in
both directions without violating the accepted state domain, so their
boundary oracle uses an admissible second-order one-sided derivative with
Richardson extrapolation under the same tolerance. This boundary semantics was
accepted for Phase C0 on 2026-09-14.
CPU/GPU source terms use
`1e-13*source_scale + 5e-12*abs(reference)`. Matched ROS2 trajectories use
`rtol=1e-9` for temperatures and major species and
`atol=1e-13, rtol=1e-8` for trace species.

`source_scale` is the maximum absolute component of the reference source
vector at the same state. An all-zero reference vector must remain exactly
zero on both paths or use an explicitly defined physical test scale rather
than an arbitrary numerical floor.

Against a converged Radau reference, temperature uses `rtol=1e-7`, major
species use `rtol=1e-6`, and trace species use `atol=1e-10`. Normalized mass
and elemental residuals are bounded by `5e-13` on CPU and `5e-12` on GPU or
full trajectories. Relative complete-total-energy drift per accepted chemistry
substep is bounded by `5e-12`.

**Consequences**

Accepted states contain no negative species and no clipping or normalization
repair. NaN, infinity, out-of-range temperature, or continuation after a
failed solve is a hard failure. Nonstiff order tests require an observed order
of at least 1.8; stiff cases require monotone error reduction and final-oracle
agreement rather than an artificial order requirement. Initial bounds may be
tightened with evidence but may not be loosened merely to obtain a pass.
