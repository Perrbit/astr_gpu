#pragma once
#include "device_state.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Called once after choosing active modules. Gradient buffers are optional.
 * Numerical launches enqueue work; check/synchronize at a consuming boundary.
 * The caller supplies same-phase physical nodes and the stencil halo.
 * A numerical failure remains latched until context destruction; no operation
 * silently clears it or retries a state. */
int astr_device_fluid_prepare(void *handle, int gradients);
int astr_device_fluid_primitives(void *handle, double const2, double const6);
/* Reconstruct physical nodes and single-axis face halos; never consume the
 * undefined double/triple-outside corners of a stencil-only halo layout. */
int astr_device_fluid_refresh_primitives(void *handle, double const2, double const6);
int astr_device_fluid_conservative(void *handle, double thermal_coefficient);
int astr_device_fluid_gradient(void *handle);
int astr_device_fluid_rhs_prepare(void *handle, int viscous);
/* Convection starts the RHS with -div(F). Viscous flux is built on physical
 * nodes from current gradients; its halo must be exchanged before divergence. */
int astr_device_fluid_convection(void *handle);
int astr_device_fluid_viscous_flux(void *handle, double reynolds, double prandtl,
                                   double const5, double tempconst, double tempconst1);
int astr_device_fluid_viscous_divergence(void *handle);
/* Full-variable ping-pong workspace for the initial migration baseline.
 * Call x: q->work, y: work->q, z: q->work, then commit. The caller supplies
 * the current source halo between axes, including distributed directions. */
int astr_device_fluid_filter_prepare(void *handle);
int astr_device_fluid_filter_axis(void *handle, int axis);
int astr_device_fluid_filter_commit(void *handle);
/* Scalar storage: q(c)->work in x, work->q(c) in y; after all y components
 * exchange q's z halo once, then q(c)->work in z and commit that component. */
int astr_device_fluid_filter_prepare_scalar(void *handle);
int astr_device_fluid_filter_workspace_components(void *handle);
int astr_device_fluid_filter_component_axis(void *handle, int axis, int component);
int astr_device_fluid_filter_component_commit(void *handle, int component);
int astr_device_fluid_rk_save(void *handle);
int astr_device_fluid_rk_update(void *handle, double dt, double a, double b, double c);
int astr_device_fluid_check(void *handle);
/* Three local directional maxima. The existing Fortran controller performs
 * MPI maxima and their sum for C_upper; this is not max of the nodal sums. */
int astr_device_fluid_cfl(void *handle, int nondimensional, double mach,
                          double gamma, double rgas, double rates[3]);
/* Unnormalised unique-node energy/enstrophy/dissipation sums, then five
 * physical-node q maxima and five minima. MPI and scaling remain in Fortran. */
int astr_device_fluid_statistics(void *handle, double reynolds, double tc, double tc1, double values[13]);
int astr_device_fluid_cfl_diagnostics(void *handle, int nondimensional, double mach, double gamma,
    double rgas, const int global_upper[3], const int origin[3], int ny,
    double values[4], int64_t keys[4], double *profile);
#ifdef __cplusplus
}
#endif
