#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Called once after choosing active modules. Periodic explicit components,
 * CFL reductions and instantaneous scalar diagnostics follow the build kind;
 * native FP32 main and other operator combinations need separate qualification.
 * Gradient buffers are optional.
 * Numerical launches enqueue work; check/synchronize at a consuming boundary.
 * The caller supplies same-phase physical nodes and the stencil halo.
 * A numerical failure remains latched until context destruction; no operation
 * silently clears it or retries a state. */
int astr_device_fluid_prepare(void *handle, int gradients);
/* Startup only: physical-exterior q/primitives and q multi-axis corners.
 * Physical nodes and valid MPI face halos are not touched. Restore follows
 * normal file upload and must not invoke this initialization again. */
int astr_device_fluid_initialize_exterior(void *handle,const int low[3],const int high[3]);
/* Only the physical y coordinate is needed for the channel integral. */
int astr_device_fluid_channel_geometry_prepare(void *handle);
int astr_device_fluid_flat_geometry_prepare(void *handle,int lower_wall);
/* Local sums: volume-weighted x momentum, wall shear integral, unique-node w^2,
 * wall heat integral and wall area. Fortran owns the global normalization. */
int astr_device_fluid_flat_statistics(void *handle,astr_numerical_real reynolds,astr_numerical_real tc,astr_numerical_real tc1,
    astr_numerical_real prandtl,astr_numerical_real c5,int lower_wall,astr_numerical_real values[5]);
int astr_device_fluid_uniform_force(void *handle, const astr_numerical_real force[3]);
int astr_device_fluid_rti_source(void *handle);
int astr_device_fluid_gradient_high_wall(void *handle,int axis,const int low[3],const int high[3]);
/* Apply existing matched-moment high-order rows in every physical direction,
 * preserving corrections already accumulated at face intersections. */
int astr_device_fluid_gradient_high_open(void *handle,const int low[3],const int high[3]);
int astr_device_fluid_viscous_high_open(void *handle,const int low[3],const int high[3]);
int astr_device_fluid_sbp63_gradient(void *handle,int total,int start,const astr_numerical_real temperature[2]);
int astr_device_fluid_sbp63_diffusion(void *handle,int total,int start,const astr_numerical_real temperature[2],
    astr_numerical_real re,astr_numerical_real pr,astr_numerical_real c5,astr_numerical_real tc,astr_numerical_real tc1);
int astr_device_fluid_flux_divergence_high_wall(void *handle,int viscous,int axis,const int low[3],
    const int high[3],const int first[3],const int last[3],int adiabatic);
int astr_device_fluid_wall_rhs(void *handle,int axis,const int low[3],const int high[3],
    const astr_numerical_real temperature[2],astr_numerical_real c2,astr_numerical_real c6,int kind);
int astr_device_fluid_primitives(void *handle, astr_numerical_real const2, astr_numerical_real const6);
/* Reconstruct physical nodes and single-axis face halos; never consume the
 * undefined double/triple-outside corners of a stencil-only halo layout. */
int astr_device_fluid_refresh_primitives(void *handle, astr_numerical_real const2, astr_numerical_real const6);
int astr_device_fluid_refresh_primitives_boundary(void *handle, astr_numerical_real const2, astr_numerical_real const6,
                                                  const int low[3], const int high[3]);
int astr_device_fluid_conservative(void *handle, astr_numerical_real thermal_coefficient);
int astr_device_fluid_gradient(void *handle);
/* Existing 643e low-order physical-end closure; MPI interfaces remain centered. */
int astr_device_fluid_gradient_basic_boundary(void *handle, const int low[3], const int high[3]);
/* Existing MP2 diagnostic candidate: independent FP64 evaluation and final
 * FP32 stores. Does not change solver gradients or RHS. Prepare is reusable;
 * the caller refreshes same-phase primitives/halos before evaluating. */
int astr_device_fluid_prepare_diagnostic_gradient(void *handle);
int astr_device_fluid_diagnostic_gradient(void *handle,const int low[3],const int high[3]);
int astr_device_fluid_rhs_prepare(void *handle, int viscous);
/* Existing MP2 candidate: FP64 state/algebra, FP32 stored stress/heat flux.
 * MPI transport buffers and the RHS remain FP64. */
int astr_device_fluid_rhs_prepare_mixed_viscous(void *handle);
/* Convection starts the RHS with -div(F). Viscous flux is built on physical
 * nodes from current gradients; its halo must be exchanged before divergence. */
int astr_device_fluid_convection(void *handle);
int astr_device_fluid_rhs_clear(void *handle);
/* Per-direction active ranges match CPU nodal convection/diffusion accumulation. */
int astr_device_fluid_flux_divergence_basic(void *handle, int viscous, const int low[3], const int high[3],
                                            const int first[3], const int last[3]);
/* Accumulate the nodal contribution only outside the prepared ME6 inner box. */
int astr_device_fluid_viscous_wall_band(void *handle, const int low[3], const int high[3],
                                        const int first[3], const int last[3]);
int astr_device_fluid_viscous_flux(void *handle, astr_numerical_real reynolds, astr_numerical_real prandtl,
                                   astr_numerical_real const5, astr_numerical_real tempconst, astr_numerical_real tempconst1);
int astr_device_fluid_viscous_divergence(void *handle);
/* Full-variable ping-pong workspace for the initial migration baseline.
 * Call x: q->work, y: work->q, z: q->work, then commit. The caller supplies
 * the current source halo between axes, including distributed directions. */
int astr_device_fluid_filter_prepare(void *handle);
int astr_device_fluid_filter_axis(void *handle, int axis);
/* Generic physical-end 0-6-6-6-8-10 closure, not the high-order-wall filter. */
int astr_device_fluid_filter_axis_basic(void *handle, int axis, int lower, int upper);
/* Existing local-f6 rows 1..4 for isothermal/adiabatic evolved-density walls.
 * component=0 selects full storage; 1..5 selects scalar storage. */
int astr_device_fluid_filter_axis_high_wall(void *handle,int axis,int component,int lower,int upper,int adiabatic);
int astr_device_fluid_filter_commit(void *handle);
/* Scalar storage: q(c)->work in x, work->q(c) in y; after all y components
 * exchange q's z halo once, then q(c)->work in z and commit that component. */
int astr_device_fluid_filter_prepare_scalar(void *handle);
int astr_device_fluid_filter_workspace_components(void *handle);
int astr_device_fluid_filter_component_axis(void *handle, int axis, int component);
int astr_device_fluid_filter_component_axis_basic(void *handle, int axis, int component, int lower, int upper);
int astr_device_fluid_filter_component_commit(void *handle, int component);
int astr_device_fluid_rk_save(void *handle);
int astr_device_fluid_rk_update(void *handle, astr_numerical_real dt, astr_numerical_real a, astr_numerical_real b, astr_numerical_real c);
int astr_device_fluid_check(void *handle);
/* Three local directional maxima. The existing Fortran controller performs
 * MPI maxima and their sum for C_upper; this is not max of the nodal sums. */
int astr_device_fluid_cfl(void *handle, int nondimensional, astr_numerical_real mach,
                          astr_numerical_real gamma, astr_numerical_real rgas, astr_numerical_real rates[3]);
/* Unnormalised unique-node energy/enstrophy/dissipation sums, then five
 * physical-node q maxima and five minima. MPI and scaling remain in Fortran. */
int astr_device_fluid_statistics(void *handle, astr_numerical_real reynolds, astr_numerical_real tc,
                                  astr_numerical_real tc1, astr_numerical_real values[13]);
/* Sum(momentum squared / density) on 0:upper-1, plus the invalid-node flag.
 * The existing Fortran monitor owns MPI reduction and reference scaling. */
int astr_device_fluid_adaptive_energy_sum(void *handle,astr_numerical_real values[2]);
/* Unnormalized local mass-flux integral, wall-shear difference and sum(w*w). */
int astr_device_fluid_channel_statistics(void *handle, astr_numerical_real reynolds, astr_numerical_real tc, astr_numerical_real tc1,
                                         int lower_wall, int upper_wall, astr_numerical_real values[3]);
int astr_device_fluid_cfl_diagnostics(void *handle, int nondimensional, astr_numerical_real mach, astr_numerical_real gamma,
    astr_numerical_real rgas, const int global_upper[3], const int origin[3], int ny,
    astr_numerical_real values[4], int64_t keys[4], astr_numerical_real *profile);
#ifdef __cplusplus
}
#endif
