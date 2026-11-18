#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Existing Cartesian AIR5 state boundaries, not the characteristic RHS.
 * Caller selects physical owners. These kernels modify q only; after global
 * status release the full boundary scheduler must perform the existing carry
 * reset and rebuild spatial halo/primitives. Failed sites do not write q.
 * Reads the same raw q interior states as the original boundary kernels.
 * Each status-producing stage resets the local failure slot: the caller must
 * perform the original global status release before entering the next stage. */
int astr_device_air5_wall_lower_y(void *handle,double temperature,int preserve_all);
int astr_device_air5_pressure_outlet_upper_x(void *handle,double pressure,int preserve_all);
/* Target copies complete before prepare returns. Explicit updates reuse fixed
 * allocations; after prepare there is no per-application target upload.
 * Profile order is Fortran (j=0..upper_y,component=1..11), j fastest. */
int astr_device_air5_fixed_boundary_prepare(void *handle,const double left[11],const double right[11]);
int astr_device_air5_profile_boundary_prepare(void *handle,const double *profile,size_t count);
/* Original postshock/normalshock branches: normal=0 fixes right state;
 * normal=1 uses pressure correction only when pressure>0, else extrapolates.
 * left/right ownership is supplied by the Fortran topology, not inferred here.
 * Like the original postshock operation, these writes do not modify carry. */
int astr_device_air5_fixed_boundary_apply(void *handle,int left,int right,int normal,double pressure,int preserve_all);
int astr_device_air5_profile_inflow_lower_x(void *handle);
/* Static top reference states and local x(i,k), with i fastest. Incident states
 * are Fortran (component=1..11,upstream/downstream); preparation owns a copy. */
int astr_device_air5_top_boundary_prepare(void *handle,const double farfield[11],const double incident[22],
    double origin,double incident_x,int incident_enabled,const double *x_top,size_t count);
int astr_device_air5_hbl_outflow_upper_x(void *handle,int physical_top,int characteristic_top,int preserve_all);
int astr_device_air5_farfield_upper_y(void *handle,int characteristic_top,int preserve_all);
/* Called only after the complete HBL boundary sequence has been released.
 * Requires carry storage; preserves dynamic-top carry except at inlet-owned nodes. */
int astr_device_air5_hbl_boundary_carry_reset(void *handle,const int lower[3],const int upper[3],int characteristic_top);
/* spacing(i,k)=y(i,jm,k)-y(i,jm-1,k) is static physical geometry.
 * Ensures the same high-normal plane used by symmetric convection exists. */
int astr_device_air5_characteristic_top_prepare(void *handle,const double *spacing,size_t count);
/* Original normal-only high/low top assembly, before the symmetric budget and
 * tangential/diffusive stages. first_i skips the inlet-owned top corner.
 * Does not advance q or reset carry; check/release status before the limiter. */
int astr_device_air5_characteristic_top_rhs(void *handle,int first_i,int preserve_all,double rate,double reference_sound);
#ifdef __cplusplus
}
#endif
