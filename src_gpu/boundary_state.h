#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Adds the existing low-order x+ characteristic divergence to resident RHS. */
int astr_device_nscbc_outlet_x(void *parent,int npdcj,int npdck,int include_lower_z,
    astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real length,astr_numerical_real roinf,astr_numerical_real uinf,astr_numerical_real vinf,astr_numerical_real winf,astr_numerical_real pinf,astr_numerical_real pout);
int astr_device_nscbc_inlet_x(void *parent,int npdcj,int npdck,astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real length,
    astr_numerical_real roinf,astr_numerical_real uinf,astr_numerical_real vinf,astr_numerical_real winf,astr_numerical_real pinf);
/* Upper-y policies 0 compatibility, 1 target relaxation, 2 nonreflecting, 3 GRCBC raw RHS.
 * Shock targets use an explicitly uploaded derived 2D conservative plane. */
int astr_device_nscbc_top_target_upload(void *parent,const astr_numerical_real *target,size_t bytes);
/* GRCBC: static exterior q(5) plus physical normal spacing, on the upper-y owner.
 * Complete projection runs after all convective/viscous/case-source contributions.
 * boundary_status retains original 1/2 errors and 4 for the unsupported subsonic corner. */
int astr_device_nscbc_incident_prepare(void *parent,const astr_numerical_real *target,size_t bytes);
int astr_device_nscbc_incident_complete(void *parent,int npdci,astr_numerical_real gamma,astr_numerical_real mach,int *boundary_status);
/* H1 static exterior primitive lines (7x5), last exterior q(5), hn and front flag.
 * Non-top ranks pass NULL/0; the per-call boundary status remains available. */
int astr_device_nscbc_high_open_prepare(void *parent,const astr_numerical_real *target,size_t bytes);
int astr_device_nscbc_high_open_complete(void *parent,const int low[3],const int high[3],
    astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real length,astr_numerical_real roinf,astr_numerical_real uinf,astr_numerical_real vinf,astr_numerical_real winf,
    astr_numerical_real pinf,astr_numerical_real pout,int *boundary_status);
int astr_device_nscbc_outlet_filter_high_open(void *parent,int npdcj);
int astr_device_high_open_wall_y(void *parent,int code,int forcing,int outlet_only,astr_numerical_real tw,astr_numerical_real c2,astr_numerical_real c6);
int astr_device_nscbc_top_filter_prepare(void *parent);
int astr_device_nscbc_top_filter(void *parent,int axis,int ntype);
/* Snapshot after convection/boundary assembly, apply after viscous assembly.
 * The snapshot is an optional derived 2D plane consumed exactly once. */
int astr_device_nscbc_top_prediff_capture(void *parent);
int astr_device_nscbc_top_viscous_source(void *parent,astr_numerical_real gamma,astr_numerical_real mach);
int astr_device_nscbc_top_mach2(void *parent,astr_numerical_real mach,astr_numerical_real *value);
int astr_device_nscbc_top_rhs(void *parent,int npdci,int npdck,int policy,int shock,int keep_corner,
    astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real length,astr_numerical_real pressure_target,const astr_numerical_real target[5],astr_numerical_real max_mach2);
/* Existing KEEP/ME6 x+/y+ intersection: project the assembled convective RHS,
 * after both normal contributions and the outlet's spanwise contribution. */
int astr_device_nscbc_top_outlet_corner(void *parent,astr_numerical_real gamma,astr_numerical_real mach);
/* Six-order transverse outlet filter. Only the owner prepares its plane;
 * the caller exchanges q halos before and after, then rebuilds primitives. */
int astr_device_nscbc_outlet_prepare(void *parent);
int astr_device_nscbc_outlet_filter(void *parent,int npdcj);
/* Five-variable low-order boundary commands, without blowing or chemistry.
 * Wall states preserve RHS; extrapolation/symmetry reset their selected face. */
int astr_device_wall41_basic_state(void *parent,int axis,int side,astr_numerical_real tw,astr_numerical_real c2,astr_numerical_real c6);
int astr_device_wall42_basic_state(void *parent,int axis,int side,astr_numerical_real c2,astr_numerical_real c6);
/* Original y-only 411/421 states, without blowing. Lower 411 uses a static
 * Fortran-order x<=xslip byte mask; geometric normals come from resident metrics. */
int astr_device_slip_mask_upload(void *parent,const int8_t *mask,size_t bytes);
int astr_device_slip_wall_y(void *parent,int code,int side,astr_numerical_real tw,astr_numerical_real c2,astr_numerical_real c6);
/* Existing RTI y-face states: lower rho/p=2/1, upper=1/2.5, zero velocity. */
int astr_device_fixed_rti_y(void *parent,int side,astr_numerical_real c2,astr_numerical_real c6);
/* Shipped LDC UDF: y-upper u=1, T=1, including lid/side-wall intersections. */
int astr_device_ldc_top_lid_y(void *parent,astr_numerical_real c2,astr_numerical_real c6);
/* State projection only: retain authoritative q density, impose no-slip/Tw.
 * This does not supply high-order derivatives, wall RHS or RK constraints. */
int astr_device_wall41_evolved_density_state(void *parent,int axis,int side,astr_numerical_real tw,astr_numerical_real c2,astr_numerical_real c6);
/* Prescribed lower-wall speed plane; no flow field is downloaded to build it. */
int astr_device_wall_forcing_prepare(void *parent);
int astr_device_wall_forcing_upload(void *parent,const astr_numerical_real *velocity,size_t bytes);
int astr_device_wall41_forcing_y(void *parent,astr_numerical_real tw,astr_numerical_real c2,astr_numerical_real c6);
/* The early x+/lower-y intersection, before imposing the rest of the wall. */
int astr_device_wall41_outlet_y(void *parent,int forcing,astr_numerical_real tw,astr_numerical_real c2,astr_numerical_real c6);
/* Existing high-order adiabatic no-slip x/y, or slip y, using authoritative q. */
int astr_device_wall_adiabatic_evolved_state(void *parent,int axis,int side,int slip,astr_numerical_real c2,astr_numerical_real c6);
int astr_device_zeroextrap_basic(void *parent,int axis,int side,astr_numerical_real c2,astr_numerical_real c6);
int astr_device_symmetry_basic(void *parent,int axis,int side,astr_numerical_real c2,astr_numerical_real c6);
/* Static x-min target: rho,u,v,w,p,T and the inward unit normal. Plane-only
 * Fortran-order storage; caller owns actual face selection and intersections. */
int astr_device_profile_inflow_prepare(void *parent);
/* Four original time-series slots of five primitive fluctuations, on the
 * x-min owner only. Transfer methods operate on one contiguous 2D slot. */
int astr_device_inflow_slices_prepare(void *parent);
int astr_device_inflow_slice_upload(void *parent,int slot,const astr_numerical_real *values,size_t bytes);
int astr_device_inflow_slice_download(void *parent,int slot,astr_numerical_real *values,size_t bytes);
int astr_device_dynamic_inflow_x(void *parent,const int slots[4],astr_numerical_real time0,astr_numerical_real delta,astr_numerical_real time,
    int mach_pressure,astr_numerical_real mach,astr_numerical_real c2,astr_numerical_real c6);
int astr_device_profile_inflow_x(void *parent,int mach_pressure,astr_numerical_real mach,astr_numerical_real c2,astr_numerical_real c6);
/* Actual legacy bc51 y-upper rule, not a characteristic farfield condition. */
int astr_device_farfield51_upper_basic(void *parent,astr_numerical_real c2,astr_numerical_real c6);
/* Original x-upper bc21: supersonic extrapolation or prescribed-pressure
 * subsonic outlet, retaining its boundary-temperature and x-velocity rules. */
int astr_device_outflow21_x_basic(void *parent,astr_numerical_real mach,astr_numerical_real c2,astr_numerical_real c6,astr_numerical_real pressure_target);
#ifdef __cplusplus
}
#endif
