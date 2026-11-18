#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Existing five-variable, nondimensional Steger-Warming + MP7/MP7-LD path.
 * Pressure, velocity, temperature and q must belong to the same RHS phase.
 * Caller supplies state/geometry face halos and current physical sensor mask.
 * A completed shock-mask exchange selects the original expanded interface rule;
 * otherwise the nonexpanded nearest-physical-node rule is retained. High-order
 * boundary overrides and crashfix critical nodes remain separate consumers. */
int astr_device_upwind_prepare(void *handle);
/* Interface values only, without RHS accumulation. The positivity operator
 * zeros unused exterior physical faces before constructing its correction. */
int astr_device_upwind_positivity_faces(void *handle,int axis,int high_order,
    astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real weight,
    const int low[3],const int high[3]);
/* Add the H1 upper x/y faces, excluded from the MP body box, using the current
 * directional interface cache and the existing normal derivative. Prescribed
 * lower faces are excluded and an upper x/y intersection is visited once. */
int astr_device_upwind_high_order_boundary_rhs(void *handle,int axis,const int low[3],const int high[3]);
/* Existing MP1: periodic physical-space MP7; F+ is rounded to FP32 before
 * adding FP64 F-, then the sum is stored FP32. State/RHS arithmetic stays FP64.
 * This is separate from viscous_flux and selective characteristic storage. */
int astr_device_upwind_prepare_mixed_flux(void *handle);
/* Existing MP3: all reconstruction/Roe arithmetic is FP64; cast only the
 * completed physical interface flux to FP32, including unmarked interfaces. */
int astr_device_upwind_prepare_mixed_characteristic_flux(void *handle);
int astr_device_upwind_axis(void *handle,int axis,int reconstruction,int characteristic,
    astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real weight,int metric_consistent,
    const int low[3],const int high[3],const int first[3],const int last[3]);
/* Production H1 MP7-LD physical-end interface reconstruction, not test candidates.
 * Only smooth, noncritical physical-end rows use the biased sixth-order flux.
 * Upper physical nodes included in first:last use the original eight-node normal
 * derivative; other contributions use interface-flux differences. The caller
 * still owns prescribed-face zeros and complete NSCBC. */
int astr_device_upwind_high_order_axis(void *handle,int axis,astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real weight,
    const int low[3],const int high[3],const int first[3],const int last[3]);
#ifdef __cplusplus
}
#endif
