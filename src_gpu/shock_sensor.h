#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Existing divergence-squared Ducros/pressure-curvature sensor. The caller
 * supplies same-phase pressure/physical gradients and exchanges the raw sensor
 * halo without shared-node averaging between raw and expand. No H1 candidate. */
int astr_device_shock_prepare(void *handle);
int astr_device_shock_raw(void *handle,const int low[3],const int high[3]);
int astr_device_shock_expand(void *handle,const int low[3],const int high[3],astr_numerical_real threshold);
/* Refresh one expanded-mask face layer using the configured transport.
 * No shared-node averaging. The original compact-mask selector calls this
 * only for 743e/643c; expand invalidates the previous face layer. */
int astr_device_shock_mask_exchange(void *handle);
/* Validation observations only. Physical markers are in Fortran order.
 * Faces are axis 0/1/2, lower then upper, cyclic transverse first index fastest. */
int astr_device_shock_mask_download(void *handle,int8_t *host,size_t bytes);
int astr_device_shock_faces_download(void *handle,int8_t *host,size_t bytes);
#ifdef __cplusplus
}
#endif
