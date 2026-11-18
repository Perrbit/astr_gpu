#pragma once
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Body bounds are inclusive. Negative begin marks a locally absent layer.
 * Coefficients are packed in Fortran spatial order over the clipped region. */
int astr_device_sponge_prepare(void *parent,const int first[3],const int last[3],
    const int begin[2],const int end[2],const astr_numerical_real *x_coefficient,const astr_numerical_real *y_coefficient);
/* Caller refreshes q halo without shared-node averaging before each axis. */
int astr_device_sponge_apply(void *parent,int axis);
#ifdef __cplusplus
}
#endif
