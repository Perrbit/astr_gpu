#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
int astr_device_me6_prepare(void *parent, int communicator, const int *lower, const int *upper,
    const int *parts, const int *periodic, const int *valid_first, const int *valid_last,
    const int *inner_first, const int *inner_last, int mode, const astr_numerical_real *ci, const astr_numerical_real *cd, const astr_numerical_real *cp);
int astr_device_me6_diffusion(void *parent, astr_numerical_real reynolds, astr_numerical_real prandtl, astr_numerical_real c5, astr_numerical_real tc, astr_numerical_real tc1);
/* Controlled device/pinned bytes and application sent/staging bytes. No device readback. */
int astr_device_me6_metrics(void *parent, int64_t values[4]);
#ifdef __cplusplus
}
#endif
