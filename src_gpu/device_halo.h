#pragma once
#include "device_state.h"
#ifdef __cplusplus
extern "C" {
#endif
/* A local periodic axis, not a replacement for a distributed-axis exchange.
 * reconcile=1 averages the two physical endpoint planes first. Transverse
 * indices remain physical; call a final halo-only pass after all averaging. */
int astr_device_periodic_axis(void *handle, int field, int axis, int reconcile);
#ifdef __cplusplus
}
#endif
