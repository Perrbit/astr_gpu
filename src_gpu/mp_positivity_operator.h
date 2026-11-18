#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional, recomputable RK scratch. No allocation or communication while off.
 * The MPI communicator and neighbors are supplied by the Fortran orchestrator. */
int astr_device_mp_positivity_prepare(void *handle,int communicator,const int lower[3],const int upper[3],int mode);
int astr_device_mp_positivity_begin(void *handle);
int astr_device_mp_positivity_axis(void *handle,int axis,int high_order,
    astr_numerical_real gamma,astr_numerical_real mach,astr_numerical_real weight,
    const int low[3],const int high[3],const int first[3],const int last[3]);
int astr_device_mp_positivity_faces_complete(void *handle);
/* Invoke after the complete viscous/boundary/source RHS, before the RK update. */
int astr_device_mp_positivity_limit(void *handle,astr_numerical_real dt,astr_numerical_real a,
    astr_numerical_real b,astr_numerical_real c,int stage,const int first[3],const int last[3],int *limited);
#ifdef __cplusplus
}
#endif
