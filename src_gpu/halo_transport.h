#pragma once
#include "device_state.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Fortran supplies the topology and an MPI Fortran communicator handle.
 * The transport owns a duplicated communicator and fixed device/pinned slots;
 * no MPI or CUDA ABI types cross this boundary. Destroy before MPI_Finalize. */
int astr_device_transport_prepare(void *handle,int communicator,const int lower[3],const int upper[3],
                                  const int sizes[3],int max_components);
int astr_device_exchange_axis(void *handle,int field,int axis,int reconcile);
#ifdef __cplusplus
}
#endif
