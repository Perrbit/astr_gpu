#pragma once
#include "device_state.h"
#ifdef __cplusplus
extern "C" {
#endif
enum AstrHaloTransportMode { ASTR_HALO_PINNED=0, ASTR_HALO_DEVICE=1 };
/* Fortran supplies topology and starts/completes exchanges on the MPI thread.
 * Fixed slots cannot be reused until end; device mode passes their actual
 * device pointers to MPI. No MPI or CUDA ABI types cross this boundary. */
int astr_device_transport_prepare(void *handle,int communicator,const int lower[3],const int upper[3],
                                  const int sizes[3],int max_components,int mode,int timing);
int astr_device_exchange_begin(void *handle,int field,int axis,int reconcile);
int astr_device_exchange_end(void *handle,int field,int axis,int reconcile);
int astr_device_exchange_axis(void *handle,int field,int axis,int reconcile);
/* Host durations: send-ready, posting, Waitall, receive/unpack enqueue;
 * followed by sent bytes and application D2H+H2D bytes. Not additive GPU time. */
int astr_device_transport_metrics(void *handle,double values[6],int reset);
#ifdef __cplusplus
}
#endif
