#pragma once
#include "astr_precision.h"
#include "device_state.h"

#ifdef __cplusplus
extern "C" {
#endif
enum AstrSampleField { ASTR_SAMPLE_VELOCITY=0,ASTR_SAMPLE_HALO=1,ASTR_SAMPLE_DIAGNOSTICS=2,ASTR_SAMPLE_STATE=3,
                      ASTR_SAMPLE_COORDINATES=4,ASTR_SAMPLE_GEOMETRY=5 };
/* Independent complete-step sample. Q-based capture takes canonical upper nodes
 * from the next rank's lower face; vector capture preserves its selected source.
 * Neither averages or changes solver fields. Parent and stream are borrowed.
 * Velocity/diagnostics are component-interleaved (3,N)/(1|14,N); halo is SoA.
 * Output views stay valid until the next begin/destroy. Canonical state is a
 * temporary SoA view between endpoint completion and velocity conversion.
 * All consumers must finish before their owner advances to either event. */
int astr_insitu_sample_create(void *parent,int communicator,const int lower[3],const int upper[3],
    const int sizes[3],const int periodic[3],int mode,int velocity,int halo,int diagnostic_components,
    int wall_sixth,void **sample);
int astr_insitu_sample_begin(void *sample);
/* Borrow an external device view with rows=local physical nodes. Columns are
 * selected by zero-based slots. Do not pass an output retired by this capture. */
int astr_insitu_sample_vector(void *sample,const AstrDeviceView *source,const int slots[3]);
/* Halo-only vector from unique owned-node rows (e.g. saved means). Complete
 * duplicate endpoints by owner copy, then exchange private halo; no q access. */
int astr_insitu_sample_owned_vector(void *sample,const AstrDeviceView *source,const int owned[3],const int slots[3]);
/* Extend physical positions without periodic wrapping or changing the velocity
 * halo. Internal MPI faces only; global exterior remains private zero padding. */
int astr_insitu_sample_coordinates(void *sample,const AstrDeviceView *source,const int offset[3],const int global[3]);
/* One-time upload of saved physical coordinates (3,N), retained with their
 * private coordinate halo. No periodic coordinate wrapping. Setup H2D=bytes,
 * separate from MPI-staging counters returned by sample_metrics. */
int astr_insitu_sample_upload_coordinates(void *sample,const astr_numerical_real *xyz,size_t bytes,
    const int offset[3],const int global[3]);
int astr_insitu_sample_canonical_axis(void *sample,int axis);
int astr_insitu_sample_velocity(void *sample);
int astr_insitu_sample_halo_axis(void *sample,int axis);
int astr_insitu_sample_finish(void *sample);
int astr_insitu_sample_view(void *sample,int field,AstrDeviceView *view);
/* Explicit host consumer/test observation, not part of device capture. */
int astr_insitu_sample_download(void *sample,int field,astr_numerical_real *host,size_t bytes);
/* device current/peak, pinned, MPI sent, staging D2H/H2D, scalar D2H,
 * explicit observation D2H, completed captures; byte counts are application-level. */
int astr_insitu_sample_metrics(void *sample,int64_t values[9]);
int astr_insitu_sample_destroy(void **sample);
/* Sparse normalized primitive probes. Indices are component-interleaved local
 * physical nodes; (-1,-1,-1) contributes zero for a non-owner. Constants are
 * const2,const6,velocity_scale,rho_ref,p_ref,T_ref. Output layout is (6,count).
 * The parent stream/state is borrowed and remains unchanged. */
int astr_insitu_probe_create(void *parent,int count,const int *indices,
    const astr_numerical_real constants[6],void **probe);
int astr_insitu_probe_capture(void *probe,astr_numerical_real *values,size_t bytes);
/* device bytes, setup-index H2D, captures, status D2H, values D2H */
int astr_insitu_probe_metrics(void *probe,int64_t values[5]);
int astr_insitu_probe_destroy(void **probe);
#ifdef __cplusplus
}
#endif
