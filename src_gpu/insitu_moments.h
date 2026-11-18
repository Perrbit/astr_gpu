#pragma once
#include "astr_precision.h"
#include "device_state.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Owned, half-open node box; a physical upper wall may add one owned node.
 * Parent/stream are borrowed. The solver q is never modified. */
int astr_insitu_moments_create(void *parent,const int owned[3],const astr_numerical_real window[2],void **moments);
/* Endpoint weights come from the shared Fortran clipped_trapezoid routine.
 * The first observation seeds previous rho/u with both weights zero. */
int astr_insitu_moments_push(void *moments,astr_numerical_real time,const astr_numerical_real weights[2]);
/* Borrowed (N,30) SoA state: previous rho/u, duration/mass, R/F means,
 * two full symmetric central matrices. Valid until push/restore/destroy. */
int astr_insitu_moments_view(void *moments,AstrDeviceView *view);
/* Explicit host consumers only. Packed state uses the existing 34-component
 * layout; derived output uses the existing 41-component field order. */
int astr_insitu_moments_pack(void *moments,astr_numerical_real *host,size_t bytes);
int astr_insitu_moments_restore(void *moments,const astr_numerical_real *host,size_t bytes,astr_numerical_real time);
int astr_insitu_moments_output(void *moments,astr_numerical_real *host,size_t bytes);
/* Static node measures upload on demand. Spatial returns only local volume,
 * weighted instantaneous u[3] and weighted temporal Reynolds variance[3].
 * Fortran retains ownership of global MPI reduction and normalization. */
int astr_insitu_moments_weights(void *moments,const astr_numerical_real *weights,size_t bytes);
int astr_insitu_moments_spatial(void *moments,astr_numerical_real local[7]);
/* Previous time, window endpoints, duration, has_previous (0 or 1). */
int astr_insitu_moments_info(void *moments,astr_numerical_real values[5]);
/* Current/peak owned device bytes, status D2H, explicit observation D2H,
 * explicit setup/restore H2D, successful observations. */
int astr_insitu_moments_metrics(void *moments,int64_t values[6]);
int astr_insitu_moments_destroy(void **moments);

/* Sparse selected-time groups. The compact state layout is the existing
 * insitu_sample_moments definition, not the full-velocity layout above.
 * Indices are (3,N), immutable local physical-node indices; views are borrowed. */
int astr_insitu_selected_create(void *parent,int nodes,const int *indices,int components,void **selected);
int astr_insitu_selected_push(void *selected,int fields,const int *ids,astr_numerical_real const2,astr_numerical_real const6,
    const astr_numerical_real weights[2],int previous,int covariance);
/* External device packet: columns=density/selected values, positions are unique
 * one-based group rows. The producer retains a non-aliasing packet until this
 * synchronized call returns. This existing wall-source path has no covariance. */
int astr_insitu_selected_scatter(void *selected,int nodes,const int *positions,const AstrDeviceView *values,
    int fields,const astr_numerical_real weights[2],int previous);
/* 0 state (N,C), 1 indices (N,3), each with explicit element stride. */
int astr_insitu_selected_view(void *selected,int field,AstrDeviceView *view);
int astr_insitu_selected_download(void *selected,astr_numerical_real *host,size_t bytes);
int astr_insitu_selected_upload(void *selected,const astr_numerical_real *host,size_t bytes);
/* (ny,nk) selected support, k-major order; returns existing profile terms
 * (2+2*fields+12*covariance,ny), not a downloaded point field.
 * Span measures are static for the lifetime of a selected group. */
int astr_insitu_selected_profile(void *selected,int ny,int nk,const astr_numerical_real *weights,int fields,int covariance,
    astr_numerical_real *profile,size_t bytes);
/* Device current/peak, update status D2H, explicit state D2H/H2D,
 * index H2D, successful updates, profile D2H/H2D. */
int astr_insitu_selected_metrics(void *selected,int64_t values[9]);
/* Actual optional profile and external-index device bytes, no allocation/readback. */
int astr_insitu_selected_workspace(void *selected,int64_t bytes[2]);
int astr_insitu_selected_destroy(void **selected);
#ifdef __cplusplus
}
#endif
