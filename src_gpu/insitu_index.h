#pragma once
#include "astr_precision.h"
#include "device_state.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AstrIndexRegion {
    int first[3],extent[3],lower[3],shape[3],origin[3],intervals[3],periodic[3];
    int cartesian_support,nondimen;
    astr_numerical_real tempconst,tempconst1,reynolds,prandtl,const5,cp;
} AstrIndexRegion;
/* Geometry is static and copied once per parent; group buffers expire at end.
 * Values use (N,fields) SoA and coordinates (3,N) interleaved, as in the existing
 * index provider. Grid indices are signed local solver indices; positions are
 * Fortran 1-based slots in the private stencil neighborhood. */
int astr_insitu_index_begin(void *parent,int nodes,int output_nodes,int raw_count,const int *raw,
    int field_count,const int *fields,const astr_numerical_real weights[49],int geometry,const astr_numerical_real *xyz,size_t xyz_bytes);
int astr_insitu_index_fill(void *parent,int nodes,const int *indices,const int *positions);
int astr_insitu_index_sample(void *parent,int nodes,const int *indices,astr_numerical_real *values);
int astr_insitu_index_receive(void *parent,int nodes,const int *positions,const astr_numerical_real *values);
int astr_insitu_index_exchange_prepare(void *parent,int capacity,int mode);
int astr_insitu_index_exchange(void *parent,int sends,const int *indices,int receives,const int *positions,
    int destination,int source,int communicator);
int astr_insitu_index_exchange_end(void *parent);
int astr_insitu_index_evaluate(void *parent,const AstrIndexRegion *region);
/* Observation fields: 0 selected values, 1 stencil neighborhood, 2 coordinates. */
int astr_insitu_index_download(void *parent,int field,astr_numerical_real *values,size_t bytes);
int astr_insitu_index_view(void *parent,void **values,void **coordinates);
/* Borrow rows=physical nodes and columns=fields/xyz. The coordinates view is
 * strided over the existing interleaved storage, without allocating a copy. */
int astr_insitu_index_source(void *parent,int coordinates,AstrDeviceView *view);
/* 0 instantaneous, 1 Reynolds, 2 Favre velocity. Borrow the owned moments;
 * mean center values use the completed unique-owner neighborhood packet. */
int astr_insitu_index_statistics(void *parent,const AstrDeviceView *state,const int owned[3],int source);
int astr_insitu_index_air5_mean(void *parent,double duration,int source);
int astr_insitu_index_selected(void *parent,const AstrDeviceView *state,const AstrDeviceView *indices,
    int fields,const int *ids,const int cells[3],int source,astr_numerical_real const2,astr_numerical_real const6);
int astr_insitu_index_usage(void *parent,int64_t values[3]);
int astr_insitu_index_end(void *parent);
#ifdef __cplusplus
}
#endif
