#pragma once
#include "device_state.h"
#include "astr_precision.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Numeric factors, workspaces and packets follow the build precision.
 * Host-prepared factors use the existing Fortran column-major layout and
 * one-based first/last/owner entries. Creation copies every table to device. */
typedef struct AstrCompactFactors {
    int n, tile_capacity, tiles, levels, reduced_n, reduced_levels, cyclic;
    int *first, *last, *owner, *local_levels;
    astr_numerical_real *initial, *left, *right, *inverse, *edge_lower, *edge_upper;
    astr_numerical_real *response_left, *response_right;
    astr_numerical_real *reduced_initial, *reduced_left, *reduced_right, *reduced_inverse;
    astr_numerical_real *cyclic_response;
    astr_numerical_real correction[4];
} AstrCompactFactors;

/* The plan borrows its parent's device/stream. Destroy plans before the parent.
 * Borrowed RHS/solution views expire when the plan is destroyed. */
int astr_compact_line_create(void *parent, const AstrCompactFactors *factors,
                             int capacity, void **handle, size_t *device_bytes);
int astr_compact_line_views(void *handle, AstrDeviceView *rhs, AstrDeviceView *solution);
int astr_compact_line_solve(void *handle, int nrhs);
/* Explicit host observation/input only: 0 upload RHS, 1 read RHS, 2 read solution.
 * Normal device callers fill/read the borrowed views on the parent stream. */
int astr_compact_line_transfer(void *handle, int mode, astr_numerical_real *host, int nrhs);
int astr_compact_line_destroy(void **handle);

typedef struct AstrCompactDistributedFactors {
    int communicator, levels, cyclic, mode, nodes;
    astr_numerical_real *response, *coefficient;
    astr_numerical_real edges[2];
} AstrCompactDistributedFactors;

/* Successful creation consumes the local handle. The caller owns the private
 * Fortran communicator until distributed destruction has completed. */
int astr_compact_distributed_create(void **local, const AstrCompactDistributedFactors *factors,
                                    void **handle, size_t *device_bytes, size_t *host_bytes);
int astr_compact_distributed_views(void *handle, AstrDeviceView *rhs,
                                  AstrDeviceView *solution, AstrDeviceView *nodes);
int astr_compact_distributed_solve(void *handle, int nrhs);
int astr_compact_distributed_restore(void *handle, int nrhs);
/* Modes 0/1/2 match local transfer; mode 3 reads restored shared nodes. */
int astr_compact_distributed_transfer(void *handle, int mode, astr_numerical_real *host, int nrhs);
int astr_compact_distributed_metrics(void *handle, int64_t values[2], int reset);
int astr_compact_distributed_destroy(void **handle);

#ifdef __cplusplus
}
#endif
