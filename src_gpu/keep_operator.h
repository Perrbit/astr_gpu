#pragma once
#include "device_state.h"
#include "compact_operator.h"

#ifdef __cplusplus
extern "C" {
#endif
int astr_device_keep_explicit(void *parent, astr_numerical_real gamma);
int astr_device_keep_sbp63_convection(void *parent,int total,int start,astr_numerical_real gamma);
/* Accumulates -div(F) only in [first,last], retaining prior boundary RHS. */
int astr_device_keep_explicit_accumulate_basic(void *parent, astr_numerical_real gamma, const int types[3],
                                              const int first[3], const int last[3]);
int astr_device_keep_prepare(void *parent, int terms, int bases, const int *left, const int *right,
                              const int *component, const int *basis, const astr_numerical_real *weight);
int astr_device_keep_compact_rhs(void *parent, void *line, const AstrCompactStencil *stencil, int nrhs, astr_numerical_real gamma);
int astr_device_keep_compact_commit(void *parent, void *line, const AstrCompactStencil *stencil, int nrhs, astr_numerical_real gamma);
#ifdef __cplusplus
}
#endif
