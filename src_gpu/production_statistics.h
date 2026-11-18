#pragma once
#include <stddef.h>
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
enum AstrProductionStatisticsBuffer {
    ASTR_PRODUCTION_MEASURE=0, ASTR_PRODUCTION_SUM=1, ASTR_PRODUCTION_ENDPOINT=2,
    ASTR_PRODUCTION_WALL_MEASURE=3, ASTR_PRODUCTION_WALL_SUM=4, ASTR_PRODUCTION_WALL_ENDPOINT=5
};
/* Immutable segment lengths: (0:im,0:jm,1:km), Fortran order.
 * Optional lower-wall frame: (0:im,0:km,6), normal then projected unit tangent.
 * Geometry is prepared once from the solver's immutable coordinates. */
int astr_device_production_prepare(void *parent,const astr_numerical_real *segments,const astr_numerical_real *wall_frame);
int astr_device_production_reset(void *parent);
int astr_device_production_plane(void *parent,astr_numerical_real interval,int has_previous);
/* constants: Re, Pr, const5, cp, tempconst, tempconst1. */
int astr_device_production_wall(void *parent,astr_numerical_real interval,int has_previous,
    int nondimensional,const astr_numerical_real constants[6]);
/* Explicit output/restart events only; upload is allowed for sums/endpoints, not geometry. */
int astr_device_production_transfer(void *parent,int buffer,int upload,astr_numerical_real *host,size_t bytes);
#ifdef __cplusplus
}
#endif
