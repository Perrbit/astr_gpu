#pragma once
#include "device_state.h"
#include "astr_precision.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AstrCompactStencil {
    int axis, first, total, periodic, kind, start, component_offset;
    /* Destination ranges only: nodal divergence uses its current axis; KEEP
     * uses the common 3D box. Never truncate RHS/system assembly with these. */
    int active_first[3], active_last[3];
    astr_numerical_real c6[4], c8[5], c10[6], boundary[21];
} AstrCompactStencil;

/* kind: 1 primitive gradient, 2 conservative convection, 3 viscous divergence,
 * 4 conservative filter. All source/destination fields remain on the device. */
int astr_compact_field_rhs(void *parent, void *line, const AstrCompactStencil *stencil, int nrhs);
int astr_compact_field_commit(void *parent, void *line, const AstrCompactStencil *stencil, int nrhs);
int astr_compact_field_commit_wall_band(void *parent, void *line, const AstrCompactStencil *stencil, int nrhs);
int astr_compact_rhs_clear(void *parent);
int astr_compact_configure_sync(void *parent, int explicit_sync);
typedef struct AstrWall5FilterPlan {
    astr_numerical_real rows[5][11], inverse[5], alpha;
} AstrWall5FilterPlan;
int astr_compact_filter_wall5(void *parent, const AstrWall5FilterPlan *plan, int walls);

#ifdef __cplusplus
}
#endif
