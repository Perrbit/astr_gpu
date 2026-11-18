#pragma once
#include "device_state.h"
#include "astr_precision.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AstrCrashfixReport {
    int candidates,repaired,invalid;
    astr_numerical_real max_change[5],minima[4];
} AstrCrashfixReport;
/* A single original ordered-neighbour pass. Unresolved states are reported,
 * not declared repaired; expansion/reconciliation belong to the caller. */
int astr_device_crashfix_prepare(void *parent,const int8_t *fluid,size_t bytes);
int astr_device_crashfix_history_upload(void *parent,const int8_t *mask,size_t bytes);
int astr_device_crashfix_history_download(void *parent,int8_t *mask,size_t bytes);
/* OR all shared/periodic copies, then refresh hm face-halo layers. Uses the
 * prepared pinned/device transport slots and never changes q. */
int astr_device_crashfix_history_exchange(void *parent);
int astr_device_crashfix_ordered_pass(void *parent,const int origin[3],const int global[3],
    astr_numerical_real c2,astr_numerical_real c6,astr_numerical_real eps_pressure,AstrCrashfixReport *report);
/* One original expanded-search round after halo-only q exchange. Diagnostics:
 * donor presence, repaired this round, cumulative radius, single-donor count. */
int astr_device_crashfix_expanded_pass(void *parent,const int lower[3],const int upper[3],
    astr_numerical_real c2,astr_numerical_real c6,astr_numerical_real eps_pressure,AstrCrashfixReport *report,int diagnostics[4]);
int astr_device_crashfix_recovery_halos(void *parent,astr_numerical_real c2,astr_numerical_real c6);
int astr_device_crashfix_reconciled_primitives(void *parent,astr_numerical_real c2,astr_numerical_real c6);
/* Original post-reconciliation test: both conservative kinetic-energy
 * orderings, not the thermal donor/repair-floor test. Does not change q. */
int astr_device_crashfix_input_invalid(void *parent,int *invalid);
#ifdef __cplusplus
}
#endif
