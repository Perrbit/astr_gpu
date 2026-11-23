#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct AstrAir5ChemistryOptions {
    int compensated,history_enabled,source_mode,source_only,preserve_all;
    int block_size,specialized_dispatch;
    double rtol,atol_factor,activity_threshold;
} AstrAir5ChemistryOptions;
typedef struct AstrAir5ChemistryReport {
    int status,max_counts[4],failure[6];
    int64_t skipped,total;
    int64_t minimum_bits[3],maximum_bits[2];
    double failure_q[11],failure_carry[6];
#ifdef ASTR_AIR5_WORKLOAD_PROBE
    int64_t sum_counts[4];
#endif
} AstrAir5ChemistryReport;
/* Prepare once after air5_prepare. Carry/history are allocated only if enabled.
 * begin resets one half-step; region may be called for interior and dynamic top.
 * Caller supplies the existing MPI active boxes and performs global failure reduction.
 * Successful cells commit q[6:11], optional carry/history and matching primitives;
 * unsuccessful cells retain their input. There is no field-wide rollback here.
 * status reads one scalar; report is requested only for diagnostics or failure. */
int astr_device_air5_chemistry_prepare(void *handle,const AstrAir5ChemistryOptions *options);
int astr_device_air5_chemistry_begin(void *handle);
int astr_device_air5_chemistry_region(void *handle,const int first[3],const int last[3],double duration,int collect);
int astr_device_air5_chemistry_status(void *handle,int *status);
int astr_device_air5_chemistry_report(void *handle,AstrAir5ChemistryReport *report);
#ifdef __cplusplus
}
#endif
