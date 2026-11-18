#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Existing compact monitor packet: (xyz3,q11,owner1), Fortran (15,count).
 * Static coordinates and local physical indices use Fortran (3,count).
 * A negative i marks a nonowner; its complete packet is positive zero.
 * Prepare once per context; collect downloads only 15*count FP64 words. */
int astr_device_air5_monitor_prepare(void *handle,int count,const int *nodes,const double *xyz);
int astr_device_air5_monitor_collect(void *handle,double *packed,size_t bytes);
/* Existing FP64 AIR5 main time statistics, not the in-situ central moments.
 * Optional physical-node SUM44 and ENDPOINT14 use the ordinary state views and
 * explicit output/restart transfers. No allocation occurs before prepare.
 * Caller owns the original sample identity, time coverage and global release;
 * sample only after primitives and physical velocity gradients are current. */
int astr_device_air5_mean_prepare(void *handle);
int astr_device_air5_mean_reset(void *handle);
/* Retire external consumers first. Release waits the owning stream, invalidates
 * the two borrowed statistic views, and leaves other solver fields allocated. */
int astr_device_air5_mean_release(void *handle);
int astr_device_air5_mean_refresh_velocity(void *handle,const int low[3],const int high[3]);
int astr_device_air5_conservation_prepare(void *handle,int64_t budget,int64_t *bytes);
int astr_device_air5_conservation_collect(void *handle,double totals[11]);
int astr_device_air5_mean_sample(void *handle,double interval,int has_previous);
/* Failed nodes do not commit; other nodes may have committed. No rollback.
 * This separate status slot does not overwrite the solver's physics status. */
int astr_device_air5_mean_check(void *handle,int *status);
#ifdef __cplusplus
}
#endif
