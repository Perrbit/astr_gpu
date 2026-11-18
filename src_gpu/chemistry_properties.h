#pragma once
#ifdef __cplusplus
extern "C" {
#endif
/* Prepare once on a fresh FP64 context, before uploading state or preparing MPI.
 * q/RHS/save have 11 components. Existing nonreacting contexts remain unchanged.
 * The refresh box is inclusive and may include face halos, not exterior corners.
 * Launches use the owning stream; check waits and returns the chemistry status.
 * No field is copied to the host by a conversion; check reads one status integer. */
int astr_device_air5_prepare(void *handle);
int astr_device_air5_conservative(void *handle,const int first[3],const int last[3]);
int astr_device_air5_primitives(void *handle,const int first[3],const int last[3],int preserve_all);
/* Refresh several inclusive boxes in one status phase. Bounds are Fortran
 * (3,regions), i.e. three coordinates per box. Clears status once; later valid
 * regions cannot erase an earlier failure. Failed nodes do not write caches.
 * API errors are fatal to the phase; no cache rollback is performed. */
int astr_device_air5_primitives_regions(void *handle,int regions,const int *first,const int *last,int preserve_all);
int astr_device_air5_check(void *handle,int *chemistry_status);
#ifdef __cplusplus
}
#endif
