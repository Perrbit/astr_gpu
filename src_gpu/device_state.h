#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum AstrDeviceField {
    ASTR_DEVICE_Q = 0, ASTR_DEVICE_RHS = 1, ASTR_DEVICE_SAVE = 2,
    ASTR_DEVICE_RHO = 3, ASTR_DEVICE_VELOCITY = 4, ASTR_DEVICE_PRESSURE = 5,
    ASTR_DEVICE_TEMPERATURE = 6, ASTR_DEVICE_METRIC = 7,
    ASTR_DEVICE_DVELOCITY = 8, ASTR_DEVICE_DTEMPERATURE = 9,
    ASTR_DEVICE_JACOBIAN = 10, ASTR_DEVICE_STRESS = 11, ASTR_DEVICE_HEAT_FLUX = 12,
    ASTR_DEVICE_FILTER_WORK = 13, ASTR_DEVICE_FIELD_COUNT = 14
};
enum { ASTR_DEVICE_INVALID_ARGUMENT = -1, ASTR_DEVICE_NOT_ALLOCATED = -2,
       ASTR_DEVICE_NONPHYSICAL = -3, ASTR_DEVICE_INVALID_SPECTRUM = -4 };

/* FP64 Fortran-order borrowed view; strides are in elements, not bytes.
 * Upload/download complete before returning. Borrowed addresses expire on destroy.
 * The owning rank keeps the selected device current throughout the context lifetime. */
typedef struct AstrDeviceView {
    void *data;
    int rank;
    int element_bytes;
    int lower[5];
    int extent[5];
    int64_t stride[5];
} AstrDeviceView;

int astr_device_bind(int device);
int astr_device_count(int *count);
int astr_device_state_create(int device, int im, int jm, int km, int halo,
                             int debug_sync, void **handle);
int astr_device_state_view(void *handle, int field, AstrDeviceView *view);
int astr_device_state_upload(void *handle, int field, const double *host, size_t bytes);
int astr_device_state_download(void *handle, int field, double *host, size_t bytes);
/* Test observation: physical nodes plus the selected axis's face halos.
 * Other entries are zero padding, not reads of undefined corner storage. */
int astr_device_state_snapshot_axis(void *handle, int field, int axis, double *host, size_t bytes);
int astr_device_state_synchronize(void *handle);
int astr_device_state_destroy(void **handle);

#ifdef __cplusplus
}
#endif
