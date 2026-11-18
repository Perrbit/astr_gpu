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
    ASTR_DEVICE_FILTER_WORK = 13, ASTR_DEVICE_COORDINATE_Y = 14,
    ASTR_DEVICE_SHOCK_SENSOR = 15, ASTR_DEVICE_INTERFACE_FLUX = 16,
    ASTR_DEVICE_INFLOW_TARGET = 17, ASTR_DEVICE_CELL_VOLUME = 18,
    ASTR_DEVICE_WALL_CELL_AREA = 19, ASTR_DEVICE_DIAGNOSTIC_DVELOCITY = 20,
    ASTR_DEVICE_DIAGNOSTIC_DTEMPERATURE = 21,
    ASTR_DEVICE_MP_DELTA = 22, ASTR_DEVICE_MP_RATIO = 23,
    ASTR_DEVICE_AIR5_TV = 24, ASTR_DEVICE_AIR5_SPECIES = 25,
    ASTR_DEVICE_AIR5_CARRY = 26, ASTR_DEVICE_AIR5_HISTORY = 27,
    ASTR_DEVICE_AIR5_ORIGIN = 28, ASTR_DEVICE_AIR5_ORIGIN_CARRY = 29,
    ASTR_DEVICE_AIR5_DTV = 30, ASTR_DEVICE_AIR5_DSPECIES = 31,
    ASTR_DEVICE_AIR5_SPECIES_FLUX = 32, ASTR_DEVICE_AIR5_EV_FLUX = 33,
    ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX = 34, ASTR_DEVICE_AIR5_CONVECTION_BASE = 35,
    ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH = 36,
    ASTR_DEVICE_AIR5_MEAN_SUM = 37, ASTR_DEVICE_AIR5_MEAN_ENDPOINT = 38,
    ASTR_DEVICE_AIR5_DIFFUSION_RATIO = 39, ASTR_DEVICE_AIR5_DIFFUSION_ENERGY_RATIO = 40,
    ASTR_DEVICE_AIR5_CONVECTION_FLUID_RATIO = 41,
    ASTR_DEVICE_FIELD_COUNT = 42
};
enum { ASTR_DEVICE_INVALID_ARGUMENT = -1, ASTR_DEVICE_NOT_ALLOCATED = -2,
       ASTR_DEVICE_NONPHYSICAL = -3, ASTR_DEVICE_INVALID_SPECTRUM = -4 };

/* Fortran-order borrowed view; element_bytes describes the actual storage.
 * q/RHS/save storage follows the shared numerical build precision. Complete
 * flow arithmetic is currently connected only for FP64; its MP2 candidates
 * may store viscous fluxes or separate diagnostic gradients in FP32.
 * Strides are in elements, not bytes.
 * Upload/download complete before returning. Borrowed addresses expire on destroy
 * or when the initial five-component allocation is prepared for AIR5.
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
int astr_device_state_upload(void *handle, int field, const void *host, size_t bytes);
int astr_device_state_download(void *handle, int field, void *host, size_t bytes);
/* Test observation: physical nodes plus the selected axis's MPI/periodic face
 * halos. low/high mark true physical ends; their exterior, and other-axis
 * halos/corners, are zero padding rather than reads of undefined storage.
 * Diagnostic output is FP64, with exact promotion of finite FP32 values. */
int astr_device_state_snapshot_axis(void *handle, int field, int axis,
    const int low[3],const int high[3],double *host,size_t bytes);
int astr_device_state_synchronize(void *handle);
int astr_device_memory_info(void *handle,size_t *free_bytes,size_t *total_bytes);
int astr_device_pci_identity(void *handle,char *identity,int capacity);
int astr_device_state_destroy(void **handle);

#ifdef __cplusplus
}
#endif
