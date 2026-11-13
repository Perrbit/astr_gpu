#pragma once
#include <cuda_runtime.h>
#include "device_state.h"

namespace astr {
constexpr int block_threads = 512;
struct Buffer {
    AstrDeviceView view{};
    size_t bytes = 0;
};
struct HaloTransport;
int release_transport(HaloTransport *transport);
bool transport_in_flight(const HaloTransport *transport);
struct State {
    cudaStream_t stream = nullptr;
    Buffer fields[ASTR_DEVICE_FIELD_COUNT];
    int upper[3]{};
    int halo = 0;
    bool debug_sync = false;
    bool fluid_ready = false;
    bool gradients_ready = false;
    bool rhs_ready = false;
    bool viscous_ready = false;
    int *fluid_failure = nullptr;
    double *cfl_partial = nullptr;
    double *cfl_values = nullptr;
    double *statistics_partial = nullptr;
    double *statistics_values = nullptr;
    void *cfl_records = nullptr;
    double *cfl_diagnostic_values = nullptr;
    int64_t *cfl_diagnostic_keys = nullptr;
    double *cfl_profile = nullptr;
    HaloTransport *transport = nullptr;
};
int allocate_field(Buffer &buffer, const int upper[3], int halo, int rank,
                   int components = 1, int directions = 1);
inline int finish_launch(State &state)
{
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    return state.debug_sync ? static_cast<int>(cudaStreamSynchronize(state.stream)) : 0;
}
__device__ inline size_t offset(const AstrDeviceView &view, int i, int j, int k,
                               int c = 1, int d = 1)
{
    return static_cast<size_t>(i - view.lower[0]) * view.stride[0]
         + static_cast<size_t>(j - view.lower[1]) * view.stride[1]
         + static_cast<size_t>(k - view.lower[2]) * view.stride[2]
         + static_cast<size_t>(c - view.lower[3]) * view.stride[3]
         + static_cast<size_t>(d - view.lower[4]) * view.stride[4];
}
__device__ inline double &at(const AstrDeviceView &view, int i, int j, int k,
                            int c = 1, int d = 1)
{
    return static_cast<double *>(view.data)[offset(view, i, j, k, c, d)];
}
}
