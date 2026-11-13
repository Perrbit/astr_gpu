#include <cuda_runtime.h>
#include "device_state.h"

namespace {
__device__ size_t offset(const AstrDeviceView &view, const int index[5])
{
    size_t result = 0;
    for (int axis = 0; axis < 5; ++axis)
        result += static_cast<size_t>(index[axis] - view.lower[axis]) * view.stride[axis];
    return result;
}

__global__ __launch_bounds__(512) void reverse_kernel(AstrDeviceView q, double *output, size_t count)
{
    const size_t point = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point < count) {
        size_t remaining = point;
        int index[5];
        for (int axis = 0; axis < 5; ++axis) {
            index[axis] = q.lower[axis] + q.extent[axis] - 1 - remaining % q.extent[axis];
            remaining /= q.extent[axis];
        }
        output[point] = static_cast<const double *>(q.data)[offset(q, index)];
    }
    return;
}

__global__ __launch_bounds__(512) void physical_kernel(AstrDeviceView q, AstrDeviceView rhs, size_t count)
{
    const size_t point = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (point < count) {
        size_t remaining = point;
        int index[5];
        for (int axis = 0; axis < 5; ++axis) {
            index[axis] = rhs.lower[axis] + remaining % rhs.extent[axis];
            remaining /= rhs.extent[axis];
        }
        static_cast<double *>(rhs.data)[point] = static_cast<const double *>(q.data)[offset(q, index)];
    }
    return;
}
}

// Independent consumer of borrowed views; it owns its diagnostic stream and buffer.
extern "C" int astr_mb1_consume(void *state, double *reversed, size_t count)
{
    AstrDeviceView q{}, rhs{};
    int code = astr_device_state_view(state, ASTR_DEVICE_Q, &q);
    if (code != 0) return code;
    code = astr_device_state_view(state, ASTR_DEVICE_RHS, &rhs);
    if (code != 0) return code;
    if (count != static_cast<size_t>(q.stride[4])) return ASTR_DEVICE_INVALID_ARGUMENT;
    double *output = nullptr;
    cudaStream_t stream = nullptr;
    cudaError_t status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return static_cast<int>(status);
    status = cudaMalloc(&output, count * sizeof(double));
    if (status == cudaSuccess) {
        reverse_kernel<<<(count + 511) / 512, 512, 0, stream>>>(q, output, count);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        const size_t physical = rhs.stride[4];
        physical_kernel<<<(physical + 511) / 512, 512, 0, stream>>>(q, rhs, physical);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaStreamSynchronize(stream);
    if (status == cudaSuccess)
        status = cudaMemcpy(reversed, output, count * sizeof(double), cudaMemcpyDeviceToHost);
    const cudaError_t completed = cudaStreamSynchronize(stream);
    if (status == cudaSuccess) status = completed;
    if (output) {
        const cudaError_t freed = cudaFree(output);
        if (status == cudaSuccess) status = freed;
    }
    const cudaError_t destroyed = cudaStreamDestroy(stream);
    if (status == cudaSuccess) status = destroyed;
    return static_cast<int>(status);
}
