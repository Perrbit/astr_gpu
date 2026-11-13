#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include "qualification_device.cuh"

namespace {
constexpr int threads = 512;

__global__ __launch_bounds__(threads)
void transform_kernel(const double *input, double *output, int n, int *width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = mb0_transform(input[i]);
    if (i == 0) *width = warpSize;
    return;
}

__global__ __launch_bounds__(threads)
void derivative_kernel(const double *field, double *derivative, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) derivative[i] = mb0_derivative(field, i, n);
    return;
}

__global__ __launch_bounds__(threads)
void reduction_kernel(const double *field, double *partial, int n)
{
    __shared__ double values[threads];
    const int lane = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + lane;
    values[lane] = i < n ? field[i] : 0.0;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (lane < stride) values[lane] += values[lane + stride];
        __syncthreads();
    }
    if (lane == 0) partial[blockIdx.x] = values[0];
    return;
}
}

extern "C" int astr_mb0_evaluate(int rank, int n, const double *input,
    double *output, double *derivative, double *sum)
{
    cudaError_t status = cudaSuccess;
    cudaStream_t producer = nullptr, consumer = nullptr;
    cudaEvent_t produced = nullptr, completed = nullptr;
    double *device_input = nullptr, *device_output = nullptr;
    double *device_derivative = nullptr, *device_partial = nullptr;
    double *host_input = nullptr, *host_output = nullptr;
    double *host_derivative = nullptr, *host_partial = nullptr;
    int *device_width = nullptr;
    int count = 0, device = 0, current_device = -1, builtin_width = 0;
    int runtime_version = 0, driver_version = 0;
    const int blocks = (n + threads - 1) / threads;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    cudaDeviceProp properties{};
#define MB0_CUDA(call) do { status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "MB0 %s: %s\n", #call, cudaGetErrorString(status)); \
    goto cleanup; } } while (0)
    MB0_CUDA(cudaGetDeviceCount(&count));
    if (rank >= count) {
        std::fprintf(stderr, "MB0 requires one visible device per rank: rank=%d count=%d\n", rank, count);
        return static_cast<int>(cudaErrorInvalidDevice);
    }
    device = rank;
    MB0_CUDA(cudaSetDevice(device));
    MB0_CUDA(cudaGetDevice(&current_device));
    if (current_device != device) {
        status = cudaErrorInvalidDevice;
        std::fprintf(stderr, "MB0 binding mismatch: requested=%d current=%d\n", device, current_device);
        goto cleanup;
    }
    MB0_CUDA(cudaGetDeviceProperties(&properties, device));
    MB0_CUDA(cudaRuntimeGetVersion(&runtime_version));
    MB0_CUDA(cudaDriverGetVersion(&driver_version));
    MB0_CUDA(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    MB0_CUDA(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    MB0_CUDA(cudaEventCreateWithFlags(&produced, cudaEventDisableTiming));
    MB0_CUDA(cudaEventCreateWithFlags(&completed, cudaEventDisableTiming));
    MB0_CUDA(cudaMallocHost(&host_input, bytes));
    MB0_CUDA(cudaMallocHost(&host_output, bytes));
    MB0_CUDA(cudaMallocHost(&host_derivative, bytes));
    MB0_CUDA(cudaMallocHost(&host_partial, blocks * sizeof(double)));
    MB0_CUDA(cudaMalloc(&device_input, bytes));
    MB0_CUDA(cudaMalloc(&device_output, bytes));
    MB0_CUDA(cudaMalloc(&device_derivative, bytes));
    MB0_CUDA(cudaMalloc(&device_partial, blocks * sizeof(double)));
    MB0_CUDA(cudaMalloc(&device_width, sizeof(int)));
    std::memcpy(host_input, input, bytes);
    MB0_CUDA(cudaMemcpyAsync(device_input, host_input, bytes, cudaMemcpyHostToDevice, producer));
    transform_kernel<<<blocks, threads, 0, producer>>>(device_input, device_output, n, device_width);
    MB0_CUDA(cudaGetLastError());
    MB0_CUDA(cudaEventRecord(produced, producer));
    MB0_CUDA(cudaStreamWaitEvent(consumer, produced, 0));
    derivative_kernel<<<blocks, threads, 0, consumer>>>(device_output, device_derivative, n);
    MB0_CUDA(cudaGetLastError());
    reduction_kernel<<<blocks, threads, 0, consumer>>>(device_output, device_partial, n);
    MB0_CUDA(cudaGetLastError());
    MB0_CUDA(cudaMemcpyAsync(host_output, device_output, bytes, cudaMemcpyDeviceToHost, consumer));
    MB0_CUDA(cudaMemcpyAsync(host_derivative, device_derivative, bytes, cudaMemcpyDeviceToHost, consumer));
    MB0_CUDA(cudaMemcpyAsync(host_partial, device_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost, consumer));
    MB0_CUDA(cudaEventRecord(completed, consumer));
    MB0_CUDA(cudaEventSynchronize(completed));
    MB0_CUDA(cudaMemcpy(&builtin_width, device_width, sizeof(int), cudaMemcpyDeviceToHost));
    std::memcpy(output, host_output, bytes);
    std::memcpy(derivative, host_derivative, bytes);
    *sum = 0.0;
    for (int block = 0; block < blocks; ++block) *sum += host_partial[block];
    std::printf("MB0 rank=%d device=%d visible=%d pci=%04x:%02x:%02x name=%s API_warp=%d kernel_warp=%d threads=%d n=%d headers=%d runtime_API=%d driver_API=%d\n",
        rank, current_device, count, static_cast<unsigned>(properties.pciDomainID),
        static_cast<unsigned>(properties.pciBusID), static_cast<unsigned>(properties.pciDeviceID),
        properties.name, properties.warpSize, builtin_width, threads, n,
        CUDART_VERSION, runtime_version, driver_version);
cleanup:
    // Synchronize even on failure before releasing outstanding pinned transfers.
    if (producer) cudaStreamSynchronize(producer);
    if (consumer) cudaStreamSynchronize(consumer);
#define MB0_RELEASE(call) do { const cudaError_t released = (call); \
    if (status == cudaSuccess && released != cudaSuccess) status = released; } while (0)
    if (device_input) MB0_RELEASE(cudaFree(device_input));
    if (device_output) MB0_RELEASE(cudaFree(device_output));
    if (device_derivative) MB0_RELEASE(cudaFree(device_derivative));
    if (device_partial) MB0_RELEASE(cudaFree(device_partial));
    if (device_width) MB0_RELEASE(cudaFree(device_width));
    if (host_input) MB0_RELEASE(cudaFreeHost(host_input));
    if (host_output) MB0_RELEASE(cudaFreeHost(host_output));
    if (host_derivative) MB0_RELEASE(cudaFreeHost(host_derivative));
    if (host_partial) MB0_RELEASE(cudaFreeHost(host_partial));
    if (produced) MB0_RELEASE(cudaEventDestroy(produced));
    if (completed) MB0_RELEASE(cudaEventDestroy(completed));
    if (producer) MB0_RELEASE(cudaStreamDestroy(producer));
    if (consumer) MB0_RELEASE(cudaStreamDestroy(consumer));
    return static_cast<int>(status);
}
