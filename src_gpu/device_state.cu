#include <cuda_runtime.h>
#include <new>
#include "device_state_internal.cuh"

namespace {
using astr::Buffer;
using astr::State;

__global__ __launch_bounds__(astr::block_threads)
void snapshot_axis_kernel(AstrDeviceView field,int axis,int im,int jm,int km,size_t count,double *result)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (id<count) {
        const size_t node=id%field.stride[3];
        const int p[3]={static_cast<int>(node%field.extent[0])+field.lower[0],
            static_cast<int>((node/field.extent[0])%field.extent[1])+field.lower[1],
            static_cast<int>(node/(static_cast<size_t>(field.extent[0])*field.extent[1]))+field.lower[2]};
        const int upper[3]={im,jm,km};
        bool keep=true;
        for (int a=0;a<3;++a) if (a!=axis) keep=keep && p[a]>=0 && p[a]<=upper[a];
        result[id]=keep ? static_cast<const double *>(field.data)[id] : 0.0;
    }
    return;
}

int release(State *state)
{
    cudaError_t status = cudaSuccess;
    if (state->stream) status = cudaStreamSynchronize(state->stream);
    const int transport_status=astr::release_transport(state->transport);
    if (status==cudaSuccess && transport_status!=0) status=static_cast<cudaError_t>(transport_status);
    for (Buffer &buffer : state->fields) {
        if (!buffer.view.data) continue;
        const cudaError_t freed = cudaFree(buffer.view.data);
        if (status == cudaSuccess) status = freed;
    }
    void *work[] = {state->fluid_failure, state->cfl_partial, state->cfl_values,
                   state->statistics_partial,state->statistics_values,state->cfl_records,
                   state->cfl_diagnostic_values,state->cfl_diagnostic_keys,state->cfl_profile};
    for (void *pointer : work) {
        if (!pointer) continue;
        const cudaError_t freed = cudaFree(pointer);
        if (status == cudaSuccess) status = freed;
    }
    if (state->stream) {
        const cudaError_t destroyed = cudaStreamDestroy(state->stream);
        if (status == cudaSuccess) status = destroyed;
    }
    delete state;
    return static_cast<int>(status);
}

int transfer(void *handle, int field, void *host, size_t bytes, cudaMemcpyKind direction)
{
    if (!handle || field < 0 || field >= ASTR_DEVICE_FIELD_COUNT) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state = *static_cast<State *>(handle);
    Buffer &buffer = state.fields[field];
    if (!buffer.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if (!host || bytes != buffer.bytes) return ASTR_DEVICE_INVALID_ARGUMENT;
    void *destination = direction == cudaMemcpyHostToDevice ? buffer.view.data : host;
    const void *source = direction == cudaMemcpyHostToDevice ? host : buffer.view.data;
    const cudaError_t status = cudaMemcpyAsync(destination, source, bytes, direction, state.stream);
    if (status != cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaStreamSynchronize(state.stream));
}
}

int astr::allocate_field(Buffer &buffer, const int upper[3], int halo, int rank,
                         int components, int directions)
{
    AstrDeviceView &view = buffer.view;
    view.rank = rank;
    view.element_bytes = sizeof(double);
    int64_t elements = 1;
    for (int axis = 0; axis < 5; ++axis) {
        view.lower[axis] = axis < 3 ? -halo : 1;
        view.extent[axis] = axis < 3 ? upper[axis] + 2 * halo + 1
                                   : (axis == 3 ? components : directions);
        view.stride[axis] = elements;
        elements *= view.extent[axis];
    }
    buffer.bytes = static_cast<size_t>(elements) * sizeof(double);
    return static_cast<int>(cudaMalloc(&view.data, buffer.bytes));
}

extern "C" int astr_device_bind(int device)
{
    const cudaError_t selected = cudaSetDevice(device);
    if (selected != cudaSuccess) return static_cast<int>(selected);
    int current = -1;
    const cudaError_t queried = cudaGetDevice(&current);
    if (queried != cudaSuccess) return static_cast<int>(queried);
    if (current != device) return static_cast<int>(cudaErrorInvalidDevice);
    return static_cast<int>(cudaFree(nullptr));
}

extern "C" int astr_device_count(int *count)
{
    if (!count) return ASTR_DEVICE_INVALID_ARGUMENT;
    return static_cast<int>(cudaGetDeviceCount(count));
}

extern "C" int astr_device_state_create(int device, int im, int jm, int km, int halo,
                                       int debug_sync, void **handle)
{
    if (!handle || im < 0 || jm < 0 || km < 0 || halo < 0)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    *handle = nullptr;
    int status = astr_device_bind(device);
    if (status != 0) return status;
    astr::State *state = new (std::nothrow) astr::State{};
    if (!state) return static_cast<int>(cudaErrorMemoryAllocation);
    state->debug_sync = debug_sync != 0;
    state->upper[0] = im;
    state->upper[1] = jm;
    state->upper[2] = km;
    state->halo = halo;
    status = static_cast<int>(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    for (int field = 0; field < 3 && status == 0; ++field)
        status = astr::allocate_field(state->fields[field], state->upper,
                                     field == ASTR_DEVICE_Q ? halo : 0, 4, 5);
    if (status != 0) {
        release(state);
        return status;
    }
    *handle = state;
    return 0;
}

extern "C" int astr_device_state_view(void *handle, int field, AstrDeviceView *view)
{
    if (!handle || !view || field < 0 || field >= ASTR_DEVICE_FIELD_COUNT) return ASTR_DEVICE_INVALID_ARGUMENT;
    if (!static_cast<State *>(handle)->fields[field].view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    *view = static_cast<State *>(handle)->fields[field].view;
    return 0;
}

extern "C" int astr_device_state_upload(void *handle, int field, const double *host, size_t bytes)
{
    return transfer(handle, field, const_cast<double *>(host), bytes, cudaMemcpyHostToDevice);
}

extern "C" int astr_device_state_download(void *handle, int field, double *host, size_t bytes)
{
    return transfer(handle, field, host, bytes, cudaMemcpyDeviceToHost);
}

extern "C" int astr_device_state_snapshot_axis(void *handle,int field,int axis,double *host,size_t bytes)
{
    if (!handle || !host || field<0 || field>=ASTR_DEVICE_FIELD_COUNT || axis<0 || axis>2)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    const Buffer &buffer=s.fields[field];
    if (!buffer.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if (bytes!=buffer.bytes) return ASTR_DEVICE_INVALID_ARGUMENT;
    double *device=nullptr;
    cudaError_t status=cudaMalloc(&device,bytes);
    if (status!=cudaSuccess) return static_cast<int>(status);
    const size_t count=bytes/sizeof(double);
    const int threads=astr::block_threads;
    snapshot_axis_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(buffer.view,axis,
        s.upper[0],s.upper[1],s.upper[2],count,device);
    int result=astr::finish_launch(s);
    if (result==0) result=static_cast<int>(cudaMemcpyAsync(host,device,bytes,cudaMemcpyDeviceToHost,s.stream));
    if (result==0) result=static_cast<int>(cudaStreamSynchronize(s.stream));
    const int freed=static_cast<int>(cudaFree(device));
    return result==0 ? freed : result;
}

extern "C" int astr_device_state_synchronize(void *handle)
{
    if (!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    return static_cast<int>(cudaStreamSynchronize(static_cast<State *>(handle)->stream));
}

extern "C" int astr_device_memory_info(void *handle,size_t *free_bytes,size_t *total_bytes)
{
    if (!handle || !free_bytes || !total_bytes) return ASTR_DEVICE_INVALID_ARGUMENT;
    return static_cast<int>(cudaMemGetInfo(free_bytes,total_bytes));
}

extern "C" int astr_device_state_destroy(void **handle)
{
    if (!handle || !*handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State *state = static_cast<State *>(*handle);
    if (astr::transport_in_flight(state->transport)) return ASTR_DEVICE_INVALID_ARGUMENT;
    *handle = nullptr;
    return release(state);
}
