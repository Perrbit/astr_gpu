#include "device_state_internal.cuh"

extern "C" int upload_probe_mask(void *handle,const int8_t *mask,size_t bytes)
{
    auto &s=*static_cast<astr::State *>(handle);
    const cudaError_t status=cudaMemcpyAsync(s.shock_mask,mask,bytes,cudaMemcpyHostToDevice,s.stream);
    return status==cudaSuccess ? cudaStreamSynchronize(s.stream) : status;
}
