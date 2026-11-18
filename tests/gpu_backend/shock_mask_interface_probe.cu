#include "device_state_internal.cuh"
#include "shock_sensor_internal.cuh"

namespace {
struct Box { int upper[3]; };
__global__ void observe_interfaces(const int8_t *mask,const int8_t *faces,Box box,int axis,
    size_t count,int8_t *result)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<count) {
        const int width=box.upper[axis]+2,n1=box.upper[(axis+1)%3]+1;
        int p[3]={};p[axis]=static_cast<int>(id%width)-1;
        p[(axis+1)%3]=(id/width)%n1;p[(axis+2)%3]=id/(width*n1);
        result[id]=astr::shock_interface_active(mask,faces,box.upper,p,axis) ? 1 : 0;
    }
    return;
}
}

extern "C" int test_shock_interfaces(void *handle,int axis,int8_t *host,size_t count)
{
    astr::State &s=*static_cast<astr::State *>(handle);Box box{};
    for(int a=0;a<3;++a) box.upper[a]=s.upper[a];
    int8_t *result=nullptr;
    cudaError_t status=cudaMalloc(&result,count);
    if(status!=cudaSuccess) return static_cast<int>(status);
    observe_interfaces<<<(count+511)/512,512,0,s.stream>>>(s.shock_mask,s.shock_faces_ready ? s.shock_faces : nullptr,
        box,axis,count,result);
    int error=astr::finish_launch(s);
    if(error==0) error=static_cast<int>(cudaMemcpyAsync(host,result,count,cudaMemcpyDeviceToHost,s.stream));
    if(error==0) error=static_cast<int>(cudaStreamSynchronize(s.stream));
    const cudaError_t freed=cudaFree(result);
    return error==0 ? static_cast<int>(freed) : error;
}
