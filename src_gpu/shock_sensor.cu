#include <cmath>
#include "device_state_internal.cuh"
#include "shock_sensor.h"
#include "shock_sensor_internal.cuh"

namespace {
using astr::State;
using astr::at;
using Real=astr::numerical_real;
constexpr int threads=astr::block_threads;
struct Ends { int low[3],high[3],upper[3]; };

size_t nodes(const State &state)
{
    return static_cast<size_t>(state.upper[0]+1)*(state.upper[1]+1)*(state.upper[2]+1);
}

int ends_for(const State &state,const int *low,const int *high,Ends &ends)
{
    if(!low || !high) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) {
        if((low[a]!=0 && low[a]!=1) || (high[a]!=0 && high[a]!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=state.upper[a];
    }
    return 0;
}

__device__ void node(size_t id,const Ends &ends,int p[3])
{
    p[0]=id%(ends.upper[0]+1);
    p[1]=(id/(ends.upper[0]+1))%(ends.upper[1]+1);
    p[2]=id/(static_cast<size_t>(ends.upper[0]+1)*(ends.upper[1]+1));
}

__device__ int neighbor(int p,int a,const Ends &ends)
{
    if(ends.low[a] && p<0) return 0;
    if(ends.high[a] && p>ends.upper[a]) return ends.upper[a];
    return p;
}

__global__ __launch_bounds__(threads)
void raw_kernel(AstrDeviceView gradient,AstrDeviceView pressure,AstrDeviceView sensor,Ends ends,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int p[3];node(id,ends,p);
        const int i=p[0],j=p[1],k=p[2];
        const Real div=at(gradient,i,j,k,1,1)+at(gradient,i,j,k,2,2)+at(gradient,i,j,k,3,3);
        const Real wx=at(gradient,i,j,k,3,2)-at(gradient,i,j,k,2,3);
        const Real wy=at(gradient,i,j,k,1,3)-at(gradient,i,j,k,3,1);
        const Real wz=at(gradient,i,j,k,2,1)-at(gradient,i,j,k,1,2);
        const Real div2=div*div,vort=wx*wx+wy*wy+wz*wz;
        const Real center=at(pressure,i,j,k);
        Real curvature=0;
        for(int a=0;a<3;++a) {
            int left[3]={i,j,k},right[3]={i,j,k};
            left[a]=neighbor(p[a]-1,a,ends);right[a]=neighbor(p[a]+1,a,ends);
            const Real minus=at(pressure,left[0],left[1],left[2]);
            const Real plus=at(pressure,right[0],right[1],right[2]);
            curvature=fmax(curvature,fabs(plus-Real(2)*center+minus)/(plus+Real(2)*center+minus));
        }
        at(sensor,i,j,k)=div2/(div2+vort+Real(1.e-30))*curvature;
    }
    return;
}

__global__ __launch_bounds__(threads)
void expand_kernel(AstrDeviceView sensor,int8_t *mask,Ends ends,int halo,Real threshold,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) {
        int p[3];node(id,ends,p);
        astr::numerical_real maximum=0;
        for(int a=0;a<3;++a) for(int h=-halo+1;h<=halo;++h) {
            int point[3]={p[0],p[1],p[2]};point[a]=neighbor(p[a]+h,a,ends);
            maximum=fmax(maximum,at(sensor,point[0],point[1],point[2]));
        }
        mask[id]=maximum>threshold ? 1 : 0;
    }
    return;
}
}

extern "C" int astr_device_shock_prepare(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state=*static_cast<State *>(handle);
    if(state.halo<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!state.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!state.fields[ASTR_DEVICE_SHOCK_SENSOR].view.data) {
        const int status=astr::allocate_field(state.fields[ASTR_DEVICE_SHOCK_SENSOR],state.upper,state.halo,3);
        if(status!=0) return status;
    }
    if(!state.shock_mask) return static_cast<int>(cudaMalloc(&state.shock_mask,nodes(state)*sizeof(int8_t)));
    return 0;
}

extern "C" int astr_device_shock_raw(void *handle,const int low[3],const int high[3])
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state=*static_cast<State *>(handle);Ends ends;
    const int status=ends_for(state,low,high,ends);
    if(status!=0) return status;
    if(!state.shock_mask) return ASTR_DEVICE_NOT_ALLOCATED;
    raw_kernel<<<(nodes(state)+threads-1)/threads,threads,0,state.stream>>>(state.fields[ASTR_DEVICE_DVELOCITY].view,
        state.fields[ASTR_DEVICE_PRESSURE].view,state.fields[ASTR_DEVICE_SHOCK_SENSOR].view,ends,nodes(state));
    return astr::finish_launch(state);
}

extern "C" int astr_device_shock_expand(void *handle,const int low[3],const int high[3],Real threshold)
{
    if(!handle || !std::isfinite(threshold)) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state=*static_cast<State *>(handle);Ends ends;
    const int status=ends_for(state,low,high,ends);
    if(status!=0) return status;
    if(!state.shock_mask) return ASTR_DEVICE_NOT_ALLOCATED;
    state.shock_faces_ready=false;
    expand_kernel<<<(nodes(state)+threads-1)/threads,threads,0,state.stream>>>(state.fields[ASTR_DEVICE_SHOCK_SENSOR].view,
        state.shock_mask,ends,state.halo,threshold,nodes(state));
    return astr::finish_launch(state);
}

extern "C" int astr_device_shock_mask_download(void *handle,int8_t *host,size_t bytes)
{
    if(!handle || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state=*static_cast<State *>(handle);
    if(!state.shock_mask) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=nodes(state)*sizeof(int8_t)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const cudaError_t status=cudaMemcpyAsync(host,state.shock_mask,bytes,cudaMemcpyDeviceToHost,state.stream);
    if(status!=cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaStreamSynchronize(state.stream));
}

extern "C" int astr_device_shock_faces_download(void *handle,int8_t *host,size_t bytes)
{
    if(!handle || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &state=*static_cast<State *>(handle);
    if(!state.shock_faces_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=astr::shock_faces_size(state.upper)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const cudaError_t status=cudaMemcpyAsync(host,state.shock_faces,bytes,cudaMemcpyDeviceToHost,state.stream);
    if(status!=cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaStreamSynchronize(state.stream));
}
