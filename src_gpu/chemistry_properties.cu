#include "device_state_internal.cuh"
#include "chemistry_properties.h"
#include "chemistry_properties_math.cuh"

namespace {
using astr::State;
using astr::at;
struct Box { int first[3],size[3],upper[3];size_t count; };

template<bool conservative>
__global__ void convert_kernel(AstrDeviceView q,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView t,AstrDeviceView tv,AstrDeviceView p,AstrDeviceView y,Box box,
    bool preserve_all,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=box.count) return;
    const int i=id%box.size[0]+box.first[0];
    const int j=(id/box.size[0])%box.size[1]+box.first[1];
    const int k=id/(static_cast<size_t>(box.size[0])*box.size[1])+box.first[2];
    if((i<0 || i>box.upper[0])+(j<0 || j>box.upper[1])+(k<0 || k>box.upper[2])>1) return;
    double state[11],density,u[3],temperature,vibration,pressure,fractions[5];
    int status;
    if(conservative) {
        for(int d=0;d<3;++d) u[d]=at(velocity,i,j,k,d+1);
        for(int s=0;s<5;++s) fractions[s]=at(y,i,j,k,s+1);
        status=astr::air5::primitive_to_conservative(at(rho,i,j,k),u,at(t,i,j,k),fractions,
                                                    at(tv,i,j,k),state);
        if(status==0) for(int c=0;c<11;++c) at(q,i,j,k,c+1)=state[c];
    } else {
        for(int c=0;c<11;++c) state[c]=at(q,i,j,k,c+1);
        status=astr::air5::conservative_to_primitive(state,preserve_all,density,u,temperature,
                                                   fractions,vibration,pressure);
        if(status==0) {
            at(rho,i,j,k)=density;at(t,i,j,k)=temperature;at(tv,i,j,k)=vibration;at(p,i,j,k)=pressure;
            for(int d=0;d<3;++d) at(velocity,i,j,k,d+1)=u[d];
            for(int s=0;s<5;++s) at(y,i,j,k,s+1)=fractions[s];
        }
    }
    if(status!=0) atomicMax(failure,status);
}

int conversion_box(const State &s,const int first[3],const int last[3],Box &box)
{
    box.count=1;
    for(int d=0;d<3;++d) {
        if(first[d]<-s.halo || last[d]>s.upper[d]+s.halo || first[d]>last[d])
            return ASTR_DEVICE_INVALID_ARGUMENT;
        box.first[d]=first[d];box.size[d]=last[d]-first[d]+1;box.upper[d]=s.upper[d];
        box.count*=box.size[d];
    }
    return 0;
}
int launch_conversion(State &s,Box box,bool conservative,bool preserve_all)
{
    if(conservative)
        convert_kernel<true><<<(box.count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
            s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
            s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_AIR5_TV].view,
            s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_AIR5_SPECIES].view,box,preserve_all,s.air5_failure);
    else
        convert_kernel<false><<<(box.count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
            s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
            s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_AIR5_TV].view,
            s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_AIR5_SPECIES].view,box,preserve_all,s.air5_failure);
    // These flags describe allocated workspaces; the caller orders cache refreshes.
    return astr::finish_launch(s);
}
int convert(void *handle,const int first[3],const int last[3],bool conservative,bool preserve_all)
{
    if(!handle || !first || !last) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);if(!s.air5_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    Box box{};int status=conversion_box(s,first,last,box);if(status)return status;
    status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    return launch_conversion(s,box,conservative,preserve_all);
}
}

extern "C" int astr_device_air5_prepare(void *handle)
{
    if(!handle || sizeof(astr::numerical_real)!=sizeof(double)) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(s.air5_ready) return 0;
    if(s.fluid_ready || s.transport) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int f=0;f<3;++f) {
        int status=static_cast<int>(cudaFree(s.fields[f].view.data));
        if(status!=0) return status;
        s.fields[f]={};
        status=astr::allocate_field(s.fields[f],s.upper,f==ASTR_DEVICE_Q?s.halo:0,4,11);
        if(status!=0) return status;
    }
    const int fields[]={ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_PRESSURE,
                        ASTR_DEVICE_TEMPERATURE,ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_AIR5_SPECIES};
    const int components[]={1,3,1,1,1,5};
    for(int i=0;i<6;++i) {
        const int status=astr::allocate_field(s.fields[fields[i]],s.upper,s.halo,components[i]==1?3:4,components[i]);
        if(status!=0) return status;
    }
    int status=static_cast<int>(cudaMalloc(&s.air5_failure,sizeof(int)));
    if(status!=0) return status;
    status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));
    if(status!=0) return status;
    s.air5_ready=true;
    return 0;
}
extern "C" int astr_device_air5_conservative(void *handle,const int first[3],const int last[3])
{
    return convert(handle,first,last,true,false);
}
extern "C" int astr_device_air5_primitives(void *handle,const int first[3],const int last[3],int preserve_all)
{
    return convert(handle,first,last,false,preserve_all!=0);
}
extern "C" int astr_device_air5_primitives_regions(void *handle,int regions,const int *first,const int *last,int preserve_all)
{
    if(!handle || !first || !last || regions<1)return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    for(int region=0;region<regions;++region) {
        Box box{};const size_t offset=3*static_cast<size_t>(region);
        status=conversion_box(s,first+offset,last+offset,box);if(status)return status;
        status=launch_conversion(s,box,false,preserve_all!=0);if(status)return status;
    }
    return 0;
}
extern "C" int astr_device_air5_check(void *handle,int *chemistry_status)
{
    if(!handle || !chemistry_status) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.air5_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    int status=static_cast<int>(cudaMemcpyAsync(chemistry_status,s.air5_failure,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
    return status!=0?status:static_cast<int>(cudaStreamSynchronize(s.stream));
}
