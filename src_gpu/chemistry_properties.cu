#include "device_state_internal.cuh"
#include "chemistry_properties.h"
#include "chemistry_properties_math.cuh"

namespace {
using astr::State;
using astr::at;
struct Box { int first[3],size[3],upper[3];size_t count; };
struct PrimitiveViews {AstrDeviceView q,rho,velocity,t,tv,p,y;};

template<bool conservative>
__device__ inline void convert_point(const PrimitiveViews &fields,int i,int j,int k,bool preserve_all,int *failure)
{
    double state[11],density,u[3],temperature,vibration,pressure,fractions[5];
    int status;
    if(conservative) {
        for(int d=0;d<3;++d) u[d]=at(fields.velocity,i,j,k,d+1);
        for(int s=0;s<5;++s) fractions[s]=at(fields.y,i,j,k,s+1);
        status=astr::air5::primitive_to_conservative(at(fields.rho,i,j,k),u,at(fields.t,i,j,k),fractions,
                                                    at(fields.tv,i,j,k),state);
        if(status==0) for(int c=0;c<11;++c) at(fields.q,i,j,k,c+1)=state[c];
    } else {
        for(int c=0;c<11;++c) state[c]=at(fields.q,i,j,k,c+1);
        status=astr::air5::conservative_to_primitive(state,preserve_all,density,u,temperature,
                                                   fractions,vibration,pressure);
        if(status==0) {
            at(fields.rho,i,j,k)=density;at(fields.t,i,j,k)=temperature;
            at(fields.tv,i,j,k)=vibration;at(fields.p,i,j,k)=pressure;
            for(int d=0;d<3;++d) at(fields.velocity,i,j,k,d+1)=u[d];
            for(int s=0;s<5;++s) at(fields.y,i,j,k,s+1)=fractions[s];
        }
    }
    if(status!=0) atomicMax(failure,status);
}
template<bool conservative>
__global__ void convert_kernel(PrimitiveViews fields,Box box,bool preserve_all,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=box.count) return;
    const int i=id%box.size[0]+box.first[0];
    const int j=(id/box.size[0])%box.size[1]+box.first[1];
    const int k=id/(static_cast<size_t>(box.size[0])*box.size[1])+box.first[2];
    if((i<0 || i>box.upper[0])+(j<0 || j>box.upper[1])+(k<0 || k>box.upper[2])>1) return;
    convert_point<conservative>(fields,i,j,k,preserve_all,failure);
}

__global__ void refresh_shell(PrimitiveViews fields,Box recovered,size_t shell_points,bool preserve_all,int *failure)
{
    size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=shell_points)return;
    const int ny=recovered.upper[1]+1,nz=recovered.upper[2]+1;
    const int nx_core=recovered.size[0],ny_core=recovered.size[1];
    const int ox=recovered.upper[0]+1-nx_core,oy=ny-ny_core;
    const size_t x_slab=static_cast<size_t>(ox)*ny,y_slab=static_cast<size_t>(nx_core)*oy;
    const size_t x_count=x_slab*nz,y_count=y_slab*nz;
    int i,j,k;
    // Three disjoint sets: x outside; x inside and y outside; x,y inside and z outside.
    if(id<x_count) {
        i=id%ox;j=(id/ox)%ny;k=id/x_slab;
        if(i>=recovered.first[0])i+=nx_core;
    } else if(id<x_count+y_count) {
        id-=x_count;
        i=recovered.first[0]+id%nx_core;j=(id/nx_core)%oy;k=id/y_slab;
        if(j>=recovered.first[1])j+=ny_core;
    } else {
        id-=x_count+y_count;
        i=recovered.first[0]+id%nx_core;j=recovered.first[1]+(id/nx_core)%ny_core;
        k=id/(static_cast<size_t>(nx_core)*ny_core);
        if(k>=recovered.first[2])k+=recovered.size[2];
    }
    convert_point<false>(fields,i,j,k,preserve_all,failure);
}

PrimitiveViews primitive_views(const State &s)
{
    return {s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_AIR5_TV].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_AIR5_SPECIES].view};
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
        convert_kernel<true><<<(box.count+255)/256,256,0,s.stream>>>(primitive_views(s),box,preserve_all,s.air5_failure);
    else
        convert_kernel<false><<<(box.count+255)/256,256,0,s.stream>>>(primitive_views(s),box,preserve_all,s.air5_failure);
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
extern "C" int astr_device_air5_primitives_shell(void *handle,const int first[3],const int last[3],int preserve_all)
{
    if(!handle || !first || !last)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<State *>(handle);
    if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    size_t physical=1;
    for(int d=0;d<3;++d) {
        if(first[d]<1 || last[d]>=s.upper[d])return ASTR_DEVICE_INVALID_ARGUMENT;
        physical*=s.upper[d]+1;
    }
    Box recovered{};
    int status=conversion_box(s,first,last,recovered);if(status)return status;
    status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t shell_points=physical-recovered.count;
    refresh_shell<<<(shell_points+255)/256,256,0,s.stream>>>(primitive_views(s),recovered,shell_points,
        preserve_all!=0,s.air5_failure);
    return astr::finish_launch(s);
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
