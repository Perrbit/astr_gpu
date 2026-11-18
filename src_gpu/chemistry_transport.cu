#include "device_state_internal.cuh"
#include "chemistry_transport.h"
#include "chemistry_compensation_math.cuh"
#include "chemistry_properties_math.cuh"
#include "fluid_explicit.h"

namespace {
// AIR5 storage is FP64 independently of the nonreacting build kind. Its
// preparation API rejects native-FP32 contexts before these kernels can run.
__device__ inline double &at(const AstrDeviceView &v,int i,int j,int k,int c=1,int d=1)
{
    return static_cast<double *>(v.data)[astr::offset(v,i,j,k,c,d)];
}
struct Box {int first[3],size[3];size_t count;};
struct Ends {int low[3],high[3],upper[3];};
__device__ double derivative(AstrDeviceView field,int i,int j,int k,int component,int axis,Ends ends)
{
    const int p[3]={i,j,k};const int index=p[axis],last=ends.upper[axis];
    const bool left=ends.low[axis] && index<3,right=ends.high[axis] && index>last-3;
    if(left || right) {
        const int depth=left?index:last-index,step=left?1:-1;
        int x[3]={i,j,k};x[axis]=left?0:last;
        double v[5];
        for(int r=0;r<(depth==2?5:3);++r) {v[r]=at(field,x[0],x[1],x[2],component);x[axis]+=step;}
        if(depth==0) return step*(-.5*v[2]+2.*v[1]-1.5*v[0]);
        if(depth==1) return step*.5*(v[2]-v[0]);
        return step*((2./3.)*(v[3]-v[1])-(1./12.)*(v[4]-v[0]));
    }
    const double *v=static_cast<const double *>(field.data);
    const size_t c=astr::offset(field,i,j,k,component),s=field.stride[axis];
    return .75*(v[c+s]-v[c-s])-.15*(v[c+2*s]-v[c-2*s])+(1./60.)*(v[c+3*s]-v[c-3*s]);
}
__global__ void species_tv_gradient_kernel(AstrDeviceView tv,AstrDeviceView y,AstrDeviceView metric,
    AstrDeviceView dtv,AstrDeviceView dy,Ends ends,bool zero_lower,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int nx=ends.upper[0]+1,ny=ends.upper[1]+1;
    const int i=id%nx,j=(id/nx)%ny,k=id/(static_cast<size_t>(nx)*ny);
    for(int c=0;c<=5;++c) {
        const AstrDeviceView field=c==0?tv:y;
        const int component=c==0?1:c;
        double d[3];for(int axis=0;axis<3;++axis) d[axis]=derivative(field,i,j,k,component,axis,ends);
        for(int physical=1;physical<=3;++physical) {
            const double value=d[0]*at(metric,i,j,k,1,physical)+d[1]*at(metric,i,j,k,2,physical)+
                d[2]*at(metric,i,j,k,3,physical);
            if(c==0) at(dtv,i,j,k,physical)=value;
            else at(dy,i,j,k,c,physical)=zero_lower && j==0 && physical==2?0.:value;
        }
    }
}
__global__ void diffusion_kernel(AstrDeviceView rho,AstrDeviceView vel,AstrDeviceView t,AstrDeviceView tv,
    AstrDeviceView p,AstrDeviceView y,AstrDeviceView dv,AstrDeviceView dt,AstrDeviceView dtv,AstrDeviceView dy,
    AstrDeviceView stress,AstrDeviceView heat,AstrDeviceView species,AstrDeviceView evflux,AstrDeviceView es,
    int nx,int ny,size_t count,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=id%nx,j=(id/nx)%ny,k=id/(static_cast<size_t>(nx)*ny);
    double u[3],fractions[5],gu[9],gt[3],gtv[3],gy[15],momentum[9],js[15],energy[3],evf[3],h[5],ev[5];
    for(int d=1;d<=3;++d) {
        u[d-1]=at(vel,i,j,k,d);gt[d-1]=at(dt,i,j,k,d);gtv[d-1]=at(dtv,i,j,k,d);
        for(int c=1;c<=3;++c) gu[c-1+3*(d-1)]=at(dv,i,j,k,c,d);
        for(int c=1;c<=5;++c) gy[c-1+5*(d-1)]=at(dy,i,j,k,c,d);
    }
    for(int c=1;c<=5;++c) fractions[c-1]=at(y,i,j,k,c);
    const int status=astr::air5::diffusive_flux(at(rho,i,j,k),u,at(t,i,j,k),at(tv,i,j,k),at(p,i,j,k),
        fractions,gu,gt,gtv,gy,momentum,js,energy,evf,h,ev);
    if(status) {atomicMax(failure,status);return;}
    const int tensor[]={0,3,6,4,7,8};
    for(int c=1;c<=6;++c) at(stress,i,j,k,c)=momentum[tensor[c-1]];
    for(int d=1;d<=3;++d) {
        at(heat,i,j,k,d)=energy[d-1];at(evflux,i,j,k,d)=evf[d-1];
        double hs=0,vs=0;
        for(int c=1;c<=5;++c) {
            const double f=js[c-1+5*(d-1)];at(species,i,j,k,c,d)=f;
            if(es.data) {hs+=h[c-1]*f;vs+=ev[c-1]*f;}
        }
        if(es.data) {at(es,i,j,k,d)=-hs;at(es,i,j,k,d+3)=-vs;}
    }
}
__global__ void rk_kernel(AstrDeviceView q,AstrDeviceView rhs,AstrDeviceView saved,AstrDeviceView jac,
    AstrDeviceView carry,AstrDeviceView origin,AstrDeviceView origin_carry,Box box,double dt,
    double a,double b,double c,bool first,bool compensated)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=box.count) return;
    const int i=id%box.size[0]+box.first[0],j=(id/box.size[0])%box.size[1]+box.first[1];
    const int k=id/(static_cast<size_t>(box.size[0])*box.size[1])+box.first[2];
    const double jacobian=at(jac,i,j,k);
    for(int component=1;component<=11;++component) {
        double &high=at(q,i,j,k,component);
        const double residual=at(rhs,i,j,k,component);
        if(first) {
            const double value=high*jacobian;
            at(saved,i,j,k,component)=value;
            if(compensated) {
                at(origin,i,j,k,component)=high;
                at(origin_carry,i,j,k,component)=at(carry,i,j,k,component);
                astr::air5::compensated_add(high,at(carry,i,j,k,component),dt*residual/jacobian);
            } else high=(value+dt*residual)/jacobian;
        } else if(compensated) {
            astr::air5::compensated_rk(high,at(carry,i,j,k,component),at(origin,i,j,k,component),
                at(origin_carry,i,j,k,component),a,b,c*dt*residual/jacobian);
        } else high=(a*at(saved,i,j,k,component)+b*high*jacobian+c*dt*residual)/jacobian;
    }
}
int allocate_velocity_temperature_gradients(astr::State &s)
{
    const int status=astr::allocate_field(s.fields[ASTR_DEVICE_DVELOCITY],s.upper,0,5,3,3,sizeof(double));
    if(status) return status;
    return astr::allocate_field(s.fields[ASTR_DEVICE_DTEMPERATURE],s.upper,0,4,3,1,sizeof(double));
}
}
extern "C" int astr_device_air5_geometry_prepare(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || s.fluid_ready) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=astr::allocate_field(s.fields[ASTR_DEVICE_METRIC],s.upper,s.halo,5,3,3);
    if(status!=0) return status;
    status=astr::allocate_field(s.fields[ASTR_DEVICE_JACOBIAN],s.upper,s.halo,3);
    if(status!=0) return status;
    status=static_cast<int>(cudaMalloc(&s.fluid_failure,sizeof(int)));
    if(status!=0) return status;
    status=static_cast<int>(cudaMemsetAsync(s.fluid_failure,0,sizeof(int),s.stream));
    if(status!=0) return status;
    s.fluid_ready=true;
    return 0;
}
extern "C" int astr_device_air5_velocity_gradients_prepare(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.gradients_ready || s.halo<3) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=allocate_velocity_temperature_gradients(s);
    if(status==0) s.gradients_ready=true;
    return status;
}
extern "C" int astr_device_air5_transport_prepare(void *handle,int layered)
{
    if(!handle || (layered!=0 && layered!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.gradients_ready || s.viscous_ready || s.halo<3) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int gradients=allocate_velocity_temperature_gradients(s);
    if(gradients) return gradients;
    const int fields[]={ASTR_DEVICE_AIR5_DTV,ASTR_DEVICE_AIR5_DSPECIES,ASTR_DEVICE_STRESS,ASTR_DEVICE_HEAT_FLUX,
        ASTR_DEVICE_AIR5_SPECIES_FLUX,ASTR_DEVICE_AIR5_EV_FLUX,ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX};
    const int components[]={3,5,6,3,5,3,6},directions[]={1,3,1,1,3,1,1};
    for(int i=0;i<(layered?7:6);++i) {
        const int h=i>=2?s.halo:0,rank=directions[i]>1?5:4;
        const int status=astr::allocate_field(s.fields[fields[i]],s.upper,h,rank,components[i],directions[i],sizeof(double));
        if(status) return status;
    }
    s.gradients_ready=true;s.viscous_ready=true;
    return 0;
}
extern "C" int astr_device_air5_gradients(void *handle,const int low[3],const int high[3],int zero_lower)
{
    if(!handle || !low || !high || (zero_lower!=0 && zero_lower!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.viscous_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    Ends ends{};size_t count=1;
    for(int a=0;a<3;++a) {
        if((low[a]!=0 && low[a]!=1) || (high[a]!=0 && high[a]!=1) ||
            ((low[a] || high[a]) && s.upper[a]<4)) return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=low[a];ends.high[a]=high[a];ends.upper[a]=s.upper[a];count*=s.upper[a]+1;
    }
    const int status=astr_device_fluid_gradient_basic_boundary(handle,low,high);
    if(status) return status;
    species_tv_gradient_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_AIR5_TV].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES].view,s.fields[ASTR_DEVICE_METRIC].view,
        s.fields[ASTR_DEVICE_AIR5_DTV].view,s.fields[ASTR_DEVICE_AIR5_DSPECIES].view,ends,zero_lower!=0,count);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_diffusive_flux(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.viscous_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));
    if(status) return status;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    diffusion_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_RHO].view,
        s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
        s.fields[ASTR_DEVICE_AIR5_TV].view,s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_AIR5_SPECIES].view,
        s.fields[ASTR_DEVICE_DVELOCITY].view,s.fields[ASTR_DEVICE_DTEMPERATURE].view,
        s.fields[ASTR_DEVICE_AIR5_DTV].view,s.fields[ASTR_DEVICE_AIR5_DSPECIES].view,
        s.fields[ASTR_DEVICE_STRESS].view,s.fields[ASTR_DEVICE_HEAT_FLUX].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES_FLUX].view,s.fields[ASTR_DEVICE_AIR5_EV_FLUX].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX].view,s.upper[0]+1,s.upper[1]+1,count,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_rk_prepare(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fields[ASTR_DEVICE_JACOBIAN].view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!s.fields[ASTR_DEVICE_AIR5_CARRY].view.data || s.fields[ASTR_DEVICE_AIR5_ORIGIN].view.data) {
        s.rhs_ready=true;return 0;
    }
    int status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_ORIGIN],s.upper,0,4,11);
    if(status!=0) return status;
    status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_ORIGIN_CARRY],s.upper,0,4,11);
    if(status==0) s.rhs_ready=true;
    return status;
}
extern "C" int astr_device_air5_rk_update(void *handle,const int first[3],const int last[3],
    double dt,double a,double b,double c,int first_stage)
{
    if(!handle || !first || !last) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const bool compensated=s.fields[ASTR_DEVICE_AIR5_CARRY].view.data!=nullptr;
    if(!s.air5_ready || !s.fields[ASTR_DEVICE_JACOBIAN].view.data ||
       (compensated && !s.fields[ASTR_DEVICE_AIR5_ORIGIN_CARRY].view.data)) return ASTR_DEVICE_NOT_ALLOCATED;
    Box box{};box.count=1;
    for(int axis=0;axis<3;++axis) {
        if(first[axis]<0 || last[axis]>s.upper[axis] || first[axis]>last[axis]) return ASTR_DEVICE_INVALID_ARGUMENT;
        box.first[axis]=first[axis];box.size[axis]=last[axis]-first[axis]+1;box.count*=box.size[axis];
    }
    rk_kernel<<<(box.count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_RHS].view,
        s.fields[ASTR_DEVICE_SAVE].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,
        s.fields[ASTR_DEVICE_AIR5_ORIGIN].view,s.fields[ASTR_DEVICE_AIR5_ORIGIN_CARRY].view,box,dt,a,b,c,first_stage!=0,compensated);
    return astr::finish_launch(s);
}
