#include "device_state_internal.cuh"
#include "chemistry_boundary.h"
#include "chemistry_boundary_math.cuh"
#include "chemistry_characteristic_math.cuh"
#include <new>

namespace astr {
struct Air5BoundaryPlan {
    double *fixed=nullptr,*profile=nullptr,*top=nullptr,*x_top=nullptr,*top_spacing=nullptr;
    double origin=0.,incident_x=0.;bool incident=false;
};
int release_air5_boundary(Air5BoundaryPlan *plan)
{
    if(!plan)return 0;int status=0;
    if(plan->fixed)status=static_cast<int>(cudaFree(plan->fixed));
    if(plan->profile) {const int freed=static_cast<int>(cudaFree(plan->profile));if(status==0)status=freed;}
    if(plan->top) {const int freed=static_cast<int>(cudaFree(plan->top));if(status==0)status=freed;}
    if(plan->x_top) {const int freed=static_cast<int>(cudaFree(plan->x_top));if(status==0)status=freed;}
    if(plan->top_spacing) {const int freed=static_cast<int>(cudaFree(plan->top_spacing));if(status==0)status=freed;}
    delete plan;return status;
}
}

namespace {
__device__ int top_target(const double *refs,double x,double origin,bool incident,double incident_x,double result[11])
{
    if(incident) {const int start=x<incident_x?11:22;for(int c=0;c<11;++c)result[c]=refs[start+c];return 0;}
    return astr::air5::boundary::similarity_farfield(refs,origin,x,result);
}
__global__ void characteristic_top_kernel(AstrDeviceView q,AstrDeviceView jac,AstrDeviceView rhs,AstrDeviceView high,
    const double *refs,const double *x,const double *spacing,int nx,int ny,int nz,int first_i,
    double origin,bool incident,double incident_x,bool preserve,double rate,double reference_sound,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=(nx-first_i)*nz)return;
    const int i=first_i+id%(nx-first_i),k=id/(nx-first_i),j=ny-1;const auto *data=static_cast<const double *>(q.data);
    double local[11],one[11],two[11],gradient[11],normal[11],target[11],remaining[11]{},result[11],p,sound;
    for(int c=0;c<11;++c) {
        local[c]=data[astr::offset(q,i,j,k,c+1)];one[c]=data[astr::offset(q,i,j-1,k,c+1)];two[c]=data[astr::offset(q,i,j-2,k,c+1)];
    }
    using namespace astr::air5;int status=characteristic::pressure_differential(local,preserve,p,sound,gradient);
    if(status) {atomicMax(failure,status);return;}
    const int at=i+nx*k;
    if(!characteristic::normal_convection(local,one,two,spacing[at],p,gradient,normal,false)) {atomicMax(failure,static_cast<int>(nonfinite));return;}
    const double jacobian=static_cast<const double *>(jac.data)[astr::offset(jac,i,j,k)];
    if(!(jacobian>0.)) {atomicMax(failure,static_cast<int>(nonfinite));return;}
    for(double &v:remaining)v/=jacobian;
    status=top_target(refs,x[at],origin,incident,incident_x,target);
    if(status) {atomicMax(failure,status);return;}
    if(!characteristic::transport_rhs(local,target,p,sound,gradient,normal,remaining,rate,result,reference_sound)) {atomicMax(failure,static_cast<int>(nonfinite));return;}
    bool valid=true;for(int c=0;c<11;++c) {result[c]*=jacobian;valid=valid && astr::air5::finite(result[c]);}
    if(!valid) {atomicMax(failure,static_cast<int>(nonfinite));return;}
    for(int c=0;c<11;++c)static_cast<double *>(high.data)[astr::offset(high,i,0,k,c+1)]=result[c];
    if(!characteristic::normal_convection(local,one,two,spacing[at],p,gradient,normal,true)) {atomicMax(failure,static_cast<int>(nonfinite));return;}
    if(!characteristic::transport_rhs(local,target,p,sound,gradient,normal,remaining,rate,result,reference_sound)) {atomicMax(failure,static_cast<int>(nonfinite));return;}
    for(int c=0;c<11;++c)static_cast<double *>(rhs.data)[astr::offset(rhs,i,j,k,c+1)]=jacobian*result[c];
}
__global__ void hbl_outflow_kernel(AstrDeviceView q,int upper,int ny,int nz,int halo,
    bool dynamic,bool physical_top,bool preserve,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=ny*nz)return;
    const int j=id%ny,k=id/ny;auto *data=static_cast<double *>(q.data);
    if(dynamic && physical_top && j==ny-1) {
        for(int c=0;c<11;++c)for(int i=upper+1;i<=upper+halo;++i)
            data[astr::offset(q,i,j,k,c+1)]=data[astr::offset(q,upper,j,k,c+1)];
        return;
    }
    double one[11],two[11],result[11];
    for(int c=0;c<11;++c) {one[c]=data[astr::offset(q,upper-1,j,k,c+1)];two[c]=data[astr::offset(q,upper-2,j,k,c+1)];}
    const int status=astr::air5::boundary::hbl_outflow(one,two,preserve,result);
    if(status) {atomicMax(failure,status);return;}
    for(int c=0;c<11;++c)for(int i=upper;i<=upper+halo;++i)data[astr::offset(q,i,j,k,c+1)]=result[c];
}
__global__ void farfield_kernel(AstrDeviceView q,const double *refs,const double *x,int nx,int upper,int nz,int halo,
    double origin,double incident_x,bool incident,bool dynamic,bool preserve,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=nx*nz)return;
    const int i=id%nx,k=id/nx;auto *data=static_cast<double *>(q.data);double result[11];int status=0;
    if(dynamic) {
        for(int c=0;c<11;++c)result[c]=data[astr::offset(q,i,upper,k,c+1)];
        double rho,u[3],t,y[5],tv,p;status=astr::air5::conservative_to_primitive(result,preserve,rho,u,t,y,tv,p);
    } else status=top_target(refs,x[id],origin,incident,incident_x,result);
    if(status) {atomicMax(failure,status);return;}
    for(int c=0;c<11;++c)for(int j=upper+(dynamic?1:0);j<=upper+halo;++j)data[astr::offset(q,i,j,k,c+1)]=result[c];
}
struct BoundaryEnds {int low[3],high[3];};
__global__ void carry_reset_kernel(AstrDeviceView carry,int nx,int ny,int nz,BoundaryEnds ends,bool dynamic)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=nx*ny*nz)return;
    const int i=id%nx,j=(id/nx)%ny,k=id/(nx*ny);
    const bool fixed=(ends.low[0] && i==0)||(ends.high[0] && i==nx-1 && !(dynamic && ends.high[1] && j==ny-1))||
        (ends.low[1] && j==0)||(ends.high[1] && j==ny-1 && !dynamic);
    if(fixed)for(int c=1;c<=11;++c)static_cast<double *>(carry.data)[astr::offset(carry,i,j,k,c)]=0.;
}
__global__ void fixed_kernel(AstrDeviceView q,const double *fixed,int upper,int ny,int nz,int halo,
    bool left,bool right,bool normal,double pressure,bool preserve,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=ny*nz)return;
    const int j=id%ny,k=id/ny;auto *data=static_cast<double *>(q.data);
    if(left)for(int c=0;c<11;++c)for(int i=-halo;i<=0;++i)data[astr::offset(q,i,j,k,c+1)]=fixed[c];
    if(right) {
        double result[11];
        if(normal && pressure>0.) {
            double inner[11];for(int c=0;c<11;++c)inner[c]=data[astr::offset(q,upper-1,j,k,c+1)];
            const int status=astr::air5::boundary::pressure_outlet(inner,pressure,preserve,result);
            if(status) {atomicMax(failure,status);return;}
        } else for(int c=0;c<11;++c)result[c]=normal?data[astr::offset(q,upper-1,j,k,c+1)]:fixed[11+c];
        for(int c=0;c<11;++c)for(int i=upper;i<=upper+halo;++i)data[astr::offset(q,i,j,k,c+1)]=result[c];
    }
}
__global__ void profile_kernel(AstrDeviceView q,const double *profile,int ny,int nz,int halo)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=ny*nz)return;
    const int j=id%ny,k=id/ny;auto *data=static_cast<double *>(q.data);
    for(int c=0;c<11;++c)for(int i=-halo;i<=0;++i)data[astr::offset(q,i,j,k,c+1)]=profile[j+ny*c];
}
int boundary_plan(astr::State &s)
{
    if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(!s.air5_boundary)s.air5_boundary=new(std::nothrow) astr::Air5BoundaryPlan{};
    return s.air5_boundary?0:static_cast<int>(cudaErrorMemoryAllocation);
}
__global__ void wall_kernel(AstrDeviceView q,int nx,int nz,int halo,double temperature,bool preserve,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=nx*nz)return;
    const int i=id%nx,k=id/nx;double one[11],two[11],result[11];
    auto *data=static_cast<double *>(q.data);
    for(int c=0;c<11;++c) {one[c]=data[astr::offset(q,i,1,k,c+1)];two[c]=data[astr::offset(q,i,2,k,c+1)];}
    const int status=astr::air5::boundary::isothermal_wall(one,two,temperature,preserve,result);
    if(status) {atomicMax(failure,status);return;}
    for(int j=-halo;j<=0;++j)for(int c=0;c<11;++c)data[astr::offset(q,i,j,k,c+1)]=result[c];
}
__global__ void outlet_kernel(AstrDeviceView q,int upper,int ny,int nz,int halo,double pressure,bool preserve,int *failure)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;if(id>=ny*nz)return;
    const int j=id%ny,k=id/ny;double inner[11],result[11];auto *data=static_cast<double *>(q.data);
    for(int c=0;c<11;++c)inner[c]=data[astr::offset(q,upper-1,j,k,c+1)];
    const int status=astr::air5::boundary::pressure_outlet(inner,pressure,preserve,result);
    if(status) {atomicMax(failure,status);return;}
    for(int i=upper;i<=upper+halo;++i)for(int c=0;c<11;++c)data[astr::offset(q,i,j,k,c+1)]=result[c];
}
}
extern "C" int astr_device_air5_characteristic_top_prepare(void *handle,const double *spacing,size_t count)
{
    if(!handle || !spacing)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(count!=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1))return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=boundary_plan(s);if(status)return status;auto &plan=*s.air5_boundary;
    if(!plan.top_spacing) {status=static_cast<int>(cudaMalloc(&plan.top_spacing,count*sizeof(double)));if(status)return status;}
    auto &high=s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH];
    if(!high.view.data) {const int upper[3]={s.upper[0],0,s.upper[2]};status=astr::allocate_field(high,upper,0,4,11,1,sizeof(double));if(status)return status;}
    status=static_cast<int>(cudaMemcpyAsync(plan.top_spacing,spacing,count*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}
extern "C" int astr_device_air5_characteristic_top_rhs(void *handle,int first_i,int preserve,double rate,double reference_sound)
{
    if(!handle || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(s.upper[1]<2 || first_i<0 || first_i>s.upper[0])return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto *plan=s.air5_boundary;
    if(!s.air5_ready || !plan || !plan->top || !plan->x_top || !plan->top_spacing ||
       !s.fields[ASTR_DEVICE_JACOBIAN].view.data || !s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH].view.data)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[0]-first_i+1)*(s.upper[2]+1);
    characteristic_top_kernel<<<(count+127)/128,128,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH].view,plan->top,plan->x_top,plan->top_spacing,
        s.upper[0]+1,s.upper[1]+1,s.upper[2]+1,first_i,plan->origin,plan->incident,plan->incident_x,preserve!=0,rate,reference_sound,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_top_boundary_prepare(void *handle,const double farfield[11],const double incident[22],
    double origin,double incident_x,int enabled,const double *x_top,size_t count)
{
    if(!handle || !farfield || !incident || !x_top || enabled<0 || enabled>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(count!=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1))return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=boundary_plan(s);if(status)return status;auto &plan=*s.air5_boundary;
    if(!plan.top) {status=static_cast<int>(cudaMalloc(&plan.top,33*sizeof(double)));if(status)return status;}
    if(!plan.x_top) {status=static_cast<int>(cudaMalloc(&plan.x_top,count*sizeof(double)));if(status)return status;}
    status=static_cast<int>(cudaMemcpyAsync(plan.top,farfield,11*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    status=static_cast<int>(cudaMemcpyAsync(plan.top+11,incident,22*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    status=static_cast<int>(cudaMemcpyAsync(plan.x_top,x_top,count*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    plan.origin=origin;plan.incident_x=incident_x;plan.incident=enabled!=0;
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}
extern "C" int astr_device_air5_hbl_outflow_upper_x(void *handle,int top,int dynamic,int preserve)
{
    if(!handle || top<0 || top>1 || dynamic<0 || dynamic>1 || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<2)return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    hbl_outflow_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.upper[0],s.upper[1]+1,s.upper[2]+1,
        s.halo,dynamic!=0,top!=0,preserve!=0,s.air5_failure);return astr::finish_launch(s);
}
extern "C" int astr_device_air5_farfield_upper_y(void *handle,int dynamic,int preserve)
{
    if(!handle || dynamic<0 || dynamic>1 || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_boundary || !s.air5_boundary->top || !s.air5_boundary->x_top)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const auto &plan=*s.air5_boundary;const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    farfield_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,plan.top,plan.x_top,s.upper[0]+1,
        s.upper[1],s.upper[2]+1,s.halo,plan.origin,plan.incident_x,plan.incident,dynamic!=0,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_hbl_boundary_carry_reset(void *handle,const int lower[3],const int upper[3],int dynamic)
{
    if(!handle || !lower || !upper || dynamic<0 || dynamic>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);const auto carry=s.fields[ASTR_DEVICE_AIR5_CARRY].view;
    if(!s.air5_ready || !carry.data)return ASTR_DEVICE_NOT_ALLOCATED;BoundaryEnds ends;
    for(int a=0;a<3;++a) {
        if(lower[a]<0 || lower[a]>1 || upper[a]<0 || upper[a]>1)return ASTR_DEVICE_INVALID_ARGUMENT;
        ends.low[a]=lower[a];ends.high[a]=upper[a];
    }
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    carry_reset_kernel<<<(count+255)/256,256,0,s.stream>>>(carry,s.upper[0]+1,s.upper[1]+1,s.upper[2]+1,ends,dynamic!=0);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_fixed_boundary_prepare(void *handle,const double left[11],const double right[11])
{
    if(!handle || !left || !right)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    int status=boundary_plan(s);if(status)return status;auto &plan=*s.air5_boundary;
    if(!plan.fixed) {status=static_cast<int>(cudaMalloc(&plan.fixed,22*sizeof(double)));if(status)return status;}
    status=static_cast<int>(cudaMemcpyAsync(plan.fixed,left,11*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    status=static_cast<int>(cudaMemcpyAsync(plan.fixed+11,right,11*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}
extern "C" int astr_device_air5_profile_boundary_prepare(void *handle,const double *profile,size_t count)
{
    if(!handle || !profile)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(count!=11*static_cast<size_t>(s.upper[1]+1))return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=boundary_plan(s);if(status)return status;auto &plan=*s.air5_boundary;
    if(!plan.profile) {status=static_cast<int>(cudaMalloc(&plan.profile,count*sizeof(double)));if(status)return status;}
    status=static_cast<int>(cudaMemcpyAsync(plan.profile,profile,count*sizeof(double),cudaMemcpyHostToDevice,s.stream));if(status)return status;
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}
extern "C" int astr_device_air5_fixed_boundary_apply(void *handle,int left,int right,int normal,double pressure,int preserve)
{
    if(!handle || left<0 || left>1 || right<0 || right>1 || normal<0 || normal>1 || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_boundary || !s.air5_boundary->fixed)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<1)return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    fixed_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.air5_boundary->fixed,s.upper[0],s.upper[1]+1,s.upper[2]+1,s.halo,left!=0,right!=0,normal!=0,pressure,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_profile_inflow_lower_x(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_boundary || !s.air5_boundary->profile)return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    profile_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.air5_boundary->profile,s.upper[1]+1,s.upper[2]+1,s.halo);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_wall_lower_y(void *handle,double temperature,int preserve)
{
    if(!handle || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[1]<2)return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    wall_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.upper[0]+1,s.upper[2]+1,s.halo,temperature,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_pressure_outlet_upper_x(void *handle,double pressure,int preserve)
{
    if(!handle || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.upper[0]<1)return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    outlet_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.upper[0],s.upper[1]+1,s.upper[2]+1,s.halo,pressure,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
