#include "device_state_internal.cuh"
#include "chemistry_statistics.h"
#include "chemistry_properties_math.cuh"
#include <climits>
#include <new>

namespace astr {
struct Air5MonitorPlan {
    int count=0;
    int *nodes=nullptr;
    double *coordinates=nullptr,*packed=nullptr;
};
int release_air5_monitor(Air5MonitorPlan *plan)
{
    if(!plan)return 0;
    int status=0;
    void *buffers[]={plan->nodes,plan->coordinates,plan->packed};
    for(void *p:buffers)if(p) {
        const int result=static_cast<int>(cudaFree(p));if(status==0)status=result;
    }
    delete plan;return status;
}
}

namespace {
__device__ inline double &at(const AstrDeviceView &v,int i,int j,int k,int c=1,int d=1)
{return static_cast<double *>(v.data)[astr::offset(v,i,j,k,c,d)];}

constexpr int conservation_components=11;
constexpr int conservation_max_tiles=4096;

__global__ void integrate_cell_tiles(AstrDeviceView q,AstrDeviceView jac,int im,int jm,int cells,int parts,double *work)
{
    const int component=threadIdx.x+1,tile=blockIdx.x;
    if(component>conservation_components)return;
    const int chunk=(cells+parts-1)/parts;
    const int first=tile*chunk,last=min((tile+1)*chunk,cells);
    double tile_total=0.;
    for(int id=first;id<last;++id) {
        const int i=id%im,j=(id/im)%jm,k=id/(im*jm);
        double cell_total=0.;
        for(int dk=0;dk<2;++dk)
            for(int dj=0;dj<2;++dj)
                for(int di=0;di<2;++di)
                    cell_total+=at(q,i+di,j+dj,k+dk,component)*at(jac,i+di,j+dj,k+dk);
        tile_total+=0.125*cell_total;
    }
    work[tile*conservation_components+component-1]=tile_total;
}

__global__ void sum_cell_tiles(int parts,double *work)
{
    const int component=threadIdx.x;
    if(component>=conservation_components)return;
    double total=0.;
    for(int tile=0;tile<parts;++tile)total+=work[tile*conservation_components+component];
    work[parts*conservation_components+component]=total;
}

__global__ void monitor_kernel(AstrDeviceView q,astr::Air5MonitorPlan plan)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=static_cast<size_t>(plan.count))return;
    double *packet=plan.packed+15*static_cast<size_t>(id);
    for(int c=0;c<15;++c)packet[c]=0.;
    const int i=plan.nodes[3*static_cast<size_t>(id)];if(i<0)return;
    const int j=plan.nodes[3*static_cast<size_t>(id)+1],k=plan.nodes[3*static_cast<size_t>(id)+2];
    for(int d=0;d<3;++d)packet[d]=plan.coordinates[3*static_cast<size_t>(id)+d];
    for(int c=0;c<11;++c)packet[c+3]=at(q,i,j,k,c+1);
    packet[14]=1.;
}

__device__ void point_moments(const double e[14],double v[44])
{
    const double r=e[0],u=e[1],v2=e[2],w=e[3],p=e[4],t=e[5];
    const double s11=e[6],s22=e[7],s33=e[8],s12=e[9],s13=e[10],s23=e[11];
    const double first[33]={r,r*u,r*v2,r*w,p,r*t,r*u*u,r*v2*v2,r*w*w,r*u*v2,r*u*w,r*v2*w,
        p*p,r*t*t,r*u*t,r*v2*t,r*w*t,r*((u*u)*u),r*((v2*v2)*v2),r*((w*w)*w),
        r*u*u*v2,r*u*u*w,r*u*v2*v2,r*u*w*w,r*v2*v2*w,r*v2*w*w,r*u*v2*w,
        u,v2,w,p*u,p*v2,p*w};
    for(int m=0;m<33;++m)v[m]=first[m];
    for(int m=0;m<8;++m)v[33+m]=e[6+m];
    v[41]=s11*u+s12*v2+s13*w;v[42]=s12*u+s22*v2+s23*w;v[43]=s13*u+s23*v2+s33*w;
}

__global__ void sample_kernel(AstrDeviceView rho,AstrDeviceView vel,AstrDeviceView pressure,
    AstrDeviceView temperature,AstrDeviceView tv,AstrDeviceView species,AstrDeviceView gradient,
    AstrDeviceView sum,AstrDeviceView endpoint,int nx,int ny,size_t count,double interval,bool previous,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count)return;
    const int i=id%nx,j=(id/nx)%ny,k=id/(static_cast<size_t>(nx)*ny);
    const double r=at(rho,i,j,k),p=at(pressure,i,j,k),t=at(temperature,i,j,k);
    const double u=at(vel,i,j,k,1),v=at(vel,i,j,k,2),w=at(vel,i,j,k,3);
    double y[5],mu,ktr,kv,mus[5],binary[25],mixture[5];
    for(int s=0;s<5;++s)y[s]=at(species,i,j,k,s+1);
    const int status=astr::air5::transport_properties(t,at(tv,i,j,k),p,y,mu,ktr,kv,mus,binary,mixture);
    if(status!=0) {atomicMax(failure,status);return;}
    const double d11=at(gradient,i,j,k,1,1),d12=at(gradient,i,j,k,1,2),d13=at(gradient,i,j,k,1,3);
    const double d21=at(gradient,i,j,k,2,1),d22=at(gradient,i,j,k,2,2),d23=at(gradient,i,j,k,2,3);
    const double d31=at(gradient,i,j,k,3,1),d32=at(gradient,i,j,k,3,2),d33=at(gradient,i,j,k,3,3);
    const double divergence=(1./3.)*(d11+d22+d33);
    const double s11=2.*mu*(d11-divergence),s22=2.*mu*(d22-divergence),s33=2.*mu*(d33-divergence);
    const double s12=mu*(d12+d21),s13=mu*(d13+d31),s23=mu*(d23+d32);
    const double current[14]={r,u,v,w,p,t,s11,s22,s33,s12,s13,s23,
        s11*d11+s12*d12+s13*d13+s12*d21+s22*d22+s23*d23+s13*d31+s23*d32+s33*d33,
        p*(d11+d22+d33)};
    double values[44],old_values[44],increment[44]={};
    point_moments(current,values);
    if(previous) {
        double old[14];for(int m=0;m<14;++m)old[m]=at(endpoint,i,j,k,m+1);
        point_moments(old,old_values);
        for(int m=0;m<44;++m)increment[m]=(.5*interval)*(old_values[m]+values[m]);
    }
    for(int m=0;m<44;++m)if(!astr::air5::finite(values[m]) || !astr::air5::finite(at(sum,i,j,k,m+1)+increment[m])) {
        atomicMax(failure,1);return;
    }
    for(int m=0;m<44;++m)at(sum,i,j,k,m+1)=at(sum,i,j,k,m+1)+increment[m];
    for(int m=0;m<14;++m)at(endpoint,i,j,k,m+1)=current[m];
}
}

extern "C" int astr_device_air5_monitor_prepare(void *handle,int count,const int *nodes,const double *xyz)
{
    if(!handle || count<1 || !nodes || !xyz)return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || s.air5_monitor)return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int p=0;p<count;++p)if(nodes[3*static_cast<size_t>(p)]>=0)
        for(int d=0;d<3;++d)if(nodes[3*static_cast<size_t>(p)+d]<0 || nodes[3*static_cast<size_t>(p)+d]>s.upper[d])
            return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *plan=new(std::nothrow) astr::Air5MonitorPlan{};
    if(!plan)return static_cast<int>(cudaErrorMemoryAllocation);
    plan->count=count;
    int status=static_cast<int>(cudaMalloc(&plan->nodes,3*static_cast<size_t>(count)*sizeof(int)));
    if(status==0)status=static_cast<int>(cudaMalloc(&plan->coordinates,3*static_cast<size_t>(count)*sizeof(double)));
    if(status==0)status=static_cast<int>(cudaMalloc(&plan->packed,15*static_cast<size_t>(count)*sizeof(double)));
    if(status==0)status=static_cast<int>(cudaMemcpyAsync(plan->nodes,nodes,3*static_cast<size_t>(count)*sizeof(int),cudaMemcpyHostToDevice,s.stream));
    if(status==0)status=static_cast<int>(cudaMemcpyAsync(plan->coordinates,xyz,3*static_cast<size_t>(count)*sizeof(double),cudaMemcpyHostToDevice,s.stream));
    const int synchronized=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0)status=synchronized;
    if(status!=0) {astr::release_air5_monitor(plan);return status;}
    s.air5_monitor=plan;return 0;
}
extern "C" int astr_device_air5_monitor_collect(void *handle,double *packed,size_t bytes)
{
    if(!handle || !packed)return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.air5_monitor)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &plan=*s.air5_monitor;
    if(bytes!=15*static_cast<size_t>(plan.count)*sizeof(double))return ASTR_DEVICE_INVALID_ARGUMENT;
    monitor_kernel<<<(static_cast<size_t>(plan.count)+127)/128,128,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,plan);
    int status=astr::finish_launch(s);if(status)return status;
    status=static_cast<int>(cudaMemcpyAsync(packed,plan.packed,bytes,cudaMemcpyDeviceToHost,s.stream));
    return status?status:static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_air5_conservation_prepare(void *handle,int64_t budget,int64_t *bytes)
{
    if(!handle || !bytes || budget<0)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fields[ASTR_DEVICE_JACOBIAN].view.data)return ASTR_DEVICE_NOT_ALLOCATED;
    const int64_t cells=static_cast<int64_t>(s.upper[0])*s.upper[1]*s.upper[2];
    if(cells<1 || cells>INT_MAX-(conservation_max_tiles-1) || s.air5_conservation_work)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    const int parts=min(conservation_max_tiles,static_cast<int>((cells+63)/64));
    *bytes=static_cast<int64_t>(parts+1)*conservation_components*sizeof(double);
    if(*bytes>budget)return ASTR_DEVICE_INVALID_ARGUMENT;
    return static_cast<int>(cudaMalloc(&s.air5_conservation_work,static_cast<size_t>(*bytes)));
}

extern "C" int astr_device_air5_conservation_collect(void *handle,double totals[11])
{
    if(!handle || !totals)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_conservation_work)return ASTR_DEVICE_NOT_ALLOCATED;
    const int cells=s.upper[0]*s.upper[1]*s.upper[2];
    const int parts=min(conservation_max_tiles,static_cast<int>((static_cast<int64_t>(cells)+63)/64));
    auto *work=s.air5_conservation_work;
    integrate_cell_tiles<<<parts,32,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.upper[0],s.upper[1],cells,parts,work);
    int status=astr::finish_launch(s);
    if(status)return status;
    sum_cell_tiles<<<1,32,0,s.stream>>>(parts,work);
    status=astr::finish_launch(s);
    if(status)return status;
    status=static_cast<int>(cudaMemcpyAsync(totals,work+parts*conservation_components,
        sizeof(double)*conservation_components,cudaMemcpyDeviceToHost,s.stream));
    return status ? status : static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_air5_mean_prepare(void *handle)
{
    if(!handle || sizeof(astr::numerical_real)!=sizeof(double))return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || s.air5_mean_failure)return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_MEAN_SUM],s.upper,0,4,44);
    if(status)return status;
    status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_MEAN_ENDPOINT],s.upper,0,4,14);
    if(status)return status;
    status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_MEAN_VELOCITY],s.upper,s.halo,4,3);
    if(status)return status;
    status=static_cast<int>(cudaMalloc(&s.air5_mean_failure,sizeof(int)));if(status)return status;
    return astr_device_air5_mean_reset(handle);
}
extern "C" int astr_device_air5_mean_reset(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.air5_mean_failure)return ASTR_DEVICE_NOT_ALLOCATED;
    const int fields[]={ASTR_DEVICE_AIR5_MEAN_SUM,ASTR_DEVICE_AIR5_MEAN_ENDPOINT};
    for(int field:fields) {
        const int status=static_cast<int>(cudaMemsetAsync(s.fields[field].view.data,0,s.fields[field].bytes,s.stream));
        if(status)return status;
    }
    return static_cast<int>(cudaMemsetAsync(s.air5_mean_failure,0,sizeof(int),s.stream));
}
extern "C" int astr_device_air5_mean_sample(void *handle,double interval,int has_previous)
{
    if(!handle || !std::isfinite(interval) || interval<0 || has_previous<0 || has_previous>1 ||
        (has_previous && interval==0))return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.air5_mean_failure || !s.gradients_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_mean_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    sample_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
        s.fields[ASTR_DEVICE_PRESSURE].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_AIR5_TV].view,
        s.fields[ASTR_DEVICE_AIR5_SPECIES].view,s.fields[ASTR_DEVICE_DVELOCITY].view,
        s.fields[ASTR_DEVICE_AIR5_MEAN_SUM].view,s.fields[ASTR_DEVICE_AIR5_MEAN_ENDPOINT].view,
        s.upper[0]+1,s.upper[1]+1,count,interval,has_previous!=0,s.air5_mean_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_mean_release(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    int status=static_cast<int>(cudaStreamSynchronize(s.stream));if(status)return status;
    const int fields[]={ASTR_DEVICE_AIR5_MEAN_SUM,ASTR_DEVICE_AIR5_MEAN_ENDPOINT,ASTR_DEVICE_AIR5_MEAN_VELOCITY};
    for(int field:fields)if(s.fields[field].view.data) {
        status=static_cast<int>(cudaFree(s.fields[field].view.data));if(status)return status;
        s.fields[field]={};
    }
    if(s.air5_mean_failure) {
        status=static_cast<int>(cudaFree(s.air5_mean_failure));if(status)return status;
        s.air5_mean_failure=nullptr;
    }
    return 0;
}
extern "C" int astr_device_air5_mean_check(void *handle,int *status)
{
    if(!handle || !status)return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(handle);
    if(!s.air5_mean_failure)return ASTR_DEVICE_NOT_ALLOCATED;
    const int code=static_cast<int>(cudaMemcpyAsync(status,s.air5_mean_failure,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
    return code?code:static_cast<int>(cudaStreamSynchronize(s.stream));
}
