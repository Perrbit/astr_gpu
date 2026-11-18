#include <cmath>
#include <new>
#include "device_state_internal.cuh"
#include "production_statistics.h"
using Real=astr::numerical_real;

namespace astr {
struct ProductionStatisticsPlan {
    size_t plane_nodes=0,wall_nodes=0;
    Real *segments=nullptr,*frame=nullptr,*measure=nullptr;
    Real *sum=nullptr,*endpoint=nullptr,*wall_sum=nullptr,*wall_endpoint=nullptr;
};

int release_production_statistics(ProductionStatisticsPlan *plan)
{
    if(!plan) return 0;
    int status=0;
    void *pointers[]={plan->segments,plan->frame,plan->measure,plan->sum,plan->endpoint,
                      plan->wall_sum,plan->wall_endpoint};
    for(void *p:pointers) if(p) {
        const int result=static_cast<int>(cudaFree(p));
        if(status==0) status=result;
    }
    delete plan;
    return status;
}
}

namespace {
using astr::at;
using astr::ProductionStatisticsPlan;

__global__ void measure_kernel(ProductionStatisticsPlan p,int km)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=p.plane_nodes) return;
    Real total=0;
    for(int k=0;k<km;++k) total+=p.segments[id+p.plane_nodes*k];
    p.measure[id]=total;
}

__device__ void plane_values(AstrDeviceView rho,AstrDeviceView velocity,AstrDeviceView pressure,
    AstrDeviceView temperature,int i,int j,int k,Real v[12])
{
    const Real r=at(rho,i,j,k),u=at(velocity,i,j,k,1),v2=at(velocity,i,j,k,2),w=at(velocity,i,j,k,3);
    v[0]=r;v[1]=r*u;v[2]=r*v2;v[3]=r*w;v[4]=r*at(temperature,i,j,k);v[5]=at(pressure,i,j,k);
    v[6]=r*u*u;v[7]=r*v2*v2;v[8]=r*w*w;v[9]=r*u*v2;v[10]=r*u*w;v[11]=r*v2*w;
}

__global__ __launch_bounds__(astr::block_threads)
void plane_kernel(ProductionStatisticsPlan p,AstrDeviceView rho,AstrDeviceView velocity,
    AstrDeviceView pressure,AstrDeviceView temperature,int im,int jm,int km,Real interval,bool previous)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y*blockDim.y+threadIdx.y;
    if(i>im || j>jm) return;
    const size_t id=i+static_cast<size_t>(im+1)*j;
    Real left[12],right[12],total[12]={};
    plane_values(rho,velocity,pressure,temperature,i,j,0,left);
    for(int k=1;k<=km;++k) {
        const Real ds=p.segments[id+p.plane_nodes*(k-1)];
        plane_values(rho,velocity,pressure,temperature,i,j,k,right);
        for(int m=0;m<12;++m) {total[m]+=Real(.5)*(left[m]+right[m])*ds;left[m]=right[m];}
    }
    for(int m=0;m<12;++m) {
        const size_t n=id+p.plane_nodes*m;
        if(previous) p.sum[n]+=(Real(.5)*interval)*(p.endpoint[n]+total[m]);
        p.endpoint[n]=total[m];
    }
}

struct WallConstants {Real re,pr,c5,cp,tc,tc1;int nondimensional;};

__device__ void wall_values(ProductionStatisticsPlan p,AstrDeviceView pressure,AstrDeviceView temperature,
    AstrDeviceView gradient,AstrDeviceView thermal,int im,int i,int k,WallConstants c,Real values[3])
{
    const size_t surface=i+static_cast<size_t>(im+1)*k;
    Real normal[3],tangent[3],stress[3][3],traction[3];
    for(int a=0;a<3;++a) {normal[a]=p.frame[surface+p.wall_nodes*a];tangent[a]=p.frame[surface+p.wall_nodes*(a+3)];}
    const Real t=at(temperature,i,0,k);
    Real mu,conductivity;
    if(c.nondimensional) {
        mu=t*sqrt(t)*c.tc1/(t+c.tc)/c.re;
        conductivity=(mu/c.pr)/c.c5;
    } else {
        const Real tn=t/Real(273.15);
        mu=Real(1.716e-5)*tn*sqrt(tn)*(Real(273.15)+Real(110.4))/(t+Real(110.4));
        conductivity=c.cp*mu/c.pr;
    }
    const Real divergence=astr::load_value(gradient,i,0,k,1,1)+astr::load_value(gradient,i,0,k,2,2)+astr::load_value(gradient,i,0,k,3,3);
    for(int a=0;a<3;++a) {
        stress[a][a]=mu*(2*astr::load_value(gradient,i,0,k,a+1,a+1)-2*divergence/3);
        for(int b=a+1;b<3;++b) {
            stress[a][b]=mu*(astr::load_value(gradient,i,0,k,a+1,b+1)+astr::load_value(gradient,i,0,k,b+1,a+1));
            stress[b][a]=stress[a][b];
        }
    }
    for(int a=0;a<3;++a) traction[a]=stress[a][0]*normal[0]+stress[a][1]*normal[1]+stress[a][2]*normal[2];
    values[0]=at(pressure,i,0,k);
    values[1]=tangent[0]*traction[0]+tangent[1]*traction[1]+tangent[2]*traction[2];
    values[2]=conductivity*(astr::load_value(thermal,i,0,k,1)*normal[0]+astr::load_value(thermal,i,0,k,2)*normal[1]+astr::load_value(thermal,i,0,k,3)*normal[2]);
}

__global__ void wall_kernel(ProductionStatisticsPlan p,AstrDeviceView pressure,AstrDeviceView temperature,
    AstrDeviceView gradient,AstrDeviceView thermal,int im,int km,Real interval,bool previous,WallConstants c)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>im) return;
    Real left[3],right[3],total[3]={};
    wall_values(p,pressure,temperature,gradient,thermal,im,i,0,c,left);
    for(int k=1;k<=km;++k) {
        const Real ds=p.segments[i+p.plane_nodes*(k-1)];
        wall_values(p,pressure,temperature,gradient,thermal,im,i,k,c,right);
        for(int m=0;m<3;++m) {total[m]+=Real(.5)*(left[m]+right[m])*ds;left[m]=right[m];}
    }
    for(int m=0;m<3;++m) {
        const size_t id=i+static_cast<size_t>(im+1)*m;
        if(previous) p.wall_sum[id]+=(Real(.5)*interval)*(p.wall_endpoint[id]+total[m]);
        p.wall_endpoint[id]=total[m];
    }
}

int reset(astr::State &s,ProductionStatisticsPlan &p)
{
    void *buffers[]={p.sum,p.endpoint,p.wall_sum,p.wall_endpoint};
    const size_t counts[]={12*p.plane_nodes,12*p.plane_nodes,
        p.frame?3*static_cast<size_t>(s.upper[0]+1):0,p.frame?3*static_cast<size_t>(s.upper[0]+1):0};
    for(int b=0;b<4;++b) if(counts[b]) {
        const int status=static_cast<int>(cudaMemsetAsync(buffers[b],0,counts[b]*sizeof(Real),s.stream));
        if(status!=0) return status;
    }
    return 0;
}
}

extern "C" int astr_device_production_prepare(void *parent,const Real *segments,const Real *wall_frame)
{
    if(!parent || !segments) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(s.production || s.upper[2]<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    ProductionStatisticsPlan *p=new(std::nothrow) ProductionStatisticsPlan;
    if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->plane_nodes=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1);
    p->wall_nodes=wall_frame?static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1):0;
    Real **buffers[]={&p->segments,&p->measure,&p->sum,&p->endpoint,&p->frame,&p->wall_sum,&p->wall_endpoint};
    const size_t counts[]={p->plane_nodes*s.upper[2],p->plane_nodes,12*p->plane_nodes,12*p->plane_nodes,
                          6*p->wall_nodes,wall_frame?3*static_cast<size_t>(s.upper[0]+1):0,
                          wall_frame?3*static_cast<size_t>(s.upper[0]+1):0};
    int status=0;
    for(int b=0;b<7 && status==0;++b) if(counts[b])
        status=static_cast<int>(cudaMalloc(buffers[b],counts[b]*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(p->segments,segments,counts[0]*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    if(status==0 && wall_frame) status=static_cast<int>(cudaMemcpyAsync(p->frame,wall_frame,counts[4]*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    if(status==0) status=reset(s,*p);
    if(status==0) {
        measure_kernel<<<(p->plane_nodes+255)/256,256,0,s.stream>>>(*p,s.upper[2]);
        status=astr::finish_launch(s);
    }
    const int synchronized=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) status=synchronized;
    if(status!=0) {astr::release_production_statistics(p);return status;}
    s.production=p;
    return 0;
}

extern "C" int astr_device_production_reset(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    return s.production?reset(s,*s.production):ASTR_DEVICE_NOT_ALLOCATED;
}

extern "C" int astr_device_production_plane(void *parent,Real interval,int has_previous)
{
    if(!parent || !std::isfinite(interval) || interval<0 || has_previous<0 || has_previous>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.production || !s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    plane_kernel<<<dim3((s.upper[0]+32)/32,(s.upper[1]+16)/16),dim3(32,16),0,s.stream>>>(*s.production,
        s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,s.upper[0],s.upper[1],s.upper[2],interval,has_previous);
    return astr::finish_launch(s);
}

extern "C" int astr_device_production_wall(void *parent,Real interval,int has_previous,
    int nondimensional,const Real constants[6])
{
    if(!parent || !constants || !std::isfinite(interval) || interval<0 || has_previous<0 || has_previous>1 ||
        nondimensional<0 || nondimensional>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.production || !s.production->frame || !s.gradients_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    WallConstants c{constants[0],constants[1],constants[2],constants[3],constants[4],constants[5],nondimensional};
    const AstrDeviceView diagnostic=s.fields[ASTR_DEVICE_DIAGNOSTIC_DVELOCITY].view;
    const AstrDeviceView thermal=s.fields[ASTR_DEVICE_DIAGNOSTIC_DTEMPERATURE].view;
    wall_kernel<<<(s.upper[0]+256)/256,256,0,s.stream>>>(*s.production,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,diagnostic.data?diagnostic:s.fields[ASTR_DEVICE_DVELOCITY].view,
        thermal.data?thermal:s.fields[ASTR_DEVICE_DTEMPERATURE].view,
        s.upper[0],s.upper[2],interval,has_previous,c);
    return astr::finish_launch(s);
}

extern "C" int astr_device_production_transfer(void *parent,int buffer,int upload,Real *host,size_t bytes)
{
    if(!parent || !host || buffer<0 || buffer>5 || upload<0 || upload>1 ||
        (upload && (buffer==ASTR_PRODUCTION_MEASURE || buffer==ASTR_PRODUCTION_WALL_MEASURE))) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.production) return ASTR_DEVICE_NOT_ALLOCATED;
    ProductionStatisticsPlan &p=*s.production;
    Real *pointers[]={p.measure,p.sum,p.endpoint,p.frame?p.measure:nullptr,p.wall_sum,p.wall_endpoint};
    const size_t counts[]={p.plane_nodes,12*p.plane_nodes,12*p.plane_nodes,
        static_cast<size_t>(s.upper[0]+1),3*static_cast<size_t>(s.upper[0]+1),3*static_cast<size_t>(s.upper[0]+1)};
    if(!pointers[buffer]) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=counts[buffer]*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=static_cast<int>(cudaMemcpyAsync(upload?pointers[buffer]:host,upload?host:pointers[buffer],bytes,
        upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,s.stream));
    const int synchronized=static_cast<int>(cudaStreamSynchronize(s.stream));
    return status==0?synchronized:status;
}
