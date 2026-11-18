#include <new>
#include "device_state_internal.cuh"
#include "chemistry_transport.h"
#include "chemistry_filter_math.cuh"
#include "chemistry_compensation_math.cuh"

namespace astr {
struct Air5FilterPlan {
    struct Parameters {int upper[3],global[3],offset[3],periodic[3],halo;double tr[5],v[5];} parameters{};
    double *strips=nullptr,*faces=nullptr,*aliases=nullptr;
    size_t area=0,strip_capacity=0;
    bool scalar=true;
};
int release_air5_filter(Air5FilterPlan *plan)
{
    if(!plan)return 0;int status=0;
    for(double *p:{plan->strips,plan->faces,plan->aliases})if(p) {
        const int freed=static_cast<int>(cudaFree(p));if(status==0)status=freed;
    }
    delete plan;return status;
}
}
namespace {
using Parameters=astr::Air5FilterPlan::Parameters;
__device__ double &at(AstrDeviceView v,const int p[3],int c)
{return static_cast<double *>(v.data)[astr::offset(v,p[0],p[1],p[2],c+1)];}
__device__ void node(int axis,int r,int t1,int t2,int p[3])
{p[0]=axis==0?r:t1;p[1]=axis==0?t1:(axis==1?r:t2);p[2]=axis==2?r:t2;}
__device__ bool index(int axis,Parameters data,int p[3])
{
    int id;
    if(axis==0) {
        id=blockIdx.x*blockDim.x+threadIdx.x;p[0]=id%(data.upper[0]+1);
        p[1]=(id/(data.upper[0]+1))%(data.upper[1]+1);p[2]=id/((data.upper[0]+1)*(data.upper[1]+1));
    } else if(axis==1) {
        id=blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
        p[0]=id%(data.upper[0]+1);p[1]=id/(data.upper[0]+1);p[2]=blockIdx.z;
    } else {
        id=blockIdx.x*blockDim.x*blockDim.z+threadIdx.z*blockDim.x+threadIdx.x;
        p[0]=id%(data.upper[0]+1);p[1]=blockIdx.y;p[2]=id/(data.upper[0]+1);
    }
    return p[0]<=data.upper[0] && p[1]<=data.upper[1] && p[2]<=data.upper[2];
}
__global__ void pack_kernel(AstrDeviceView q,AstrDeviceView carry,double *strips,size_t capacity,
    Parameters data,int axis,int n1,int n2,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    const size_t area=static_cast<size_t>(n1+1)*(n2+1);const int t=id%area,layer=(id/area)%(data.halo+1),side=id/(area*(data.halo+1));
    int p[3];node(axis,side==0?layer:data.upper[axis]-data.halo+layer,t%(n1+1),t/(n1+1),p);
    const size_t base=22*(layer+(data.halo+1)*static_cast<size_t>(t))+side*capacity;
    for(int c=0;c<11;++c) {strips[base+c]=at(q,p,c);strips[base+11+c]=carry.data?at(carry,p,c):0.;}
}
__device__ void get(AstrDeviceView q,AstrDeviceView carry,const double *strips,size_t capacity,
    Parameters data,int axis,int n1,int r,int t1,int t2,bool has_upper,double h[11],double c[11])
{
    const int n=data.upper[axis],t=t1+(n1+1)*t2;
    if(r<0 || (r>=n && has_upper)) {
        const int layer=r<0?r+data.halo:r-n,side=r<0?2:3;
        const size_t base=side*capacity+22*(layer+(data.halo+1)*static_cast<size_t>(t));
        for(int s=0;s<11;++s) {h[s]=strips[base+s];c[s]=strips[base+11+s];}
    } else {
        int p[3];node(axis,r,t1,t2,p);
        for(int s=0;s<11;++s) {h[s]=at(q,p,s);c[s]=carry.data?at(carry,p,s):0.;}
    }
}
__global__ __launch_bounds__(512) void faces_kernel(AstrDeviceView q,AstrDeviceView carry,AstrDeviceView rhs,
    const double *strips,size_t capacity,double *faces,Parameters data,int axis,int n1,bool has_upper,int *failure)
{
    int p[3];if(!index(axis,data,p))return;
    const int r=p[axis],g=data.offset[axis]+r,t1=axis==0?p[1]:p[0],t2=axis==2?p[1]:p[2],t=t1+(n1+1)*t2;
    double flux[11]{};int status=0;
    if(r<data.upper[axis] && (data.periodic[axis] || (g>0 && g<data.global[axis]-1))) {
        double stencil[110]{},low[110]{},lh[11],lc[11],rh[11],rc[11],left[11],right[11],lm[11],rm[11],high[11],candidate[11],lower[5],upper[5],mass,ratio;
        for(int layer=-4;layer<=5;++layer) {
            if(!data.periodic[axis] && (g+layer<1 || g+layer>data.global[axis]-1))continue;
            get(q,carry,strips,capacity,data,axis,n1,r+layer,t1,t2,has_upper,lh,lc);
            for(int s=0;s<11;++s) {stencil[11*(layer+4)+s]=lh[s];low[11*(layer+4)+s]=lc[s];}
        }
        get(q,carry,strips,capacity,data,axis,n1,r,t1,t2,has_upper,lh,lc);
        get(q,carry,strips,capacity,data,axis,n1,r+1,t1,t2,has_upper,rh,rc);
        for(int s=0;s<11;++s) {left[s]=carry.data?lh[s]-lc[s]:lh[s];right[s]=carry.data?rh[s]-rc[s]:rh[s];lm[s]=fabs(lh[s]);rm[s]=fabs(rh[s]);}
        if(carry.data) {astr::air5::filter::state_magnitude(lh,lc,lm);astr::air5::filter::state_magnitude(rh,rc,rm);}
        astr::air5::filter::high_face(stencil,low,g,data.global[axis],data.periodic[axis]!=0,carry.data!=nullptr,high);
        astr::air5::filter::face_bounds(left,right,high,lower,upper,mass);
        for(int s=0;s<11;++s)candidate[s]=high[s];candidate[0]=mass;
        status=astr::air5::close_species_flux(high+5,lower,upper,mass,candidate+5);
        if(status==0)status=astr::air5::filter::thermal_limit(left,right,candidate,data.tr,data.v,lm,rm,carry.data!=nullptr,flux,ratio);
        if(status)atomicMax(failure,status);
    }
    for(int s=0;s<11;++s) {at(rhs,p,s)=flux[s];if(r==data.upper[axis]-1)faces[s+11*static_cast<size_t>(t)]=flux[s];}
}
__global__ __launch_bounds__(512) void commit_kernel(AstrDeviceView q,AstrDeviceView carry,AstrDeviceView rhs,
    AstrDeviceView work,const double *left_face,Parameters data,int axis,int n1,bool has_lower,bool has_upper,bool scalar)
{
    int p[3];if(!index(axis,data,p))return;const int r=p[axis],g=data.offset[axis]+r;
    if(r==data.upper[axis] && has_upper)return;
    if(!data.periodic[axis] && (g==0 || g==data.global[axis]))return;
    const int t1=axis==0?p[1]:p[0],t2=axis==2?p[1]:p[2],t=t1+(n1+1)*t2;
    int b[3]={p[0],p[1],p[2]};--b[axis];
    for(int s=0;s<11;++s) {
        const double fr=at(rhs,p,s),fl=r==0?(has_lower?left_face[s+11*static_cast<size_t>(t)]:0.):at(rhs,b,s);
        double h=at(q,p,s);
        if(fr!=0 || fl!=0) {
            if(carry.data) {double c=at(carry,p,s);astr::air5::compensated_add(h,c,fr);astr::air5::compensated_add(h,c,-fl);at(carry,p,s)=c;}
            else h=(h+fr)-fl;
        }
        at(work,p,scalar?0:s)=h;at(q,p,s)=at(work,p,scalar?0:s);
    }
}
__global__ void alias_kernel(AstrDeviceView q,AstrDeviceView carry,double *aliases,size_t area,
    Parameters data,int axis,int n1,size_t count,bool unpack)
{
    const size_t t=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=count)return;
    int p[3];node(axis,unpack?data.upper[axis]:0,t%(n1+1),t/(n1+1),p);
    for(int s=0;s<11;++s) {
        if(unpack) {at(q,p,s)=aliases[22*area+22*t+s];if(carry.data)at(carry,p,s)=aliases[22*area+22*t+11+s];}
        else {aliases[22*t+s]=at(q,p,s);aliases[22*t+11+s]=carry.data?at(carry,p,s):0.;}
    }
}
__global__ void check_kernel(AstrDeviceView q,AstrDeviceView carry,Parameters data,size_t count,bool preserve,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    const int p[3]={static_cast<int>(id%(data.upper[0]+1)),static_cast<int>((id/(data.upper[0]+1))%(data.upper[1]+1)),static_cast<int>(id/((data.upper[0]+1)*(data.upper[1]+1)))};
    double state[11],low[11];for(int s=0;s<11;++s) {state[s]=at(q,p,s);low[s]=carry.data?at(carry,p,s):0.;}
    if(!astr::air5::filter::pressure_represented_admissible(state,low)) {atomicMax(failure,3);return;}
    for(int s=0;s<11;++s)state[s]-=low[s];
    double density,u[3],t,y[5],tv,pres;
    const int status=astr::air5::conservative_to_primitive(state,preserve,density,u,t,y,tv,pres);
    if(status)atomicMax(failure,status);
}
struct VibrationalFloor {double value[5];};

__device__ int legacy_species_repair(AstrDeviceView q,AstrDeviceView rhs,AstrDeviceView carry,
    const int p[3],const double floor[5])
{
    using namespace astr::air5;
    double state[11];for(int m=0;m<11;++m)state[m]=at(q,p,m);
    const double rho_base=at(rhs,p,0),rho=state[0];
    if(!astr::air5::finite(rho_base) || !astr::air5::finite(rho))return nonfinite;
    if(rho_base<=0. || rho<=0.)return invalid_density;
    double base[5],species[5],base_sum=0.;int closure=0;
    for(int s=0;s<5;++s) {
        base[s]=at(rhs,p,5+s);species[s]=state[5+s];
        if(!astr::air5::finite(base[s]) || !astr::air5::finite(species[s]))return nonfinite;
        if(base[s]<0.)return invalid_composition;
        base_sum+=base[s];if(base[s]>base[closure])closure=s;
    }
    if(base_sum<=DBL_MIN || fabs(base_sum-rho_base)>1.e-10*fmax(rho_base,1.))return invalid_composition;
    double other=0.;
    for(int s=0;s<5;++s)if(s!=closure)other+=species[s];
    species[closure]=rho-other;
    bool limited=false;for(int s=0;s<5;++s)limited=limited || species[s]<0.;
    if(limited) {
        double theta=1.,low[5];
        for(int s=0;s<5;++s) {
            low[s]=rho*base[s]/base_sum;
            if(species[s]-low[s]<0.)theta=fmin(theta,low[s]/(low[s]-species[s]));
        }
        theta=interior_ratio(theta);
        // NVHPC contracts this legacy blend; retain it with global contraction off.
        for(int s=0;s<5;++s)species[s]=fma(theta,species[s]-low[s],low[s]);
    }
    other=0.;for(int s=0;s<5;++s)if(s!=closure)other+=species[s];
    species[closure]=rho-other;
    for(int s=0;s<5;++s)if(species[s]<0.)return invalid_composition;
    if(carry.data) {
        double low[11];for(int m=0;m<11;++m)low[m]=at(carry,p,m);
        compensated_filter_projection(state,low,species,closure);
        for(int m=0;m<11;++m)if(!astr::air5::finite(low[m]))return nonfinite;
        for(int s=5;s<10;++s)if(state[s]<low[s])return invalid_composition;
        for(int m=0;m<11;++m) {at(q,p,m)=state[m];at(carry,p,m)=low[m];state[m]-=low[m];}
    } else {
        for(int s=0;s<5;++s) {state[5+s]=species[s];at(q,p,5+s)=species[s];}
    }
    // Match the original commit-before-domain-check order; the caller stops on failure.
    return state_admissible(state,floor)?ok:invalid_composition;
}

template<bool save>
__global__ void legacy_species_kernel(AstrDeviceView q,AstrDeviceView rhs,AstrDeviceView carry,
    int nx,int ny,size_t count,VibrationalFloor floor,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count)return;
    const int p[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
    if(save) {
        at(rhs,p,0)=at(q,p,0);
        for(int s=5;s<10;++s)at(rhs,p,s)=at(q,p,s);
    } else {
        const int status=legacy_species_repair(q,rhs,carry,p,floor.value);
        if(status)atomicMax(failure,status);
    }
}

template<bool save>
int legacy_species(void *handle,const double *floor)
{
    if(!handle || (!save && !floor))return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    VibrationalFloor data{};
    if(!save) {
        const int rc=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(rc)return rc;
        for(int n=0;n<5;++n)data.value[n]=floor[n];
    }
    const int nx=s.upper[0]+1,ny=s.upper[1]+1;
    const size_t count=static_cast<size_t>(nx)*ny*(s.upper[2]+1);
    legacy_species_kernel<save><<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,nx,ny,count,data,s.air5_failure);
    return astr::finish_launch(s);
}

template<bool swap>
__global__ void legacy_compensation_kernel(AstrDeviceView q,AstrDeviceView carry,int nx,int ny,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count)return;
    const int p[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
    for(int m=0;m<11;++m) {
        double &high=at(q,p,m),&low=at(carry,p,m);
        if(swap) {const double previous=high;high=low;low=previous;}
        else astr::air5::compensated_normalize(high,low);
    }
}
void launch_shape(astr::State &s,int axis,dim3 &block,dim3 &grid,int &n1,int &n2)
{
    n1=s.upper[axis==0?1:0];n2=s.upper[axis==2?1:2];
    if(axis==0) {block=dim3(512,1,1);grid=dim3(((s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1)+511)/512,1,1);}
    else if(axis==1) {block=dim3(32,16,1);grid=dim3(((s.upper[0]+1)*(s.upper[1]+1)+511)/512,1,s.upper[2]+1);}
    else {block=dim3(64,1,8);grid=dim3(((s.upper[0]+1)*(s.upper[2]+1)+511)/512,s.upper[1]+1,1);}
}
}
extern "C" int astr_device_air5_legacy_filter_save(void *handle)
{return legacy_species<true>(handle,nullptr);}

extern "C" int astr_device_air5_legacy_filter_repair(void *handle,const double floor[5])
{return legacy_species<false>(handle,floor);}

extern "C" int astr_device_air5_legacy_filter_compensation(void *handle,int swap)
{
    if(!handle || swap<0 || swap>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    const AstrDeviceView q=s.fields[ASTR_DEVICE_Q].view,carry=s.fields[ASTR_DEVICE_AIR5_CARRY].view;
    if(!carry.data)return ASTR_DEVICE_NOT_ALLOCATED;
    const int nx=s.upper[0]+1,ny=s.upper[1]+1;
    const size_t count=static_cast<size_t>(nx)*ny*(s.upper[2]+1),grid=(count+255)/256;
    if(swap)legacy_compensation_kernel<true><<<grid,256,0,s.stream>>>(q,carry,nx,ny,count);
    else legacy_compensation_kernel<false><<<grid,256,0,s.stream>>>(q,carry,nx,ny,count);
    return astr::finish_launch(s);
}

extern "C" int astr_device_air5_filter_prepare(void *handle,const double tr[5],const double v[5],
    const int global[3],const int offset[3],const int periodic[3],int scalar)
{
    if(!handle || !tr || !v || !global || !offset || !periodic || scalar<0 || scalar>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.rhs_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.air5_filter || s.fields[ASTR_DEVICE_FILTER_WORK].view.data || s.halo<5)return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a)if(s.upper[a]<s.halo || global[a]<7 || offset[a]<0 || offset[a]+s.upper[a]>global[a] || periodic[a]<0 || periodic[a]>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    s.air5_filter=new(std::nothrow) astr::Air5FilterPlan{};if(!s.air5_filter)return static_cast<int>(cudaErrorMemoryAllocation);
    auto &plan=*s.air5_filter;plan.scalar=scalar!=0;
    for(int a=0;a<3;++a) {
        plan.parameters.upper[a]=s.upper[a];plan.parameters.global[a]=global[a];plan.parameters.offset[a]=offset[a];plan.parameters.periodic[a]=periodic[a];
        const size_t area=static_cast<size_t>(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1);if(area>plan.area)plan.area=area;
    }
    plan.parameters.halo=s.halo;for(int i=0;i<5;++i) {plan.parameters.tr[i]=tr[i];plan.parameters.v[i]=v[i];}
    plan.strip_capacity=22*(s.halo+1)*plan.area;
    int status=astr::allocate_field(s.fields[ASTR_DEVICE_FILTER_WORK],s.upper,s.halo,scalar?3:4,scalar?1:11,1,sizeof(double));if(status)return status;
    status=static_cast<int>(cudaMalloc(&plan.strips,4*plan.strip_capacity*sizeof(double)));if(status)return status;
    status=static_cast<int>(cudaMalloc(&plan.faces,22*plan.area*sizeof(double)));if(status)return status;
    return static_cast<int>(cudaMalloc(&plan.aliases,44*plan.area*sizeof(double)));
}
extern "C" int astr_device_air5_filter_buffers(void *handle,void *buffers[8],size_t *capacity,size_t *area)
{
    if(!handle || !buffers || !capacity || !area)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);if(!s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;auto &plan=*s.air5_filter;
    for(int i=0;i<4;++i)buffers[i]=plan.strips+i*plan.strip_capacity;
    buffers[4]=plan.faces;buffers[5]=plan.faces+11*plan.area;buffers[6]=plan.aliases;buffers[7]=plan.aliases+22*plan.area;
    *capacity=plan.strip_capacity;*area=plan.area;return 0;
}
extern "C" int astr_device_air5_filter_pack(void *handle,int axis)
{
    if(!handle || axis<1 || axis>3)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;auto &plan=*s.air5_filter;int n1,n2;dim3 block,grid;launch_shape(s,axis-1,block,grid,n1,n2);
    const size_t count=2*static_cast<size_t>(s.halo+1)*(n1+1)*(n2+1);
    pack_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,plan.strips,plan.strip_capacity,plan.parameters,axis-1,n1,n2,count);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_filter_faces(void *handle,int axis,int upper)
{
    if(!handle || axis<1 || axis>3 || upper<0 || upper>1)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;auto &plan=*s.air5_filter;int n1,n2;dim3 block,grid;launch_shape(s,axis-1,block,grid,n1,n2);
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    faces_kernel<<<grid,block,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,s.fields[ASTR_DEVICE_RHS].view,plan.strips,plan.strip_capacity,plan.faces,plan.parameters,axis-1,n1,upper!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_filter_commit(void *handle,int axis,int lower,int upper)
{
    if(!handle || axis<1 || axis>3 || lower<0 || lower>1 || upper<0 || upper>1)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;auto &plan=*s.air5_filter;int n1,n2;dim3 block,grid;launch_shape(s,axis-1,block,grid,n1,n2);
    commit_kernel<<<grid,block,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_FILTER_WORK].view,plan.faces+11*plan.area,plan.parameters,axis-1,n1,lower!=0,upper!=0,plan.scalar);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_filter_alias(void *handle,int axis,int unpack)
{
    if(!handle || axis<1 || axis>3 || unpack<0 || unpack>1)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;auto &plan=*s.air5_filter;int n1,n2;dim3 block,grid;launch_shape(s,axis-1,block,grid,n1,n2);
    const size_t area=static_cast<size_t>(n1+1)*(n2+1);
    alias_kernel<<<(area+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,plan.aliases,plan.area,plan.parameters,axis-1,n1,area,unpack!=0);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_filter_check(void *handle,int preserve)
{
    if(!handle || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_filter)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    check_kernel<<<(count+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_AIR5_CARRY].view,s.air5_filter->parameters,count,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
