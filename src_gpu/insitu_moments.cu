#include <algorithm>
#include <cmath>
#include <new>
#include "insitu_moments.h"
#include "device_state_internal.cuh"
using Real=astr::numerical_real;

namespace {
using astr::at;
struct Moments {
    astr::State *parent=nullptr;
    int owned[3]{};
    int groups=1;
    size_t nodes=0;
    Real *state=nullptr;
    Real *weights=nullptr,*spatial=nullptr;
    int *invalid=nullptr;
    Real window[2]{},time=0,duration=0;
    Real volume=0;
    bool previous=false,borrowed=false;
    int64_t peak=0,scalar=0,observation=0,upload=0,samples=0;
};
int64_t bytes(const Moments &p)
{
    return 30*p.groups*p.nodes*sizeof(Real)+sizeof(int)+(p.weights?p.nodes*sizeof(Real):0)+
        (p.spatial?(p.groups==4?7:6)*sizeof(Real):0);
}
int release(Moments *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    if(p->state) {const int freed=static_cast<int>(cudaFree(p->state));if(status==0) status=freed;}
    if(p->weights) {const int freed=static_cast<int>(cudaFree(p->weights));if(status==0) status=freed;}
    if(p->spatial) {const int freed=static_cast<int>(cudaFree(p->spatial));if(status==0) status=freed;}
    if(p->invalid) {const int freed=static_cast<int>(cudaFree(p->invalid));if(status==0) status=freed;}
    if(p->borrowed) --p->parent->moment_plans;
    delete p;
    return status;
}
__device__ void result(const Real s[30],Real r[41])
{
    for(int n=0;n<41;++n) r[n]=0;
    if(s[4]>0) {
        r[0]=s[4];r[1]=s[5]/s[4];
        for(int a=0;a<3;++a) {r[a+2]=s[a+6];r[a+5]=s[a+9];}
        for(int a=0;a<9;++a) {
            r[a+14]=s[a+12]/s[4];r[a+23]=s[a+21]/s[5];r[a+32]=s[a+21]/s[4];
        }
        for(int a=0;a<3;++a) {r[a+8]=sqrt(r[14+4*a]);r[a+11]=sqrt(r[23+4*a]);}
    }
}
struct Air5Samples {AstrDeviceView rho,vel,t,tv,species;};
template<bool air5>
__global__ __launch_bounds__(512) void push(AstrDeviceView q,Air5Samples samples,Real *state,int nx,int ny,size_t nodes,
    Real w0,Real w1,bool previous,int *invalid)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    const int i=n%nx,j=(n/nx)%ny,k=n/(static_cast<size_t>(nx)*ny);
    Real current_rho=1,current[3];
    if(air5) {
        const int g=blockIdx.y;
        state+=30*g*nodes;
        if(g==0) {
            current_rho=at(samples.rho,i,j,k);
            for(int a=0;a<3;++a)current[a]=at(samples.vel,i,j,k,a+1);
        } else if(g==1) {
            current[0]=at(samples.t,i,j,k);current[1]=at(samples.tv,i,j,k);current[2]=at(samples.species,i,j,k);
        } else if(g==2) {
            for(int a=0;a<3;++a)current[a]=at(samples.species,i,j,k,a+2);
        } else {
            current[0]=at(samples.species,i,j,k,5);current[1]=0;current[2]=0;
        }
    } else {
        current_rho=at(q,i,j,k);
    }
    if(!(isfinite(current_rho) && current_rho>0)) {atomicExch(invalid,1);return;}
    if(!air5)for(int a=0;a<3;++a)current[a]=at(q,i,j,k,a+2)/current_rho;
    Real s[30]{};
    if(previous) for(int a=0;a<30;++a) s[a]=state[n+nodes*a];
    for(int e=0;e<2;++e) {
        Real w=e==0?w0:w1;
        if(w==0) continue;
        const Real rho=e==0?s[0]:current_rho;
        Real u[3];
        for(int a=0;a<3;++a) u[a]=e==0?s[a+1]:current[a];
        for(int f=0;f<2;++f) {
            const int total_slot=4+f,mean_slot=6+3*f,central_slot=12+9*f;
            if(f==1) w=w*rho;
            const Real total=s[total_slot],next=total+w;
            if(!(isfinite(w) && w>0 && isfinite(next) && next>0)) {atomicExch(invalid,1);return;}
            if(total==0) for(int a=0;a<3;++a) s[mean_slot+a]=u[a];
            else {
                Real delta[3];
                for(int a=0;a<3;++a) delta[a]=u[a]-s[mean_slot+a];
                const Real ratio=w/next,factor=(total/next)*w;
                for(int a=0;a<3;++a) s[mean_slot+a]+=ratio*delta[a];
                for(int b=0;b<3;++b) for(int a=0;a<=b;++a) {
                    const Real entry=s[central_slot+3*b+a]+(factor*delta[a])*delta[b];
                    s[central_slot+3*b+a]=entry;s[central_slot+3*a+b]=entry;
                }
            }
            s[total_slot]=next;
        }
    }
    s[0]=current_rho;
    for(int a=0;a<3;++a) s[a+1]=current[a];
    for(int a=0;a<30;++a) if(!isfinite(s[a])) {atomicExch(invalid,1);return;}
    for(int a=0;a<3;++a) if(s[12+4*a]<0 || s[21+4*a]<0) {atomicExch(invalid,1);return;}
    if(!air5) {
        Real r[41];result(s,r);
        for(int a=0;a<41;++a) if(!isfinite(r[a])) {atomicExch(invalid,1);return;}
    }
    for(int a=0;a<30;++a) state[n+nodes*a]=s[a];
}
__global__ __launch_bounds__(512) void output(const Real *state,Real *values,size_t nodes)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    Real s[30],r[41];
    for(int a=0;a<30;++a) s[a]=state[n+nodes*a];
    result(s,r);
    for(int a=0;a<41;++a) values[n+nodes*a]=r[a];
}
template<bool include_measure>
__global__ void spatial(const Real *state,const Real *weights,Real *values,size_t nodes)
{
    __shared__ Real scratch[256];
    const int lane=threadIdx.x,component=static_cast<int>(blockIdx.x)-(include_measure?1:0);
    Real value=0;
    for(size_t n=lane;n<nodes;n+=256) {
        Real entry=component<0?Real(1):state[n+nodes*(component+1)];
        if(component>=3) entry=state[n+4*nodes]>0?state[n+nodes*(12+4*(component-3))]/state[n+4*nodes]:Real(0.);
        value+=entry*weights[n];
    }
    scratch[lane]=value;__syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(lane<stride) scratch[lane]+=scratch[lane+stride];
        __syncthreads();
    }
    if(lane==0) values[blockIdx.x]=scratch[0];
}
int create(void *parent,const int owned[3],const Real window[2],int groups,void **moments)
{
    if(!parent || !owned || !window || !moments || !std::isfinite(window[0]) ||
        !std::isfinite(window[1]) || window[1]<=window[0]) return ASTR_DEVICE_INVALID_ARGUMENT;
    *moments=nullptr;
    auto &s=*static_cast<astr::State *>(parent);
    if(groups==4 && (!s.air5_ready || sizeof(Real)!=sizeof(double)))return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) if(owned[a]<1 || owned[a]>s.upper[a]+1) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *p=new(std::nothrow) Moments;
    if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->parent=&s;p->nodes=1;p->groups=groups;
    for(int a=0;a<3;++a) {p->owned[a]=owned[a];p->nodes*=owned[a];}
    p->window[0]=window[0];p->window[1]=window[1];
    int status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->state),30*groups*p->nodes*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->invalid),sizeof(int)));
    if(status!=0) {release(p);return status;}
    p->borrowed=true;++s.moment_plans;p->peak=bytes(*p);*moments=p;
    return 0;
}
}
extern "C" int astr_insitu_moments_create(void *parent,const int owned[3],const Real window[2],void **moments)
{return create(parent,owned,window,1,moments);}
extern "C" int astr_insitu_moments_create_air5(void *parent,const int owned[3],const Real window[2],void **moments)
{return create(parent,owned,window,4,moments);}

extern "C" int astr_insitu_moments_push(void *moments,Real time,const Real weights[2])
{
    if(!moments || !weights) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);auto &s=*p.parent;
    if(!std::isfinite(time) || (p.previous && time<=p.time) || !std::isfinite(weights[0]) ||
        !std::isfinite(weights[1]) || weights[0]<0 || weights[1]<0 ||
        (!p.previous && (weights[0]!=0 || weights[1]!=0))) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    if(status!=0) return status;
    if(p.groups==4) {
        const Air5Samples sample{s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,
            s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_AIR5_TV].view,s.fields[ASTR_DEVICE_AIR5_SPECIES].view};
        push<true><<<dim3((p.nodes+511)/512,4),512,0,s.stream>>>(AstrDeviceView{},sample,p.state,p.owned[0],p.owned[1],
            p.nodes,weights[0],weights[1],p.previous,p.invalid);
    } else {
        push<false><<<(p.nodes+511)/512,512,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,Air5Samples{},p.state,
            p.owned[0],p.owned[1],p.nodes,weights[0],weights[1],p.previous,p.invalid);
    }
    status=astr::finish_launch(s);
    int bad=0;
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status!=0) return status;
    p.scalar+=sizeof(int);
    if(bad) return ASTR_DEVICE_NONPHYSICAL;
    p.time=time;p.previous=true;p.duration+=weights[0];p.duration+=weights[1];++p.samples;
    return 0;
}
extern "C" int astr_insitu_moments_view(void *moments,AstrDeviceView *view)
{
    if(!moments || !view) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);
    if(!p.previous || p.groups!=1) return ASTR_DEVICE_NOT_ALLOCATED;
    *view=AstrDeviceView{};view->data=p.state;view->rank=2;view->element_bytes=sizeof(Real);
    int64_t stride=1;
    for(int a=0;a<5;++a) {
        view->lower[a]=1;view->extent[a]=a==0?static_cast<int>(p.nodes):(a==1?30:1);
        view->stride[a]=stride;stride*=view->extent[a];
    }
    return 0;
}
extern "C" int astr_insitu_moments_pack(void *moments,Real *host,size_t count)
{
    if(!moments || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);
    const auto N=p.nodes;
    if(!p.previous || count!=34*p.groups*N*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=0;
    for(int g=0;g<p.groups && status==0;++g) {
        Real *group=host+34*g*N;
        std::fill(group,group+N,p.window[0]);std::fill(group+N,group+2*N,p.window[1]);
        std::fill(group+2*N,group+3*N,p.time);std::fill(group+33*N,group+34*N,Real(1.));
        status=static_cast<int>(cudaMemcpyAsync(group+3*N,p.state+30*g*N,30*N*sizeof(Real),
            cudaMemcpyDeviceToHost,p.parent->stream));
    }
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) p.observation+=30*p.groups*N*sizeof(Real);
    return status;
}
extern "C" int astr_insitu_moments_restore(void *moments,const Real *host,size_t count,Real time)
{
    if(!moments || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);
    if(count!=34*p.groups*p.nodes*sizeof(Real) || !std::isfinite(time)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto N=p.nodes;const Real previous=host[2*N];
    if(!std::isfinite(previous) || previous>time) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int g=0;g<p.groups;++g) for(size_t n=0;n<N;++n) {
        const Real *group=host+34*g*N+n;
        for(int a=0;a<34;++a) if(!std::isfinite(group[N*a])) return ASTR_DEVICE_NONPHYSICAL;
        if(group[0]!=p.window[0] || group[N]!=p.window[1] || group[2*N]!=previous || group[33*N]!=Real(1.))
            return ASTR_DEVICE_INVALID_ARGUMENT;
        const Real *s=group+3*N;
        if(s[0]<=0 || s[4*N]<0 || s[5*N]<0 || ((s[4*N]==0)!=(s[5*N]==0))) return ASTR_DEVICE_NONPHYSICAL;
        for(int base:{12,21}) for(int b=0;b<3;++b) {
            if(s[N*(base+4*b)]<0) return ASTR_DEVICE_NONPHYSICAL;
            for(int a=0;a<b;++a) if(s[N*(base+3*b+a)]!=s[N*(base+3*a+b)]) return ASTR_DEVICE_NONPHYSICAL;
        }
        if(s[4*N]==0) for(int a=6;a<30;++a) if(s[N*a]!=0) return ASTR_DEVICE_NONPHYSICAL;
    }
    int status=0;
    for(int g=0;g<p.groups && status==0;++g)
        status=static_cast<int>(cudaMemcpyAsync(p.state+30*g*N,host+(34*g+3)*N,30*N*sizeof(Real),
            cudaMemcpyHostToDevice,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) {p.time=previous;p.duration=host[7*N];p.previous=true;p.upload+=30*p.groups*N*sizeof(Real);}
    return status;
}
extern "C" int astr_insitu_moments_output(void *moments,Real *host,size_t count)
{
    if(!moments || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);auto &s=*p.parent;
    if(!p.previous || p.groups!=1 || count!=41*p.nodes*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    Real *values=nullptr;
    int status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&values),count));
    if(status!=0) return status;
    p.peak=std::max(p.peak,bytes(p)+static_cast<int64_t>(count));
    output<<<(p.nodes+511)/512,512,0,s.stream>>>(p.state,values,p.nodes);
    status=astr::finish_launch(s);
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(host,values,count,cudaMemcpyDeviceToHost,s.stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(s.stream));
    const int freed=static_cast<int>(cudaFree(values));
    if(status==0) {p.observation+=count;status=freed;}
    return status;
}
extern "C" int astr_insitu_moments_info(void *moments,Real values[5])
{
    if(!moments || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Moments *>(moments);
    values[0]=p.time;values[1]=p.window[0];values[2]=p.window[1];values[3]=p.duration;values[4]=p.previous?1:0;
    return 0;
}
extern "C" int astr_insitu_moments_weights(void *moments,const Real *weights,size_t count)
{
    if(!moments || !weights) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);
    if(count!=p.nodes*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    Real volume=0;
    for(size_t n=0;n<p.nodes;++n) {
        if(!std::isfinite(weights[n]) || weights[n]<0) return ASTR_DEVICE_INVALID_ARGUMENT;
        volume+=weights[n];
    }
    if(!std::isfinite(volume) || volume<=0) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=0;
    if(!p.weights) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p.weights),count));
    if(status==0 && !p.spatial) status=static_cast<int>(
        cudaMalloc(reinterpret_cast<void **>(&p.spatial),(p.groups==4?7:6)*sizeof(Real)));
    if(status!=0) return status;
    p.peak=std::max(p.peak,bytes(p));
    status=static_cast<int>(cudaMemcpyAsync(p.weights,weights,count,cudaMemcpyHostToDevice,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) {p.volume=volume;p.upload+=count;}
    return status;
}
extern "C" int astr_insitu_moments_spatial(void *moments,Real local[7])
{
    if(!moments || !local) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Moments *>(moments);auto &s=*p.parent;
    if(!p.previous || !p.weights || !p.spatial) return ASTR_DEVICE_NOT_ALLOCATED;
    const bool volume_on_device=p.groups==4;
    if(volume_on_device)spatial<true><<<7,256,0,s.stream>>>(p.state,p.weights,p.spatial,p.nodes);
    else spatial<false><<<6,256,0,s.stream>>>(p.state,p.weights,p.spatial,p.nodes);
    int status=astr::finish_launch(s);
    local[0]=p.volume;
    const size_t transfer_bytes=(volume_on_device?7:6)*sizeof(Real);
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(local+(volume_on_device?0:1),p.spatial,
        transfer_bytes,cudaMemcpyDeviceToHost,s.stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) {
        p.scalar+=transfer_bytes;
        for(int a=0;a<7;++a) if(!std::isfinite(local[a])) return ASTR_DEVICE_NONPHYSICAL;
    }
    return status;
}
extern "C" int astr_insitu_moments_metrics(void *moments,int64_t values[6])
{
    if(!moments || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Moments *>(moments);
    values[0]=bytes(p);values[1]=p.peak;values[2]=p.scalar;values[3]=p.observation;values[4]=p.upload;values[5]=p.samples;
    return 0;
}
extern "C" int astr_insitu_moments_destroy(void **moments)
{
    if(!moments || !*moments) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=release(static_cast<Moments *>(*moments));*moments=nullptr;
    return status;
}

namespace {
struct Selected {
    astr::State *parent=nullptr;
    Real *state=nullptr,*profile_weights=nullptr,*profile=nullptr;
    int *indices=nullptr,*invalid=nullptr,*positions=nullptr;
    int nodes=0,capacity=0,components=0,ny=0,nk=0,terms=0;
    bool borrowed=false;
    int64_t peak=0,scalar=0,download=0,upload=0,index_upload=0,updates=0,profile_download=0,profile_upload=0;
};
struct SelectedFields {int nf,ids[5],slots[3],components;};
int64_t selected_bytes(const Selected &p)
{
    int64_t bytes=static_cast<int64_t>(p.capacity)*(3*sizeof(int)+p.components*sizeof(Real))+sizeof(int);
    if(p.positions) bytes+=static_cast<int64_t>(p.capacity)*sizeof(int);
    if(p.profile_weights) bytes+=static_cast<int64_t>(p.nk)*sizeof(Real);
    if(p.profile) bytes+=static_cast<int64_t>(p.ny)*p.terms*sizeof(Real);
    return bytes;
}
int release_selected(Selected *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    void *buffers[]={p->state,p->indices,p->invalid,p->positions,p->profile_weights,p->profile};
    for(void *b:buffers) if(b) {
        const int freed=static_cast<int>(cudaFree(b));if(status==0) status=freed;
    }
    if(p->borrowed) --p->parent->moment_plans;
    delete p;return status;
}
bool selected_fields(int nf,const int *ids,bool covariance,int components,SelectedFields &f)
{
    if(nf<0 || nf>5 || components!=3+3*nf+(covariance?12:0) || (covariance && nf<3) || (nf>0 && !ids)) return false;
    f=SelectedFields{};f.nf=nf;f.components=components;
    for(int c=0;c<nf;++c) {
        if(ids[c]<2 || ids[c]>6) return false;
        f.ids[c]=ids[c];
        if(covariance && ids[c]<=4) f.slots[ids[c]-2]=c+1;
    }
    if(covariance) for(int c=0;c<3;++c) if(f.slots[c]==0) return false;
    return true;
}
__device__ bool selected_update(Real *state,const Real *values,Real rho,const SelectedFields &f,
    Real w0,Real w1,bool previous)
{
    if(!(isfinite(rho) && rho>0) || !isfinite(w0) || !isfinite(w1) || w0<0 || w1<0) return false;
    for(int c=0;c<f.nf;++c) if(!isfinite(values[c])) return false;
    Real next[30]{};
    if(previous) {
        for(int c=0;c<f.components;++c) {if(!isfinite(state[c])) return false;next[c]=state[c];}
        if(next[0]<=0 || next[1]<0 || next[2]<0 || ((next[1]==0)!=(next[2]==0))) return false;
    } else if(w0!=0 || w1!=0) return false;
    const bool covariance=f.slots[0]!=0 || f.slots[1]!=0 || f.slots[2]!=0;
    for(int e=0;e<2;++e) {
        const Real endpoint_weight=e==0?w0:w1;
        if(endpoint_weight==0) continue;
        const Real density=e==0?next[0]:rho;
        Real sample[5];
        for(int c=0;c<f.nf;++c) sample[c]=e==0?next[3+c]:values[c];
        for(int kind=0;kind<2;++kind) {
            const Real weight=kind==0?endpoint_weight:endpoint_weight*density;
            const Real total=next[1+kind],new_total=total+weight;
            if(!(isfinite(weight) && weight>0 && isfinite(new_total) && new_total>0)) return false;
            const int mean=3+(kind+1)*f.nf;
            Real delta[5];for(int c=0;c<f.nf;++c) delta[c]=sample[c]-next[mean+c];
            if(total==0) for(int c=0;c<f.nf;++c) next[mean+c]=sample[c];
            else {
                const Real ratio=weight/new_total,factor=(total/new_total)*weight;
                for(int c=0;c<f.nf;++c) next[mean+c]+=ratio*delta[c];
                if(covariance) {
                    int entry=3+3*f.nf+6*kind;
                    for(int b=0;b<3;++b) for(int a=0;a<=b;++a)
                        next[entry++]+=(factor*delta[f.slots[a]-1])*delta[f.slots[b]-1];
                }
            }
            next[1+kind]=new_total;
        }
    }
    next[0]=rho;for(int c=0;c<f.nf;++c) next[3+c]=values[c];
    for(int c=0;c<f.components;++c) if(!isfinite(next[c])) return false;
    for(int c=0;c<f.components;++c) state[c]=next[c];
    return true;
}
__device__ bool selected_primitives(AstrDeviceView q,int i,int j,int k,const SelectedFields &f,
    Real const2,Real const6,Real *values,Real &rho)
{
    rho=at(q,i,j,k);
    if(!(isfinite(rho) && rho>0)) return false;
    bool thermal=false;for(int c=0;c<f.nf;++c) thermal=thermal || f.ids[c]>=5;
    Real pressure=0;
    if(thermal) {
        if(!(isfinite(const6) && const6>0)) return false;
        Real velocity[3];
        for(int c=0;c<3;++c) {
            const Real momentum=at(q,i,j,k,c+2);if(!isfinite(momentum)) return false;
            velocity[c]=momentum/rho;
        }
        const Real energy=at(q,i,j,k,5);if(!isfinite(energy)) return false;
        pressure=(energy-Real(.5)*rho*(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2]))/const6;
    }
    for(int c=0;c<f.nf;++c) {
        const int id=f.ids[c];
        if(id<=4) values[c]=at(q,i,j,k,id)/rho;
        else if(id==5) values[c]=pressure;
        else {
            if(!(isfinite(const2) && const2>0)) return false;
            values[c]=(pressure/rho)*const2;
        }
        if(!isfinite(values[c])) return false;
    }
    return true;
}
__global__ __launch_bounds__(256) void selected_push(AstrDeviceView q,const int *indices,Real *state,
    int nodes,int capacity,SelectedFields f,Real const2,Real const6,Real w0,Real w1,bool previous,
    int im,int jm,int km,int *invalid)
{
    const int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nodes) return;
    const int i=indices[3*n],j=indices[3*n+1],k=indices[3*n+2];
    if(i<0 || i>im || j<0 || j>jm || k<0 || k>km) {atomicExch(invalid,1);return;}
    Real values[5],rho,local[30];
    if(!selected_primitives(q,i,j,k,f,const2,const6,values,rho)) {atomicExch(invalid,1);return;}
    for(int c=0;c<f.components;++c) local[c]=state[n+static_cast<size_t>(capacity)*c];
    if(!selected_update(local,values,rho,f,w0,w1,previous)) {atomicExch(invalid,1);return;}
    for(int c=0;c<f.components;++c) state[n+static_cast<size_t>(capacity)*c]=local[c];
}
__global__ __launch_bounds__(256) void selected_scatter(AstrDeviceView input,const int *positions,Real *state,
    int nodes,int capacity,SelectedFields f,Real w0,Real w1,bool previous,int *invalid)
{
    const int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=nodes) return;
    const int target=positions[n]-1;
    if(target<0 || target>=capacity) {atomicExch(invalid,1);return;}
    const auto *source=static_cast<const Real *>(input.data);
    const Real rho=source[static_cast<size_t>(n)*input.stride[0]];
    Real values[5],local[30];
    for(int c=0;c<f.nf;++c) values[c]=source[static_cast<size_t>(n)*input.stride[0]+(c+1)*input.stride[1]];
    for(int c=0;c<f.components;++c) local[c]=state[target+static_cast<size_t>(capacity)*c];
    if(!selected_update(local,values,rho,f,w0,w1,previous)) {atomicExch(invalid,1);return;}
    for(int c=0;c<f.components;++c) state[target+static_cast<size_t>(capacity)*c]=local[c];
}
__device__ bool selected_profile_terms(const Real *s,Real weight,int nf,bool covariance,Real *terms)
{
    for(int c=0;c<24;++c) terms[c]=0;
    if(!(isfinite(weight) && weight>=0 && isfinite(s[1]) && s[1]>0 && isfinite(s[2]) && s[2]>0)) return false;
    const Real weighted_density=weight*(s[2]/s[1]);
    terms[0]=weight;terms[1]=weighted_density;
    for(int c=0;c<nf;++c) {terms[2+c]=weight*s[3+nf+c];terms[2+nf+c]=weighted_density*s[3+2*nf+c];}
    if(covariance) for(int c=0;c<6;++c) {
        terms[2+2*nf+c]=weight*(s[3+3*nf+c]/s[1]);
        terms[2+2*nf+6+c]=weight*(s[3+3*nf+6+c]/s[1]);
    }
    const int count=2+2*nf+(covariance?12:0);
    for(int c=0;c<count;++c) if(!isfinite(terms[c])) return false;
    return true;
}
__global__ __launch_bounds__(128) void selected_profile(const Real *state,const Real *weights,Real *profile,
    int ny,int nk,int components,int terms,int nf,bool covariance,int *invalid)
{
    __shared__ Real partial[24][128];
    const int lane=threadIdx.x,j=blockIdx.x,capacity=ny*nk;
    Real total[24]{};
    for(int k=lane;k<nk;k+=128) {
        const int n=k*ny+j;Real s[30],values[24];
        for(int c=0;c<components;++c) s[c]=state[n+static_cast<size_t>(capacity)*c];
        if(!selected_profile_terms(s,weights[k],nf,covariance,values)) atomicExch(invalid,1);
        for(int c=0;c<terms;++c) total[c]+=values[c];
    }
    for(int c=0;c<terms;++c) partial[c][lane]=total[c];
    __syncthreads();
    for(int stride=64;stride>0;stride/=2) {
        if(lane<stride) for(int c=0;c<terms;++c) partial[c][lane]+=partial[c][lane+stride];
        __syncthreads();
    }
    if(lane==0) for(int c=0;c<terms;++c) {
        const Real value=partial[c][0];profile[c+terms*j]=value;
        if(!isfinite(value)) atomicExch(invalid,1);
    }
}
int selected_status(Selected &p,bool profile)
{
    int status=astr::finish_launch(*p.parent),bad=0;
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status!=0) return status;
    if(profile) p.profile_download+=sizeof(int);else p.scalar+=sizeof(int);
    return bad?ASTR_DEVICE_NONPHYSICAL:0;
}
}

extern "C" int astr_insitu_selected_create(void *parent,int nodes,const int *indices,int components,void **selected)
{
    if(!parent || !selected || nodes<0 || components<1 || components>30 || (nodes>0 && !indices)) return ASTR_DEVICE_INVALID_ARGUMENT;
    *selected=nullptr;auto *p=new(std::nothrow) Selected;
    if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->parent=static_cast<astr::State *>(parent);p->nodes=nodes;p->capacity=std::max(1,nodes);p->components=components;
    const size_t state_bytes=static_cast<size_t>(p->capacity)*components*sizeof(Real);
    int status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->state),state_bytes));
    if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->indices),static_cast<size_t>(p->capacity)*3*sizeof(int)));
    if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->invalid),sizeof(int)));
    if(status==0) status=static_cast<int>(cudaMemsetAsync(p->state,0,state_bytes,p->parent->stream));
    if(status==0 && nodes>0) status=static_cast<int>(cudaMemcpyAsync(p->indices,indices,static_cast<size_t>(nodes)*3*sizeof(int),
        cudaMemcpyHostToDevice,p->parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    if(status!=0) {release_selected(p);return status;}
    p->index_upload=static_cast<int64_t>(nodes)*3*sizeof(int);p->borrowed=true;++p->parent->moment_plans;
    p->peak=selected_bytes(*p);*selected=p;return 0;
}
extern "C" int astr_insitu_selected_push(void *selected,int fields,const int *ids,Real const2,Real const6,
    const Real weights[2],int previous,int covariance)
{
    if(!selected || !weights) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Selected *>(selected);auto &s=*p.parent;SelectedFields f{};
    if(!selected_fields(fields,ids,covariance!=0,p.components,f)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(p.nodes==0) return 0;
    int status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));if(status!=0) return status;
    selected_push<<<(p.nodes+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,p.indices,p.state,p.nodes,p.capacity,
        f,const2,const6,weights[0],weights[1],previous!=0,s.upper[0],s.upper[1],s.upper[2],p.invalid);
    status=selected_status(p,false);if(status==0) ++p.updates;return status;
}
extern "C" int astr_insitu_selected_scatter(void *selected,int nodes,const int *positions,const AstrDeviceView *values,
    int fields,const Real weights[2],int previous)
{
    if(!selected || !weights) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Selected *>(selected);auto &s=*p.parent;SelectedFields f{};
    if(fields<0 || fields>5 || p.components!=3+3*fields || nodes<0 || nodes>p.nodes) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(nodes==0) return 0;
    if(!positions || !values || !values->data || values->rank!=2 || values->element_bytes!=sizeof(Real) ||
        values->extent[0]!=nodes || values->extent[1]!=fields+1) return ASTR_DEVICE_INVALID_ARGUMENT;
    f.nf=fields;f.components=p.components;
    int status=0;
    if(!p.positions) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p.positions),static_cast<size_t>(p.capacity)*sizeof(int)));
    p.peak=std::max(p.peak,selected_bytes(p));if(status!=0) return status;
    status=static_cast<int>(cudaMemcpyAsync(p.positions,positions,static_cast<size_t>(nodes)*sizeof(int),cudaMemcpyHostToDevice,s.stream));
    if(status!=0) return status;
    p.index_upload+=static_cast<int64_t>(nodes)*sizeof(int);
    status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));if(status!=0) return status;
    selected_scatter<<<(nodes+255)/256,256,0,s.stream>>>(*values,p.positions,p.state,nodes,p.capacity,f,
        weights[0],weights[1],previous!=0,p.invalid);
    status=selected_status(p,false);if(status==0) ++p.updates;return status;
}
extern "C" int astr_insitu_selected_view(void *selected,int field,AstrDeviceView *view)
{
    if(!selected || !view || field<0 || field>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Selected *>(selected);
    *view=AstrDeviceView{};view->data=field==0?static_cast<void *>(p.state):static_cast<void *>(p.indices);
    view->rank=2;view->element_bytes=field==0?sizeof(Real):sizeof(int);
    for(int a=0;a<5;++a) {view->lower[a]=1;view->extent[a]=1;view->stride[a]=1;}
    view->extent[0]=p.nodes;view->extent[1]=field==0?p.components:3;
    view->stride[0]=field==0?1:3;view->stride[1]=field==0?p.capacity:1;
    for(int a=2;a<5;++a) view->stride[a]=static_cast<int64_t>(p.capacity)*view->extent[1];
    return 0;
}
extern "C" int astr_insitu_selected_download(void *selected,Real *host,size_t count)
{
    if(!selected) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Selected *>(selected);
    if(count!=static_cast<size_t>(p.nodes)*p.components*sizeof(Real) || (count>0 && !host)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(count==0) return 0;
    int status=static_cast<int>(cudaMemcpyAsync(host,p.state,count,cudaMemcpyDeviceToHost,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) p.download+=count;return status;
}
extern "C" int astr_insitu_selected_upload(void *selected,const Real *host,size_t count)
{
    if(!selected) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Selected *>(selected);
    if(count!=static_cast<size_t>(p.nodes)*p.components*sizeof(Real) || (count>0 && !host)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(count==0) return 0;
    int status=static_cast<int>(cudaMemcpyAsync(p.state,host,count,cudaMemcpyHostToDevice,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) p.upload+=count;return status;
}
extern "C" int astr_insitu_selected_profile(void *selected,int ny,int nk,const Real *weights,int fields,int covariance,
    Real *profile,size_t count)
{
    if(!selected || fields<0 || fields>5 || (covariance!=0 && fields<3)) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Selected *>(selected);auto &s=*p.parent;
    const int terms=2+2*fields+(covariance?12:0);
    if(p.components!=3+3*fields+(covariance?12:0) || ny<0 || nk<0 || static_cast<int64_t>(ny)*nk!=p.nodes ||
        count!=static_cast<size_t>(ny)*terms*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(ny==0) return 0;
    if(nk==0 || !weights || !profile) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=0;
    if(!p.profile_weights) {
        p.ny=ny;p.nk=nk;p.terms=terms;
        status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p.profile_weights),static_cast<size_t>(nk)*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p.profile),count));
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(p.profile_weights,weights,static_cast<size_t>(nk)*sizeof(Real),
            cudaMemcpyHostToDevice,s.stream));
        if(status==0) p.profile_upload+=static_cast<int64_t>(nk)*sizeof(Real);
        p.peak=std::max(p.peak,selected_bytes(p));
        if(status!=0) {
            if(p.profile) cudaFree(p.profile);
            if(p.profile_weights) cudaFree(p.profile_weights);
            p.profile=nullptr;p.profile_weights=nullptr;p.ny=0;p.nk=0;p.terms=0;
            return status;
        }
    }
    if(p.ny!=ny || p.nk!=nk || p.terms!=terms) return ASTR_DEVICE_INVALID_ARGUMENT;
    status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));if(status!=0) return status;
    selected_profile<<<ny,128,0,s.stream>>>(p.state,p.profile_weights,p.profile,ny,nk,p.components,terms,fields,covariance!=0,p.invalid);
    status=selected_status(p,true);if(status!=0) return status;
    status=static_cast<int>(cudaMemcpyAsync(profile,p.profile,count,cudaMemcpyDeviceToHost,s.stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) p.profile_download+=count;return status;
}
extern "C" int astr_insitu_selected_metrics(void *selected,int64_t values[9])
{
    if(!selected || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Selected *>(selected);
    values[0]=selected_bytes(p);values[1]=p.peak;values[2]=p.scalar;values[3]=p.download;values[4]=p.upload;
    values[5]=p.index_upload;values[6]=p.updates;values[7]=p.profile_download;values[8]=p.profile_upload;
    return 0;
}
extern "C" int astr_insitu_selected_destroy(void **selected)
{
    if(!selected || !*selected) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int status=release_selected(static_cast<Selected *>(*selected));*selected=nullptr;return status;
}
extern "C" int astr_insitu_selected_workspace(void *selected,int64_t bytes[2])
{
    if(!selected || !bytes) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Selected *>(selected);
    bytes[0]=(p.profile_weights?static_cast<int64_t>(p.nk)*sizeof(Real):0)+
        (p.profile?static_cast<int64_t>(p.ny)*p.terms*sizeof(Real):0);
    bytes[1]=p.positions?static_cast<int64_t>(p.capacity)*sizeof(int):0;
    return 0;
}
