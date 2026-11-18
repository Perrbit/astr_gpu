#include <cfloat>
#include <cmath>
#include <new>
#include <limits>
#include "crashfix_operator.h"
#include "device_state_internal.cuh"
using Real=astr::numerical_real;
constexpr Real largest=std::numeric_limits<Real>::max();

namespace astr {
struct CrashfixPlan {
    int8_t *fluid=nullptr;
    int *first=nullptr,*last=nullptr;
    int *expansion=nullptr;
    int *input_invalid=nullptr;
    AstrCrashfixReport *report=nullptr;
};
int release_crashfix(CrashfixPlan *plan)
{
    if(!plan) return 0;
    cudaError_t status=cudaSuccess;
    void *buffers[]={plan->fluid,plan->first,plan->last,plan->report,plan->expansion,plan->input_invalid};
    for(void *buffer:buffers) if(buffer) {
        const cudaError_t freed=cudaFree(buffer);
        if(status==cudaSuccess) status=freed;
    }
    delete plan;
    return static_cast<int>(status);
}
}

namespace {
using astr::at;
struct Fields {AstrDeviceView q,rho,vel,prs,tmp;};
struct Limits {int upper[3],origin[3],global[3];Real c2,c6,epsp;};
struct Bounds {int lower[3],upper[3];};

__device__ bool admissible(const Real q[5],Real u[3],Real squared[3])
{
    Real products[3];bool valid=true;
    for(int c=0;c<5;++c) valid=valid && isfinite(q[c]);
    valid=valid && q[0]>0;
    if(valid) {
        for(int c=0;c<3;++c) {
            u[c]=q[c+1]/q[0];products[c]=Real(.5)*q[c+1]*u[c];squared[c]=u[c]*u[c];
            valid=valid && isfinite(u[c]) && isfinite(products[c]) && isfinite(squared[c]);
        }
        const Real kinetic=products[0]+products[1]+products[2];
        const Real primitive_kinetic=Real(.5)*q[0]*(squared[0]+squared[1]+squared[2]);
        valid=valid && isfinite(kinetic) && isfinite(primitive_kinetic) && q[4]>kinetic && q[4]>primitive_kinetic;
    }
    return valid;
}

__device__ bool recovery_valid(const Real q[5],Real c2,Real c6)
{
    Real u[3],squared[3];
    if(!admissible(q,u,squared)) return false;
    const Real pressure=(q[4]-Real(.5)*q[0]*(squared[0]+squared[1]+squared[2]))/c6;
    const Real temperature=pressure/q[0]*c2;
    return isfinite(pressure) && pressure>0 && isfinite(temperature) && temperature>0;
}

__device__ bool refresh(Fields f,int i,int j,int k,Real c2,Real c6)
{
    Real q[5],u[3],squared[3];
    for(int c=0;c<5;++c) q[c]=at(f.q,i,j,k,c+1);
    if(admissible(q,u,squared)) {
        const Real pressure=(q[4]-Real(.5)*q[0]*(squared[0]+squared[1]+squared[2]))/c6;
        const Real temperature=pressure/q[0]*c2;
        at(f.rho,i,j,k)=q[0];for(int c=0;c<3;++c) at(f.vel,i,j,k,c+1)=u[c];
        at(f.prs,i,j,k)=pressure;at(f.tmp,i,j,k)=temperature;
        return isfinite(pressure) && pressure>0 && isfinite(temperature) && temperature>0;
    }
    at(f.rho,i,j,k)=q[0];for(int c=1;c<=3;++c) at(f.vel,i,j,k,c)=0;
    at(f.prs,i,j,k)=-1;at(f.tmp,i,j,k)=-1;
    return false;
}

__global__ void initialize_scan(astr::CrashfixPlan plan,int im,size_t rows,bool begin)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<rows) {plan.first[id]=im+1;plan.last[id]=-1;}
    if(id==0) {
        if(begin) {
            *plan.report=AstrCrashfixReport{};
            plan.expansion[2]=0;plan.expansion[3]=0;
        }
        plan.report->candidates=0;plan.report->invalid=0;
        plan.expansion[0]=0;plan.expansion[1]=0;
        for(int c=0;c<4;++c) plan.report->minima[c]=largest;
    }
}

__global__ void inspect(Fields f,astr::CrashfixPlan plan,Limits limits,bool collect,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int nx=limits.upper[0]+1,ny=limits.upper[1]+1;
    const int i=id%nx,j=(id/nx)%ny,k=id/(static_cast<size_t>(nx)*ny);
    const bool valid=refresh(f,i,j,k,limits.c2,limits.c6);
    if(!plan.fluid[id]) return;
    if(!valid) atomicExch(&plan.report->invalid,1);
    const bool bad=!(at(f.rho,i,j,k)>=Real(1.e-5) && at(f.prs,i,j,k)>=limits.epsp && at(f.tmp,i,j,k)>=Real(1.e-5));
    if(collect && bad) {
        atomicAdd(&plan.report->candidates,1);atomicMin(plan.first+j+ny*k,i);atomicMax(plan.last+j+ny*k,i);
    }
}

__global__ void ordered(Fields f,astr::CrashfixPlan plan,int8_t *critical,Limits limits)
{
    if(plan.report->candidates==0) return;
    const int im=limits.upper[0],jm=limits.upper[1],km=limits.upper[2];
    for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) {
        const int row=j+(jm+1)*k;
        for(int i=plan.first[row];i<=plan.last[row];++i) {
            if(!plan.fluid[i+(im+1)*row]) continue;
            if(at(f.rho,i,j,k)>=Real(1.e-5) && at(f.prs,i,j,k)>=limits.epsp && at(f.tmp,i,j,k)>=Real(1.e-5)) continue;
            const size_t node=(i-f.q.lower[0])*f.q.stride[0]+(j-f.q.lower[1])*f.q.stride[1]+(k-f.q.lower[2])*f.q.stride[2];
            critical[node]=1;
            Real average[5]={};int n=0;
            for(int kk=-1;kk<=1;++kk) for(int jj=-1;jj<=1;++jj) for(int ii=-1;ii<=1;++ii) {
                if((ii==0 && jj==0 && kk==0) || limits.origin[0]+i+ii<0 || limits.origin[0]+i+ii>limits.global[0] ||
                    limits.origin[1]+j+jj<0 || limits.origin[1]+j+jj>limits.global[1]) continue;
                if(at(f.rho,i+ii,j+jj,k+kk)>=0 && at(f.prs,i+ii,j+jj,k+kk)>=0 && at(f.tmp,i+ii,j+jj,k+kk)>=0) {
                    for(int c=0;c<5;++c) average[c]+=at(f.q,i+ii,j+jj,k+kk,c+1);
                    ++n;
                }
            }
            if(n==0) continue;
            Real previous[5];
            for(int c=0;c<5;++c) {previous[c]=at(f.q,i,j,k,c+1);at(f.q,i,j,k,c+1)=average[c]/static_cast<Real>(n);}
            if(refresh(f,i,j,k,limits.c2,limits.c6)) {
                ++plan.report->repaired;
                for(int c=0;c<5;++c) {
                    const Real baseline=isfinite(previous[c]) ? previous[c] : 0;
                    const Real difference=fabs(at(f.q,i,j,k,c+1)-baseline);
                    plan.report->max_change[c]=fmax(plan.report->max_change[c],isfinite(difference) ? difference : largest);
                }
            }
        }
    }
}

__device__ bool usable_donor(Fields f,astr::CrashfixPlan plan,Limits limits,int i,int j,int k)
{
    const int im=limits.upper[0],jm=limits.upper[1],km=limits.upper[2];
    if(i>=0 && i<=im && j>=0 && j<=jm && k>=0 && k<=km && !plan.fluid[i+(im+1)*(j+(jm+1)*k)]) return false;
    if((i<0 || i>im)+(j<0 || j>jm)+(k<0 || k>km)>1) return false;
    Real q[5];for(int c=0;c<5;++c) q[c]=at(f.q,i,j,k,c+1);
    return recovery_valid(q,limits.c2,limits.c6);
}

__global__ void find_donor(Fields f,astr::CrashfixPlan plan,Limits limits,Bounds bounds,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int nx=bounds.upper[0]-bounds.lower[0]+1,ny=bounds.upper[1]-bounds.lower[1]+1;
    const int i=bounds.lower[0]+id%nx,j=bounds.lower[1]+(id/nx)%ny,k=bounds.lower[2]+id/(static_cast<size_t>(nx)*ny);
    if(usable_donor(f,plan,limits,i,j,k)) atomicExch(plan.expansion,1);
}

__device__ void add_donor(Fields f,astr::CrashfixPlan plan,Limits limits,int i,int j,int k,
    Real average[5],Real first[5],int &n)
{
    if(!usable_donor(f,plan,limits,i,j,k)) return;
    const Real weight=Real(1.0)/static_cast<Real>(++n);
    for(int c=0;c<5;++c) {
        const Real q=at(f.q,i,j,k,c+1);
        if(n==1) first[c]=q;
        average[c]=(Real(1.0)-weight)*average[c]+weight*q;
    }
}

__global__ void expanded(Fields f,astr::CrashfixPlan plan,int8_t *critical,Limits limits,Bounds bounds)
{
    if(!plan.expansion[0]) return;
    const int im=limits.upper[0],jm=limits.upper[1],km=limits.upper[2];
    for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) {
        const int row=j+(jm+1)*k;
        for(int i=plan.first[row];i<=plan.last[row];++i) {
            if(!plan.fluid[i+(im+1)*row]) continue;
            Real previous[5];for(int c=0;c<5;++c) previous[c]=at(f.q,i,j,k,c+1);
            if(recovery_valid(previous,limits.c2,limits.c6)) continue;
            Real average[5]={},first[5]={};int n=0,used_radius=0;bool valid=false;
            const int radius_max=max(max(max(i-bounds.lower[0],bounds.upper[0]-i),max(j-bounds.lower[1],bounds.upper[1]-j)),
                max(k-bounds.lower[2],bounds.upper[2]-k));
            for(int radius=1;radius<=radius_max;++radius) {
                used_radius=radius;
                for(int kk=max(bounds.lower[2],k-radius);kk<=min(bounds.upper[2],k+radius);++kk)
                for(int jj=max(bounds.lower[1],j-radius);jj<=min(bounds.upper[1],j+radius);++jj) {
                    if(abs(kk-k)==radius || abs(jj-j)==radius) {
                        for(int ii=max(bounds.lower[0],i-radius);ii<=min(bounds.upper[0],i+radius);++ii)
                            add_donor(f,plan,limits,ii,jj,kk,average,first,n);
                    } else {
                        const int left=i-radius,right=i+radius;
                        if(left>=bounds.lower[0]) add_donor(f,plan,limits,left,jj,kk,average,first,n);
                        if(right<=bounds.upper[0]) add_donor(f,plan,limits,right,jj,kk,average,first,n);
                    }
                }
                if(n==0 || !recovery_valid(average,limits.c2,limits.c6)) continue;
                for(int c=0;c<5;++c) at(f.q,i,j,k,c+1)=average[c];
                valid=refresh(f,i,j,k,limits.c2,limits.c6);
                if(valid) break;
                for(int c=0;c<5;++c) at(f.q,i,j,k,c+1)=previous[c];
            }
            if(!valid && n>0) {
                // Preserve the original approved one-point mean, not a new recovery policy.
                for(int c=0;c<5;++c) at(f.q,i,j,k,c+1)=first[c];
                valid=refresh(f,i,j,k,limits.c2,limits.c6);
                if(valid) ++plan.expansion[3];
            }
            if(valid) {
                const size_t node=(i-f.q.lower[0])*f.q.stride[0]+(j-f.q.lower[1])*f.q.stride[1]+(k-f.q.lower[2])*f.q.stride[2];
                critical[node]=1;++plan.report->repaired;++plan.expansion[1];
                plan.expansion[2]=max(plan.expansion[2],used_radius);
                for(int c=0;c<5;++c) {
                    const Real baseline=isfinite(previous[c]) ? previous[c] : 0;
                    const Real difference=fabs(at(f.q,i,j,k,c+1)-baseline);
                    plan.report->max_change[c]=fmax(plan.report->max_change[c],isfinite(difference) ? difference : largest);
                }
            } else {
                for(int c=0;c<5;++c) at(f.q,i,j,k,c+1)=previous[c];
                refresh(f,i,j,k,limits.c2,limits.c6);
            }
        }
    }
}

__global__ void reset_final_scan(astr::CrashfixPlan plan)
{
    plan.report->invalid=0;
}

template<bool reconciled>
__global__ void refresh_recovery_cache(Fields f,Limits limits,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int nx=f.q.extent[0],ny=f.q.extent[1];
    const int i=id%nx+f.q.lower[0],j=(id/nx)%ny+f.q.lower[1],k=id/(static_cast<size_t>(nx)*ny)+f.q.lower[2];
    const int outside=(i<0 || i>limits.upper[0])+(j<0 || j>limits.upper[1])+(k<0 || k>limits.upper[2]);
    if(!reconciled) {
        if(outside==0) return;
        if(outside>1) {
            at(f.rho,i,j,k)=-1;at(f.prs,i,j,k)=-1;at(f.tmp,i,j,k)=-1;
            for(int c=1;c<=3;++c) at(f.vel,i,j,k,c)=0;
        } else refresh(f,i,j,k,limits.c2,limits.c6);
        return;
    }
    // Match the original final primitive-halo phase, including defined corner
    // values. Zero-density unset diagonal q keeps its inactive cache.
    const Real den=at(f.q,i,j,k,1);
    if(den==0 && outside>1) return;
    const Real u=at(f.q,i,j,k,2)/den,v=at(f.q,i,j,k,3)/den,w=at(f.q,i,j,k,4)/den;
    const Real pressure=(at(f.q,i,j,k,5)-Real(.5)*den*(u*u+v*v+w*w))/limits.c6;
    at(f.rho,i,j,k)=den;at(f.vel,i,j,k,1)=u;at(f.vel,i,j,k,2)=v;at(f.vel,i,j,k,3)=w;
    at(f.prs,i,j,k)=pressure;at(f.tmp,i,j,k)=pressure/den*limits.c2;
}

__global__ void check_input(AstrDeviceView q,const int im,const int jm,size_t count,int *invalid)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=id%(im+1),j=(id/(im+1))%(jm+1),k=id/(static_cast<size_t>(im+1)*(jm+1));
    Real state[5],u[3],squared[3];
    for(int c=0;c<5;++c) state[c]=at(q,i,j,k,c+1);
    if(!admissible(state,u,squared)) atomicExch(invalid,1);
}

template<bool reconciled> int refresh_cache(void *parent,Real c2,Real c6)
{
    if(!parent || !(c2>0) || !(c6>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.crashfix) return ASTR_DEVICE_NOT_ALLOCATED;
    Limits limits{};
    for(int a=0;a<3;++a) limits.upper[a]=s.upper[a];
    limits.c2=c2;limits.c6=c6;
    Fields fields{s.fields[0].view,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view};
    const size_t count=s.fields[0].view.stride[3];
    refresh_recovery_cache<reconciled><<<(count+255)/256,256,0,s.stream>>>(fields,limits,count);
    return astr::finish_launch(s);
}

__global__ void final_report(Fields f,astr::CrashfixPlan plan,Limits limits,size_t count)
{
    __shared__ Real minimum[4][256];
    const int thread=threadIdx.x;
    for(int c=0;c<4;++c) minimum[c][thread]=largest;
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id<count) {
        const int nx=limits.upper[0]+1,ny=limits.upper[1]+1;
        const int i=id%nx,j=(id/nx)%ny,k=id/(static_cast<size_t>(nx)*ny);
        const bool valid=refresh(f,i,j,k,limits.c2,limits.c6);
        if(plan.fluid[id]) {
            if(!valid) atomicExch(&plan.report->invalid,1);
            else {
                const Real den=at(f.rho,i,j,k),values[4]={den,at(f.prs,i,j,k),at(f.tmp,i,j,k),
                    at(f.q,i,j,k,5)-(Real(.5)*at(f.q,i,j,k,2)*(at(f.q,i,j,k,2)/den)+
                        Real(.5)*at(f.q,i,j,k,3)*(at(f.q,i,j,k,3)/den)+Real(.5)*at(f.q,i,j,k,4)*(at(f.q,i,j,k,4)/den))};
                for(int c=0;c<4;++c) {
                    if(!(isfinite(values[c]) && values[c]>0)) atomicExch(&plan.report->invalid,1);
                    else minimum[c][thread]=values[c];
                }
            }
        }
    }
    __syncthreads();
    for(int stride=128;stride>0;stride/=2) {
        if(thread<stride) for(int c=0;c<4;++c) minimum[c][thread]=fmin(minimum[c][thread],minimum[c][thread+stride]);
        __syncthreads();
    }
    if(thread==0) for(int c=0;c<4;++c) {
        // Positive finite values have the same ordering as their unsigned bits.
#ifdef ASTR_REAL32
        atomicMin(reinterpret_cast<unsigned int *>(plan.report->minima+c),__float_as_uint(minimum[c][0]));
#else
        atomicMin(reinterpret_cast<unsigned long long *>(plan.report->minima+c),
            static_cast<unsigned long long>(__double_as_longlong(minimum[c][0])));
#endif
    }
}

int history_transfer(void *parent,int8_t *mask,size_t bytes,bool upload)
{
    if(!parent || !mask) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.critical_mask) return ASTR_DEVICE_NOT_ALLOCATED;
    if(bytes!=static_cast<size_t>(s.fields[ASTR_DEVICE_Q].view.stride[3])) return ASTR_DEVICE_INVALID_ARGUMENT;
    const cudaError_t status=upload ? cudaMemcpyAsync(s.critical_mask,mask,bytes,cudaMemcpyHostToDevice,s.stream) :
        cudaMemcpyAsync(mask,s.critical_mask,bytes,cudaMemcpyDeviceToHost,s.stream);
    if(status!=cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}
}

extern "C" int astr_device_crashfix_prepare(void *parent,const int8_t *fluid,size_t bytes)
{
    if(!parent || !fluid) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.crashfix || s.halo<1 || bytes!=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *plan=new(std::nothrow) astr::CrashfixPlan{};
    if(!plan) return static_cast<int>(cudaErrorMemoryAllocation);
    const size_t rows=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1);
    cudaError_t status=cudaMalloc(&plan->fluid,bytes);
    if(status==cudaSuccess) status=cudaMalloc(&plan->first,rows*sizeof(int));
    if(status==cudaSuccess) status=cudaMalloc(&plan->last,rows*sizeof(int));
    if(status==cudaSuccess) status=cudaMalloc(&plan->report,sizeof(AstrCrashfixReport));
    if(status==cudaSuccess) status=cudaMalloc(&plan->expansion,4*sizeof(int));
    if(status==cudaSuccess) status=cudaMalloc(&plan->input_invalid,sizeof(int));
    if(status==cudaSuccess) status=cudaMemsetAsync(plan->report,0,sizeof(AstrCrashfixReport),s.stream);
    if(status==cudaSuccess) status=cudaMemcpyAsync(plan->fluid,fluid,bytes,cudaMemcpyHostToDevice,s.stream);
    const size_t nodes=s.fields[ASTR_DEVICE_Q].view.stride[3];
    if(status==cudaSuccess) status=cudaMalloc(&s.critical_mask,nodes);
    if(status==cudaSuccess) status=cudaMemsetAsync(s.critical_mask,0,nodes,s.stream);
    if(status==cudaSuccess) status=cudaStreamSynchronize(s.stream);
    if(status!=cudaSuccess) {astr::release_crashfix(plan);return static_cast<int>(status);}
    s.crashfix=plan;
    return 0;
}

extern "C" int astr_device_crashfix_history_upload(void *parent,const int8_t *mask,size_t bytes)
{return history_transfer(parent,const_cast<int8_t *>(mask),bytes,true);}
extern "C" int astr_device_crashfix_history_download(void *parent,int8_t *mask,size_t bytes)
{return history_transfer(parent,mask,bytes,false);}

extern "C" int astr_device_crashfix_recovery_halos(void *parent,Real c2,Real c6)
{return refresh_cache<false>(parent,c2,c6);}
extern "C" int astr_device_crashfix_reconciled_primitives(void *parent,Real c2,Real c6)
{return refresh_cache<true>(parent,c2,c6);}

extern "C" int astr_device_crashfix_input_invalid(void *parent,int *invalid)
{
    if(!parent || !invalid) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.crashfix) return ASTR_DEVICE_NOT_ALLOCATED;
    const cudaError_t cleared=cudaMemsetAsync(s.crashfix->input_invalid,0,sizeof(int),s.stream);
    if(cleared!=cudaSuccess) return static_cast<int>(cleared);
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    check_input<<<(count+255)/256,256,0,s.stream>>>(s.fields[0].view,s.upper[0],s.upper[1],count,s.crashfix->input_invalid);
    const int status=astr::finish_launch(s);if(status) return status;
    const cudaError_t copied=cudaMemcpyAsync(invalid,s.crashfix->input_invalid,sizeof(int),cudaMemcpyDeviceToHost,s.stream);
    if(copied!=cudaSuccess) return static_cast<int>(copied);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_crashfix_ordered_pass(void *parent,const int origin[3],const int global[3],
    Real c2,Real c6,Real eps_pressure,AstrCrashfixReport *report)
{
    if(!parent || !origin || !global || !report || !(c2>0) || !(c6>0) || !(eps_pressure>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.crashfix) return ASTR_DEVICE_NOT_ALLOCATED;
    Limits limits{};
    for(int a=0;a<3;++a) {limits.upper[a]=s.upper[a];limits.origin[a]=origin[a];limits.global[a]=global[a];}
    limits.c2=c2;limits.c6=c6;limits.epsp=eps_pressure;
    Fields fields{s.fields[0].view,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view};
    const size_t rows=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1),count=rows*(s.upper[0]+1);
    initialize_scan<<<(rows+255)/256,256,0,s.stream>>>(*s.crashfix,s.upper[0],rows,true);
    int status=astr::finish_launch(s);if(status) return status;
    inspect<<<(count+255)/256,256,0,s.stream>>>(fields,*s.crashfix,limits,true,count);
    status=astr::finish_launch(s);if(status) return status;
    ordered<<<1,1,0,s.stream>>>(fields,*s.crashfix,s.critical_mask,limits);
    status=astr::finish_launch(s);if(status) return status;
    reset_final_scan<<<1,1,0,s.stream>>>(*s.crashfix);
    status=astr::finish_launch(s);if(status) return status;
    final_report<<<(count+255)/256,256,0,s.stream>>>(fields,*s.crashfix,limits,count);
    status=astr::finish_launch(s);if(status) return status;
    cudaError_t copied=cudaMemcpyAsync(report,s.crashfix->report,sizeof(*report),cudaMemcpyDeviceToHost,s.stream);
    if(copied!=cudaSuccess) return static_cast<int>(copied);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}

extern "C" int astr_device_crashfix_expanded_pass(void *parent,const int lower[3],const int upper[3],
    Real c2,Real c6,Real eps_pressure,AstrCrashfixReport *report,int diagnostics[4])
{
    if(!parent || !lower || !upper || !report || !diagnostics || !(c2>0) || !(c6>0) || !(eps_pressure>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.crashfix) return ASTR_DEVICE_NOT_ALLOCATED;
    Bounds bounds{};Limits limits{};size_t donors=1;
    for(int a=0;a<3;++a) {
        if(lower[a]<-s.halo || upper[a]>s.upper[a]+s.halo || lower[a]>upper[a]) return ASTR_DEVICE_INVALID_ARGUMENT;
        bounds.lower[a]=lower[a];bounds.upper[a]=upper[a];limits.upper[a]=s.upper[a];
        donors*=static_cast<size_t>(upper[a]-lower[a]+1);
    }
    limits.c2=c2;limits.c6=c6;limits.epsp=eps_pressure;
    Fields fields{s.fields[0].view,s.fields[3].view,s.fields[4].view,s.fields[5].view,s.fields[6].view};
    const size_t rows=static_cast<size_t>(s.upper[1]+1)*(s.upper[2]+1),count=rows*(s.upper[0]+1);
    initialize_scan<<<(rows+255)/256,256,0,s.stream>>>(*s.crashfix,s.upper[0],rows,false);
    int status=astr::finish_launch(s);if(status) return status;
    inspect<<<(count+255)/256,256,0,s.stream>>>(fields,*s.crashfix,limits,true,count);
    status=astr::finish_launch(s);if(status) return status;
    find_donor<<<(donors+255)/256,256,0,s.stream>>>(fields,*s.crashfix,limits,bounds,donors);
    status=astr::finish_launch(s);if(status) return status;
    expanded<<<1,1,0,s.stream>>>(fields,*s.crashfix,s.critical_mask,limits,bounds);
    status=astr::finish_launch(s);if(status) return status;
    reset_final_scan<<<1,1,0,s.stream>>>(*s.crashfix);
    status=astr::finish_launch(s);if(status) return status;
    final_report<<<(count+255)/256,256,0,s.stream>>>(fields,*s.crashfix,limits,count);
    status=astr::finish_launch(s);if(status) return status;
    cudaError_t copied=cudaMemcpyAsync(report,s.crashfix->report,sizeof(*report),cudaMemcpyDeviceToHost,s.stream);
    if(copied==cudaSuccess) copied=cudaMemcpyAsync(diagnostics,s.crashfix->expansion,4*sizeof(int),cudaMemcpyDeviceToHost,s.stream);
    if(copied!=cudaSuccess) return static_cast<int>(copied);
    return static_cast<int>(cudaStreamSynchronize(s.stream));
}
