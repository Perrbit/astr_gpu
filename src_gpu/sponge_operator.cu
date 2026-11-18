#include <algorithm>
#include <new>
#include "device_state_internal.cuh"
#include "sponge_operator.h"
using Real=astr::numerical_real;

namespace astr {
struct SpongeRegion {
    int first[3]{},extent[3]{};
    size_t cells=0;
    Real *coefficient=nullptr;
};
struct SpongePlan { SpongeRegion region[2];Real *work=nullptr; };

int release_sponge(SpongePlan *plan)
{
    if(!plan) return 0;
    int status=0;
    void *pointers[]={plan->region[0].coefficient,plan->region[1].coefficient,plan->work};
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
template<bool commit>
__global__ void sponge_kernel(AstrDeviceView q,astr::SpongeRegion region,Real *work)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=5*region.cells) return;
    const size_t node=id%region.cells;
    const int i=region.first[0]+node%region.extent[0],
        j=region.first[1]+(node/region.extent[0])%region.extent[1],
        k=region.first[2]+node/(static_cast<size_t>(region.extent[0])*region.extent[1]),c=id/region.cells+1;
    if(commit) at(q,i,j,k,c)=work[id];
    else {
        const Real sigma=region.coefficient[node];
        work[id]=(1-sigma)*at(q,i,j,k,c)+(Real(1.0)/6)*sigma*(at(q,i+1,j,k,c)+at(q,i-1,j,k,c)+
            at(q,i,j+1,k,c)+at(q,i,j-1,k,c)+at(q,i,j,k+1,c)+at(q,i,j,k-1,c));
    }
}
}

extern "C" int astr_device_sponge_prepare(void *parent,const int first[3],const int last[3],
    const int begin[2],const int end[2],const Real *x_coefficient,const Real *y_coefficient)
{
    if(!parent || !first || !last || !begin || !end) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(s.sponge) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int d=0;d<3;++d) if(first[d]<0 || last[d]>s.upper[d] || first[d]>last[d]) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::SpongePlan *plan=new(std::nothrow) astr::SpongePlan;
    if(!plan) return static_cast<int>(cudaErrorMemoryAllocation);
    const Real *host[2]={x_coefficient,y_coefficient};
    size_t maximum=0;int status=0;
    for(int axis=0;axis<2 && status==0;++axis) {
        astr::SpongeRegion &r=plan->region[axis];
        if(begin[axis]<0 || end[axis]<begin[axis]) continue;
        r.cells=1;
        for(int d=0;d<3;++d) {
            r.first[d]=d==axis ? std::max(first[d],begin[axis]) : first[d];
            const int last_node=d==axis ? std::min(last[d],end[axis]) : last[d];
            r.extent[d]=std::max(0,last_node-r.first[d]+1);r.cells*=r.extent[d];
        }
        maximum=std::max(maximum,r.cells);
        if(!r.cells) continue;
        if(!host[axis]) {status=ASTR_DEVICE_INVALID_ARGUMENT;break;}
        status=static_cast<int>(cudaMalloc(&r.coefficient,r.cells*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(r.coefficient,host[axis],r.cells*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    }
    if(status==0 && maximum) status=static_cast<int>(cudaMalloc(&plan->work,5*maximum*sizeof(Real)));
    const int synchronized=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) status=synchronized;
    if(status!=0) {astr::release_sponge(plan);return status;}
    s.sponge=plan;
    return 0;
}

extern "C" int astr_device_sponge_apply(void *parent,int axis)
{
    if(!parent || axis<0 || axis>1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.sponge) return ASTR_DEVICE_NOT_ALLOCATED;
    const astr::SpongeRegion &r=s.sponge->region[axis];
    if(!r.cells) return 0;
    const size_t blocks=(5*r.cells+255)/256;
    sponge_kernel<false><<<blocks,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,r,s.sponge->work);
    int status=astr::finish_launch(s);
    if(status!=0) return status;
    sponge_kernel<true><<<blocks,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,r,s.sponge->work);
    return astr::finish_launch(s);
}
