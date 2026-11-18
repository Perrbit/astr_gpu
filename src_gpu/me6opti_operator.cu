#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cmath>
#include <climits>
#include <new>
#include "me6opti_operator.h"
#include "halo_transport.h"
#include "device_state_internal.cuh"

using Real=astr::numerical_real;

namespace astr {
struct Me6Plan {
    Buffer mu,kappa;
    Real *ci=nullptr,*cd=nullptr,*cp=nullptr;
    MPI_Comm communicator=MPI_COMM_NULL;
    int lower[3]{},upper[3]{},parts[3]{},periodic[3]{},first[3]{},last[3]{},inner_first[3]{},inner_last[3]{};
    int mode=ASTR_HALO_PINNED;
    int64_t device_bytes=0,host_bytes=0,sent_bytes=0,staging_bytes=0;
    Real *send[2]{},*receive[2]{},*send_host[2]{},*receive_host[2]{};
};

int me6_inner_bounds(const State &state,int first[3],int last[3])
{
    if(!state.me6) return ASTR_DEVICE_NOT_ALLOCATED;
    for(int a=0;a<3;++a) { first[a]=state.me6->inner_first[a];last[a]=state.me6->inner_last[a]; }
    return 0;
}

int release_me6(Me6Plan *p)
{
    if(!p) return 0;
    int status=0;
    void *buffers[]={p->mu.view.data,p->kappa.view.data,p->ci,p->cd,p->cp,p->send[0],p->send[1],p->receive[0],p->receive[1]};
    for(void *buffer:buffers) if(buffer) {
        const int freed=static_cast<int>(cudaFree(buffer));
        if(status==0) status=freed;
    }
    for(int side=0;side<2;++side) {
        Real *host[]={p->send_host[side],p->receive_host[side]};
        for(Real *buffer:host) if(buffer) {
            const int freed=static_cast<int>(cudaFreeHost(buffer));
            if(status==0) status=freed;
        }
    }
    if(p->communicator!=MPI_COMM_NULL) {
        const int freed=MPI_Comm_free(&p->communicator);
        if(status==0) status=freed;
    }
    delete p;
    return status;
}
}

namespace {
using astr::at;
const MPI_Datatype me6_mpi_type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
struct Bounds { int first[3],last[3]; };

__device__ bool inside(const int *node,Bounds bounds)
{
    return node[0]>=bounds.first[0] && node[0]<=bounds.last[0] &&
        node[1]>=bounds.first[1] && node[1]<=bounds.last[1] && node[2]>=bounds.first[2] && node[2]<=bounds.last[2];
}

__device__ void edge_point(size_t id,int halo,int nx,int nt,int axis,const int *upper,int *lo,int *hi,int *lg,int *hg,int &component)
{
    const int h=id%halo+1,i=(id/halo)%nx,j=(id/(static_cast<size_t>(halo)*nx))%nt;
    component=id/(static_cast<size_t>(halo)*nx*nt);
    lo[0]=i-halo;lo[1]=axis==1 ? h : j-halo;lo[2]=axis==1 ? j : h;
    for(int a=0;a<3;++a) { hi[a]=lo[a];lg[a]=lo[a];hg[a]=lo[a]; }
    hi[axis]=upper[axis]-halo+h-1;lg[axis]=h-halo-1;hg[axis]=upper[axis]+h;
}

template<int operation>
__global__ void edges_kernel(AstrDeviceView vel,AstrDeviceView tmp,int halo,int nx,int nt,int axis,int im,int jm,int km,
    Bounds valid,size_t count,Real *send_low,Real *send_high,const Real *receive_low,const Real *receive_high,
    bool apply_low,bool apply_high)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int upper[3]={im,jm,km};int lo[3],hi[3],lg[3],hg[3],c;
    edge_point(id,halo,nx,nt,axis,upper,lo,hi,lg,hg,c);
    if(lo[0]<valid.first[0] || lo[0]>valid.last[0] ||
       (axis==2 && (lo[1]<valid.first[1] || lo[1]>valid.last[1]))) {
        if(operation==0) { send_low[id]=0;send_high[id]=0; }
        return;
    }
    const AstrDeviceView field=c<3 ? vel : tmp;
    const int component=c<3 ? c+1 : 1;
    if(operation==0) {
        send_low[id]=at(field,lo[0],lo[1],lo[2],component);
        send_high[id]=at(field,hi[0],hi[1],hi[2],component);
    } else if(operation==1) {
        if(apply_low) at(field,lg[0],lg[1],lg[2],component)=receive_low[id];
        if(apply_high) at(field,hg[0],hg[1],hg[2],component)=receive_high[id];
    } else {
        at(field,lg[0],lg[1],lg[2],component)=at(field,hi[0],hi[1],hi[2],component);
        at(field,hg[0],hg[1],hg[2],component)=at(field,lo[0],lo[1],lo[2],component);
    }
}

int extend_edges(astr::State &s,astr::Me6Plan &p)
{
    Bounds valid;
    for(int a=0;a<3;++a) { valid.first[a]=p.first[a];valid.last[a]=p.last[a]; }
    for(int axis=1;axis<3;++axis) {
        if(p.parts[axis]==1 && !p.periodic[axis]) continue;
        const int nx=s.upper[0]+2*s.halo+1,nt=axis==1 ? s.upper[2]+1 : s.upper[1]+2*s.halo+1;
        const size_t count=4*static_cast<size_t>(s.halo)*nx*nt,bytes=count*sizeof(Real);
        const int blocks=(count+255)/256;
        if(p.parts[axis]==1) {
            edges_kernel<2><<<blocks,256,0,s.stream>>>(s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
                s.halo,nx,nt,axis,s.upper[0],s.upper[1],s.upper[2],valid,count,nullptr,nullptr,nullptr,nullptr,true,true);
            const int status=astr::finish_launch(s);
            if(status!=0) return status;
            continue;
        }
        edges_kernel<0><<<blocks,256,0,s.stream>>>(s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
            s.halo,nx,nt,axis,s.upper[0],s.upper[1],s.upper[2],valid,count,p.send[0],p.send[1],nullptr,nullptr,false,false);
        int status=astr::finish_launch(s);
        if(status!=0) return status;
        if(p.mode==ASTR_HALO_PINNED) for(int side=0;side<2;++side) {
            status=static_cast<int>(cudaMemcpyAsync(p.send_host[side],p.send[side],bytes,cudaMemcpyDeviceToHost,s.stream));
            if(status!=0) return status;
        }
        status=static_cast<int>(cudaStreamSynchronize(s.stream));
        if(status!=0) return status;
        Real **send=p.mode==ASTR_HALO_DEVICE ? p.send : p.send_host;
        Real **receive=p.mode==ASTR_HALO_DEVICE ? p.receive : p.receive_host;
        MPI_Request requests[4];
        status=MPI_Irecv(receive[0],static_cast<int>(count),me6_mpi_type,p.lower[axis],1,p.communicator,&requests[0]);
        if(status!=MPI_SUCCESS) return status;
        status=MPI_Irecv(receive[1],static_cast<int>(count),me6_mpi_type,p.upper[axis],0,p.communicator,&requests[1]);
        if(status!=MPI_SUCCESS) return status;
        status=MPI_Isend(send[0],static_cast<int>(count),me6_mpi_type,p.lower[axis],0,p.communicator,&requests[2]);
        if(status!=MPI_SUCCESS) return status;
        status=MPI_Isend(send[1],static_cast<int>(count),me6_mpi_type,p.upper[axis],1,p.communicator,&requests[3]);
        if(status!=MPI_SUCCESS) return status;
        status=MPI_Waitall(4,requests,MPI_STATUSES_IGNORE);
        if(status!=MPI_SUCCESS) return status;
        const int peers=(p.lower[axis]!=MPI_PROC_NULL)+(p.upper[axis]!=MPI_PROC_NULL);
        p.sent_bytes+=peers*bytes;
        if(p.mode==ASTR_HALO_PINNED) p.staging_bytes+=(2+peers)*bytes;
        if(p.mode==ASTR_HALO_PINNED) for(int side=0;side<2;++side) {
            if((side==0 ? p.lower[axis] : p.upper[axis])==MPI_PROC_NULL) continue;
            status=static_cast<int>(cudaMemcpyAsync(p.receive[side],p.receive_host[side],bytes,cudaMemcpyHostToDevice,s.stream));
            if(status!=0) return status;
        }
        edges_kernel<1><<<blocks,256,0,s.stream>>>(s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_TEMPERATURE].view,
            s.halo,nx,nt,axis,s.upper[0],s.upper[1],s.upper[2],valid,count,nullptr,nullptr,p.receive[0],p.receive[1],
            p.lower[axis]!=MPI_PROC_NULL,p.upper[axis]!=MPI_PROC_NULL);
        status=astr::finish_launch(s);
        if(status!=0) return status;
    }
    return 0;
}

__global__ void properties_kernel(AstrDeviceView tmp,AstrDeviceView mu,AstrDeviceView kappa,Bounds valid,
                                   size_t count,Real re,Real pr,Real c5,Real tc,Real tc1)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int node[3]={static_cast<int>(id%tmp.extent[0])+tmp.lower[0],
        static_cast<int>((id/tmp.extent[0])%tmp.extent[1])+tmp.lower[1],
        static_cast<int>(id/(static_cast<size_t>(tmp.extent[0])*tmp.extent[1]))+tmp.lower[2]};
    if(!inside(node,valid)) return;
    const Real t=at(tmp,node[0],node[1],node[2]);
    Real value=t*sqrt(t)*tc1/(t+tc);value/=re;
    at(mu,node[0],node[1],node[2])=value;at(kappa,node[0],node[1],node[2])=(value/pr)/c5;
}

__global__ __launch_bounds__(512)
void mapped_kernel(AstrDeviceView vel,AstrDeviceView tmp,AstrDeviceView mu_field,AstrDeviceView kappa_field,
    AstrDeviceView metric_field,AstrDeviceView jac,AstrDeviceView rhs,Bounds inner,int nx,int ny,size_t count,int axis,
    const Real *ci_table,const Real *cd_table,const Real *cp_table)
{
    const int lane=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    const size_t id=static_cast<size_t>(blockIdx.x)*512+lane;
    if(id>=count) return;
    const int node[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
    if(!inside(node,inner)) return;
    const Real central[3]={Real(.75),-Real(.15),Real(1.0)/Real(60.0)},outer[3]={Real(75.0)/Real(64.0),-Real(25.0)/Real(384.0),Real(3.0)/Real(640.0)};
    Real divergence[4]={};
    for(int h=0;h<3;++h) {
        Real flux[2][4];
        for(int side=0;side<2;++side) {
            const int column=2*h+side;
            Real mu=0,kappa=0,u[3]={},metric[3][3]={},area[3]={},computational[4][3]={},penalty[4]={};
            for(int p=-4;p<=4;++p) {
                int point[3]={node[0],node[1],node[2]};point[axis]+=p;
                const int row=p+4+9*column;
                const Real ci=ci_table[row],cd=cd_table[row],cp=cp_table[row];
                Real value[4];
                for(int c=0;c<3;++c) value[c]=at(vel,point[0],point[1],point[2],c+1);
                value[3]=at(tmp,point[0],point[1],point[2]);
                mu+=ci*at(mu_field,point[0],point[1],point[2]);kappa+=ci*at(kappa_field,point[0],point[1],point[2]);
                for(int b=0;b<3;++b) {
                    u[b]+=ci*value[b];
                    for(int d=0;d<3;++d) metric[d][b]+=ci*at(metric_field,point[0],point[1],point[2],d+1,b+1);
                    area[b]+=ci*at(jac,point[0],point[1],point[2])*at(metric_field,point[0],point[1],point[2],axis+1,b+1);
                }
                for(int c=0;c<4;++c) { computational[c][axis]+=cd*value[c];penalty[c]+=cp*value[c]; }
                for(int d=0;d<3;++d) if(d!=axis) {
                    Real transverse[4]={};
                    for(int l=1;l<=3;++l) {
                        int plus[3]={point[0],point[1],point[2]},minus[3]={point[0],point[1],point[2]};
                        plus[d]+=l;minus[d]-=l;
                        for(int c=0;c<3;++c) transverse[c]+=central[l-1]*(at(vel,plus[0],plus[1],plus[2],c+1)-at(vel,minus[0],minus[1],minus[2],c+1));
                        transverse[3]+=central[l-1]*(at(tmp,plus[0],plus[1],plus[2])-at(tmp,minus[0],minus[1],minus[2]));
                    }
                    for(int c=0;c<4;++c) computational[c][d]+=ci*transverse[c];
                }
            }
            Real norms[3],physical[4][3]={},stress[3][3];
            for(int d=0;d<3;++d) norms[d]=sqrt(metric[d][0]*metric[d][0]+metric[d][1]*metric[d][1]+metric[d][2]*metric[d][2]);
            for(int d=0;d<3;++d) if(d!=axis)
                for(int c=0;c<4;++c) computational[c][d]+=(norms[axis]/norms[d])*penalty[c];
            for(int b=0;b<3;++b) for(int c=0;c<4;++c) for(int d=0;d<3;++d) physical[c][b]+=computational[c][d]*metric[d][b];
            const Real trace=physical[0][0]+physical[1][1]+physical[2][2];
            for(int b=0;b<3;++b) {
                for(int c=0;c<3;++c) stress[c][b]=mu*(physical[c][b]+physical[b][c]);
                stress[b][b]-=2*mu*trace/3;
            }
            for(int c=0;c<3;++c) flux[side][c]=stress[c][0]*area[0]+stress[c][1]*area[1]+stress[c][2]*area[2];
            flux[side][3]=kappa*(physical[3][0]*area[0]+physical[3][1]*area[1]+physical[3][2]*area[2])
                +u[0]*flux[side][0]+u[1]*flux[side][1]+u[2]*flux[side][2];
        }
        for(int c=0;c<4;++c) divergence[c]+=outer[h]*(flux[1][c]-flux[0][c]);
    }
    for(int c=0;c<4;++c) at(rhs,node[0],node[1],node[2],c+2)+=divergence[c];
}
}

extern "C" int astr_device_me6_prepare(void *parent,int communicator,const int *lower,const int *upper,
    const int *parts,const int *periodic,const int *valid_first,const int *valid_last,
    const int *inner_first,const int *inner_last,int mode,const Real *ci,const Real *cd,const Real *cp)
{
    if(!parent || (mode!=ASTR_HALO_PINNED && mode!=ASTR_HALO_DEVICE)) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(s.me6 || s.halo<4) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::Me6Plan *p=new(std::nothrow) astr::Me6Plan{};
    if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->mode=mode;
    size_t capacity=0;
    for(int a=0;a<3;++a) {
        p->lower[a]=lower[a];p->upper[a]=upper[a];p->parts[a]=parts[a];p->periodic[a]=periodic[a];
        p->first[a]=valid_first[a];p->last[a]=valid_last[a];p->inner_first[a]=inner_first[a];p->inner_last[a]=inner_last[a];
        if(a>0 && parts[a]>1) {
            const int nt=a==1 ? s.upper[2]+1 : s.upper[1]+2*s.halo+1;
            const size_t count=4*static_cast<size_t>(s.halo)*(s.upper[0]+2*s.halo+1)*nt;
            if(count>capacity) capacity=count;
        }
    }
    if(capacity>INT_MAX) { astr::release_me6(p);return ASTR_DEVICE_INVALID_ARGUMENT; }
    int status=astr::allocate_field(p->mu,s.upper,s.halo,3);
    if(status==0) status=astr::allocate_field(p->kappa,s.upper,s.halo,3);
    Real **tables[]={&p->ci,&p->cd,&p->cp};const Real *source[]={ci,cd,cp};
    for(int t=0;t<3;++t) {
        if(status==0) status=static_cast<int>(cudaMalloc(tables[t],54*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(*tables[t],source[t],54*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    }
    if(status==0 && capacity>0) status=MPI_Comm_dup(MPI_Comm_f2c(static_cast<MPI_Fint>(communicator)),&p->communicator);
    for(int side=0;side<2 && capacity>0;++side) {
        if(status==0) status=static_cast<int>(cudaMalloc(&p->send[side],capacity*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMalloc(&p->receive[side],capacity*sizeof(Real)));
        if(mode==ASTR_HALO_PINNED) {
            if(status==0) status=static_cast<int>(cudaMallocHost(&p->send_host[side],capacity*sizeof(Real)));
            if(status==0) status=static_cast<int>(cudaMallocHost(&p->receive_host[side],capacity*sizeof(Real)));
        }
    }
    const int synced=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) status=synced;
    if(status!=0) { astr::release_me6(p);return status; }
    p->device_bytes=p->mu.bytes+p->kappa.bytes+162*sizeof(Real)+4*capacity*sizeof(Real);
    p->host_bytes=mode==ASTR_HALO_PINNED ? 4*capacity*sizeof(Real) : 0;
    s.me6=p;
    return 0;
}

extern "C" int astr_device_me6_metrics(void *parent,int64_t values[4])
{
    if(!parent || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.me6) return ASTR_DEVICE_NOT_ALLOCATED;
    values[0]=s.me6->device_bytes;values[1]=s.me6->host_bytes;
    values[2]=s.me6->sent_bytes;values[3]=s.me6->staging_bytes;
    return 0;
}

extern "C" int astr_device_me6_diffusion(void *parent,Real re,Real pr,Real c5,Real tc,Real tc1)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.me6) return ASTR_DEVICE_NOT_ALLOCATED;
    astr::Me6Plan &p=*s.me6;
    int status=extend_edges(s,p);
    if(status!=0) return status;
    Bounds valid,inner;
    for(int a=0;a<3;++a) {
        valid.first[a]=p.first[a];valid.last[a]=p.last[a];inner.first[a]=p.inner_first[a];inner.last[a]=p.inner_last[a];
    }
    const AstrDeviceView tmp=s.fields[ASTR_DEVICE_TEMPERATURE].view;
    const size_t count=tmp.stride[3],physical=s.fields[ASTR_DEVICE_RHS].view.stride[3];
    properties_kernel<<<(count+255)/256,256,0,s.stream>>>(tmp,p.mu.view,p.kappa.view,valid,count,re,pr,c5,tc,tc1);
    status=astr::finish_launch(s);
    if(status!=0) return status;
    for(int axis=0;axis<3;++axis) {
        const dim3 block=axis==0 ? dim3(512,1,1) : (axis==1 ? dim3(32,16,1) : dim3(64,1,8));
        mapped_kernel<<<(physical+511)/512,block,0,s.stream>>>(s.fields[ASTR_DEVICE_VELOCITY].view,tmp,p.mu.view,p.kappa.view,
            s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_RHS].view,
            inner,s.upper[0]+1,s.upper[1]+1,physical,axis,p.ci,p.cd,p.cp);
        status=astr::finish_launch(s);
        if(status!=0) return status;
    }
    return 0;
}
