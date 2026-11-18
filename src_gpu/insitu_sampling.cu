#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cmath>
#include <climits>
#include <initializer_list>
#include <new>
#include "insitu_sampling.h"
#include "halo_transport.h"
#include "device_state_internal.cuh"
using Real=astr::numerical_real;
const MPI_Datatype sampling_mpi_type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;

namespace {
using astr::at;
using astr::Buffer;
struct Sample {
    astr::State *parent=nullptr;
    MPI_Comm comm=MPI_COMM_NULL;
    int rank=0,lower[3]{},upper[3]{},sizes[3]{},periodic[3]{},mode=0,components=0;
    bool velocity_requested=false,halo_requested=false,wall_sixth=false,valid=false,borrowed=false;
    Buffer state,velocity,halo,diagnostics,coordinates,geometry;
    Real *send=nullptr,*receive=nullptr,*send_host=nullptr,*receive_host=nullptr;
    int *invalid=nullptr;
    size_t capacity=0,nodes=0;
    int64_t peak=0,sent=0,d2h=0,h2d=0,scalar=0,observation=0,frames=0;
};

int64_t device_bytes(const Sample &p)
{
    const Buffer *buffers[]={&p.state,&p.velocity,&p.halo,&p.diagnostics,&p.coordinates,&p.geometry};
    int64_t bytes=2*p.capacity*sizeof(Real)+sizeof(int);
    for(const Buffer *b:buffers) if(b->view.data) bytes+=b->bytes;
    return bytes;
}
void peak(Sample &p) {const auto current=device_bytes(p);if(current>p.peak) p.peak=current;}
int free_buffer(Buffer &b)
{
    if(!b.view.data) return 0;
    const int status=static_cast<int>(cudaFree(b.view.data));
    b=Buffer{};
    return status;
}
int allocate_packed(Buffer &b,int components,size_t nodes)
{
    b.view.rank=2;b.view.element_bytes=sizeof(Real);
    int64_t stride=1;
    for(int a=0;a<5;++a) {
        b.view.lower[a]=1;b.view.extent[a]=a==0 ? components : (a==1 ? static_cast<int>(nodes) : 1);
        b.view.stride[a]=stride;stride*=b.view.extent[a];
    }
    b.bytes=stride*sizeof(Real);
    return static_cast<int>(cudaMalloc(&b.view.data,b.bytes));
}

// Agreement precedes the next distributed operation. Preserve the first local
// error code rather than turning a failed peer into success or a backend switch.
int agree(MPI_Comm comm,int rank,int local)
{
    int bad=local!=0,any=0;
    int status=MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
    if(status!=MPI_SUCCESS || !any) return status;
    int root=local!=0 ? rank : INT_MAX,first=INT_MAX;
    status=MPI_Allreduce(&root,&first,1,MPI_INT,MPI_MIN,comm);
    if(status!=MPI_SUCCESS) return status;
    status=MPI_Bcast(&local,1,MPI_INT,first,comm);
    return status==MPI_SUCCESS ? local : status;
}
int ready(Sample &p,int status)
{
    const int synced=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    return agree(p.comm,p.rank,status!=0 ? status : synced);
}
int kernel_status() {return static_cast<int>(cudaGetLastError());}
int release(Sample *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    Buffer *buffers[]={&p->state,&p->velocity,&p->halo,&p->diagnostics,&p->coordinates,&p->geometry};
    for(Buffer *b:buffers) {const int freed=free_buffer(*b);if(status==0) status=freed;}
    void *device[]={p->send,p->receive,p->invalid};
    for(void *pointer:device) if(pointer) {
        const int freed=static_cast<int>(cudaFree(pointer));if(status==0) status=freed;
    }
    void *host[]={p->send_host,p->receive_host};
    for(void *pointer:host) if(pointer) {
        const int freed=static_cast<int>(cudaFreeHost(pointer));if(status==0) status=freed;
    }
    if(p->comm!=MPI_COMM_NULL) {const int freed=MPI_Comm_free(&p->comm);if(status==0) status=freed;}
    if(p->borrowed) --p->parent->sample_plans;
    delete p;
    return status;
}

__device__ void point(size_t id,int nx,int ny,int *p)
{
    p[0]=id%nx;p[1]=(id/nx)%ny;p[2]=id/(static_cast<size_t>(nx)*ny);
}
__global__ void copy_state(AstrDeviceView q,AstrDeviceView snapshot,int nx,int ny,size_t nodes)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=nodes) return;
    int p[3];point(id,nx,ny,p);
    for(int c=1;c<=4;++c) at(snapshot,p[0],p[1],p[2],c)=at(q,p[0],p[1],p[2],c);
}
template<bool unpack>
__global__ void endpoint(AstrDeviceView state,int axis,int last,int n1,int n2,size_t count,Real *buffer)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    int p[3]={};p[(axis+1)%3]=id%n1;p[(axis+2)%3]=(id/n1)%n2;p[axis]=unpack ? last : 0;
    const int c=id/(static_cast<size_t>(n1)*n2)+1;
    if(unpack) at(state,p[0],p[1],p[2],c)=buffer[id];
    else buffer[id]=at(state,p[0],p[1],p[2],c);
}
__global__ void convert_velocity(AstrDeviceView state,Real *velocity,
    int nx,int ny,size_t nodes,int *invalid)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=nodes) return;
    int p[3];point(id,nx,ny,p);
    const Real rho=at(state,p[0],p[1],p[2]);
    Real v[3];bool valid=isfinite(rho) && rho>0;
    for(int c=0;c<3;++c) {
        v[c]=at(state,p[0],p[1],p[2],c+2)/rho;valid=valid && isfinite(v[c]);
    }
    if(!valid) {atomicExch(invalid,1);return;}
    for(int c=0;c<3;++c) velocity[3*id+c]=v[c];
}
__global__ void copy_halo(const Real *velocity,AstrDeviceView halo,int nx,int ny,size_t nodes)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=nodes) return;
    int p[3];point(id,nx,ny,p);
    for(int c=0;c<3;++c) at(halo,p[0],p[1],p[2],c+1)=velocity[3*id+c];
}
struct VectorSlots {int column[3];};
struct OwnedShape {int n[3];};
__global__ void import_owned_vector(AstrDeviceView source,VectorSlots slots,OwnedShape owned,AstrDeviceView halo,
    size_t nodes,int *invalid)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=nodes) return;
    int p[3];point(id,owned.n[0],owned.n[1],p);
    const auto *values=static_cast<const Real *>(source.data);
    for(int c=0;c<3;++c) {
        const Real value=values[id*source.stride[0]+slots.column[c]*source.stride[1]];
        if(!isfinite(value)) atomicExch(invalid,1);
        at(halo,p[0],p[1],p[2],c+1)=value;
    }
}
__global__ void import_vector(AstrDeviceView source,VectorSlots slots,Real *packed,AstrDeviceView halo,
    int nx,int ny,size_t nodes,int *invalid)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=nodes) return;
    int p[3];point(id,nx,ny,p);
    const auto *values=static_cast<const Real *>(source.data);
    for(int c=0;c<3;++c) {
        const Real value=values[id*source.stride[0]+slots.column[c]*source.stride[1]];
        if(!isfinite(value)) atomicExch(invalid,1);
        if(packed) packed[3*id+c]=value;
        if(halo.data) at(halo,p[0],p[1],p[2],c+1)=value;
    }
}
template<bool unpack>
__global__ void halo_packet(AstrDeviceView field,int axis,int side,int last,int n1,int n2,size_t count,Real *buffer)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const size_t area=static_cast<size_t>(n1)*n2;
    const int node=id%area,layer=(id/area)%3+1,c=id/(3*area)+1;
    int p[3]={};p[(axis+1)%3]=node%n1-3;p[(axis+2)%3]=node/n1-3;
    p[axis]=unpack ? (side==0 ? last+layer : layer-4) : (side==0 ? layer : last-4+layer);
    if(unpack) at(field,p[0],p[1],p[2],c)=buffer[id];
    else buffer[id]=at(field,p[0],p[1],p[2],c);
}

int exchange(Sample &p,int axis,int side,size_t count,bool &received,const int *neighbors=nullptr)
{
    received=false;
    const size_t bytes=count*sizeof(Real);
    if(p.sizes[axis]==1) {
        if(!p.periodic[axis] || neighbors) return 0;
        received=true;
        return static_cast<int>(cudaMemcpyAsync(p.receive,p.send,bytes,cudaMemcpyDeviceToDevice,p.parent->stream));
    }
    const int lower=neighbors ? neighbors[0] : p.lower[axis],upper=neighbors ? neighbors[1] : p.upper[axis];
    const int destination=side==0 ? lower : upper;
    const int source=side==0 ? upper : lower;
    received=source!=MPI_PROC_NULL;
    int status=0;
    if(p.mode==ASTR_HALO_PINNED && destination!=MPI_PROC_NULL) {
        status=static_cast<int>(cudaMemcpyAsync(p.send_host,p.send,bytes,cudaMemcpyDeviceToHost,p.parent->stream));
        if(status==0) p.d2h+=bytes;
    }
    status=ready(p,status);
    if(status!=0) return status;
    MPI_Request requests[2];const int tag=axis+1+3*side;
    Real *send=p.mode==ASTR_HALO_PINNED ? p.send_host : p.send;
    Real *receive=p.mode==ASTR_HALO_PINNED ? p.receive_host : p.receive;
    status=MPI_Irecv(receive,static_cast<int>(count),sampling_mpi_type,source,tag,p.comm,&requests[0]);
    if(status!=MPI_SUCCESS) return status;
    status=MPI_Isend(send,static_cast<int>(count),sampling_mpi_type,destination,tag,p.comm,&requests[1]);
    if(status!=MPI_SUCCESS) return status;
    status=MPI_Waitall(2,requests,MPI_STATUSES_IGNORE);
    if(status!=MPI_SUCCESS) return status;
    if(destination!=MPI_PROC_NULL) p.sent+=bytes;
    if(p.mode==ASTR_HALO_PINNED && received) {
        status=static_cast<int>(cudaMemcpyAsync(p.receive,p.receive_host,bytes,cudaMemcpyHostToDevice,p.parent->stream));
        if(status==0) p.h2d+=bytes;
    }
    return ready(p,status);
}

int extend_axis(Sample &p,AstrDeviceView field,int axis,const int *neighbors=nullptr)
{
    auto &s=*p.parent;
    if(p.sizes[axis]==1 && (!p.periodic[axis] || neighbors)) return 0;
    const int n1=s.upper[(axis+1)%3]+7,n2=s.upper[(axis+2)%3]+7;
    const size_t count=9*static_cast<size_t>(n1)*n2;
    for(int side=0;side<2;++side) {
        halo_packet<false><<<(count+255)/256,256,0,s.stream>>>(field,axis,side,s.upper[axis],n1,n2,count,p.send);
        int status=ready(p,kernel_status());bool received=false;
        if(status==0) status=exchange(p,axis,side,count,received,neighbors);
        if(status==0 && received) {
            halo_packet<true><<<(count+255)/256,256,0,s.stream>>>(field,axis,side,s.upper[axis],n1,n2,count,p.receive);
            status=kernel_status();
        }
        status=ready(p,status);if(status!=0) return status;
    }
    return 0;
}
int reset_outputs(Sample &p)
{
    int status=astr::transport_in_flight(p.parent->transport) || p.state.view.data ? ASTR_DEVICE_INVALID_ARGUMENT : 0;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    p.valid=false;
    Buffer *outputs[]={&p.velocity,&p.halo,&p.diagnostics,&p.coordinates,&p.geometry};
    for(Buffer *b:outputs) {const int freed=free_buffer(*b);if(status==0) status=freed;}
    return agree(p.comm,p.rank,status);
}
int vector_identity(const Sample &p,const AstrDeviceView *source,const int slots[3])
{
    if(!source || !source->data || !slots || source->rank!=2 || source->element_bytes!=sizeof(Real) ||
        source->extent[0]!=p.nodes || source->stride[0]<1 || source->stride[1]<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<3;++c) if(slots[c]<0 || slots[c]>=source->extent[1]) return ASTR_DEVICE_INVALID_ARGUMENT;
    return 0;
}
int import_selected(Sample &p,const AstrDeviceView &source,VectorSlots slots,Real *packed,AstrDeviceView halo)
{
    auto &s=*p.parent;
    int status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    import_vector<<<(p.nodes+255)/256,256,0,s.stream>>>(source,slots,packed,halo,s.upper[0]+1,s.upper[1]+1,p.nodes,p.invalid);
    status=ready(p,kernel_status());int invalid=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&invalid,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);}
    if(status==0 && invalid) status=ASTR_DEVICE_NONPHYSICAL;
    return agree(p.comm,p.rank,status);
}

__device__ Real value(AstrDeviceView halo,int *node,int c,int axis,int position)
{
    int p[3]={node[0],node[1],node[2]};p[axis]=position;
    return at(halo,p[0],p[1],p[2],c);
}
__device__ Real derivative(AstrDeviceView halo,int *node,int c,int axis,int last,bool low,bool high,bool sixth)
{
    const int n=node[axis];
    if(sixth && ((low && n<3) || (high && n>last-3))) {
        const int start=low && n<3 ? 0 : last-6;
        int row=n-start;Real f[7];
        for(int r=0;r<7;++r) f[r]=value(halo,node,c,axis,start+r);
        Real sign=1;
        if(row>3) {row=6-row;sign=-1;for(int r=0;r<3;++r) {const Real t=f[r];f[r]=f[6-r];f[6-r]=t;}}
        const Real w[3][7]={{-147,360,-450,400,-225,72,-10},{-10,-77,150,-100,50,-15,2},{2,-24,-35,80,-30,8,-1}};
        Real result=0,center=at(halo,node[0],node[1],node[2],c);
        for(int r=0;r<7;++r) result+=w[row][r]*(f[r]-center);
        return sign*(Real(1.0)/Real(60.0))*result;
    }
    if(low && n==0) return -Real(.5)*value(halo,node,c,axis,2)+2*value(halo,node,c,axis,1)-Real(1.5)*value(halo,node,c,axis,0);
    if((low && n==1) || (high && n==last-1)) return Real(.5)*(value(halo,node,c,axis,n+1)-value(halo,node,c,axis,n-1));
    if((low && n==2) || (high && n==last-2)) return (Real(2.0)/Real(3.0))*(value(halo,node,c,axis,n+1)-value(halo,node,c,axis,n-1))
        -(Real(1.0)/Real(12.0))*(value(halo,node,c,axis,n+2)-value(halo,node,c,axis,n-2));
    if(high && n==last) return Real(.5)*value(halo,node,c,axis,last-2)-2*value(halo,node,c,axis,last-1)+Real(1.5)*value(halo,node,c,axis,last);
    return Real(.75)*(value(halo,node,c,axis,n+1)-value(halo,node,c,axis,n-1))
        -Real(.15)*(value(halo,node,c,axis,n+2)-value(halo,node,c,axis,n-2))
        +(Real(1.0)/Real(60.0))*(value(halo,node,c,axis,n+3)-value(halo,node,c,axis,n-3));
}
struct Ends {int lower[3],upper[3],last[3];};
__global__ void derive(AstrDeviceView halo,AstrDeviceView metric,Real *output,int components,
    int nx,int ny,size_t nodes,Ends ends,bool sixth,int *invalid)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=nodes) return;
    int p[3];point(id,nx,ny,p);Real gradient[3][3],fields[14];
    for(int c=0;c<3;++c) {
        Real directional[3];
        for(int a=0;a<3;++a) directional[a]=derivative(halo,p,c+1,a,ends.last[a],ends.lower[a],ends.upper[a],sixth);
        for(int d=0;d<3;++d) {
            gradient[c][d]=directional[0]*at(metric,p[0],p[1],p[2],1,d+1)
                +directional[1]*at(metric,p[0],p[1],p[2],2,d+1)+directional[2]*at(metric,p[0],p[1],p[2],3,d+1);
            fields[c+3*d]=gradient[c][d];
        }
    }
    Real trace=0;
    for(int d=0;d<3;++d) for(int c=0;c<3;++c) trace+=gradient[c][d]*gradient[d][c];
    fields[9]=-Real(.5)*trace;fields[10]=gradient[0][0]+gradient[1][1]+gradient[2][2];
    fields[11]=gradient[2][1]-gradient[1][2];fields[12]=gradient[0][2]-gradient[2][0];fields[13]=gradient[1][0]-gradient[0][1];
    for(int c=0;c<14;++c) {
        if(!isfinite(fields[c])) atomicExch(invalid,1);
        if(components==14) output[14*id+c]=fields[c];
    }
    if(components==1) output[id]=fields[9];
}
}

extern "C" int astr_insitu_sample_create(void *parent,int communicator,const int lower[3],const int upper[3],
    const int sizes[3],const int periodic[3],int mode,int velocity,int halo,int components,int wall_sixth,void **sample)
{
    if(!sample || !parent || !lower || !upper || !sizes || !periodic) return ASTR_DEVICE_INVALID_ARGUMENT;
    *sample=nullptr;
    astr::State &s=*static_cast<astr::State *>(parent);
    const MPI_Comm comm=MPI_Comm_f2c(static_cast<MPI_Fint>(communicator));
    int rank=0,ranks=0;MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);
    int status=0;size_t capacity=0,nodes=1;int64_t parts=1;
    if(!s.fluid_ready || astr::transport_in_flight(s.transport) || (mode!=ASTR_HALO_PINNED && mode!=ASTR_HALO_DEVICE) ||
        (components!=0 && components!=1 && components!=14) || (components>0 && !halo) || (!velocity && !halo))
        status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) {
        if(s.upper[a]<(halo ? 3 : 1) || sizes[a]<1 || (periodic[a]!=0 && periodic[a]!=1)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        if(components>0 && !periodic[a] && wall_sixth && (lower[a]==MPI_PROC_NULL || upper[a]==MPI_PROC_NULL) && s.upper[a]<6)
            status=ASTR_DEVICE_INVALID_ARGUMENT;
        for(int peer:{lower[a],upper[a]}) if(peer!=MPI_PROC_NULL && (peer<0 || peer>=ranks)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        if(periodic[a] && sizes[a]>1 && (lower[a]==MPI_PROC_NULL || upper[a]==MPI_PROC_NULL)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        if(!periodic[a] && sizes[a]==1 && (lower[a]!=MPI_PROC_NULL || upper[a]!=MPI_PROC_NULL)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        nodes*=s.upper[a]+1;parts*=sizes[a];
        if(sizes[a]>1 || periodic[a]) {
            const size_t n1=s.upper[(a+1)%3]+1,n2=s.upper[(a+2)%3]+1;
            size_t count=4*n1*n2;
            if(halo && 9*(n1+6)*(n2+6)>count) count=9*(n1+6)*(n2+6);
            if(count>capacity) capacity=count;
        }
    }
    if(parts!=ranks || capacity>INT_MAX || nodes>INT_MAX) status=ASTR_DEVICE_INVALID_ARGUMENT;
    status=agree(comm,rank,status);
    if(status!=0) return status;
    Sample *p=new(std::nothrow) Sample{};
    status=agree(comm,rank,p ? 0 : static_cast<int>(cudaErrorMemoryAllocation));
    if(status!=0) {delete p;return status;}
    p->parent=&s;p->rank=rank;p->nodes=nodes;p->capacity=capacity;p->mode=mode;p->components=components;
    p->velocity_requested=velocity!=0;p->halo_requested=halo!=0;p->wall_sixth=wall_sixth!=0;
    for(int a=0;a<3;++a) {p->lower[a]=lower[a];p->upper[a]=upper[a];p->sizes[a]=sizes[a];p->periodic[a]=periodic[a];}
    status=MPI_Comm_dup(comm,&p->comm);
    if(status==MPI_SUCCESS) status=static_cast<int>(cudaMalloc(&p->invalid,sizeof(int)));
    if(status==0 && capacity>0) status=static_cast<int>(cudaMalloc(&p->send,capacity*sizeof(Real)));
    if(status==0 && capacity>0) status=static_cast<int>(cudaMalloc(&p->receive,capacity*sizeof(Real)));
    if(mode==ASTR_HALO_PINNED && ranks>1 && capacity>0) {
        if(status==0) status=static_cast<int>(cudaMallocHost(&p->send_host,capacity*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMallocHost(&p->receive_host,capacity*sizeof(Real)));
    }
    status=agree(comm,rank,status);
    if(status!=0) {release(p);return status;}
    p->borrowed=true;++s.sample_plans;peak(*p);*sample=p;
    return 0;
}
extern "C" int astr_insitu_sample_begin(void *sample)
{
    if(!sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    int status=reset_outputs(p);if(status!=0) return status;
    if(status==0) status=astr::allocate_field(p.state,s.upper,0,4,4);
    status=agree(p.comm,p.rank,status);
    if(status!=0) return status;
    peak(p);
    copy_state<<<(p.nodes+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,p.state.view,s.upper[0]+1,s.upper[1]+1,p.nodes);
    return ready(p,kernel_status());
}
extern "C" int astr_insitu_sample_vector(void *sample,const AstrDeviceView *source,const int slots[3])
{
    if(!sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    int status=agree(p.comm,p.rank,vector_identity(p,source,slots));if(status!=0) return status;
    status=reset_outputs(p);if(status!=0) return status;
    if(p.velocity_requested) status=allocate_packed(p.velocity,3,p.nodes);
    if(status==0 && p.halo_requested) {
        status=astr::allocate_field(p.halo,s.upper,3,4,3);
        if(status==0) status=static_cast<int>(cudaMemsetAsync(p.halo.view.data,0,p.halo.bytes,s.stream));
    }
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    peak(p);VectorSlots selected{{slots[0],slots[1],slots[2]}};
    return import_selected(p,*source,selected,static_cast<Real *>(p.velocity.view.data),p.halo.view);
}
extern "C" int astr_insitu_sample_owned_vector(void *sample,const AstrDeviceView *source,const int owned[3],const int slots[3])
{
    if(!sample || !source || !owned || !slots) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    size_t nodes=1;int status=0;OwnedShape shape{};
    for(int a=0;a<3;++a) {
        const int expected=s.upper[a]+(!p.periodic[a] && p.upper[a]==MPI_PROC_NULL ? 1 : 0);
        if(owned[a]!=expected) status=ASTR_DEVICE_INVALID_ARGUMENT;
        shape.n[a]=owned[a];nodes*=owned[a];
    }
    if(!p.halo_requested || p.velocity_requested || p.components!=0 || !source->data || source->rank!=2 ||
        source->element_bytes!=sizeof(Real) || source->extent[0]!=nodes || source->stride[0]<1 || source->stride[1]<1)
        status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<3;++c) if(slots[c]<0 || slots[c]>=source->extent[1]) status=ASTR_DEVICE_INVALID_ARGUMENT;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    status=reset_outputs(p);if(status!=0) return status;
    status=astr::allocate_field(p.halo,s.upper,3,4,3);
    if(status==0) status=static_cast<int>(cudaMemsetAsync(p.halo.view.data,0,p.halo.bytes,s.stream));
    if(status==0) status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    peak(p);VectorSlots selected{{slots[0],slots[1],slots[2]}};
    import_owned_vector<<<(nodes+255)/256,256,0,s.stream>>>(*source,selected,shape,p.halo.view,nodes,p.invalid);
    status=ready(p,kernel_status());int invalid=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&invalid,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);}
    if(status==0 && invalid) status=ASTR_DEVICE_NONPHYSICAL;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    // Copy unique owners into duplicate upper nodes before extending the halo.
    for(int a=0;a<3;++a) {
        if(p.sizes[a]==1 && !p.periodic[a]) continue;
        const int n1=s.upper[(a+1)%3]+1,n2=s.upper[(a+2)%3]+1;
        const size_t count=3*static_cast<size_t>(n1)*n2;
        endpoint<false><<<(count+255)/256,256,0,s.stream>>>(p.halo.view,a,s.upper[a],n1,n2,count,p.send);
        status=ready(p,kernel_status());bool received=false;
        if(status==0) status=exchange(p,a,0,count,received);
        if(status==0 && received) {
            endpoint<true><<<(count+255)/256,256,0,s.stream>>>(p.halo.view,a,s.upper[a],n1,n2,count,p.receive);
            status=kernel_status();
        }
        status=ready(p,status);if(status!=0) return status;
    }
    for(int a=0;a<3;++a) {status=extend_axis(p,p.halo.view,a);if(status!=0) return status;}
    return astr_insitu_sample_finish(sample);
}
extern "C" int astr_insitu_sample_coordinates(void *sample,const AstrDeviceView *source,const int offset[3],const int global[3])
{
    if(!sample || !offset || !global) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);auto &s=*p.parent;const int slots[3]={0,1,2};
    int status=vector_identity(p,source,slots);
    if(!p.valid || !p.halo_requested || astr::transport_in_flight(s.transport)) status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) if(offset[a]<0 || global[a]<1 || offset[a]+s.upper[a]>global[a]) status=ASTR_DEVICE_INVALID_ARGUMENT;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    p.valid=false;status=free_buffer(p.coordinates);
    if(status==0) status=astr::allocate_field(p.coordinates,s.upper,3,4,3);
    if(status==0) status=static_cast<int>(cudaMemsetAsync(p.coordinates.view.data,0,p.coordinates.bytes,s.stream));
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    peak(p);status=import_selected(p,*source,VectorSlots{{0,1,2}},nullptr,p.coordinates.view);
    if(status!=0) return status;
    for(int a=0;a<3;++a) {
        const int neighbors[2]={offset[a]==0 ? MPI_PROC_NULL : p.lower[a],
            offset[a]+s.upper[a]==global[a] ? MPI_PROC_NULL : p.upper[a]};
        status=extend_axis(p,p.coordinates.view,a,neighbors);if(status!=0) return status;
    }
    p.valid=true;return 0;
}
extern "C" int astr_insitu_sample_upload_coordinates(void *sample,const Real *xyz,size_t bytes,
    const int offset[3],const int global[3])
{
    if(!sample || !xyz || !offset || !global) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    int status=bytes!=3*p.nodes*sizeof(Real) || !p.valid || !p.halo_requested || p.geometry.view.data ?
        ASTR_DEVICE_INVALID_ARGUMENT : 0;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    status=allocate_packed(p.geometry,3,p.nodes);
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(p.geometry.view.data,xyz,bytes,cudaMemcpyHostToDevice,s.stream));
    status=ready(p,status);if(status!=0) return status;
    peak(p);
    AstrDeviceView source=p.geometry.view;
    source.extent[0]=static_cast<int>(p.nodes);source.extent[1]=3;
    source.stride[0]=3;source.stride[1]=1;
    return astr_insitu_sample_coordinates(sample,&source,offset,global);
}
extern "C" int astr_insitu_sample_canonical_axis(void *sample,int axis)
{
    if(!sample || axis<0 || axis>2) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    if(!p.state.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    if(p.sizes[axis]==1 && !p.periodic[axis]) return 0;
    const int n1=s.upper[(axis+1)%3]+1,n2=s.upper[(axis+2)%3]+1;
    const size_t count=4*static_cast<size_t>(n1)*n2;
    endpoint<false><<<(count+255)/256,256,0,s.stream>>>(p.state.view,axis,s.upper[axis],n1,n2,count,p.send);
    int status=ready(p,kernel_status());bool received=false;
    if(status==0) status=exchange(p,axis,0,count,received);
    if(status==0 && received) {
        endpoint<true><<<(count+255)/256,256,0,s.stream>>>(p.state.view,axis,s.upper[axis],n1,n2,count,p.receive);
        status=kernel_status();
    }
    return ready(p,status);
}
extern "C" int astr_insitu_sample_velocity(void *sample)
{
    if(!sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    if(!p.state.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    int status=0;
    // As in the mature sampler, retire canonical q before allocating the halo.
    // Streamline-only consumers discard this temporary packed velocity afterward.
    status=allocate_packed(p.velocity,3,p.nodes);
    if(status==0) status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    status=agree(p.comm,p.rank,status);
    if(status!=0) return status;
    peak(p);
    convert_velocity<<<(p.nodes+255)/256,256,0,s.stream>>>(p.state.view,
        static_cast<Real *>(p.velocity.view.data),s.upper[0]+1,s.upper[1]+1,p.nodes,p.invalid);
    status=ready(p,kernel_status());
    int invalid=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&invalid,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);}
    if(status==0 && invalid) status=ASTR_DEVICE_NONPHYSICAL;
    status=agree(p.comm,p.rank,status);
    if(status!=0) return status;
    status=agree(p.comm,p.rank,free_buffer(p.state));
    if(status!=0) return status;
    if(p.halo_requested) {
        status=astr::allocate_field(p.halo,s.upper,3,4,3);
        if(status==0) status=static_cast<int>(cudaMemsetAsync(p.halo.view.data,0,p.halo.bytes,s.stream));
        status=agree(p.comm,p.rank,status);
        if(status!=0) return status;
        peak(p);
        copy_halo<<<(p.nodes+255)/256,256,0,s.stream>>>(static_cast<const Real *>(p.velocity.view.data),p.halo.view,
            s.upper[0]+1,s.upper[1]+1,p.nodes);
        status=ready(p,kernel_status());
        if(status!=0) return status;
    }
    if(!p.velocity_requested) status=free_buffer(p.velocity);
    return agree(p.comm,p.rank,status);
}
extern "C" int astr_insitu_sample_halo_axis(void *sample,int axis)
{
    if(!sample || axis<0 || axis>2) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);
    if(!p.halo.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    return extend_axis(p,p.halo.view,axis);
}
extern "C" int astr_insitu_sample_finish(void *sample)
{
    if(!sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);auto &s=*p.parent;
    int status=0;
    if(p.components>0) {
        if(!p.halo.view.data) return ASTR_DEVICE_NOT_ALLOCATED;
        status=allocate_packed(p.diagnostics,p.components,p.nodes);
        if(status==0) status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
        status=agree(p.comm,p.rank,status);
        if(status!=0) return status;
        peak(p);Ends ends{};
        for(int a=0;a<3;++a) {
            ends.last[a]=s.upper[a];ends.lower[a]=!p.periodic[a] && p.lower[a]==MPI_PROC_NULL;
            ends.upper[a]=!p.periodic[a] && p.upper[a]==MPI_PROC_NULL;
        }
        derive<<<(p.nodes+255)/256,256,0,s.stream>>>(p.halo.view,s.fields[ASTR_DEVICE_METRIC].view,
            static_cast<Real *>(p.diagnostics.view.data),p.components,s.upper[0]+1,s.upper[1]+1,p.nodes,ends,p.wall_sixth,p.invalid);
        status=kernel_status();
    }
    status=ready(p,status);
    if(status==0 && p.components>0) {
        int invalid=0;status=static_cast<int>(cudaMemcpy(&invalid,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));
        p.scalar+=sizeof(int);if(status==0 && invalid) status=ASTR_DEVICE_NONPHYSICAL;
    }
    status=agree(p.comm,p.rank,status);
    if(status==0) {p.valid=true;++p.frames;}
    return status;
}

namespace {
Buffer *output(Sample &p,int field)
{
    if(field==ASTR_SAMPLE_VELOCITY) return &p.velocity;
    if(field==ASTR_SAMPLE_HALO) return &p.halo;
    if(field==ASTR_SAMPLE_DIAGNOSTICS) return &p.diagnostics;
    if(field==ASTR_SAMPLE_STATE) return &p.state;
    if(field==ASTR_SAMPLE_COORDINATES) return &p.coordinates;
    if(field==ASTR_SAMPLE_GEOMETRY) return &p.geometry;
    return nullptr;
}
}
extern "C" int astr_insitu_sample_view(void *sample,int field,AstrDeviceView *view)
{
    if(!sample || !view) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);Buffer *b=output(p,field);
    if(!b) return ASTR_DEVICE_INVALID_ARGUMENT;
    if((field!=ASTR_SAMPLE_STATE && !p.valid) || !b->view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    *view=b->view;return 0;
}
extern "C" int astr_insitu_sample_download(void *sample,int field,Real *host,size_t bytes)
{
    if(!sample || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample &p=*static_cast<Sample *>(sample);Buffer *b=output(p,field);
    if(!b || bytes!=b->bytes) return ASTR_DEVICE_INVALID_ARGUMENT;
    if((field!=ASTR_SAMPLE_STATE && !p.valid) || !b->view.data) return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemcpy(host,b->view.data,bytes,cudaMemcpyDeviceToHost));
    if(status==0) p.observation+=bytes;
    return status;
}
extern "C" int astr_insitu_sample_metrics(void *sample,int64_t values[9])
{
    if(!sample || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const Sample &p=*static_cast<Sample *>(sample);
    const int64_t v[9]={device_bytes(p),p.peak,p.send_host ? static_cast<int64_t>(2*p.capacity*sizeof(Real)) : 0,
        p.sent,p.d2h,p.h2d,p.scalar,p.observation,p.frames};
    for(int c=0;c<9;++c) values[c]=v[c];
    return 0;
}
extern "C" int astr_insitu_sample_destroy(void **sample)
{
    if(!sample || !*sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    Sample *p=static_cast<Sample *>(*sample);*sample=nullptr;return release(p);
}

namespace {
struct ProbeConstants {Real c[6];};
struct Probe {
    astr::State *parent=nullptr;
    int count=0,*indices=nullptr,*invalid=nullptr;
    Real *values=nullptr;
    ProbeConstants constants{};
    bool borrowed=false;
    int64_t captures=0,status_d2h=0,values_d2h=0;
};
int release_probe(Probe *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    void *buffers[]={p->indices,p->invalid,p->values};
    for(void *buffer:buffers) if(buffer) {
        const int freed=static_cast<int>(cudaFree(buffer));if(status==0) status=freed;
    }
    if(p->borrowed) --p->parent->sample_plans;
    delete p;return status;
}
__global__ void gather_primitive_probes(AstrDeviceView q,int count,const int *indices,
    ProbeConstants material,Real *values,int *invalid)
{
    const int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=count) return;
    for(int c=0;c<6;++c) values[6*n+c]=0;
    const int i=indices[3*n],j=indices[3*n+1],k=indices[3*n+2];
    if(i<0) return;
    const Real rho=at(q,i,j,k);
    if(!(isfinite(rho) && rho>0)) {atomicExch(invalid,1);return;}
    const Real u=at(q,i,j,k,2)/rho,v=at(q,i,j,k,3)/rho,w=at(q,i,j,k,4)/rho;
    const Real internal=at(q,i,j,k,5)-Real(.5)*rho*(u*u+v*v+w*w);
    const Real pressure=internal/material.c[1],temperature=(pressure/rho)*material.c[0];
    const Real normalized[6]={u/material.c[2],v/material.c[2],w/material.c[2],
        rho/material.c[3],pressure/material.c[4],temperature/material.c[5]};
    for(int c=0;c<6;++c) {
        values[6*n+c]=normalized[c];if(!isfinite(normalized[c])) atomicExch(invalid,1);
    }
}
}
extern "C" int astr_insitu_probe_create(void *parent,int count,const int *indices,const Real constants[6],void **probe)
{
    if(!parent || !probe || !indices || !constants || count<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    *probe=nullptr;
    auto &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || astr::transport_in_flight(s.transport)) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<6;++c) if(!(std::isfinite(constants[c]) && constants[c]>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int n=0;n<count;++n) {
        bool absent=true,physical=true;
        for(int a=0;a<3;++a) {
            const int index=indices[3*n+a];absent=absent && index==-1;
            physical=physical && index>=0 && index<=s.upper[a];
        }
        if(!absent && !physical) return ASTR_DEVICE_INVALID_ARGUMENT;
    }
    auto *p=new(std::nothrow) Probe{};if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->parent=&s;p->count=count;
    for(int c=0;c<6;++c) p->constants.c[c]=constants[c];
    const size_t index_bytes=3*static_cast<size_t>(count)*sizeof(int);
    int status=static_cast<int>(cudaMalloc(&p->indices,index_bytes));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->values,6*static_cast<size_t>(count)*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->invalid,sizeof(int)));
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(p->indices,indices,index_bytes,cudaMemcpyHostToDevice,s.stream));
    const int synced=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0) status=synced;
    if(status!=0) {release_probe(p);return status;}
    ++s.sample_plans;p->borrowed=true;*probe=p;return 0;
}
extern "C" int astr_insitu_probe_capture(void *probe,Real *values,size_t bytes)
{
    if(!probe || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Probe *>(probe);auto &s=*p.parent;
    if(bytes!=6*static_cast<size_t>(p.count)*sizeof(Real) || astr::transport_in_flight(s.transport))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    if(status==0) {
        gather_primitive_probes<<<(p.count+127)/128,128,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
            p.count,p.indices,p.constants,p.values,p.invalid);status=static_cast<int>(cudaGetLastError());
    }
    int invalid=0;
    if(status==0) {
        status=static_cast<int>(cudaMemcpyAsync(&invalid,p.invalid,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
        if(status==0) p.status_d2h+=sizeof(int);
    }
    const int synced=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0) status=synced;
    if(status!=0) return status;
    if(invalid) return ASTR_DEVICE_NONPHYSICAL;
    status=static_cast<int>(cudaMemcpyAsync(values,p.values,bytes,cudaMemcpyDeviceToHost,s.stream));
    const int finished=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0) status=finished;
    if(status==0) {p.values_d2h+=bytes;++p.captures;}
    return status;
}
extern "C" int astr_insitu_probe_metrics(void *probe,int64_t values[5])
{
    if(!probe || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Probe *>(probe);
    values[0]=static_cast<int64_t>(p.count)*(3*sizeof(int)+6*sizeof(Real))+sizeof(int);
    values[1]=3*static_cast<int64_t>(p.count)*sizeof(int);values[2]=p.captures;
    values[3]=p.status_d2h;values[4]=p.values_d2h;return 0;
}
extern "C" int astr_insitu_probe_destroy(void **probe)
{
    if(!probe || !*probe) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *p=static_cast<Probe *>(*probe);*probe=nullptr;return release_probe(p);
}
