#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <new>
#include "insitu_wall.h"
#include "halo_transport.h"
#include "device_state_internal.cuh"
using Real=astr::numerical_real;

namespace {
using astr::at;
struct Wall {
    astr::State *parent=nullptr;
    MPI_Comm comm=MPI_COMM_NULL;
    int rank=0,nx=0,nz=0,nw=0,first=0,layers=3,mode=0;
    int lower[3]{},upper[3]{},sizes[3]{},periodic[3]{};
    bool curved=false,borrowed=false,valid=false,previous=false;
    size_t nodes=0,capacity=0;
    Real *state=nullptr,*primitive=nullptr,*geometry=nullptr,*fields=nullptr,*line=nullptr;
    Real *send=nullptr,*receive=nullptr,*send_host=nullptr,*receive_host=nullptr;
    Real *moments=nullptr;
    int *invalid=nullptr;
    int64_t d2h=0,h2d=0,sent=0,scalar=0,observed=0,captures=0,setup=0;
    Real material[7]{};
};
struct Material {Real c[7];};
const MPI_Datatype real_type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;
size_t primitive_nodes(const Wall &p) {return static_cast<size_t>(p.nx+6)*(p.nz+6)*p.nw;}
int64_t device_bytes(const Wall &p)
{
    return sizeof(Real)*((5*p.layers+13)*p.nodes+(p.curved?4*primitive_nodes(p):0)+
        6*p.nx+2*p.capacity+(p.moments?30*p.nodes:0))+sizeof(int);
}
int agree(MPI_Comm comm,int rank,int local)
{
    int bad=local!=0,any=0;
    int status=MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
    if(status!=MPI_SUCCESS || !any) return status;
    int root=local!=0?rank:INT_MAX,first=INT_MAX;
    status=MPI_Allreduce(&root,&first,1,MPI_INT,MPI_MIN,comm);
    if(status==MPI_SUCCESS) status=MPI_Bcast(&local,1,MPI_INT,first,comm);
    return status==MPI_SUCCESS?local:status;
}
int ready(Wall &p,int status)
{
    const int synced=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    return agree(p.comm,p.rank,status!=0?status:synced);
}
int release(Wall *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    void *device[]={p->state,p->primitive,p->geometry,p->fields,p->line,p->send,p->receive,p->invalid,p->moments};
    for(void *v:device) if(v) {const int code=static_cast<int>(cudaFree(v));if(status==0) status=code;}
    for(void *v:{static_cast<void *>(p->send_host),static_cast<void *>(p->receive_host)}) if(v) {
        const int code=static_cast<int>(cudaFreeHost(v));if(status==0) status=code;
    }
    if(p->comm!=MPI_COMM_NULL) {const int code=MPI_Comm_free(&p->comm);if(status==0) status=code;}
    if(p->borrowed) --p->parent->sample_plans;
    delete p;return status;
}
__device__ size_t primitive_index(int i,int k,int w,int c,int nx,int nz,int nw)
{
    return i+3+static_cast<size_t>(nx+6)*(k+3+(nz+6)*(w+nw*c));
}
__global__ void pack(AstrDeviceView q,Real *state,int nx,int nz,int nw,int first,int jm,int layers,size_t nodes)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    const int i=n%nx,k=(n/nx)%nz,w=n/(static_cast<size_t>(nx)*nz);
    const int j=w==0?first:jm,side=j==jm?-1:1;
    for(int m=0;m<layers;++m) for(int c=0;c<5;++c) state[n+nodes*(5*m+c)]=at(q,i,j+side*m,k,c+1);
}
template<bool unpack>
__global__ void endpoint(Real *state,Real *buffer,int nx,int nz,int nw,int axis,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int length=axis==0?nz:nx,w=(id/length)%nw,c=id/(static_cast<size_t>(length)*nw);
    const int i=axis==0?(unpack?nx-1:0):id%length,k=axis==0?id%length:(unpack?nz-1:0);
    const size_t n=i+static_cast<size_t>(nx)*(k+nz*w),nodes=static_cast<size_t>(nx)*nz*nw;
    if(unpack) state[n+nodes*c]=buffer[id];else buffer[id]=state[n+nodes*c];
}
__device__ bool convert(const Real *state,size_t n,size_t nodes,int m,Material material,Real v[3],Real &p,Real &t)
{
    const Real rho=state[n+nodes*(5*m)];
    if(!(isfinite(rho) && rho>0)) return false;
    Real kinetic=0;
    for(int a=0;a<3;++a) {v[a]=state[n+nodes*(5*m+a+1)]/rho;kinetic+=v[a]*v[a];}
    p=(state[n+nodes*(5*m+4)]-Real(.5)*rho*kinetic)/material.c[2];t=p/rho*material.c[0];
    return isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]) && isfinite(p) && isfinite(t) && t>0;
}
__global__ void make_primitive(const Real *state,Real *primitive,int nx,int nz,int nw,size_t nodes,Material material,int *bad)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    const int i=n%nx,k=(n/nx)%nz,w=n/(static_cast<size_t>(nx)*nz);
    Real v[3],p,t;
    if(!convert(state,n,nodes,0,material,v,p,t)) {atomicExch(bad,1);return;}
    for(int c=0;c<4;++c) primitive[primitive_index(i,k,w,c,nx,nz,nw)]=c==3?t:v[c];
}
template<bool unpack>
__global__ void halo_packet(Real *primitive,Real *buffer,int nx,int nz,int nw,int axis,int side,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int length=axis==0?nz:nx;
    const int layer=axis==0?id%3:(id/length)%3;
    const int along=axis==0?(id/3)%length:id%length;
    const int w=(id/(3*length))%nw,c=id/(static_cast<size_t>(3)*length*nw);
    const int last=axis==0?nx-1:nz-1;
    const int index=unpack?(side==0?last+1+layer:-3+layer):(side==0?1+layer:last-3+layer);
    const int i=axis==0?index:along,k=axis==0?along:index;
    const size_t n=primitive_index(i,k,w,c,nx,nz,nw);
    if(unpack) primitive[n]=buffer[id];else buffer[id]=primitive[n];
}
__global__ void derive(const Real *state,const Real *primitive,const Real *geometry,Real *fields,
    AstrDeviceView metric,int nx,int nz,int nw,int first,int jm,int layers,bool curved,size_t nodes,Material material,int *bad)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    const int i=n%nx,k=(n/nx)%nz,w=n/(static_cast<size_t>(nx)*nz),j=w==0?first:jm,side=j==jm?-1:1;
    Real velocity[8][3],pressure[8],temperature[8];
    for(int m=0;m<layers;++m) if(!convert(state,n,nodes,m,material,velocity[m],pressure[m],temperature[m])) {
        atomicExch(bad,1);return;
    }
    const Real mu=temperature[0]*sqrt(temperature[0])*material.c[6]/(temperature[0]+material.c[5])/material.c[3];
    const Real normal_metric=at(metric,i,j,k,2,2);
    if(!(isfinite(normal_metric) && normal_metric>0)) {atomicExch(bad,1);return;}
    Real shear,heat,ny=side;
    if(!curved) {
        Real du=-Real(.5)*velocity[2][0]+2*velocity[1][0]-Real(1.5)*velocity[0][0];
        Real dt=-Real(.5)*temperature[2]+2*temperature[1]-Real(1.5)*temperature[0];
        if(layers==8) {
            const Real weight[8]={-Real(156)/60,Real(423)/60,-Real(639)/60,Real(715)/60,
                -Real(540)/60,Real(261)/60,-Real(73)/60,Real(9)/60};
            du=0;dt=0;
            for(int m=0;m<8;++m) {
                du+=weight[m]*(velocity[m][0]-velocity[0][0]);dt+=weight[m]*(temperature[m]-temperature[0]);
            }
        }
        shear=mu*(du*normal_metric);heat=-((mu/material.c[4])/material.c[1])*(dt*normal_metric);
    } else {
        Real directional[3][3]{},thermal[3]{},gradient[3][3],normal[3],tangent[3],physical_t[3]{};
        for(int c=0;c<3;++c) directional[c][1]=Real(side)*(-Real(.5)*velocity[2][c]+2*velocity[1][c]-Real(1.5)*velocity[0][c]);
        thermal[1]=Real(side)*(-Real(.5)*temperature[2]+2*temperature[1]-Real(1.5)*temperature[0]);
        const Real weight[3]={Real(.75),-Real(.15),Real(1)/60};
        for(int s=1;s<=3;++s) for(int c=0;c<4;++c) {
            const Real dx=weight[s-1]*(primitive[primitive_index(i+s,k,w,c,nx,nz,nw)]-primitive[primitive_index(i-s,k,w,c,nx,nz,nw)]);
            const Real dz=weight[s-1]*(primitive[primitive_index(i,k+s,w,c,nx,nz,nw)]-primitive[primitive_index(i,k-s,w,c,nx,nz,nw)]);
            if(c==3) {thermal[0]+=dx;thermal[2]+=dz;}else {directional[c][0]+=dx;directional[c][2]+=dz;}
        }
        for(int d=0;d<3;++d) {
            normal[d]=geometry[n+nodes*(3+d)];tangent[d]=geometry[n+nodes*(6+d)];
            for(int c=0;c<3;++c) gradient[c][d]=directional[c][0]*at(metric,i,j,k,1,d+1)+
                directional[c][1]*at(metric,i,j,k,2,d+1)+directional[c][2]*at(metric,i,j,k,3,d+1);
            physical_t[d]=thermal[0]*at(metric,i,j,k,1,d+1)+thermal[1]*at(metric,i,j,k,2,d+1)+thermal[2]*at(metric,i,j,k,3,d+1);
        }
        const Real trace=gradient[0][0]+gradient[1][1]+gradient[2][2];
        Real traction[3]{};
        for(int c=0;c<3;++c) for(int d=0;d<3;++d) {
            Real stress=mu*(gradient[c][d]+gradient[d][c]);
            if(c==d) stress-=(Real(2)/3)*mu*trace;
            traction[c]+=stress*normal[d];
        }
        shear=tangent[0]*traction[0]+tangent[1]*traction[1]+tangent[2]*traction[2];
        heat=-((mu/material.c[4])/material.c[1])*(physical_t[0]*normal[0]+physical_t[1]*normal[1]+physical_t[2]*normal[2]);
        ny=normal[1];
    }
    const Real values[4]={pressure[0],shear,heat,ny};
    for(int c=0;c<4;++c) {if(!isfinite(values[c])) atomicExch(bad,1);fields[n+nodes*c]=values[c];}
}
__global__ void selected_line(const Real *geometry,const Real *fields,Real *line,int nx,int nz,int nw,int k,int w)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nx) return;
    const size_t nodes=static_cast<size_t>(nx)*nz*nw,n=i+static_cast<size_t>(nx)*(k+nz*w);
    for(int c=0;c<6;++c) line[6*i+c]=c<3?geometry[n+nodes*c]:fields[n+nodes*(c-3)];
}
__global__ void push_moments(const Real *fields,Real *moments,size_t nodes,Real w0,Real w1,bool previous,int *bad)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    Real s[30]{};
    if(previous) for(int c=0;c<30;++c) s[c]=moments[n+nodes*c];
    for(int e=0;e<2;++e) {
        const Real weight=e==0?w0:w1;
        if(weight==0) continue;
        Real u[3];for(int c=0;c<3;++c) u[c]=e==0?s[c+1]:fields[n+nodes*c];
        const Real total=s[4],next=total+weight;
        if(!(isfinite(weight) && weight>0 && isfinite(next) && next>0)) {atomicExch(bad,1);return;}
        if(total==0) for(int c=0;c<3;++c) s[c+6]=u[c];
        else {
            Real delta[3];for(int c=0;c<3;++c) delta[c]=u[c]-s[c+6];
            const Real ratio=weight/next,factor=(total/next)*weight;
            for(int c=0;c<3;++c) s[c+6]+=ratio*delta[c];
            for(int b=0;b<3;++b) for(int a=0;a<=b;++a) {
                const Real entry=s[12+3*b+a]+(factor*delta[a])*delta[b];
                s[12+3*b+a]=entry;s[12+3*a+b]=entry;
            }
        }
        s[4]=next;
    }
    s[0]=1;s[5]=s[4];
    for(int c=0;c<3;++c) {s[c+1]=fields[n+nodes*c];s[c+9]=s[c+6];}
    for(int c=0;c<9;++c) s[c+21]=s[c+12];
    for(int c=0;c<30;++c) if(!isfinite(s[c])) {atomicExch(bad,1);return;}
    for(int c=0;c<3;++c) if(s[12+4*c]<0) {atomicExch(bad,1);return;}
    for(int c=0;c<30;++c) moments[n+nodes*c]=s[c];
}
__global__ void mean_fields(const Real *moments,Real *fields,size_t nodes,Real duration,int *bad)
{
    const size_t n=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(n>=nodes) return;
    if(moments[n+4*nodes]!=duration) {atomicExch(bad,1);return;}
    for(int c=0;c<3;++c) {
        const Real value=moments[n+nodes*(c+6)];
        if(!isfinite(value)) atomicExch(bad,1);
        fields[n+nodes*c]=value;
    }
}
int exchange(Wall &p,int axis,int side,size_t count,bool &received)
{
    received=false;
    if(p.sizes[axis]==1) {
        if(!p.periodic[axis]) return 0;
        received=true;return static_cast<int>(cudaMemcpyAsync(p.receive,p.send,count*sizeof(Real),cudaMemcpyDeviceToDevice,p.parent->stream));
    }
    const int destination=side==0?p.lower[axis]:p.upper[axis],source=side==0?p.upper[axis]:p.lower[axis];
    const size_t bytes=count*sizeof(Real);received=source!=MPI_PROC_NULL;
    int status=0;
    if(p.mode==ASTR_HALO_PINNED && destination!=MPI_PROC_NULL) {
        status=static_cast<int>(cudaMemcpyAsync(p.send_host,p.send,bytes,cudaMemcpyDeviceToHost,p.parent->stream));
        if(status==0) p.d2h+=bytes;
    }
    status=ready(p,status);if(status!=0) return status;
    Real *send=p.mode==ASTR_HALO_PINNED?p.send_host:p.send,*receive=p.mode==ASTR_HALO_PINNED?p.receive_host:p.receive;
    status=MPI_Sendrecv(send,static_cast<int>(count),real_type,destination,axis+3*side,receive,
        static_cast<int>(count),real_type,source,axis+3*side,p.comm,MPI_STATUS_IGNORE);
    if(status==MPI_SUCCESS && destination!=MPI_PROC_NULL) p.sent+=bytes;
    if(status==MPI_SUCCESS && received && p.mode==ASTR_HALO_PINNED) {
        status=static_cast<int>(cudaMemcpyAsync(p.receive,p.receive_host,bytes,cudaMemcpyHostToDevice,p.parent->stream));
        if(status==0) p.h2d+=bytes;
    }
    return ready(p,status);
}
int allocate_moments(Wall &p)
{
    int status=0;
    if(!p.moments && p.nodes>0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p.moments),30*p.nodes*sizeof(Real)));
    return agree(p.comm,p.rank,status);
}
}

extern "C" int astr_insitu_wall_create(void *parent,int communicator,const int lower[3],const int upper[3],
    const int sizes[3],const int periodic[3],int mode,int first,int nw,int layers,int curved,
    const Real material[7],const Real *geometry,void **wall)
{
    if(!parent || !lower || !upper || !sizes || !periodic || !material || !wall) return ASTR_DEVICE_INVALID_ARGUMENT;
    *wall=nullptr;auto &s=*static_cast<astr::State *>(parent);
    const MPI_Comm comm=MPI_Comm_f2c(static_cast<MPI_Fint>(communicator));int rank=0,ranks=0;
    MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);
    int status=0;int64_t parts=1;
    if(!s.fluid_ready || astr::transport_in_flight(s.transport) || nw<0 || nw>2 ||
        (first!=0 && first!=s.upper[1]) || (nw==2 && first!=0) || (layers!=3 && layers!=8) || s.upper[1]+1<layers ||
        (mode!=ASTR_HALO_PINNED && mode!=ASTR_HALO_DEVICE) || (curved!=0 && curved!=1) || (nw>0 && !geometry))
        status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) {
        if(sizes[a]<1 || (periodic[a]!=0 && periodic[a]!=1)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        for(int peer:{lower[a],upper[a]}) if(peer!=MPI_PROC_NULL && (peer<0 || peer>=ranks)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        if(periodic[a] && sizes[a]>1 && (lower[a]==MPI_PROC_NULL || upper[a]==MPI_PROC_NULL)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        if(!periodic[a] && sizes[a]==1 && (lower[a]!=MPI_PROC_NULL || upper[a]!=MPI_PROC_NULL)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        parts*=sizes[a];
    }
    if(parts!=ranks || (curved && (!periodic[0] || !periodic[2] || s.upper[0]<3 || s.upper[2]<3))) status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<7;++c) if(!(std::isfinite(material[c]) && material[c]>0)) status=ASTR_DEVICE_INVALID_ARGUMENT;
    status=agree(comm,rank,status);if(status!=0) return status;
    auto *p=new(std::nothrow) Wall;
    status=agree(comm,rank,p?0:static_cast<int>(cudaErrorMemoryAllocation));if(status!=0) {delete p;return status;}
    p->parent=&s;p->rank=rank;p->nx=s.upper[0]+1;p->nz=s.upper[2]+1;p->nw=nw;p->first=first;
    p->layers=layers;p->curved=curved!=0;p->mode=mode;p->nodes=static_cast<size_t>(p->nx)*p->nz*nw;
    for(int a=0;a<3;++a) {p->lower[a]=lower[a];p->upper[a]=upper[a];p->sizes[a]=sizes[a];p->periodic[a]=periodic[a];}
    std::copy(material,material+7,p->material);
    for(int axis:{0,2}) {
        p->capacity=std::max(p->capacity,static_cast<size_t>(std::max(5*layers,12))*nw*(axis==0?p->nz:p->nx));
        int remote=nw;
        if(sizes[axis]>1) status=MPI_Sendrecv(&nw,1,MPI_INT,lower[axis],8+axis,&remote,1,MPI_INT,upper[axis],8+axis,comm,MPI_STATUS_IGNORE);
        if(status==MPI_SUCCESS && upper[axis]!=MPI_PROC_NULL && remote!=nw) status=ASTR_DEVICE_INVALID_ARGUMENT;
        status=agree(comm,rank,status);if(status!=0) {delete p;return status;}
    }
    if(p->capacity>INT_MAX) status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(size_t n=0;n<9*p->nodes;++n) if(!std::isfinite(geometry[n])) status=ASTR_DEVICE_INVALID_ARGUMENT;
    status=agree(comm,rank,status);if(status!=0) {delete p;return status;}
    status=MPI_Comm_dup(comm,&p->comm);
    auto allocate=[&](Real **v,size_t count) {if(status==0 && count>0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(v),count*sizeof(Real)));};
    allocate(&p->state,5*layers*p->nodes);allocate(&p->geometry,9*p->nodes);allocate(&p->fields,4*p->nodes);
    allocate(&p->line,6*p->nx);allocate(&p->send,p->capacity);allocate(&p->receive,p->capacity);
    if(p->curved) allocate(&p->primitive,4*primitive_nodes(*p));
    if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->invalid),sizeof(int)));
    if(status==0 && mode==ASTR_HALO_PINNED && ranks>1 && p->capacity>0) {
        status=static_cast<int>(cudaMallocHost(reinterpret_cast<void **>(&p->send_host),p->capacity*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMallocHost(reinterpret_cast<void **>(&p->receive_host),p->capacity*sizeof(Real)));
    }
    if(status==0 && p->nodes>0) status=static_cast<int>(cudaMemcpyAsync(p->geometry,geometry,9*p->nodes*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    status=ready(*p,status);if(status!=0) {release(p);return status;}
    p->setup=9*p->nodes*sizeof(Real);p->borrowed=true;++s.sample_plans;*wall=p;return 0;
}
extern "C" int astr_insitu_wall_capture(void *wall)
{
    if(!wall) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);auto &s=*p.parent;p.valid=false;
    int status=agree(p.comm,p.rank,astr::transport_in_flight(s.transport)?ASTR_DEVICE_INVALID_ARGUMENT:0);
    if(status!=0) return status;
    status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    if(status==0 && p.nodes>0) {
        pack<<<(p.nodes+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,p.state,p.nx,p.nz,p.nw,p.first,s.upper[1],p.layers,p.nodes);
        status=static_cast<int>(cudaGetLastError());
    }
    status=ready(p,status);if(status!=0) return status;
    for(int axis:{0,2}) {
        if(p.sizes[axis]==1 && !p.periodic[axis]) continue;
        const size_t count=static_cast<size_t>(5*p.layers)*p.nw*(axis==0?p.nz:p.nx);
        if(count>0) endpoint<false><<<(count+255)/256,256,0,s.stream>>>(p.state,p.send,p.nx,p.nz,p.nw,axis,count);
        status=ready(p,static_cast<int>(cudaGetLastError()));bool received=false;
        if(status==0) status=exchange(p,axis,0,count,received);
        if(status==0 && received && count>0) endpoint<true><<<(count+255)/256,256,0,s.stream>>>(p.state,p.receive,p.nx,p.nz,p.nw,axis,count);
        status=ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));if(status!=0) return status;
    }
    Material material;std::copy(p.material,p.material+7,material.c);
    if(p.curved) {
        if(p.nodes>0) {
            status=static_cast<int>(cudaMemsetAsync(p.primitive,0,4*primitive_nodes(p)*sizeof(Real),s.stream));
            if(status==0) make_primitive<<<(p.nodes+255)/256,256,0,s.stream>>>(p.state,p.primitive,p.nx,p.nz,p.nw,p.nodes,material,p.invalid);
        }
        status=ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));if(status!=0) return status;
        int bad=0;
        status=static_cast<int>(cudaMemcpy(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);
        if(status==0 && bad) status=ASTR_DEVICE_NONPHYSICAL;
        status=agree(p.comm,p.rank,status);if(status!=0) return status;
        for(int axis:{0,2}) for(int side=0;side<2;++side) {
            const size_t count=static_cast<size_t>(12)*p.nw*(axis==0?p.nz:p.nx);
            if(count>0) halo_packet<false><<<(count+255)/256,256,0,s.stream>>>(p.primitive,p.send,p.nx,p.nz,p.nw,axis,side,count);
            status=ready(p,static_cast<int>(cudaGetLastError()));bool received=false;
            if(status==0) status=exchange(p,axis,side,count,received);
            if(status==0 && received && count>0) halo_packet<true><<<(count+255)/256,256,0,s.stream>>>(p.primitive,p.receive,p.nx,p.nz,p.nw,axis,side,count);
            status=ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));if(status!=0) return status;
        }
    }
    if(p.nodes>0) derive<<<(p.nodes+255)/256,256,0,s.stream>>>(p.state,p.primitive,p.geometry,p.fields,
        s.fields[ASTR_DEVICE_METRIC].view,p.nx,p.nz,p.nw,p.first,s.upper[1],p.layers,p.curved,p.nodes,material,p.invalid);
    status=ready(p,static_cast<int>(cudaGetLastError()));int bad=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);}
    if(status==0 && bad) status=ASTR_DEVICE_NONPHYSICAL;
    status=agree(p.comm,p.rank,status);if(status==0) {p.valid=true;++p.captures;}return status;
}
extern "C" int astr_insitu_wall_view(void *wall,int geometry,AstrDeviceView *view)
{
    if(!wall || !view || (geometry!=0 && geometry!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);if(!p.valid) return ASTR_DEVICE_NOT_ALLOCATED;
    *view=AstrDeviceView{};view->data=geometry?p.geometry:p.fields;view->rank=4;view->element_bytes=sizeof(Real);
    int64_t stride=1;
    for(int a=0;a<5;++a) {view->lower[a]=0;view->extent[a]=a==0?p.nx:a==1?p.nz:a==2?p.nw:a==3?(geometry?9:4):1;view->stride[a]=stride;stride*=view->extent[a];}
    return 0;
}
extern "C" int astr_insitu_wall_line(void *wall,int k,int side,Real *host,size_t bytes)
{
    if(!wall || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);
    if(!p.valid || k<0 || k>=p.nz || side<0 || side>=p.nw || bytes!=6*p.nx*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    selected_line<<<(p.nx+255)/256,256,0,p.parent->stream>>>(p.geometry,p.fields,p.line,p.nx,p.nz,p.nw,k,side);
    int status=astr::finish_launch(*p.parent);
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(host,p.line,bytes,cudaMemcpyDeviceToHost,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) p.observed+=bytes;return status;
}
extern "C" int astr_insitu_wall_download(void *wall,Real *host,size_t bytes)
{
    if(!wall || !host) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);if(!p.valid || bytes!=4*p.nodes*sizeof(Real)) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=0;
    if(bytes>0) status=static_cast<int>(cudaMemcpyAsync(host,p.fields,bytes,cudaMemcpyDeviceToHost,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) p.observed+=bytes;return status;
}
extern "C" int astr_insitu_wall_metrics(void *wall,int64_t values[10])
{
    if(!wall || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<Wall *>(wall);
    values[0]=device_bytes(p);values[1]=values[0];values[2]=p.send_host?2*p.capacity*sizeof(Real):0;
    values[3]=p.sent;values[4]=p.d2h;values[5]=p.h2d;values[6]=p.scalar;values[7]=p.observed;values[8]=p.captures;values[9]=p.setup;
    return 0;
}
extern "C" int astr_insitu_wall_statistics_push(void *wall,const Real weights[2],int previous)
{
    if(!wall || !weights) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);auto &s=*p.parent;
    int status=!p.valid || (previous!=0 && previous!=1) || (previous!=0)!=p.previous ||
        !std::isfinite(weights[0]) || !std::isfinite(weights[1]) || weights[0]<0 || weights[1]<0 ||
        (!p.previous && (weights[0]!=0 || weights[1]!=0))?ASTR_DEVICE_INVALID_ARGUMENT:0;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    status=allocate_moments(p);if(status!=0) return status;
    status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    if(status==0 && p.nodes>0) push_moments<<<(p.nodes+255)/256,256,0,s.stream>>>(
        p.fields,p.moments,p.nodes,weights[0],weights[1],p.previous,p.invalid);
    status=ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));int bad=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);}
    if(status==0 && bad) status=ASTR_DEVICE_NONPHYSICAL;
    status=agree(p.comm,p.rank,status);if(status==0) p.previous=true;return status;
}
extern "C" int astr_insitu_wall_statistics_pack(void *wall,Real *host,size_t bytes,const Real window[2],Real time)
{
    if(!wall || !window) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);const size_t N=p.nodes;
    if(!p.previous || bytes!=34*N*sizeof(Real) || (bytes>0 && !host)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(N==0) return 0;
    std::fill(host,host+N,window[0]);std::fill(host+N,host+2*N,window[1]);
    std::fill(host+2*N,host+3*N,time);std::fill(host+33*N,host+34*N,Real(1));
    int status=static_cast<int>(cudaMemcpyAsync(host+3*N,p.moments,30*N*sizeof(Real),cudaMemcpyDeviceToHost,p.parent->stream));
    if(status==0) status=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    if(status==0) p.observed+=30*N*sizeof(Real);return status;
}
extern "C" int astr_insitu_wall_statistics_restore(void *wall,const Real *host,size_t bytes)
{
    if(!wall) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(wall);const size_t N=p.nodes;
    int status=p.previous || bytes!=34*N*sizeof(Real) || (bytes>0 && !host)?ASTR_DEVICE_INVALID_ARGUMENT:0;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    status=allocate_moments(p);if(status!=0) return status;
    if(N>0) status=static_cast<int>(cudaMemcpyAsync(p.moments,host+3*N,30*N*sizeof(Real),cudaMemcpyHostToDevice,p.parent->stream));
    status=ready(p,status);if(status==0) p.previous=true;return status;
}
extern "C" int astr_insitu_wall_statistics_mean(void *destination,void *statistics,Real duration)
{
    if(!destination || !statistics) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<Wall *>(destination);auto &history=*static_cast<Wall *>(statistics);auto &s=*p.parent;
    int status=!p.valid || !history.previous || p.parent!=history.parent || p.nx!=history.nx || p.nz!=history.nz ||
        p.nw!=history.nw || p.first!=history.first || !(std::isfinite(duration) && duration>0)?ASTR_DEVICE_INVALID_ARGUMENT:0;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    if(status==0 && p.nodes>0) mean_fields<<<(p.nodes+255)/256,256,0,s.stream>>>(history.moments,p.fields,p.nodes,duration,p.invalid);
    status=ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));int bad=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.scalar+=sizeof(int);}
    if(status==0 && bad) status=ASTR_DEVICE_NONPHYSICAL;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    for(int axis:{0,2}) {
        if(p.sizes[axis]==1 && !p.periodic[axis]) continue;
        const size_t count=static_cast<size_t>(4)*p.nw*(axis==0?p.nz:p.nx);
        if(count>0) endpoint<false><<<(count+255)/256,256,0,s.stream>>>(p.fields,p.send,p.nx,p.nz,p.nw,axis,count);
        status=ready(p,static_cast<int>(cudaGetLastError()));bool received=false;
        if(status==0) status=exchange(p,axis,0,count,received);
        if(status==0 && received && count>0) endpoint<true><<<(count+255)/256,256,0,s.stream>>>(p.fields,p.receive,p.nx,p.nz,p.nw,axis,count);
        status=ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));if(status!=0) return status;
    }
    return 0;
}
extern "C" int astr_insitu_wall_destroy(void **wall)
{
    if(!wall) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!*wall) return 0;
    const int status=release(static_cast<Wall *>(*wall));*wall=nullptr;return status;
}

namespace {
struct SbliWall {
    astr::State *parent=nullptr;
    MPI_Comm comm=MPI_COMM_NULL;
    int rank=0,nx=0,lower=0,upper=0,parts=1,mode=0,first=0;
    bool borrowed=false;
    Real fraction=0,material[7]{},reference[3]{};
    Real *state=nullptr,*x=nullptr,*line=nullptr,*send=nullptr,*receive=nullptr,*send_host=nullptr,*receive_host=nullptr;
    int *invalid=nullptr;
    int64_t sent=0,d2h=0,h2d=0,status_d2h=0,curve_d2h=0;
};
int release_sbli(SbliWall *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    for(void *v:{static_cast<void *>(p->state),static_cast<void *>(p->x),static_cast<void *>(p->line),
        static_cast<void *>(p->send),static_cast<void *>(p->receive),static_cast<void *>(p->invalid)}) if(v) {
        const int code=static_cast<int>(cudaFree(v));if(status==0) status=code;
    }
    for(void *v:{static_cast<void *>(p->send_host),static_cast<void *>(p->receive_host)}) if(v) {
        const int code=static_cast<int>(cudaFreeHost(v));if(status==0) status=code;
    }
    if(p->comm!=MPI_COMM_NULL) {const int code=MPI_Comm_free(&p->comm);if(status==0) status=code;}
    if(p->borrowed) --p->parent->sample_plans;
    delete p;return status;
}
int sbli_ready(SbliWall &p,int status)
{
    const int synced=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    return agree(p.comm,p.rank,status!=0?status:synced);
}
__global__ void pack_sbli(AstrDeviceView q,Real *state,int nx,int first)
{
    const int n=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=14*nx) return;
    const int i=n%nx,j=(n/nx)%7,k=n/(7*nx);
    for(int c=0;c<5;++c) state[n+14*nx*c]=at(q,i,j,first+k,c+1);
}
template<bool unpack>
__global__ void sbli_endpoint(Real *state,Real *buffer,int nx)
{
    const int n=threadIdx.x;
    if(n>=70) return;
    if(unpack) state[n*nx+nx-1]=buffer[n];else buffer[n]=state[n*nx];
}
struct SbliMaterial {Real c[7],reference[3];};
__global__ void sbli_values(const Real *state,const Real *x,AstrDeviceView metrics,Real *line,int nx,int first,
    Real fraction,SbliMaterial material,int *bad)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nx) return;
    Real output[4]{};
    for(int k=0;k<2;++k) {
        Real u[7],temperature[7],wall_pressure=0;
        for(int j=0;j<7;++j) {
            const int n=i+nx*(j+7*k),nodes=14*nx;
            const Real rho=state[n];
            if(!(isfinite(rho) && rho>0)) {atomicExch(bad,1);return;}
            Real velocity[3],kinetic=0;
            for(int c=0;c<3;++c) {velocity[c]=state[n+nodes*(c+1)]/rho;kinetic+=velocity[c]*velocity[c];}
            const Real pressure=(state[n+4*nodes]-Real(.5)*rho*kinetic)/material.c[2];
            u[j]=velocity[0];temperature[j]=material.c[0]*pressure/rho;
            if(j==0) wall_pressure=pressure;
            if(!isfinite(velocity[0]) || !isfinite(velocity[1]) || !isfinite(velocity[2]) ||
                !(isfinite(pressure) && pressure>0 && isfinite(temperature[j]) && temperature[j]>0)) {atomicExch(bad,1);return;}
        }
        const Real w[7]={-147,360,-450,400,-225,72,-10};
        Real du=0,dt=0;
        for(int j=0;j<7;++j) {du+=w[j]*(u[j]-u[0]);dt+=w[j]*(temperature[j]-temperature[0]);}
        const Real metric=at(metrics,i,0,first+k,2,2);
        du=((Real(1)/60)*du)*metric;dt=((Real(1)/60)*dt)*metric;
        const Real temp=temperature[0];
        const Real mu=(temp*sqrt(temp))*material.c[6]/((temp+material.c[5])*material.c[3]);
        const Real rinf=material.reference[0],uinf=material.reference[1],pinf=material.reference[2];
        const Real cf=(mu*du)/(Real(.5)*rinf*uinf*uinf);
        const Real heat=((mu/material.c[4])/material.c[1])*dt/((rinf*uinf)*(uinf*uinf));
        const Real weight=k==0?1-fraction:fraction;
        const Real values[4]={x[i],wall_pressure/pinf,cf,heat};
        for(int c=0;c<4;++c) output[c]+=weight*values[c];
    }
    for(int c=0;c<4;++c) {if(!isfinite(output[c])) atomicExch(bad,1);line[4*i+c]=output[c];}
}
}

extern "C" int astr_insitu_sbli_wall_create(void *parent,int communicator,int lower,int upper,int parts,int mode,
    int first,Real fraction,const Real material[7],const Real reference[3],const Real *x,void **sample)
{
    if(!parent || !material || !reference || !x || !sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    *sample=nullptr;auto &s=*static_cast<astr::State *>(parent);
    const MPI_Comm comm=MPI_Comm_f2c(static_cast<MPI_Fint>(communicator));int rank=0,ranks=0;
    MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&ranks);
    int status=!s.fluid_ready || astr::transport_in_flight(s.transport) || s.upper[1]<6 || first<0 || first>=s.upper[2] ||
        parts!=ranks || (mode!=ASTR_HALO_PINNED && mode!=ASTR_HALO_DEVICE) ||
        !std::isfinite(fraction) || fraction<0 || fraction>1?ASTR_DEVICE_INVALID_ARGUMENT:0;
    for(int peer:{lower,upper}) if(peer!=MPI_PROC_NULL && (peer<0 || peer>=ranks)) status=ASTR_DEVICE_INVALID_ARGUMENT;
    if(parts==1 && (lower!=MPI_PROC_NULL || upper!=MPI_PROC_NULL)) status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<7;++c) if(!(std::isfinite(material[c]) && material[c]>0)) status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<3;++c) if(!(std::isfinite(reference[c]) && reference[c]>0)) status=ASTR_DEVICE_INVALID_ARGUMENT;
    for(int i=0;i<=s.upper[0];++i) if(!std::isfinite(x[i])) status=ASTR_DEVICE_INVALID_ARGUMENT;
    status=agree(comm,rank,status);if(status!=0) return status;
    auto *p=new(std::nothrow) SbliWall;
    status=agree(comm,rank,p?0:static_cast<int>(cudaErrorMemoryAllocation));if(status!=0) {delete p;return status;}
    p->parent=&s;p->rank=rank;p->nx=s.upper[0]+1;p->lower=lower;p->upper=upper;p->parts=parts;
    p->mode=mode;p->first=first;p->fraction=fraction;
    std::copy(material,material+7,p->material);std::copy(reference,reference+3,p->reference);
    status=MPI_Comm_dup(comm,&p->comm);
    auto allocate=[&](Real **v,size_t count) {if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(v),count*sizeof(Real)));};
    allocate(&p->state,70*static_cast<size_t>(p->nx));allocate(&p->x,p->nx);allocate(&p->line,4*static_cast<size_t>(p->nx));
    allocate(&p->send,70);allocate(&p->receive,70);
    if(status==0) status=static_cast<int>(cudaMalloc(reinterpret_cast<void **>(&p->invalid),sizeof(int)));
    if(mode==ASTR_HALO_PINNED && ranks>1) {
        if(status==0) status=static_cast<int>(cudaMallocHost(reinterpret_cast<void **>(&p->send_host),70*sizeof(Real)));
        if(status==0) status=static_cast<int>(cudaMallocHost(reinterpret_cast<void **>(&p->receive_host),70*sizeof(Real)));
    }
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(p->x,x,p->nx*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    status=sbli_ready(*p,status);if(status!=0) {release_sbli(p);return status;}
    ++s.sample_plans;p->borrowed=true;*sample=p;return 0;
}
extern "C" int astr_insitu_sbli_wall_capture(void *sample,Real *line,size_t bytes)
{
    if(!sample || !line) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<SbliWall *>(sample);auto &s=*p.parent;
    int status=bytes!=4*static_cast<size_t>(p.nx)*sizeof(Real) || astr::transport_in_flight(s.transport)?ASTR_DEVICE_INVALID_ARGUMENT:0;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    pack_sbli<<<(14*p.nx+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,p.state,p.nx,p.first);
    status=sbli_ready(p,static_cast<int>(cudaGetLastError()));if(status!=0) return status;
    if(p.parts>1) {
        sbli_endpoint<false><<<1,128,0,s.stream>>>(p.state,p.send,p.nx);
        status=static_cast<int>(cudaGetLastError());
        if(status==0 && p.mode==ASTR_HALO_PINNED && p.lower!=MPI_PROC_NULL) {
            status=static_cast<int>(cudaMemcpyAsync(p.send_host,p.send,70*sizeof(Real),cudaMemcpyDeviceToHost,s.stream));
            if(status==0) p.d2h+=70*sizeof(Real);
        }
        status=sbli_ready(p,status);if(status!=0) return status;
        Real *send=p.mode==ASTR_HALO_PINNED?p.send_host:p.send,*receive=p.mode==ASTR_HALO_PINNED?p.receive_host:p.receive;
        status=MPI_Sendrecv(send,70,real_type,p.lower,1,receive,70,real_type,p.upper,1,p.comm,MPI_STATUS_IGNORE);
        if(status==MPI_SUCCESS && p.lower!=MPI_PROC_NULL) p.sent+=70*sizeof(Real);
        if(status==MPI_SUCCESS && p.upper!=MPI_PROC_NULL) {
            if(p.mode==ASTR_HALO_PINNED) {
                status=static_cast<int>(cudaMemcpyAsync(p.receive,p.receive_host,70*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
                if(status==0) p.h2d+=70*sizeof(Real);
            }
            if(status==0) sbli_endpoint<true><<<1,128,0,s.stream>>>(p.state,p.receive,p.nx);
        }
        status=sbli_ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));if(status!=0) return status;
    }
    status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    SbliMaterial material;std::copy(p.material,p.material+7,material.c);std::copy(p.reference,p.reference+3,material.reference);
    if(status==0) sbli_values<<<(p.nx+255)/256,256,0,s.stream>>>(p.state,p.x,s.fields[ASTR_DEVICE_METRIC].view,
        p.line,p.nx,p.first,p.fraction,material,p.invalid);
    status=sbli_ready(p,status!=0?status:static_cast<int>(cudaGetLastError()));int bad=0;
    if(status==0) {status=static_cast<int>(cudaMemcpy(&bad,p.invalid,sizeof(int),cudaMemcpyDeviceToHost));p.status_d2h+=sizeof(int);}
    if(status==0 && bad) status=ASTR_DEVICE_NONPHYSICAL;
    status=agree(p.comm,p.rank,status);if(status!=0) return status;
    status=static_cast<int>(cudaMemcpyAsync(line,p.line,bytes,cudaMemcpyDeviceToHost,s.stream));
    status=sbli_ready(p,status);if(status==0) p.curve_d2h+=bytes;return status;
}
extern "C" int astr_insitu_sbli_wall_metrics(void *sample,int64_t values[8])
{
    if(!sample || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<SbliWall *>(sample);
    values[0]=(75*static_cast<int64_t>(p.nx)+140)*sizeof(Real)+sizeof(int);
    values[1]=p.send_host?140*sizeof(Real):0;values[2]=p.sent;values[3]=p.d2h;values[4]=p.h2d;
    values[5]=p.status_d2h;values[6]=p.curve_d2h;values[7]=p.nx*sizeof(Real);return 0;
}
extern "C" int astr_insitu_sbli_wall_destroy(void **sample)
{
    if(!sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!*sample) return 0;
    const int status=release_sbli(static_cast<SbliWall *>(*sample));*sample=nullptr;return status;
}

namespace {
struct ProfileMaterial {Real c[7],reference[2];};
struct ProfileWall {
    astr::State *parent=nullptr;
    ProfileMaterial material{};
    int nx=0,nz=0,offset=0,global=0;
    Real *weights=nullptr,*line=nullptr;
    int *invalid=nullptr;
    bool borrowed=false;
    int64_t captures=0,status_d2h=0,curve_d2h=0;
};
int release_profile_wall(ProfileWall *p)
{
    int status=static_cast<int>(cudaStreamSynchronize(p->parent->stream));
    void *buffers[]={p->weights,p->line,p->invalid};
    for(void *buffer:buffers) if(buffer) {
        const int freed=static_cast<int>(cudaFree(buffer));if(status==0) status=freed;
    }
    if(p->borrowed) --p->parent->sample_plans;
    delete p;return status;
}
__device__ bool profile_primitive(AstrDeviceView q,int i,int j,int k,ProfileMaterial material,
    Real velocity[3],Real &temperature)
{
    const Real rho=at(q,i,j,k);
    if(!(isfinite(rho) && rho>0)) return false;
    Real kinetic=0;
    for(int c=0;c<3;++c) {velocity[c]=at(q,i,j,k,c+2)/rho;kinetic+=velocity[c]*velocity[c];}
    const Real pressure=(at(q,i,j,k,5)-Real(.5)*rho*kinetic)/material.c[2];
    temperature=(pressure/rho)*material.c[0];
    return isfinite(velocity[0]) && isfinite(velocity[1]) && isfinite(velocity[2]) && isfinite(temperature);
}
__device__ Real profile_dx(const Real *v,int i,int last,int offset,int global)
{
    const bool left=offset==0,right=offset+last==global;
    if(left && i==0) return -Real(.5)*v[5]+2*v[4]-Real(1.5)*v[3];
    if((left && i==1) || (right && i==last-1)) return Real(.5)*(v[4]-v[2]);
    if((left && i==2) || (right && i==last-2)) return (Real(2)/3)*(v[4]-v[2])-(Real(1)/12)*(v[5]-v[1]);
    if(right && i==last) return Real(.5)*v[1]-2*v[2]+Real(1.5)*v[3];
    return Real(.75)*(v[4]-v[2])-Real(.15)*(v[5]-v[1])+(Real(1)/60)*(v[6]-v[0]);
}
__global__ void profile_wall_values(AstrDeviceView q,AstrDeviceView metric,const Real *weights,Real *line,
    int nx,int nz,int offset,int global,ProfileMaterial material,int *bad)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nx) return;
    Real span=0,shear=0,heat=0;
    for(int k=0;k<nz;++k) {
        Real u[3],t[3],normal[7],v[3],temp;
        for(int j=0;j<3;++j) {
            if(!profile_primitive(q,i,j,k,material,v,temp)) {atomicExch(bad,1);return;}
            u[j]=v[0];t[j]=temp;
        }
        for(int m=-3;m<=3;++m) {
            if((i+m<0 && offset==0) || (i+m>=nx && offset+nx-1==global)) normal[m+3]=0;
            else {
                if(!profile_primitive(q,i+m,0,k,material,v,temp)) {atomicExch(bad,1);return;}
                normal[m+3]=v[1];
            }
        }
        const Real du=(-Real(1.5)*u[0]+2*u[1]-Real(.5)*u[2])*at(metric,i,0,k,2,2);
        const Real dt=(-Real(1.5)*t[0]+2*t[1]-Real(.5)*t[2])*at(metric,i,0,k,2,2);
        const Real dv=profile_dx(normal,i,nx-1,offset,global)*at(metric,i,0,k,1,1);
        const Real mu=t[0]*sqrt(t[0])*material.c[6]/(t[0]+material.c[5])/material.c[3];
        const Real rinf=material.reference[0],uinf=material.reference[1],weight=weights[i+nx*k];
        span+=weight;
        shear+=weight*mu*(du+dv)/(Real(.5)*rinf*uinf*uinf);
        heat+=weight*((mu/material.c[4])/material.c[1])*dt/(rinf*uinf*uinf*uinf);
    }
    const Real values[3]={span,shear/span,heat/span};
    for(int c=0;c<3;++c) {line[3*i+c]=values[c];if(!isfinite(values[c])) atomicExch(bad,1);}
}
}
extern "C" int astr_insitu_profile_wall_create(void *parent,int offset,int global,const Real material[7],
    const Real reference[2],const Real *weights,void **sample)
{
    if(!parent || !sample || !material || !reference || !weights) return ASTR_DEVICE_INVALID_ARGUMENT;
    *sample=nullptr;auto &s=*static_cast<astr::State *>(parent);
    if(!s.fluid_ready || astr::transport_in_flight(s.transport)) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(s.halo<3 || s.upper[0]<4 || s.upper[1]<2 || s.upper[2]<1 || offset<0 || offset+s.upper[0]>global)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<7;++c) if(!(std::isfinite(material[c]) && material[c]>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int c=0;c<2;++c) if(!(std::isfinite(reference[c]) && reference[c]>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t nodes=static_cast<size_t>(s.upper[0]+1)*s.upper[2];
    for(size_t n=0;n<nodes;++n) if(!(std::isfinite(weights[n]) && weights[n]>0)) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *p=new(std::nothrow) ProfileWall{};if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->parent=&s;p->nx=s.upper[0]+1;p->nz=s.upper[2];p->offset=offset;p->global=global;
    for(int c=0;c<7;++c) p->material.c[c]=material[c];
    for(int c=0;c<2;++c) p->material.reference[c]=reference[c];
    int status=static_cast<int>(cudaMalloc(&p->weights,nodes*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->line,3*static_cast<size_t>(p->nx)*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->invalid,sizeof(int)));
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(p->weights,weights,nodes*sizeof(Real),cudaMemcpyHostToDevice,s.stream));
    const int synced=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0) status=synced;
    if(status!=0) {release_profile_wall(p);return status;}
    ++s.sample_plans;p->borrowed=true;*sample=p;return 0;
}
extern "C" int astr_insitu_profile_wall_capture(void *sample,Real *line,size_t bytes)
{
    if(!sample || !line) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &p=*static_cast<ProfileWall *>(sample);auto &s=*p.parent;
    if(bytes!=3*static_cast<size_t>(p.nx)*sizeof(Real) || astr::transport_in_flight(s.transport)) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=static_cast<int>(cudaMemsetAsync(p.invalid,0,sizeof(int),s.stream));
    if(status==0) {
        profile_wall_values<<<(p.nx+255)/256,256,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
            s.fields[ASTR_DEVICE_METRIC].view,p.weights,p.line,p.nx,p.nz,p.offset,p.global,p.material,p.invalid);
        status=static_cast<int>(cudaGetLastError());
    }
    int invalid=0;
    if(status==0) {
        status=static_cast<int>(cudaMemcpyAsync(&invalid,p.invalid,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
        if(status==0) p.status_d2h+=sizeof(int);
    }
    const int synced=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0) status=synced;
    if(status!=0) return status;
    if(invalid) return ASTR_DEVICE_NONPHYSICAL;
    status=static_cast<int>(cudaMemcpyAsync(line,p.line,bytes,cudaMemcpyDeviceToHost,s.stream));
    const int finished=static_cast<int>(cudaStreamSynchronize(s.stream));if(status==0) status=finished;
    if(status==0) {p.curve_d2h+=bytes;++p.captures;}
    return status;
}
extern "C" int astr_insitu_profile_wall_metrics(void *sample,int64_t values[5])
{
    if(!sample || !values) return ASTR_DEVICE_INVALID_ARGUMENT;
    const auto &p=*static_cast<ProfileWall *>(sample);
    values[0]=sizeof(Real)*static_cast<int64_t>(p.nx)*(p.nz+3)+sizeof(int);
    values[1]=sizeof(Real)*static_cast<int64_t>(p.nx)*p.nz;values[2]=p.captures;
    values[3]=p.status_d2h;values[4]=p.curve_d2h;return 0;
}
extern "C" int astr_insitu_profile_wall_destroy(void **sample)
{
    if(!sample || !*sample) return ASTR_DEVICE_INVALID_ARGUMENT;
    auto *p=static_cast<ProfileWall *>(*sample);*sample=nullptr;return release_profile_wall(p);
}
