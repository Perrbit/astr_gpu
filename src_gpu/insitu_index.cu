#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <cmath>
#include <climits>
#include <new>
#include "insitu_index.h"
#include "device_state_internal.cuh"
#include "../src/insitu_vortex_math.h"
using Real=astr::numerical_real;
const MPI_Datatype index_mpi_type=sizeof(Real)==sizeof(float)?MPI_FLOAT:MPI_DOUBLE;

namespace astr {
struct IndexPlan {
    State *parent=nullptr;
    Buffer geometry;
    Real *neighborhood=nullptr,*result=nullptr,*coordinates=nullptr,*send=nullptr,*receive=nullptr;
    Real *send_host=nullptr,*receive_host=nullptr;
    int *indices=nullptr,*positions=nullptr,*invalid=nullptr;
    int nodes=0,outputs=0,raw_count=0,field_count=0,raw[12]{},fields[36]{},capacity=0,mode=0;
    AstrDeviceView moments{};
    int source=0,owned[3]{};
    AstrDeviceView selected_indices{};
    int selected_fields[5]{},selected_count=0;
    Real const2=0,const6=0;
    Real weights[49]{};
    int64_t d2h=0,h2d=0,sent=0;
};
}
namespace {
using astr::IndexPlan;
using astr::State;
using astr::at;
struct Selection {int nodes,outputs,raw_count,field_count,raw[12],fields[36];Real weights[49];};
int synchronize(IndexPlan &p,int status=0)
{
    const int finished=static_cast<int>(cudaStreamSynchronize(p.parent->stream));
    return status!=0 ? status : finished;
}
int agree(int communicator,int status)
{
    const auto comm=MPI_Comm_f2c(static_cast<MPI_Fint>(communicator));
    int rank=0,any=0,bad=status!=0;
    int code=MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
    if(code!=MPI_SUCCESS || !any) return code;
    MPI_Comm_rank(comm,&rank);int candidate=bad ? rank : INT_MAX,first=INT_MAX;
    code=MPI_Allreduce(&candidate,&first,1,MPI_INT,MPI_MIN,comm);
    if(code==MPI_SUCCESS) code=MPI_Bcast(&status,1,MPI_INT,first,comm);
    return code==MPI_SUCCESS ? status : code;
}
template<class T> int free_device(T *&p)
{
    if(!p) return 0;
    const int status=static_cast<int>(cudaFree(p));p=nullptr;return status;
}
int free_host(Real *&p)
{
    if(!p) return 0;
    const int status=static_cast<int>(cudaFreeHost(p));p=nullptr;return status;
}
void retain_error(int &first,int status) {if(first==0) first=status;}
int end_exchange(IndexPlan &p)
{
    int status=synchronize(p);
    retain_error(status,free_device(p.send));retain_error(status,free_device(p.receive));
    retain_error(status,free_device(p.indices));retain_error(status,free_device(p.positions));
    retain_error(status,free_host(p.send_host));retain_error(status,free_host(p.receive_host));
    p.capacity=0;return status;
}
int end_group(IndexPlan &p)
{
    int status=end_exchange(p);
    retain_error(status,free_device(p.neighborhood));retain_error(status,free_device(p.result));
    retain_error(status,free_device(p.coordinates));retain_error(status,free_device(p.invalid));
    p.nodes=0;p.outputs=0;p.raw_count=0;p.field_count=0;
    p.source=0;p.moments=AstrDeviceView{};
    return status;
}
IndexPlan *plan(void *parent) {return parent ? static_cast<State *>(parent)->index : nullptr;}
Selection selection(const IndexPlan &p)
{
    Selection s{};s.nodes=p.nodes;s.outputs=p.outputs;s.raw_count=p.raw_count;s.field_count=p.field_count;
    for(int i=0;i<12;++i) s.raw[i]=p.raw[i];
    for(int i=0;i<36;++i) s.fields[i]=p.fields[i];
    for(int i=0;i<49;++i) s.weights[i]=p.weights[i];
    return s;
}
struct Sources {
    AstrDeviceView density,velocity,pressure,temperature,tv,species,moments,selected_indices,q;
    int source,owned[3],selected_fields[5],selected_count;
    Real const2,const6;
};
Sources sources(const IndexPlan &p)
{
    const auto &s=*p.parent;
    Sources out{};
    out.density=s.fields[ASTR_DEVICE_RHO].view;out.velocity=s.fields[ASTR_DEVICE_VELOCITY].view;
    out.pressure=s.fields[ASTR_DEVICE_PRESSURE].view;out.temperature=s.fields[ASTR_DEVICE_TEMPERATURE].view;
    out.tv=s.fields[ASTR_DEVICE_AIR5_TV].view;out.species=s.fields[ASTR_DEVICE_AIR5_SPECIES].view;
    out.moments=p.moments;out.selected_indices=p.selected_indices;out.q=s.fields[ASTR_DEVICE_Q].view;out.source=p.source;
    for(int a=0;a<3;++a) out.owned[a]=p.owned[a];
    for(int a=0;a<5;++a) out.selected_fields[a]=p.selected_fields[a];
    out.selected_count=p.selected_count;out.const2=p.const2;out.const6=p.const6;return out;
}
__device__ Real source_value(const Sources &s,int i,int j,int k,int field)
{
    if(s.source>=5) {
        const int64_t key=i+static_cast<int64_t>(s.owned[0]+1)*(j+static_cast<int64_t>(s.owned[1]+1)*k);
        const auto *indices=static_cast<const int *>(s.selected_indices.data);
        const auto *values=static_cast<const Real *>(s.moments.data);
        const int64_t column=s.moments.stride[1];
        int lo=0,hi=s.moments.extent[0]-1;Real mean=NAN;
        const bool favre=s.source==6 || s.source==8;
        while(lo<=hi) {
            const int mid=lo+(hi-lo)/2;
            const int64_t current=indices[3*mid]+static_cast<int64_t>(s.owned[0]+1)*
                (indices[3*mid+1]+static_cast<int64_t>(s.owned[1]+1)*indices[3*mid+2]);
            if(current<key) lo=mid+1;
            else if(current>key) hi=mid-1;
            else {
                if(field==1 && !favre) mean=values[mid+2*column]/values[mid+column];
                else for(int c=0;c<s.selected_count;++c) if(s.selected_fields[c]==field)
                    mean=values[mid+(3+s.selected_count*(favre?2:1)+c)*column];
                break;
            }
        }
        if(s.source<=6) return mean;
        const Real rho=at(s.q,i,j,k);
        if(field==1) return rho-mean;
        if(field<=4) return at(s.q,i,j,k,field)/rho-mean;
        const Real u=at(s.q,i,j,k,2)/rho,v=at(s.q,i,j,k,3)/rho,w=at(s.q,i,j,k,4)/rho;
        const Real pressure=(at(s.q,i,j,k,5)-Real(.5)*rho*(u*u+v*v+w*w))/s.const6;
        return (field==5?pressure:(pressure/rho)*s.const2)-mean;
    }
    if(s.source!=0) {
        const size_t n=i+static_cast<size_t>(s.owned[0])*(j+static_cast<size_t>(s.owned[1])*k);
        const auto *values=static_cast<const Real *>(s.moments.data);
        const size_t column=s.moments.stride[1];
        if(field==1) return values[n+5*column]/values[n+4*column];
        return values[n+(field+4+(s.source==2?3:0))*column];
    }
    if(field==1) return at(s.density,i,j,k);
    if(field>=2 && field<=4) return at(s.velocity,i,j,k,field-1);
    if(field==5) return at(s.pressure,i,j,k);
    if(field==7) return at(s.tv,i,j,k);
    if(field>=8 && field<=12) return at(s.species,i,j,k,field-7);
    return at(s.temperature,i,j,k);
}
__global__ void gather(Sources src,Selection s,const int *indices,const int *positions,int count,Real *result)
{
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=count) return;
    const int i=indices[3*node],j=indices[3*node+1],k=indices[3*node+2];
    for(int m=0;m<s.raw_count;++m) result[(positions ? positions[node]-1 : node)+
        static_cast<size_t>(positions ? s.nodes : count)*m]=source_value(src,i,j,k,s.raw[m]);
}
__global__ void scatter(const Real *packet,const int *positions,int count,int raw_count,int nodes,Real *result)
{
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=count) return;
    for(int m=0;m<raw_count;++m) result[positions[node]-1+static_cast<size_t>(nodes)*m]=packet[node+static_cast<size_t>(count)*m];
}
__device__ Real material(Real t,AstrIndexRegion r)
{
    if(r.nondimen) return t*sqrt(t)*r.tempconst1/(t+r.tempconst)/r.reynolds;
    const Real ratio=t/Real(273.15);
    return Real(1.716e-5)*ratio*sqrt(ratio)*(Real(273.15)+Real(110.4))/(t+Real(110.4));
}
__global__ void evaluate(Sources src,AstrDeviceView metric,AstrDeviceView geometry,Selection s,AstrIndexRegion r,
    const Real *neighborhood,Real *result,Real *coordinates,int *invalid,int im,int jm,int km)
{
    const int node=blockIdx.x*blockDim.x+threadIdx.x;
    const int n=r.extent[0]*r.extent[1]*r.extent[2];if(node>=n) return;
    const int point[3]={r.first[0]+node%r.extent[0],r.first[1]+(node/r.extent[0])%r.extent[1],
        r.first[2]+node/(r.extent[0]*r.extent[1])};
    int local[3],stencil[3],offset[3],center=0;const int last[3]={im,jm,km};
    bool derived=false,thermal_needed=false,wall=false,wall_mean=false;int vortex_mask=0;
    for(int m=0;m<s.field_count;++m) wall_mean=wall_mean || (src.source==5 && s.fields[m]>=31 && s.fields[m]<=34);
    for(int m=0;m<s.field_count;++m) {
        const int f=s.fields[m];derived=derived || (!wall_mean && ((f>=13 && f<=33) || f>=35));
        thermal_needed=thermal_needed || (f>=28 && f<=30) || f==33;
        wall=wall || (!wall_mean && f>=31 && f<=34);
        if(f==35) vortex_mask|=1;if(f==36) vortex_mask|=2;
    }
    for(int a=0;a<3;++a) {
        local[a]=point[a]-r.origin[a];stencil[a]=point[a]-3;
        if(!r.periodic[a]) stencil[a]=min(max(stencil[a],0),r.intervals[a]-6);
        offset[a]=point[a]-stencil[a];
    }
    Real directional[5][3]{},density_gradient[3]{},thermal[3]{},diagnostic[18]{};
    astr_insitu::VortexReal gradient[9]{};
    if(derived) {
        for(int a=0;a<3;++a) for(int k=0;k<7;++k) {
            int sample[3];for(int d=0;d<3;++d) sample[d]=point[d]-r.lower[d];
            sample[a]=stencil[a]+k-r.lower[a];
            const int position=sample[0]+r.shape[0]*(sample[1]+r.shape[1]*sample[2]);
            for(int m=0;m<s.raw_count;++m) {
                const int field=s.raw[m],c=field==6 && thermal_needed ? 5 : field;
                if(c>5 || (c==5 && field!=6)) continue;
                directional[c-1][a]+=s.weights[k+7*offset[a]]*neighborhood[position+static_cast<size_t>(s.nodes)*m];
            }
        }
        for(int d=0;d<3;++d) {
            Real coefficients[3];
            for(int a=0;a<3;++a) {
                int sample[3];for(int c=0;c<3;++c) sample[c]=r.cartesian_support ? min(max(local[c],0),last[c]) : local[c];
                sample[a]=local[a];coefficients[a]=at(metric,sample[0],sample[1],sample[2],a+1,d+1);
            }
            for(int c=0;c<3;++c) for(int a=0;a<3;++a) gradient[c+3*d]+=directional[c+1][a]*coefficients[a];
            for(int a=0;a<3;++a) {density_gradient[d]+=directional[0][a]*coefficients[a];thermal[d]+=directional[4][a]*coefficients[a];}
        }
    }
    for(int c=0;c<9;++c) diagnostic[c]=gradient[c];
    Real trace_square=0;
    for(int d=0;d<3;++d) for(int c=0;c<3;++c) trace_square+=gradient[c+3*d]*gradient[d+3*c];
    diagnostic[9]=-Real(.5)*trace_square;diagnostic[10]=gradient[0]+gradient[4]+gradient[8];
    diagnostic[11]=gradient[5]-gradient[7];diagnostic[12]=gradient[6]-gradient[2];diagnostic[13]=gradient[1]-gradient[3];
    diagnostic[14]=sqrt(density_gradient[0]*density_gradient[0]+density_gradient[1]*density_gradient[1]+density_gradient[2]*density_gradient[2]);
    for(int c=0;c<3;++c) diagnostic[15+c]=thermal[c];
    const int sample[3]={point[0]-r.lower[0],point[1]-r.lower[1],point[2]-r.lower[2]};
    center=sample[0]+r.shape[0]*(sample[1]+r.shape[1]*sample[2]);
    Real mu=0,heat=0,shear_x=0,shear_z=0;
    if(wall) {
        Real t=0;for(int m=0;m<s.raw_count;++m) if(s.raw[m]==6) t=neighborhood[center+static_cast<size_t>(s.nodes)*m];
        bool valid=isfinite(t) && t>0;
        if(r.nondimen) valid=valid && isfinite(r.tempconst) && isfinite(r.tempconst1) && isfinite(r.reynolds)
            && t+r.tempconst>0 && r.tempconst1>0 && r.reynolds>0;
        if(thermal_needed) valid=valid && isfinite(r.prandtl) && r.prandtl>0 &&
            (r.nondimen ? isfinite(r.const5) && r.const5>0 : isfinite(r.cp) && r.cp>0);
        if(valid) mu=material(t,r);
        valid=valid && isfinite(mu) && mu>0;Real kappa=0;
        if(valid && thermal_needed) kappa=r.nondimen ? (mu/r.prandtl)/r.const5 : r.cp*mu/r.prandtl;
        if(!valid || !isfinite(mu) || mu<=0 || (thermal_needed && (!isfinite(kappa) || kappa<=0))) atomicExch(invalid,1);
        shear_x=mu*(gradient[3]+gradient[1]);shear_z=mu*(gradient[5]+gradient[7]);heat=kappa*thermal[1];
        if(!isfinite(shear_x) || !isfinite(shear_z) || !isfinite(heat)) atomicExch(invalid,1);
        if(derived) {
            for(int c=0;c<9;++c) if(!isfinite(gradient[c])) atomicExch(invalid,1);
            for(int c=0;c<3;++c) if(!isfinite(thermal[c])) atomicExch(invalid,1);
        }
    }
    astr_insitu::VortexValues vortex{};
    if(vortex_mask) {
        vortex=astr_insitu::vortex_values(gradient,(vortex_mask&1)!=0,(vortex_mask&2)!=0);
        if(((vortex_mask&1) && vortex.lambda2_status!=astr_insitu::VortexOk) ||
           ((vortex_mask&2) && vortex.lambda_ci_status!=astr_insitu::VortexOk)) atomicExch(invalid,1);
    }
    for(int m=0;m<s.field_count;++m) {
        const int f=s.fields[m];Real value=0;
        if(wall_mean) {
            for(int c=0;c<s.raw_count;++c) if(s.raw[c]==f) value=neighborhood[center+static_cast<size_t>(s.nodes)*c];
        }
        else if(f<=12) {
            if(src.source==0) value=source_value(src,local[0],local[1],local[2],f);
            else for(int c=0;c<s.raw_count;++c) if(s.raw[c]==f)
                value=neighborhood[center+static_cast<size_t>(s.nodes)*c];
        }
        else if(f<=30) value=diagnostic[f-13];
        else if(f==31) value=shear_x;else if(f==32) value=shear_z;else if(f==33) value=heat;
        else if(f==34) value=mu;else if(f==35) value=vortex.lambda2;else if(f==36) value=vortex.lambda_ci;
        result[node+static_cast<size_t>(s.outputs)*m]=value;
    }
    if(coordinates) for(int d=0;d<3;++d) {
        int p[3];for(int a=0;a<3;++a) p[a]=r.cartesian_support ? min(max(local[a],0),last[a]) : local[a];
        p[d]=local[d];coordinates[3*node+d]=at(geometry,p[0],p[1],p[2],d+1);
    }
}
int upload_indices(IndexPlan &p,int nodes,const int *indices,const int *positions,int *&idx,int *&pos)
{
    int status=0;
    if(indices) {
        status=static_cast<int>(cudaMalloc(&idx,3*static_cast<size_t>(nodes)*sizeof(int)));
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(idx,indices,3*static_cast<size_t>(nodes)*sizeof(int),cudaMemcpyHostToDevice,p.parent->stream));
    }
    if(positions && status==0) {
        status=static_cast<int>(cudaMalloc(&pos,static_cast<size_t>(nodes)*sizeof(int)));
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(pos,positions,static_cast<size_t>(nodes)*sizeof(int),cudaMemcpyHostToDevice,p.parent->stream));
    }
    return status;
}
}

int astr::release_insitu_index(IndexPlan *p)
{
    if(!p) return 0;
    int status=end_group(*p);
    if(p->geometry.view.data) retain_error(status,static_cast<int>(cudaFree(p->geometry.view.data)));
    delete p;return status;
}
extern "C" int astr_insitu_index_begin(void *parent,int nodes,int output_nodes,int raw_count,const int *raw,
    int field_count,const int *fields,const Real weights[49],int geometry,const Real *xyz,size_t xyz_bytes)
{
    if(!parent || nodes<0 || output_nodes<0 || raw_count<0 || raw_count>12 || field_count<1 || field_count>36)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(parent);
    if(!s.fluid_ready || astr::transport_in_flight(s.transport)) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int primitives=s.air5_ready ? 12 : 6;
    for(int i=0;i<raw_count;++i) if(raw[i]<1 || (raw[i]>primitives && (raw[i]<31 || raw[i]>34)))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int i=0;i<field_count;++i) if(fields[i]<1 || fields[i]>36 || (fields[i]>primitives && fields[i]<=12))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.index) {s.index=new(std::nothrow) IndexPlan{};if(!s.index) return static_cast<int>(cudaErrorMemoryAllocation);s.index->parent=&s;}
    IndexPlan &p=*s.index;int status=end_group(p);if(status!=0) return status;
    p.d2h=0;p.h2d=0;p.sent=0;p.nodes=nodes>0 ? nodes : 1;p.outputs=output_nodes>0 ? output_nodes : 1;
    p.raw_count=raw_count;p.field_count=field_count;
    for(int m=0;m<raw_count;++m) p.raw[m]=raw[m];for(int m=0;m<field_count;++m) p.fields[m]=fields[m];
    for(int i=0;i<49;++i) p.weights[i]=weights[i];
    status=static_cast<int>(cudaMalloc(&p.neighborhood,static_cast<size_t>(p.nodes)*(raw_count>0 ? raw_count : 1)*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMalloc(&p.result,static_cast<size_t>(p.outputs)*field_count*sizeof(Real)));
    if(status==0) status=static_cast<int>(cudaMalloc(&p.invalid,sizeof(int)));
    if(status==0 && geometry) status=static_cast<int>(cudaMalloc(&p.coordinates,3*static_cast<size_t>(p.outputs)*sizeof(Real)));
    bool geometry_created=false;
    if(status==0 && geometry && !p.geometry.view.data) {
        status=astr::allocate_field(p.geometry,s.upper,s.halo,4,3);
        geometry_created=p.geometry.view.data!=nullptr;
        if(status==0 && (!xyz || xyz_bytes!=p.geometry.bytes)) status=ASTR_DEVICE_INVALID_ARGUMENT;
        if(status==0) status=static_cast<int>(cudaMemcpyAsync(p.geometry.view.data,xyz,xyz_bytes,cudaMemcpyHostToDevice,s.stream));
    }
    status=synchronize(p,status);
    if(status!=0 && geometry_created) {
        const int freed=static_cast<int>(cudaFree(p.geometry.view.data));p.geometry=astr::Buffer{};
        retain_error(status,freed);
    }
    return status;
}
extern "C" int astr_insitu_index_fill(void *parent,int nodes,const int *indices,const int *positions)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;if(nodes==0) return 0;
    int *idx=nullptr,*pos=nullptr;int status=upload_indices(*p,nodes,indices,positions,idx,pos);
    if(status==0) {gather<<<(nodes+255)/256,256,0,p->parent->stream>>>(sources(*p),selection(*p),idx,pos,nodes,p->neighborhood);status=static_cast<int>(cudaGetLastError());}
    status=synchronize(*p,status);retain_error(status,free_device(idx));retain_error(status,free_device(pos));return status;
}
extern "C" int astr_insitu_index_sample(void *parent,int nodes,const int *indices,Real *values)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;if(nodes==0 || p->raw_count==0) return 0;
    int *idx=nullptr,*unused=nullptr;Real *packet=nullptr;int status=upload_indices(*p,nodes,indices,nullptr,idx,unused);
    const size_t bytes=static_cast<size_t>(nodes)*p->raw_count*sizeof(Real);
    if(status==0) status=static_cast<int>(cudaMalloc(&packet,bytes));
    if(status==0) {gather<<<(nodes+255)/256,256,0,p->parent->stream>>>(sources(*p),selection(*p),idx,nullptr,nodes,packet);status=static_cast<int>(cudaGetLastError());}
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(values,packet,bytes,cudaMemcpyDeviceToHost,p->parent->stream));
    status=synchronize(*p,status);retain_error(status,free_device(idx));retain_error(status,free_device(packet));return status;
}
extern "C" int astr_insitu_index_receive(void *parent,int nodes,const int *positions,const Real *values)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;if(nodes==0 || p->raw_count==0) return 0;
    int *unused=nullptr,*pos=nullptr;Real *packet=nullptr;int status=upload_indices(*p,nodes,nullptr,positions,unused,pos);
    const size_t bytes=static_cast<size_t>(nodes)*p->raw_count*sizeof(Real);
    if(status==0) status=static_cast<int>(cudaMalloc(&packet,bytes));
    if(status==0) status=static_cast<int>(cudaMemcpyAsync(packet,values,bytes,cudaMemcpyHostToDevice,p->parent->stream));
    if(status==0) {scatter<<<(nodes+255)/256,256,0,p->parent->stream>>>(packet,pos,nodes,p->raw_count,p->nodes,p->neighborhood);status=static_cast<int>(cudaGetLastError());}
    status=synchronize(*p,status);retain_error(status,free_device(pos));retain_error(status,free_device(packet));return status;
}
extern "C" int astr_insitu_index_exchange_prepare(void *parent,int capacity,int mode)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;
    if(capacity<1 || (mode!=0 && mode!=1) || static_cast<int64_t>(capacity)*p->raw_count>INT_MAX) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=end_exchange(*p);if(status!=0) return status;p->mode=mode;p->capacity=capacity;
    const size_t bytes=static_cast<size_t>(capacity)*p->raw_count*sizeof(Real);
    status=static_cast<int>(cudaMalloc(&p->send,bytes));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->receive,bytes));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->indices,3*static_cast<size_t>(capacity)*sizeof(int)));
    if(status==0) status=static_cast<int>(cudaMalloc(&p->positions,static_cast<size_t>(capacity)*sizeof(int)));
    if(status==0 && mode==0) status=static_cast<int>(cudaMallocHost(&p->send_host,bytes));
    if(status==0 && mode==0) status=static_cast<int>(cudaMallocHost(&p->receive_host,bytes));
    return status;
}
extern "C" int astr_insitu_index_exchange(void *parent,int sends,const int *indices,int receives,const int *positions,
    int destination,int source,int communicator)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;
    int status=sends<0 || receives<0 || sends>p->capacity || receives>p->capacity ? ASTR_DEVICE_INVALID_ARGUMENT : 0;
    const size_t send_bytes=static_cast<size_t>(sends)*p->raw_count*sizeof(Real),receive_bytes=static_cast<size_t>(receives)*p->raw_count*sizeof(Real);
    if(status==0 && sends>0) {
        status=static_cast<int>(cudaMemcpyAsync(p->indices,indices,3*static_cast<size_t>(sends)*sizeof(int),cudaMemcpyHostToDevice,p->parent->stream));
        if(status==0) {gather<<<(sends+255)/256,256,0,p->parent->stream>>>(sources(*p),selection(*p),p->indices,nullptr,sends,p->send);status=static_cast<int>(cudaGetLastError());}
        if(status==0 && p->mode==0) {status=static_cast<int>(cudaMemcpyAsync(p->send_host,p->send,send_bytes,cudaMemcpyDeviceToHost,p->parent->stream));if(status==0) p->d2h+=send_bytes;}
    }
    status=agree(communicator,synchronize(*p,status));if(status!=0) return status;
    status=MPI_Sendrecv(p->mode==0 ? p->send_host : p->send,sends*p->raw_count,index_mpi_type,destination,1,
        p->mode==0 ? p->receive_host : p->receive,receives*p->raw_count,index_mpi_type,source,1,
        MPI_Comm_f2c(static_cast<MPI_Fint>(communicator)),MPI_STATUS_IGNORE);
    if(status!=MPI_SUCCESS) return status;
    if(destination!=MPI_PROC_NULL) p->sent+=send_bytes;
    if(receives>0 && source!=MPI_PROC_NULL) {
        status=static_cast<int>(cudaMemcpyAsync(p->positions,positions,static_cast<size_t>(receives)*sizeof(int),cudaMemcpyHostToDevice,p->parent->stream));
        if(status==0 && p->mode==0) {status=static_cast<int>(cudaMemcpyAsync(p->receive,p->receive_host,receive_bytes,cudaMemcpyHostToDevice,p->parent->stream));if(status==0) p->h2d+=receive_bytes;}
        if(status==0) {scatter<<<(receives+255)/256,256,0,p->parent->stream>>>(p->receive,p->positions,receives,p->raw_count,p->nodes,p->neighborhood);status=static_cast<int>(cudaGetLastError());}
    }
    return agree(communicator,synchronize(*p,status));
}
extern "C" int astr_insitu_index_exchange_end(void *parent)
{
    auto *p=plan(parent);return p ? end_exchange(*p) : 0;
}
extern "C" int astr_insitu_index_evaluate(void *parent,const AstrIndexRegion *region)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;if(!region) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int nodes=region->extent[0]*region->extent[1]*region->extent[2];if(nodes==0) return 0;
    if(nodes!=p->outputs) return ASTR_DEVICE_INVALID_ARGUMENT;
    int status=static_cast<int>(cudaMemsetAsync(p->invalid,0,sizeof(int),p->parent->stream));
    if(status==0) {evaluate<<<(nodes+255)/256,256,0,p->parent->stream>>>(sources(*p),p->parent->fields[ASTR_DEVICE_METRIC].view,
        p->geometry.view,selection(*p),*region,p->neighborhood,p->result,p->coordinates,p->invalid,
        p->parent->upper[0],p->parent->upper[1],p->parent->upper[2]);status=static_cast<int>(cudaGetLastError());}
    status=synchronize(*p,status);int invalid=0;
    if(status==0) status=static_cast<int>(cudaMemcpy(&invalid,p->invalid,sizeof(int),cudaMemcpyDeviceToHost));
    return status!=0 ? status : (invalid ? ASTR_DEVICE_NONPHYSICAL : 0);
}
extern "C" int astr_insitu_index_download(void *parent,int field,Real *values,size_t bytes)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;if(bytes==0) return 0;
    if(field<0 || field>2) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t expected=field==1 ? static_cast<size_t>(p->nodes)*p->raw_count*sizeof(Real) :
        static_cast<size_t>(p->outputs)*(field==2 ? 3 : p->field_count)*sizeof(Real);
    if(!values || bytes!=expected) return ASTR_DEVICE_INVALID_ARGUMENT;
    const Real *source=field==1 ? p->neighborhood : (field==2 ? p->coordinates : p->result);
    if(!source) return ASTR_DEVICE_NOT_ALLOCATED;
    return static_cast<int>(cudaMemcpy(values,source,bytes,cudaMemcpyDeviceToHost));
}
extern "C" int astr_insitu_index_view(void *parent,void **values,void **coordinates)
{
    auto *p=plan(parent);if(!p || !values || !coordinates) return ASTR_DEVICE_NOT_ALLOCATED;
    *values=p->result;*coordinates=p->coordinates;return 0;
}
extern "C" int astr_insitu_index_usage(void *parent,int64_t values[3])
{
    auto *p=plan(parent);if(!p || !values) return ASTR_DEVICE_NOT_ALLOCATED;
    values[0]=p->d2h;values[1]=p->h2d;values[2]=p->sent;return 0;
}
extern "C" int astr_insitu_index_source(void *parent,int coordinates,AstrDeviceView *view)
{
    auto *p=plan(parent);if(!p || !view) return ASTR_DEVICE_NOT_ALLOCATED;
    if(coordinates!=0 && coordinates!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    void *data=coordinates ? p->coordinates : p->result;
    if(!data) return ASTR_DEVICE_NOT_ALLOCATED;
    *view=AstrDeviceView{};view->data=data;view->rank=2;view->element_bytes=sizeof(Real);
    for(int a=0;a<5;++a) {view->lower[a]=1;view->extent[a]=1;view->stride[a]=1;}
    view->extent[0]=p->outputs;view->extent[1]=coordinates ? 3 : p->field_count;
    view->stride[0]=coordinates ? 3 : 1;view->stride[1]=coordinates ? 1 : p->outputs;
    for(int a=2;a<5;++a) view->stride[a]=static_cast<int64_t>(view->extent[0])*view->extent[1];
    return 0;
}
extern "C" int astr_insitu_index_end(void *parent)
{
    auto *p=plan(parent);return p ? end_group(*p) : 0;
}
extern "C" int astr_insitu_index_statistics(void *parent,const AstrDeviceView *state,const int owned[3],int source)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;
    if(source==0) {
        for(int i=0;i<p->raw_count;++i) if(p->raw[i]>6) return ASTR_DEVICE_INVALID_ARGUMENT;
        p->source=0;p->moments=AstrDeviceView{};return 0;
    }
    if((source!=1 && source!=2) || !state || !state->data || !owned || state->rank!=2 || state->element_bytes!=sizeof(Real) ||
        state->extent[1]!=30 || state->stride[0]!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    int64_t nodes=1;
    for(int a=0;a<3;++a) {
        if(owned[a]<1 || owned[a]>p->parent->upper[a]+1) return ASTR_DEVICE_INVALID_ARGUMENT;
        nodes*=owned[a];
    }
    if(state->extent[0]!=nodes || state->stride[1]!=nodes) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int n=0;n<p->raw_count;++n) if(p->raw[n]<1 || p->raw[n]>4 || (source==2 && p->raw[n]==1))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int n=0;n<p->field_count;++n) if((p->fields[n]>=5 && p->fields[n]<=12) || p->fields[n]>=28 ||
        (source==2 && (p->fields[n]==1 || p->fields[n]==27))) return ASTR_DEVICE_INVALID_ARGUMENT;
    p->moments=*state;p->source=source;
    for(int a=0;a<3;++a) p->owned[a]=owned[a];
    return 0;
}
extern "C" int astr_insitu_index_selected(void *parent,const AstrDeviceView *state,const AstrDeviceView *indices,
    int fields,const int *ids,const int cells[3],int source,Real const2,Real const6)
{
    auto *p=plan(parent);if(!p) return ASTR_DEVICE_NOT_ALLOCATED;
    if(!state || !indices || !state->data || !indices->data || !cells || !ids || fields<0 || fields>5 ||
        source<5 || source>8 || state->rank!=2 || state->element_bytes!=sizeof(Real) || state->stride[0]!=1 ||
        indices->rank!=2 || indices->element_bytes!=sizeof(int) || indices->extent[0]!=state->extent[0] ||
        indices->extent[1]!=3 || indices->stride[0]!=3 || indices->stride[1]!=1) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int basic=3+3*fields;
    if(state->extent[1]!=basic && !(fields>=3 && state->extent[1]==basic+12)) return ASTR_DEVICE_INVALID_ARGUMENT;
    for(int a=0;a<3;++a) if(cells[a]!=p->parent->upper[a]) return ASTR_DEVICE_INVALID_ARGUMENT;
    bool wall_group=false;
    for(int a=0;a<fields;++a) {
        if(ids[a]<2 || (ids[a]>6 && (ids[a]<31 || ids[a]>34))) return ASTR_DEVICE_INVALID_ARGUMENT;
        wall_group=wall_group || ids[a]>=31;
    }
    for(int a=0;a<p->raw_count;++a) {
        const int field=p->raw[a];bool present=field==1 && (source==5 || source==7);
        for(int c=0;c<fields;++c) present=present || field==ids[c];
        if(!present) return ASTR_DEVICE_INVALID_ARGUMENT;
    }
    if(wall_group) {
        if(source!=5) return ASTR_DEVICE_INVALID_ARGUMENT;
        for(int a=0;a<p->field_count;++a) if(p->fields[a]!=1 && p->fields[a]!=6 &&
            (p->fields[a]<31 || p->fields[a]>34)) return ASTR_DEVICE_INVALID_ARGUMENT;
    } else for(int a=0;a<p->field_count;++a) if(p->fields[a]>=31) return ASTR_DEVICE_INVALID_ARGUMENT;
    p->source=source;p->moments=*state;p->selected_indices=*indices;p->selected_count=fields;
    for(int a=0;a<5;++a) p->selected_fields[a]=a<fields?ids[a]:0;
    for(int a=0;a<3;++a) p->owned[a]=cells[a];
    p->const2=const2;p->const6=const6;return 0;
}
