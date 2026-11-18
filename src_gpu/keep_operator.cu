#include <new>
#include "keep_operator.h"
#include "compact_line_solver.h"
#include "device_state_internal.cuh"
#include "sbp63_math.cuh"

using Real=astr::numerical_real;

namespace astr {
struct KeepPlan {
    int terms=0,bases=0;
    int *left=nullptr,*right=nullptr,*component=nullptr,*basis=nullptr;
    Real *weight=nullptr;
};

int release_keep(KeepPlan *plan)
{
    if(!plan) return 0;
    int status=0;
    void *buffers[]={plan->left,plan->right,plan->component,plan->basis,plan->weight};
    for(void *buffer:buffers) if(buffer) {
        const int freed=static_cast<int>(cudaFree(buffer));
        if(status==0) status=freed;
    }
    delete plan;
    return status;
}
}

namespace {
using astr::at;
struct Fields { AstrDeviceView rho,vel,prs,jac,metric,q; };
struct BasicVolume { int types[3],first[3],last[3],upper[3]; };

Fields fields(const astr::State &s)
{
    return {s.fields[ASTR_DEVICE_RHO].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_Q].view};
}

__device__ void variables(Fields f,int i,int j,int k,int axis,Real gamma,Real *v)
{
    v[0]=at(f.rho,i,j,k);
    for(int c=0;c<3;++c) {
        v[c+1]=at(f.vel,i,j,k,c+1);
        v[c+6]=at(f.jac,i,j,k)*at(f.metric,i,j,k,axis+1,c+1);
    }
    v[4]=at(f.prs,i,j,k)/v[0]/(gamma-Real(1.0));
    v[5]=at(f.prs,i,j,k);
}

__device__ Real monomial(const Real *v,const int *powers)
{
    Real value=1;
    for(int c=0;c<9;++c) for(int p=0;p<powers[c];++p) value*=v[c];
    return value;
}

__device__ void pair(Fields f,const int *node,const int *other,int axis,Real gamma,Real *flux)
{
    Real vl[3],vr[3],velocity[3],metric[3],work[3];
    const Real pl=at(f.prs,node[0],node[1],node[2]),pr=at(f.prs,other[0],other[1],other[2]);
    Real normal=0,kinetic=0,pressure_work=0;
    for(int c=0;c<3;++c) {
        vl[c]=at(f.vel,node[0],node[1],node[2],c+1);
        vr[c]=at(f.vel,other[0],other[1],other[2],c+1);
        velocity[c]=Real(.5)*(vl[c]+vr[c]);
        metric[c]=Real(.5)*(at(f.jac,node[0],node[1],node[2])*at(f.metric,node[0],node[1],node[2],axis+1,c+1)
                     +at(f.jac,other[0],other[1],other[2])*at(f.metric,other[0],other[1],other[2],axis+1,c+1));
        work[c]=Real(.5)*(vl[c]*pr+vr[c]*pl);
        normal+=velocity[c]*metric[c];kinetic+=vl[c]*vr[c];pressure_work+=work[c]*metric[c];
    }
    const Real rl=at(f.rho,node[0],node[1],node[2]),rr=at(f.rho,other[0],other[1],other[2]);
    const Real mass=Real(.5)*(rl+rr)*normal,internal=Real(.5)*(pl/rl+pr/rr)/(gamma-1);
    flux[0]=mass;
    for(int c=0;c<3;++c) flux[c+1]=mass*velocity[c]+Real(.5)*(pl+pr)*metric[c];
    flux[4]=mass*(Real(.5)*kinetic+internal)+pressure_work;
}

__device__ void physical_flux(Fields f,const int *node,int axis,Real *flux)
{
    Real metric[3],normal=0;
    for(int c=0;c<3;++c) {
        metric[c]=at(f.jac,node[0],node[1],node[2])*at(f.metric,node[0],node[1],node[2],axis+1,c+1);
        normal+=metric[c]*at(f.vel,node[0],node[1],node[2],c+1);
    }
    const Real pressure=at(f.prs,node[0],node[1],node[2]);
    flux[0]=at(f.q,node[0],node[1],node[2],1)*normal;
    for(int c=1;c<4;++c) flux[c]=at(f.q,node[0],node[1],node[2],c+1)*normal+pressure*metric[c-1];
    flux[4]=(at(f.q,node[0],node[1],node[2],5)+pressure)*normal;
}

__global__ void sbp63_convection(Fields f,AstrDeviceView rhs,int nx,int ny,size_t count,int total,int start,Real gamma)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int node[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
    const int globalj=start+node[1];
    Real result[5]={};
    for(int axis=0;axis<3;++axis) for(int offset=-5;offset<=5;++offset) {
        const Real weight=astr::sbp63::axis_weight(axis,globalj,offset,total);
        if(weight==0) continue;
        int other[3]={node[0],node[1],node[2]};other[axis]+=offset;
        Real flux[5];pair(f,node,other,axis,gamma,flux);
        for(int c=0;c<5;++c) result[c]+=2*weight*flux[c];
    }
    if(globalj==0 || globalj==total) {
        const Real sign=globalj==0 ? -1 : 1,h=astr::sbp63::norm(globalj,total);
        Real m[3],area=0,wall[5]={},flux[5];
        for(int c=0;c<3;++c) {m[c]=at(f.jac,node[0],node[1],node[2])*at(f.metric,node[0],node[1],node[2],2,c+1);area+=m[c]*m[c];}
        area=sqrt(area);
        for(int c=0;c<3;++c) wall[c+1]=at(f.prs,node[0],node[1],node[2])*(sign*m[c]/area);
        pair(f,node,node,1,gamma,flux);
        for(int c=0;c<5;++c) result[c]+=sign*(sign*area*wall[c]-flux[c])/h;
    }
    for(int c=0;c<5;++c) at(rhs,node[0],node[1],node[2],c+1)-=result[c];
}

template<bool physical>
__global__ void explicit_kernel(Fields f,AstrDeviceView rhs,int nx,int ny,size_t count,int axis,Real gamma,BasicVolume volume)
{
    const int lane=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
    const size_t id=static_cast<size_t>(blockIdx.x)*512+lane;
    if(id>=count) return;
    const int node[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
    int distance=3;
    if(physical) {
        for(int a=0;a<3;++a) if(node[a]<volume.first[a] || node[a]>volume.last[a]) return;
        if(volume.types[axis]==1 || volume.types[axis]==4) distance=min(distance,node[axis]);
        if(volume.types[axis]==2 || volume.types[axis]==4) distance=min(distance,volume.upper[axis]-node[axis]);
    }
    const Real a[3]={Real(.75),-Real(.15),Real(1.0)/Real(60.0)};
    Real result[5]={};
    for(int r=1;r<=distance;++r) {
        int plus[3]={node[0],node[1],node[2]},minus[3]={node[0],node[1],node[2]};
        plus[axis]+=r;minus[axis]-=r;
        Real fp[5],fm[5];
        Real coefficient=2*a[r-1];
        if(physical && distance<3) {
            physical_flux(f,plus,axis,fp);physical_flux(f,minus,axis,fm);
            coefficient=distance==1 ? Real(.5) : (r==1 ? Real(2.0)/Real(3.0) : -Real(1.0)/Real(12.0));
        } else {
            pair(f,node,plus,axis,gamma,fp);pair(f,node,minus,axis,gamma,fm);
        }
        for(int c=0;c<5;++c) result[c]+=coefficient*(fp[c]-fm[c]);
    }
    for(int c=0;c<5;++c) at(rhs,node[0],node[1],node[2],c+1)-=result[c];
}

__device__ void point(int axis,int position,int line,int nx,int ny,int *node)
{
    if(axis==0) { node[0]=position;node[1]=line%ny;node[2]=line/ny; }
    else if(axis==1) { node[0]=line%nx;node[1]=position;node[2]=line/nx; }
    else { node[0]=line%nx;node[1]=line/nx;node[2]=position; }
}

__device__ Real basis_value(Fields f,astr::KeepPlan plan,AstrCompactStencil s,int row,int column,
                              int nx,int ny,int lines,Real gamma)
{
    const int index=s.start+column;
    int node[3];Real v[9];
    point(s.axis,row,index%lines,nx,ny,node);
    variables(f,node[0],node[1],node[2],s.axis,gamma,v);
    return monomial(v,plan.right+9*(index/lines));
}

__global__ void compact_rhs_kernel(Fields f,astr::KeepPlan plan,AstrCompactStencil s,int n,int nx,int ny,int lines,
                                   Real gamma,astr::numerical_real *rhs)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x,column=blockIdx.y;
    if(row>=n) return;
    const int g=s.first+row;
    Real value;
    if(!s.periodic && g==0) value=2*(basis_value(f,plan,s,row+1,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row,column,nx,ny,lines,gamma))
        +Real(.5)*(basis_value(f,plan,s,row+2,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row,column,nx,ny,lines,gamma));
    else if(!s.periodic && g==s.total) value=2*(basis_value(f,plan,s,row,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row-1,column,nx,ny,lines,gamma))
        +Real(.5)*(basis_value(f,plan,s,row,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row-2,column,nx,ny,lines,gamma));
    else if(!s.periodic && (g==1 || g==s.total-1)) value=Real(.75)*(basis_value(f,plan,s,row+1,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row-1,column,nx,ny,lines,gamma));
    else value=(Real(7.0)/Real(9.0))*(basis_value(f,plan,s,row+1,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row-1,column,nx,ny,lines,gamma))
        +(Real(1.0)/Real(36.0))*(basis_value(f,plan,s,row+2,column,nx,ny,lines,gamma)-basis_value(f,plan,s,row-2,column,nx,ny,lines,gamma));
    rhs[row+static_cast<size_t>(n)*column]=value;
}

__global__ void compact_commit_kernel(Fields f,astr::KeepPlan plan,AstrCompactStencil s,AstrDeviceView nodes,
                                      AstrDeviceView rhs,int nx,int ny,size_t count,int lines,int nrhs,Real gamma)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const int i=id%nx,j=(id/nx)%ny,k=id/(static_cast<size_t>(nx)*ny),node[3]={i,j,k};
    for(int a=0;a<3;++a) if(node[a]<s.active_first[a] || node[a]>s.active_last[a]) return;
    const int line=s.axis==0 ? j+ny*k : (s.axis==1 ? i+nx*k : i+nx*j);
    Real v[9],result[5];
    variables(f,i,j,k,s.axis,gamma,v);
    for(int c=0;c<5;++c) result[c]=at(rhs,i,j,k,c+1);
    for(int t=0;t<plan.terms;++t) {
        const int column=(plan.basis[t]-1)*lines+line-s.start;
        if(column<0 || column>=nrhs) continue;
        Real factor=plan.weight[t];
        for(int c=0;c<9;++c) for(int p=0;p<plan.left[c+9*t];++p) factor*=v[c];
        const Real df=static_cast<const astr::numerical_real *>(nodes.data)[node[s.axis]+nodes.stride[1]*column];
        result[plan.component[t]-1]-=factor*df;
    }
    for(int c=0;c<5;++c) at(rhs,i,j,k,c+1)=result[c];
}

template<class T> int copy_table(astr::State &s,T *&device,const T *host,size_t count)
{
    cudaError_t status=cudaMalloc(&device,count*sizeof(T));
    if(status!=cudaSuccess) return static_cast<int>(status);
    return static_cast<int>(cudaMemcpyAsync(device,host,count*sizeof(T),cudaMemcpyHostToDevice,s.stream));
}

int finish_compact(astr::State &s)
{
    const cudaError_t status=cudaGetLastError();
    if(status!=cudaSuccess) return static_cast<int>(status);
    return s.compact_sync || s.debug_sync ? static_cast<int>(cudaStreamSynchronize(s.stream)) : 0;
}
}

extern "C" int astr_device_keep_sbp63_convection(void *parent,int total,int start,Real gamma)
{
    if(!parent || total<11 || start<0) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.rhs_ready) return ASTR_DEVICE_NOT_ALLOCATED;
    if(start+s.upper[1]>total || s.halo<5) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);
    sbp63_convection<<<(count+255)/256,256,0,s.stream>>>(fields(s),s.fields[ASTR_DEVICE_RHS].view,
        s.upper[0]+1,s.upper[1]+1,count,total,start,gamma);
    return astr::finish_launch(s);
}

extern "C" int astr_device_keep_prepare(void *parent,int terms,int bases,const int *left,const int *right,
                                        const int *component,const int *basis,const Real *weight)
{
    if(!parent || terms<1 || bases<1) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(s.keep) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::KeepPlan *p=new(std::nothrow) astr::KeepPlan{};
    if(!p) return static_cast<int>(cudaErrorMemoryAllocation);
    p->terms=terms;p->bases=bases;
    int status=copy_table(s,p->left,left,9*static_cast<size_t>(terms));
    if(status==0) status=copy_table(s,p->right,right,9*static_cast<size_t>(bases));
    if(status==0) status=copy_table(s,p->component,component,terms);
    if(status==0) status=copy_table(s,p->basis,basis,terms);
    if(status==0) status=copy_table(s,p->weight,weight,terms);
    const int synced=static_cast<int>(cudaStreamSynchronize(s.stream));
    if(status==0) status=synced;
    if(status!=0) { astr::release_keep(p);return status; }
    s.keep=p;
    return 0;
}

extern "C" int astr_device_keep_explicit(void *parent,Real gamma)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.rhs_ready || s.halo<3) return ASTR_DEVICE_NOT_ALLOCATED;
    int status=static_cast<int>(cudaMemsetAsync(s.fields[ASTR_DEVICE_RHS].view.data,0,s.fields[ASTR_DEVICE_RHS].bytes,s.stream));
    if(status!=0) return status;
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3];
    for(int axis=0;axis<3;++axis) {
        const dim3 block=axis==0 ? dim3(512,1,1) : (axis==1 ? dim3(32,16,1) : dim3(64,1,8));
        explicit_kernel<false><<<(count+511)/512,block,0,s.stream>>>(fields(s),s.fields[ASTR_DEVICE_RHS].view,
            s.upper[0]+1,s.upper[1]+1,count,axis,gamma,BasicVolume{});
        status=astr::finish_launch(s);
        if(status!=0) return status;
    }
    return 0;
}

extern "C" int astr_device_keep_explicit_accumulate_basic(void *parent,Real gamma,const int types[3],
                                                          const int first[3],const int last[3])
{
    if(!parent || !types || !first || !last) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.rhs_ready || s.halo<3) return ASTR_DEVICE_NOT_ALLOCATED;
    BasicVolume volume{};
    for(int a=0;a<3;++a) {
        if(types[a]<1 || types[a]>4 || first[a]<0 || last[a]>s.upper[a] || first[a]>last[a])
            return ASTR_DEVICE_INVALID_ARGUMENT;
        volume.types[a]=types[a];volume.first[a]=first[a];volume.last[a]=last[a];volume.upper[a]=s.upper[a];
    }
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3];
    for(int axis=0;axis<3;++axis) {
        const dim3 block=axis==0 ? dim3(512,1,1) : (axis==1 ? dim3(32,16,1) : dim3(64,1,8));
        explicit_kernel<true><<<(count+511)/512,block,0,s.stream>>>(fields(s),s.fields[ASTR_DEVICE_RHS].view,
            s.upper[0]+1,s.upper[1]+1,count,axis,gamma,volume);
        const int status=astr::finish_launch(s);
        if(status!=0) return status;
    }
    return 0;
}

extern "C" int astr_device_keep_compact_rhs(void *parent,void *line,const AstrCompactStencil *stencil,int nrhs,Real gamma)
{
    if(!parent || !stencil) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.keep) return ASTR_DEVICE_NOT_ALLOCATED;
    AstrDeviceView rhs,solution,nodes;
    int status=astr_compact_distributed_views(line,&rhs,&solution,&nodes);
    if(status!=0) return status;
    if(nrhs<1 || nrhs>rhs.extent[1]) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int lines=(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1)/(s.upper[stencil->axis]+1);
    compact_rhs_kernel<<<dim3((rhs.extent[0]+127)/128,nrhs),128,0,s.stream>>>(fields(s),*s.keep,*stencil,
        rhs.extent[0],s.upper[0]+1,s.upper[1]+1,lines,gamma,static_cast<astr::numerical_real *>(rhs.data));
    return finish_compact(s);
}

extern "C" int astr_device_keep_compact_commit(void *parent,void *line,const AstrCompactStencil *stencil,int nrhs,Real gamma)
{
    if(!parent || !stencil) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    if(!s.keep) return ASTR_DEVICE_NOT_ALLOCATED;
    AstrDeviceView rhs,solution,nodes;
    int status=astr_compact_distributed_views(line,&rhs,&solution,&nodes);
    if(status!=0) return status;
    if(nrhs<1 || nrhs>rhs.extent[1]) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t count=s.fields[ASTR_DEVICE_RHS].view.stride[3];
    const int lines=(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1)/(s.upper[stencil->axis]+1);
    compact_commit_kernel<<<(count+255)/256,256,0,s.stream>>>(fields(s),*s.keep,*stencil,nodes,
        s.fields[ASTR_DEVICE_RHS].view,s.upper[0]+1,s.upper[1]+1,count,lines,nrhs,gamma);
    return finish_compact(s);
}
