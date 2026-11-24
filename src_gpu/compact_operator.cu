#include "compact_operator.h"
#include "compact_line_solver.h"
#include "device_state_internal.cuh"

namespace {
using astr::at;
using Real=astr::numerical_real;
struct Fields {
    AstrDeviceView q,vel,prs,tmp,metric,jac,stress,heat;
};
struct NodalBand { int first[3],last[3]; };

__device__ void point(int axis,int position,int line,int nx,int ny,int &i,int &j,int &k)
{
    if(axis==0) { i=position;j=line%ny;k=line/ny; }
    else if(axis==1) { i=line%nx;j=position;k=line/nx; }
    else { i=line%nx;j=line/nx;k=position; }
}

__device__ Real value(const Fields &f,const AstrCompactStencil &s,int position,int i,int j,int k,int m)
{
    if(s.axis==0) i=position;
    else if(s.axis==1) j=position;
    else k=position;
    if(s.kind==1) return m<=3 ? at(f.vel,i,j,k,m) : at(f.tmp,i,j,k);
    if(s.kind==4) return at(f.q,i,j,k,m+s.component_offset-1);
    Real metric[3];
    for(int b=0;b<3;++b) metric[b]=at(f.metric,i,j,k,s.axis+1,b+1);
    if(s.kind==2) {
        const Real u=metric[0]*at(f.vel,i,j,k,1)+metric[1]*at(f.vel,i,j,k,2)+metric[2]*at(f.vel,i,j,k,3);
        const Real p=at(f.prs,i,j,k),jac=at(f.jac,i,j,k);
        if(m==1) return jac*at(f.q,i,j,k,1)*u;
        if(m==5) return jac*(at(f.q,i,j,k,5)+p)*u;
        return jac*(at(f.q,i,j,k,m)*u+metric[m-2]*p);
    }
    const int tensor[3][3]={{1,2,3},{2,4,5},{3,5,6}};
    Real flux[3];
    for(int b=0;b<3;++b) flux[b]=m==4 ? at(f.heat,i,j,k,b+1) : at(f.stress,i,j,k,tensor[m-1][b]);
    return (flux[0]*metric[0]+flux[1]*metric[1]+flux[2]*metric[2])*at(f.jac,i,j,k);
}

__global__ void rhs_kernel(Fields fields,AstrCompactStencil s,int n,int nx,int ny,int lines,Real *rhs)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x,column=blockIdx.y;
    if(row>=n) return;
    const int index=s.start+column,m=index/lines+1;
    int i,j,k;
    point(s.axis,0,index%lines,nx,ny,i,j,k);
    const int g=s.first+row;
    Real result=0;
    if(s.kind!=4) {
        if(!s.periodic && g==0) result=2*(value(fields,s,row+1,i,j,k,m)-value(fields,s,row,i,j,k,m))
            +Real(.5)*(value(fields,s,row+2,i,j,k,m)-value(fields,s,row,i,j,k,m));
        else if(!s.periodic && g==s.total) result=2*(value(fields,s,row,i,j,k,m)-value(fields,s,row-1,i,j,k,m))
            +Real(.5)*(value(fields,s,row,i,j,k,m)-value(fields,s,row-2,i,j,k,m));
        else if(!s.periodic && (g==1 || g==s.total-1)) result=Real(.75)*(value(fields,s,row+1,i,j,k,m)-value(fields,s,row-1,i,j,k,m));
        else result=(Real(7)/Real(9))*(value(fields,s,row+1,i,j,k,m)-value(fields,s,row-1,i,j,k,m))
            +(Real(1)/Real(36))*(value(fields,s,row+2,i,j,k,m)-value(fields,s,row-2,i,j,k,m));
    } else if(!s.periodic && (g<=2 || s.total-g<=2)) {
        const int distance=g<=2 ? g : s.total-g;
        for(int r=0;r<=6;++r) result+=s.boundary[distance+3*r]*
            value(fields,s,g<=2 ? row-distance+r : row+distance-r,i,j,k,m);
    } else {
        const int distance=g<s.total-g ? g : s.total-g;
        const int radius=!s.periodic && distance==3 ? 3 : (!s.periodic && distance==4 ? 4 : 5);
        const Real *c=radius==3 ? s.c6 : (radius==4 ? s.c8 : s.c10);
        const Real center=value(fields,s,row,i,j,k,m);
        result=c[0]*(center+center);
        for(int r=1;r<=radius;++r) result+=c[r]*(value(fields,s,row+r,i,j,k,m)+value(fields,s,row-r,i,j,k,m));
    }
    rhs[row+static_cast<size_t>(n)*column]=result;
}

__global__ void commit_kernel(AstrCompactStencil s,AstrDeviceView nodes,AstrDeviceView q,AstrDeviceView rhs,
                              AstrDeviceView metric,AstrDeviceView dvel,AstrDeviceView dtmp,
                              int dim,int nx,int ny,int lines,bool wall_band,NodalBand inner)
{
    const int position=blockIdx.x*blockDim.x+threadIdx.x,column=blockIdx.y;
    if(position>dim) return;
    int i,j,k;
    const int index=s.start+column,m=index/lines+1;
    point(s.axis,position,index%lines,nx,ny,i,j,k);
    if(s.kind==2 && (i<s.active_first[0] || i>s.active_last[0] || j<s.active_first[1] || j>s.active_last[1] ||
                    k<s.active_first[2] || k>s.active_last[2])) return;
    if(s.kind==3 && (position<s.active_first[s.axis] || position>s.active_last[s.axis])) return;
    if(wall_band && i>=inner.first[0] && i<=inner.last[0] && j>=inner.first[1] && j<=inner.last[1] &&
       k>=inner.first[2] && k<=inner.last[2]) return;
    const Real df=static_cast<const Real *>(nodes.data)[position+nodes.stride[1]*column];
    if(s.kind==1) for(int b=1;b<=3;++b) {
        const Real contribution=df*at(metric,i,j,k,s.axis+1,b);
        astr::numerical_real &destination=m<=3 ? at(dvel,i,j,k,m,b) : at(dtmp,i,j,k,b);
        if(s.axis==0) destination=0;
        destination+=contribution;
    }
    else if(s.kind==2) at(rhs,i,j,k,m)-=df;
    else if(s.kind==3) at(rhs,i,j,k,m+1)+=df;
    else if(s.periodic || (s.first+position!=0 && s.first+position!=s.total))
        at(q,i,j,k,m+s.component_offset-1)=df;
}

int finish(astr::State &s)
{
    const cudaError_t status=cudaGetLastError();
    if(status!=cudaSuccess) return static_cast<int>(status);
    return s.compact_sync || s.debug_sync ? static_cast<int>(cudaStreamSynchronize(s.stream)) : 0;
}

__device__ void wall5_line(AstrDeviceView q,AstrWall5FilterPlan const &plan,
                           int i,int k,int m,int wall,int direction,Real (&out)[5])
{
    Real input[11];
    #pragma unroll
    for(int j=0;j<11;++j) input[j]=at(q,i,wall+direction*j,k,m);
    #pragma unroll
    for(int j=0;j<5;++j) {
        Real rhs=0;
        #pragma unroll
        for(int r=0;r<11;++r) rhs+=plan.rows[j][r]*input[r];
        out[j]=(rhs-(j>0 ? plan.alpha*out[j-1] : Real(0)))*plan.inverse[j];
    }
    #pragma unroll
    for(int j=3;j>=0;--j) out[j]-=plan.alpha*plan.inverse[j]*out[j+1];
}

__global__ void wall5_kernel(AstrDeviceView q,AstrWall5FilterPlan plan,int walls,int nx,int ny,int nz)
{
    const int line=blockIdx.x*blockDim.x+threadIdx.x;
    if(line>=nx*nz*5) return;
    const int i=line%nx,k=(line/nx)%nz,m=line/(nx*nz)+1;
    Real lower[5],upper[5];
    // Both sides see the original line, even on meshes where their input stencils overlap.
    if(walls&1) wall5_line(q,plan,i,k,m,0,1,lower);
    if(walls&2) wall5_line(q,plan,i,k,m,ny,-1,upper);
    #pragma unroll
    for(int j=1;j<=5;++j) {
        if(walls&1) at(q,i,j,k,m)=lower[j-1];
        if(walls&2) at(q,i,ny-j,k,m)=upper[j-1];
    }
}
}

extern "C" int astr_compact_filter_wall5(void *parent,const AstrWall5FilterPlan *plan,int walls)
{
    astr::State &s=*static_cast<astr::State *>(parent);
    const int nx=s.upper[0]+1,nz=s.upper[2]+1;
    constexpr int threads=128;
    wall5_kernel<<<(nx*nz*5+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,*plan,
        walls,nx,s.upper[1],nz);
    return finish(s);
}

extern "C" int astr_compact_field_rhs(void *parent,void *line,const AstrCompactStencil *stencil,int nrhs)
{
    if(!parent || !stencil || stencil->axis<0 || stencil->axis>2 || stencil->kind<1 || stencil->kind>4)
        return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    AstrDeviceView rhs,solution,nodes;
    int status=astr_compact_distributed_views(line,&rhs,&solution,&nodes);
    if(status!=0) return status;
    if(nrhs<1 || nrhs>rhs.extent[1]) return ASTR_DEVICE_INVALID_ARGUMENT;
    Fields f{s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_VELOCITY].view,s.fields[ASTR_DEVICE_PRESSURE].view,
        s.fields[ASTR_DEVICE_TEMPERATURE].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.fields[ASTR_DEVICE_STRESS].view,s.fields[ASTR_DEVICE_HEAT_FLUX].view};
    const int lines=(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1)/(s.upper[stencil->axis]+1);
    rhs_kernel<<<dim3((rhs.extent[0]+127)/128,nrhs),128,0,s.stream>>>(
        f,*stencil,rhs.extent[0],s.upper[0]+1,s.upper[1]+1,lines,static_cast<Real *>(rhs.data));
    return finish(s);
}

static int field_commit(void *parent,void *line,const AstrCompactStencil *stencil,int nrhs,bool wall_band)
{
    if(!parent || !stencil || stencil->axis<0 || stencil->axis>2) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    NodalBand inner{};
    if(wall_band) {
        if(stencil->kind!=3) return ASTR_DEVICE_INVALID_ARGUMENT;
        const int status=astr::me6_inner_bounds(s,inner.first,inner.last);
        if(status!=0) return status;
    }
    AstrDeviceView rhs,solution,nodes;
    int status=astr_compact_distributed_views(line,&rhs,&solution,&nodes);
    if(status!=0) return status;
    if(nrhs<1 || nrhs>rhs.extent[1]) return ASTR_DEVICE_INVALID_ARGUMENT;
    const int dim=s.upper[stencil->axis],lines=(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1)/(dim+1);
    commit_kernel<<<dim3((dim+128)/128,nrhs),128,0,s.stream>>>(*stencil,nodes,s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_DVELOCITY].view,
        s.fields[ASTR_DEVICE_DTEMPERATURE].view,dim,s.upper[0]+1,s.upper[1]+1,lines,wall_band,inner);
    return finish(s);
}

extern "C" int astr_compact_field_commit(void *parent,void *line,const AstrCompactStencil *stencil,int nrhs)
{
    return field_commit(parent,line,stencil,nrhs,false);
}

extern "C" int astr_compact_field_commit_wall_band(void *parent,void *line,const AstrCompactStencil *stencil,int nrhs)
{
    return field_commit(parent,line,stencil,nrhs,true);
}

extern "C" int astr_compact_rhs_clear(void *parent)
{
    if(!parent) return ASTR_DEVICE_INVALID_ARGUMENT;
    astr::State &s=*static_cast<astr::State *>(parent);
    return static_cast<int>(cudaMemsetAsync(s.fields[ASTR_DEVICE_RHS].view.data,0,s.fields[ASTR_DEVICE_RHS].bytes,s.stream));
}

extern "C" int astr_compact_configure_sync(void *parent,int explicit_sync)
{
    if(!parent || (explicit_sync!=0 && explicit_sync!=1)) return ASTR_DEVICE_INVALID_ARGUMENT;
    static_cast<astr::State *>(parent)->compact_sync=explicit_sync!=0;
    return 0;
}
