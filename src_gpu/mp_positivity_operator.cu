#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <new>
#include "device_state_internal.cuh"
#include "halo_transport.h"
#include "mp_positivity_operator.h"
#include "mp_positivity_math.cuh"
#include "upwind_operator.h"

namespace astr {
struct MpFailure {int code=0,point[3]{};numerical_real values[5]{};};
struct MpPositivityPlan {
    MPI_Comm communicator=MPI_COMM_NULL;
    int lower[3]{},upper[3]{},mode=ASTR_HALO_PINNED;
    numerical_real *send[2]{},*receive[2]{},*send_host[2]{},*receive_host[2]{};
    MpFailure *failure=nullptr;
    int *limited=nullptr;
};

int release_mp_positivity(MpPositivityPlan *plan)
{
    if(!plan) return 0;
    int status=0;
    for(int side=0;side<2;++side) {
        for(void *p:{plan->send[side],plan->receive[side]}) if(p) {
            const int next=cudaFree(p);if(!status) status=next;
        }
        for(void *p:{plan->send_host[side],plan->receive_host[side]}) if(p) {
            const int next=cudaFreeHost(p);if(!status) status=next;
        }
    }
    if(plan->failure) {const int next=cudaFree(plan->failure);if(!status) status=next;}
    if(plan->limited) {const int next=cudaFree(plan->limited);if(!status) status=next;}
    if(plan->communicator!=MPI_COMM_NULL) {
        const int next=MPI_Comm_free(&plan->communicator);if(!status) status=next;
    }
    delete plan;
    return status;
}
}

namespace {
using Real=astr::numerical_real;
using astr::State;
using astr::at;
using astr::MpFailure;
using astr::MpPositivityPlan;
constexpr int threads=256;
struct Box {int upper[3],low[3],high[3],first[3],last[3];};
__device__ void fail(MpFailure *failure,int code,const int p[3],const Real q[5])
{
    if(atomicCAS(&failure->code,0,code)==0) {
        for(int a=0;a<3;++a) failure->point[a]=p[a];
        for(int c=0;c<5;++c) failure->values[c]=q[c];
    }
}
__device__ void physical_point(size_t id,Box box,int p[3])
{
    p[0]=id%(box.upper[0]+1);p[1]=(id/(box.upper[0]+1))%(box.upper[1]+1);
    p[2]=id/(static_cast<size_t>(box.upper[0]+1)*(box.upper[1]+1));
}
__device__ bool inside(const int p[3],Box box)
{
    for(int a=0;a<3;++a) if(p[a]<box.first[a] || p[a]>box.last[a]) return false;
    return true;
}
__global__ void low_difference(AstrDeviceView q,AstrDeviceView metric,AstrDeviceView jac,
    AstrDeviceView high,AstrDeviceView delta,Box box,int axis,Real gamma,int nx,int ny,size_t count,MpFailure *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    int p[3]={static_cast<int>(id%nx),static_cast<int>((id/nx)%ny),static_cast<int>(id/(static_cast<size_t>(nx)*ny))};
    --p[axis];
    if((box.low[axis] && p[axis]<0) || (box.high[axis] && p[axis]>=box.upper[axis])) {
        for(int c=1;c<=5;++c) {at(high,p[0],p[1],p[2],c)=0;at(delta,p[0],p[1],p[2],c,axis+1)=0;}
        return;
    }
    int left[3]={p[0],p[1],p[2]},right[3]={p[0],p[1],p[2]};++right[axis];
    if(box.low[axis]) {left[axis]=max(0,left[axis]);right[axis]=max(0,right[axis]);}
    if(box.high[axis]) {left[axis]=min(box.upper[axis],left[axis]);right[axis]=min(box.upper[axis],right[axis]);}
    Real ql[5],qr[5],ul[3],ur[3],normal[3];
    for(int c=0;c<5;++c) {ql[c]=at(q,left[0],left[1],left[2],c+1);qr[c]=at(q,right[0],right[1],right[2],c+1);}
    if(!astr::mp_admissible(ql)) {fail(failure,1,p,ql);return;}
    if(!astr::mp_admissible(qr)) {fail(failure,1,p,qr);return;}
    for(int a=0;a<3;++a) {
        normal[a]=Real(.5)*(at(jac,left[0],left[1],left[2])*at(metric,left[0],left[1],left[2],axis+1,a+1)
            +at(jac,right[0],right[1],right[2])*at(metric,right[0],right[1],right[2],axis+1,a+1));
        ul[a]=ql[a+1]/ql[0];ur[a]=qr[a+1]/qr[0];
    }
    const Real pl=(gamma-Real(1))*(ql[4]-Real(.5)*((ql[1]*ul[0]+ql[2]*ul[1])+ql[3]*ul[2]));
    const Real pr=(gamma-Real(1))*(qr[4]-Real(.5)*((qr[1]*ur[0]+qr[2]*ur[1])+qr[3]*ur[2]));
    const Real cl=sqrt(gamma*pl/ql[0]),cr=sqrt(gamma*pr/qr[0]);
    const Real vl=(normal[0]*ul[0]+normal[1]*ul[1])+normal[2]*ul[2];
    const Real vr=(normal[0]*ur[0]+normal[1]*ur[1])+normal[2]*ur[2];
    const Real length=sqrt((normal[0]*normal[0]+normal[1]*normal[1])+normal[2]*normal[2]);
    const Real alpha=fmax(fabs(vl)+cl*length,fabs(vr)+cr*length);
    Real low[5],high_values[5];
    for(int c=0;c<5;++c) low[c]=Real(.5)*(vl*ql[c]+vr*qr[c]-alpha*(qr[c]-ql[c]));
    for(int a=0;a<3;++a) low[a+1]+=Real(.5)*(pl+pr)*normal[a];
    low[4]+=Real(.5)*(vl*pl+vr*pr);
    for(int c=0;c<5;++c) {
        high_values[c]=at(high,p[0],p[1],p[2],c+1);
        at(delta,p[0],p[1],p[2],c+1,axis+1)=high_values[c]-low[c];
    }
    for(int c=0;c<5;++c) {
        if(!isfinite(low[c])) fail(failure,1,p,low);
        if(!isfinite(high_values[c])) fail(failure,1,p,high_values);
    }
}

// kind=0 reconciles shared interface fluxes; kind=1 takes the minimum theta
// at shared physical nodes. Ratio face halos use the common hm-layer transport.
template<bool unpack> __global__ void planes(AstrDeviceView high,AstrDeviceView delta,AstrDeviceView ratio,
    Box box,int axis,int kind,int n1,size_t count,Real *lower,Real *upper,bool has_lower,bool has_upper)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    int p[3]={};p[(axis+1)%3]=id%n1;p[(axis+2)%3]=id/n1;
    if(kind==0) {
        for(int side=0;side<2;++side) {
            if(unpack && !(side==0 ? has_lower : has_upper)) continue;
            p[axis]=unpack ? (side==0 ? -1 : box.upper[axis]) : (side==0 ? 0 : box.upper[axis]-1);
            Real *packet=side==0 ? lower : upper;
            for(int c=1;c<=5;++c) {
                if(unpack) {
                    at(high,p[0],p[1],p[2],c)=packet[id+(c-1)*count];
                    at(delta,p[0],p[1],p[2],c,axis+1)=packet[id+(c+4)*count];
                } else {
                    packet[id+(c-1)*count]=at(high,p[0],p[1],p[2],c);
                    packet[id+(c+4)*count]=at(delta,p[0],p[1],p[2],c,axis+1);
                }
            }
        }
    } else {
        for(int side=0;side<2;++side) {
            if(unpack && !(side==0 ? has_lower : has_upper)) continue;
            p[axis]=side==0 ? 0 : box.upper[axis];
            Real *packet=side==0 ? lower : upper;
            if(unpack) at(ratio,p[0],p[1],p[2])=fmin(at(ratio,p[0],p[1],p[2]),packet[id]);
            else packet[id]=at(ratio,p[0],p[1],p[2]);
        }
    }
}
__global__ void add_high(AstrDeviceView high,AstrDeviceView rhs,Box box,int axis,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    int p[3],left[3];physical_point(id,box,p);if(!inside(p,box)) return;
    for(int a=0;a<3;++a) left[a]=p[a];--left[axis];
    for(int c=1;c<=5;++c) at(rhs,p[0],p[1],p[2],c)-=
        at(high,p[0],p[1],p[2],c)-at(high,left[0],left[1],left[2],c);
}
__global__ void ones(Real *data,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id<count) data[id]=1;
}
__global__ void ratios(AstrDeviceView q,AstrDeviceView saved,AstrDeviceView rhs,AstrDeviceView jac,
    AstrDeviceView delta,AstrDeviceView ratio,Box box,Real dt,Real a,Real b,Real c,int stage,
    size_t count,MpFailure *failure,int *limited)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    int p[3];physical_point(id,box,p);if(!inside(p,box)) return;
    Real base[5],residual[5],correction[5];
    const Real volume=at(jac,p[0],p[1],p[2]);
    for(int m=0;m<5;++m) base[m]=at(q,p[0],p[1],p[2],m+1);
    if(!(isfinite(volume) && volume>0)) {fail(failure,2,p,base);return;}
    for(int m=0;m<5;++m) residual[m]=at(rhs,p[0],p[1],p[2],m+1);
    for(int axis=0;axis<3;++axis) {
        int left[3]={p[0],p[1],p[2]};--left[axis];
        for(int m=0;m<5;++m) residual[m]=residual[m]+at(delta,p[0],p[1],p[2],m+1,axis+1)
            -at(delta,left[0],left[1],left[2],m+1,axis+1);
    }
    const Real scale=c*dt/volume;
    for(int m=0;m<5;++m) base[m]=stage==1 ? base[m]+scale*residual[m]
        : (a*at(saved,p[0],p[1],p[2],m+1)+b*base[m]*volume+c*dt*residual[m])/volume;
    if(!astr::mp_admissible(base)) {fail(failure,2,p,base);return;}
    Real theta=1;
    for(int axis=0;axis<3;++axis) {
        int left[3]={p[0],p[1],p[2]};--left[axis];
        for(int m=0;m<5;++m) correction[m]=Real(6)*scale*at(delta,left[0],left[1],left[2],m+1,axis+1);
        Real face=astr::mp_face_budget(base,correction);
        if(face<0) {fail(failure,2,p,base);return;}theta=fmin(theta,face);
        for(int m=0;m<5;++m) correction[m]=-Real(6)*scale*at(delta,p[0],p[1],p[2],m+1,axis+1);
        face=astr::mp_face_budget(base,correction);
        if(face<0) {fail(failure,2,p,base);return;}theta=fmin(theta,face);
    }
    at(ratio,p[0],p[1],p[2])=theta;
    if(theta<Real(1)) atomicAdd(limited,1);
}
__global__ void correct(AstrDeviceView delta,AstrDeviceView ratio,AstrDeviceView rhs,Box box,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    int p[3];physical_point(id,box,p);if(!inside(p,box)) return;
    const Real center=at(ratio,p[0],p[1],p[2]);
    for(int axis=0;axis<3;++axis) {
        int left[3]={p[0],p[1],p[2]},right[3]={p[0],p[1],p[2]};--left[axis];++right[axis];
        const Real tl=fmin(center,at(ratio,left[0],left[1],left[2]));
        const Real tr=fmin(center,at(ratio,right[0],right[1],right[2]));
        if(tl<Real(1) || tr<Real(1)) for(int m=1;m<=5;++m) {
            // Match the two contracted updates in the original NVHPC kernel.
            const Real updated=fma(Real(1)-tr,at(delta,p[0],p[1],p[2],m,axis+1),at(rhs,p[0],p[1],p[2],m));
            at(rhs,p[0],p[1],p[2],m)=fma(-(Real(1)-tl),at(delta,left[0],left[1],left[2],m,axis+1),updated);
        }
    }
}
__global__ void check_update(AstrDeviceView q,AstrDeviceView saved,AstrDeviceView rhs,AstrDeviceView jac,
    Box box,Real dt,Real a,Real b,Real c,int stage,size_t count,MpFailure *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*threads+threadIdx.x;
    if(id>=count) return;
    int p[3];physical_point(id,box,p);const Real volume=at(jac,p[0],p[1],p[2]);
    Real trial[5];
    for(int m=0;m<5;++m) trial[m]=stage==1 ?
        (at(q,p[0],p[1],p[2],m+1)*volume+at(rhs,p[0],p[1],p[2],m+1)*dt)/volume
        : (a*at(saved,p[0],p[1],p[2],m+1)+b*at(q,p[0],p[1],p[2],m+1)*volume
            +c*at(rhs,p[0],p[1],p[2],m+1)*dt)/volume;
    if(!astr::mp_admissible(trial)) fail(failure,3,p,trial);
}

Box box_for(const State &s,const int first[3],const int last[3],const int *low=nullptr,const int *high=nullptr)
{
    Box box{};
    for(int a=0;a<3;++a) {
        box.upper[a]=s.upper[a];box.first[a]=first[a];box.last[a]=last[a];
        if(low) box.low[a]=low[a];if(high) box.high[a]=high[a];
    }
    return box;
}
size_t nodes(const State &s) {return static_cast<size_t>(s.upper[0]+1)*(s.upper[1]+1)*(s.upper[2]+1);}
int check_failure(State &s,const char *phase)
{
    MpFailure local{};MpPositivityPlan &plan=*s.mp_positivity;
    int status=cudaMemcpyAsync(&local,plan.failure,sizeof(local),cudaMemcpyDeviceToHost,s.stream);
    if(status) return status;
    status=cudaStreamSynchronize(s.stream);if(status) return status;
    int global=0;status=MPI_Allreduce(&local.code,&global,1,MPI_INT,MPI_MAX,plan.communicator);
    if(status || global==0) return status;
    if(local.code) {
        int rank=0;MPI_Comm_rank(plan.communicator,&rank);
        std::fprintf(stderr,"ASTR_CPP_MP_POSITIVITY_FAILURE rank=%d phase=%s status=%d point=%d,%d,%d values=%.17g %.17g %.17g %.17g %.17g\n",
            rank,phase,local.code,local.point[0],local.point[1],local.point[2],double(local.values[0]),
            double(local.values[1]),double(local.values[2]),double(local.values[3]),double(local.values[4]));
    }
    return ASTR_DEVICE_NONPHYSICAL;
}
int exchange_planes(State &s,int axis,int kind,Box box)
{
    MpPositivityPlan &plan=*s.mp_positivity;
    const int n1=s.upper[(axis+1)%3]+1;
    const size_t count=static_cast<size_t>(n1)*(s.upper[(axis+2)%3]+1),values=count*(kind==0 ? 10 : 1);
    if(values>INT_MAX) return ASTR_DEVICE_INVALID_ARGUMENT;
    const size_t bytes=values*sizeof(Real);
    const auto high=s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,delta=s.fields[ASTR_DEVICE_MP_DELTA].view,ratio=s.fields[ASTR_DEVICE_MP_RATIO].view;
    planes<false><<<(count+threads-1)/threads,threads,0,s.stream>>>(high,delta,ratio,box,axis,kind,n1,count,
        plan.send[0],plan.send[1],false,false);
    int status=astr::finish_launch(s);if(status) return status;
    if(plan.mode==ASTR_HALO_PINNED) for(int side=0;side<2;++side) {
        status=cudaMemcpyAsync(plan.send_host[side],plan.send[side],bytes,cudaMemcpyDeviceToHost,s.stream);
        if(status) return status;
    }
    status=cudaStreamSynchronize(s.stream);if(status) return status;
    Real *send[2],*receive[2];
    for(int side=0;side<2;++side) {
        send[side]=plan.mode==ASTR_HALO_PINNED ? plan.send_host[side] : plan.send[side];
        receive[side]=plan.mode==ASTR_HALO_PINNED ? plan.receive_host[side] : plan.receive[side];
    }
    const MPI_Datatype type=sizeof(Real)==sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
    const int tag=21401+2*axis;
    MPI_Request request[4]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
    status=MPI_Irecv(receive[0],int(values),type,plan.lower[axis],tag+1,plan.communicator,&request[0]);if(status) return status;
    status=MPI_Irecv(receive[1],int(values),type,plan.upper[axis],tag,plan.communicator,&request[1]);if(status) return status;
    status=MPI_Isend(send[0],int(values),type,plan.lower[axis],tag,plan.communicator,&request[2]);if(status) return status;
    status=MPI_Isend(send[1],int(values),type,plan.upper[axis],tag+1,plan.communicator,&request[3]);if(status) return status;
    status=MPI_Waitall(4,request,MPI_STATUSES_IGNORE);if(status) return status;
    const bool has[2]={plan.lower[axis]!=MPI_PROC_NULL,plan.upper[axis]!=MPI_PROC_NULL};
    if(plan.mode==ASTR_HALO_PINNED) for(int side=0;side<2;++side) if(has[side]) {
        status=cudaMemcpyAsync(plan.receive[side],plan.receive_host[side],bytes,cudaMemcpyHostToDevice,s.stream);
        if(status) return status;
    }
    planes<true><<<(count+threads-1)/threads,threads,0,s.stream>>>(high,delta,ratio,box,axis,kind,n1,count,
        plan.receive[0],plan.receive[1],has[0],has[1]);
    return astr::finish_launch(s);
}
}

extern "C" int astr_device_mp_positivity_prepare(void *handle,int communicator,const int lower[3],const int upper[3],int mode)
{
    if(!handle || !lower || !upper || (mode!=ASTR_HALO_PINNED && mode!=ASTR_HALO_DEVICE)) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(s.mp_positivity) return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.rhs_ready || !s.fields[ASTR_DEVICE_INTERFACE_FLUX].view.data || s.upwind_storage!=0) return ASTR_DEVICE_NOT_ALLOCATED;
    auto *plan=new(std::nothrow) MpPositivityPlan;
    if(!plan) return int(cudaErrorMemoryAllocation);
    s.mp_positivity=plan;plan->mode=mode;
    int status=MPI_Comm_dup(MPI_Comm_f2c(static_cast<MPI_Fint>(communicator)),&plan->communicator);
    if(status) return status;
    size_t capacity=0;
    for(int a=0;a<3;++a) {
        plan->lower[a]=lower[a];plan->upper[a]=upper[a];
        capacity=std::max(capacity,10*static_cast<size_t>(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1));
    }
    if(capacity>INT_MAX) return ASTR_DEVICE_INVALID_ARGUMENT;
    status=astr::allocate_field(s.fields[ASTR_DEVICE_MP_DELTA],s.upper,1,5,5,3);if(status) return status;
    status=astr::allocate_field(s.fields[ASTR_DEVICE_MP_RATIO],s.upper,s.halo,4);if(status) return status;
    for(int side=0;side<2;++side) {
        status=cudaMalloc(&plan->send[side],capacity*sizeof(Real));if(status) return status;
        status=cudaMalloc(&plan->receive[side],capacity*sizeof(Real));if(status) return status;
        if(mode==ASTR_HALO_PINNED) {
            status=cudaMallocHost(&plan->send_host[side],capacity*sizeof(Real));if(status) return status;
            status=cudaMallocHost(&plan->receive_host[side],capacity*sizeof(Real));if(status) return status;
        }
    }
    status=cudaMalloc(&plan->failure,sizeof(MpFailure));if(status) return status;
    return cudaMalloc(&plan->limited,sizeof(int));
}
extern "C" int astr_device_mp_positivity_begin(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.mp_positivity) return ASTR_DEVICE_NOT_ALLOCATED;
    return cudaMemsetAsync(s.mp_positivity->failure,0,sizeof(MpFailure),s.stream);
}
extern "C" int astr_device_mp_positivity_axis(void *handle,int axis,int high_order,Real gamma,Real mach,Real weight,
    const int low[3],const int high[3],const int first[3],const int last[3])
{
    if(!handle || axis<0 || axis>2 || !first || !last) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.mp_positivity) return ASTR_DEVICE_NOT_ALLOCATED;
    int status=astr_device_upwind_positivity_faces(handle,axis,high_order,gamma,mach,weight,low,high);if(status) return status;
    const Box box=box_for(s,first,last,low,high);
    int shape[3]={s.upper[0]+1,s.upper[1]+1,s.upper[2]+1};++shape[axis];
    const size_t faces=static_cast<size_t>(shape[0])*shape[1]*shape[2];
    low_difference<<<(faces+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,
        s.fields[ASTR_DEVICE_METRIC].view,s.fields[ASTR_DEVICE_JACOBIAN].view,
        s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,s.fields[ASTR_DEVICE_MP_DELTA].view,
        box,axis,gamma,shape[0],shape[1],faces,s.mp_positivity->failure);
    status=astr::finish_launch(s);if(status) return status;
    status=exchange_planes(s,axis,0,box);if(status) return status;
    add_high<<<(nodes(s)+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,
        s.fields[ASTR_DEVICE_RHS].view,box,axis,nodes(s));
    status=astr::finish_launch(s);if(status) return status;
    if(high_order) return astr_device_upwind_high_order_boundary_rhs(handle,axis,low,high);
    return 0;
}
extern "C" int astr_device_mp_positivity_faces_complete(void *handle)
{
    if(!handle) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);
    if(!s.mp_positivity) return ASTR_DEVICE_NOT_ALLOCATED;
    return check_failure(s,"face input/flux");
}
extern "C" int astr_device_mp_positivity_limit(void *handle,Real dt,Real a,Real b,Real c,int stage,
    const int first[3],const int last[3],int *limited)
{
    if(!handle || !first || !last || !limited || stage<1 || stage>3 || !std::isfinite(dt) || dt<=0 ||
        !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || std::min(a,std::min(b,c))<0) return ASTR_DEVICE_INVALID_ARGUMENT;
    State &s=*static_cast<State *>(handle);if(!s.mp_positivity) return ASTR_DEVICE_NOT_ALLOCATED;
    MpPositivityPlan &plan=*s.mp_positivity;
    const auto ratio=s.fields[ASTR_DEVICE_MP_RATIO].view;
    const size_t ratio_count=s.fields[ASTR_DEVICE_MP_RATIO].bytes/sizeof(Real);
    ones<<<(ratio_count+threads-1)/threads,threads,0,s.stream>>>(static_cast<Real *>(ratio.data),ratio_count);
    int status=astr::finish_launch(s);if(status) return status;
    status=cudaMemsetAsync(plan.failure,0,sizeof(MpFailure),s.stream);if(status) return status;
    status=cudaMemsetAsync(plan.limited,0,sizeof(int),s.stream);if(status) return status;
    const Box box=box_for(s,first,last);
    ratios<<<(nodes(s)+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_SAVE].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_JACOBIAN].view,s.fields[ASTR_DEVICE_MP_DELTA].view,
        ratio,box,dt,a,b,c,stage,nodes(s),plan.failure,plan.limited);
    status=astr::finish_launch(s);if(status) return status;
    status=check_failure(s,"complete low-order baseline");if(status) return status;
    for(int axis=0;axis<3;++axis) {status=exchange_planes(s,axis,1,box);if(status) return status;}
    for(int axis=0;axis<3;++axis) {status=astr_device_exchange_axis(handle,ASTR_DEVICE_MP_RATIO,axis,0);if(status) return status;}
    correct<<<(nodes(s)+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_MP_DELTA].view,ratio,
        s.fields[ASTR_DEVICE_RHS].view,box,nodes(s));
    status=astr::finish_launch(s);if(status) return status;
    check_update<<<(nodes(s)+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_Q].view,s.fields[ASTR_DEVICE_SAVE].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_JACOBIAN].view,box,dt,a,b,c,stage,nodes(s),plan.failure);
    status=astr::finish_launch(s);if(status) return status;
    status=check_failure(s,"complete limited RK update");if(status) return status;
    int local=0;status=cudaMemcpyAsync(&local,plan.limited,sizeof(local),cudaMemcpyDeviceToHost,s.stream);if(status) return status;
    status=cudaStreamSynchronize(s.stream);if(status) return status;
    return MPI_Allreduce(&local,limited,1,MPI_INT,MPI_SUM,plan.communicator);
}
