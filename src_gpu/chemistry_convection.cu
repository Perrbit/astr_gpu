#include <new>
#include "chemistry_face_math.cuh"
#include "chemistry_flux_math.cuh"
#include "chemistry_transport.h"

namespace astr {
struct Air5ConvectionPlan {
    struct Floors {double value[5];} floor{};
    AstrAir5TransportFailure *failure=nullptr;
    double *planes=nullptr;
    size_t capacity=0;
};
int release_air5_convection(Air5ConvectionPlan *plan)
{
    if(!plan)return 0;
    const int first=static_cast<int>(cudaFree(plan->failure));
    const int second=static_cast<int>(cudaFree(plan->planes));delete plan;
    return first?first:second;
}
}
namespace {
using namespace astr::air5;
using Floors=astr::Air5ConvectionPlan::Floors;
constexpr int threads=256;
struct Shape {int upper[3],type[3];bool top;size_t count;};
__device__ double &at(AstrDeviceView v,const int p[3],int c=1)
{return static_cast<double *>(v.data)[astr::offset(v,p[0],p[1],p[2],c)];}
__device__ void point(size_t id,Shape shape,int p[3])
{
    p[0]=id%(shape.upper[0]+1);p[1]=(id/(shape.upper[0]+1))%(shape.upper[1]+1);
    p[2]=id/(static_cast<size_t>(shape.upper[0]+1)*(shape.upper[1]+1));
}
__device__ bool active(const int p[3],Shape shape)
{
    if(shape.top)return p[1]==shape.upper[1] && (shape.type[1]==2 || shape.type[1]==4) &&
        !(p[0]==0 && (shape.type[0]==1 || shape.type[0]==4));
    for(int d=0;d<3;++d)
        if((p[d]==0 && (shape.type[d]==1 || shape.type[d]==4)) ||
           (p[d]==shape.upper[d] && (shape.type[d]==2 || shape.type[d]==4)))return false;
    return true;
}
__device__ void low_face(FaceFields fields,const int p[3],int axis,int face,Shape shape,double flux[11],int &status)
{
    ConvectiveLine line{fields,{p[0],p[1],p[2]},axis};
    if(shape.top && axis==0 && face==shape.upper[0] && (shape.type[0]==2 || shape.type[0]==4))line.load(p[0],flux);
    else convective_face(line,face,shape.upper[axis],shape.type[axis],false,true,true,flux,status);
}
__device__ void high_face(FaceFields fields,const int8_t *mask,const int p[3],int axis,int face,Shape shape,
    bool capturing,double flux[11],int &status)
{
    int position[3]={p[0],p[1],p[2]};position[axis]=face;
    const bool shock=capturing && air5_shock_face(mask,shape.upper,position,axis,shape.type[axis]);
    ConvectiveLine line{fields,{p[0],p[1],p[2]},axis};
    convective_face(line,face,shape.upper[axis],shape.type[axis],shock,false,true,flux,status);
}
__global__ void base_kernel(FaceFields fields,AstrDeviceView saved,AstrDeviceView rhs,AstrDeviceView base,
    Shape shape,Floors floor,double a,double b,double cdt,bool first,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=shape.count)return;
    int p[3];point(id,shape,p);if(!active(p,shape))return;
    double left[3][11]{},right[3][11]{},state[11];int status=0;
    for(int d=0;d<3;++d)if(!shape.top || d!=1) {
        low_face(fields,p,d,p[d]-1,shape,left[d],status);low_face(fields,p,d,p[d],shape,right[d],status);
    }
    const double jac=face_value(fields.jacobian,p);
    for(int c=0;c<11;++c) {
        double low_rhs=0;for(int d=0;d<3;++d)low_rhs+=left[d][c]-right[d][c];
        if(shape.top)low_rhs=at(rhs,p,c+1)+(left[0][c]-right[0][c])+(left[2][c]-right[2][c]);
        const double origin=first?face_value(fields.q,p,c+1)*jac:at(saved,p,c+1);
        state[c]=a*origin+b*face_value(fields.q,p,c+1)*jac+cdt*low_rhs;at(base,p,c+1)=state[c];
        if(!shape.top)at(rhs,p,c+1)=0;
    }
    if(!state_admissible(state,floor.value))status=201;
    if(status)atomicMax(failure,status);
}
__global__ void owned_faces_kernel(FaceFields fields,AstrDeviceView base,AstrDeviceView faces,const int8_t *mask,
    Shape shape,Floors floor,int axis,bool capturing,double cdt,bool preserve_all,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=shape.count)return;
    int p[3];point(id,shape,p);if(!active(p,shape))return;
    if(p[axis]==shape.upper[axis] && !(shape.top && axis==0 && (shape.type[0]==2 || shape.type[0]==4)))return;
    int right[3]={p[0],p[1],p[2]};++right[axis];
    double high[11],low[11],flux[11],beta,left_base[11],right_base[11],ql[11],qr[11];int status=0;
    high_face(fields,mask,p,axis,p[axis],shape,capturing,high,status);low_face(fields,p,axis,p[axis],shape,low,status);
    for(int c=0;c<11;++c) {
        left_base[c]=at(base,p,c+1);right_base[c]=at(base,right,c+1);
        ql[c]=face_value(fields.q,p,c+1);qr[c]=face_value(fields.q,right,c+1);
    }
    bool check_right=!(right[axis]==shape.upper[axis] && (shape.type[axis]==2 || shape.type[axis]==4));
    if(shape.top)check_right=!(axis==0 && p[0]==shape.upper[0] && (shape.type[0]==2 || shape.type[0]==4));
    if(status)atomicMax(failure,status);
    status=symmetric_face_trial(high,low,left_base,right_base,ql,qr,cdt,preserve_all,true,check_right,floor.value,flux,beta);
    if(status)atomicMax(failure,status);
    for(int c=0;c<11;++c)at(faces,p,c+1)=flux[c];at(faces,p,12)=beta;
    if(p[axis]==1 && (shape.type[axis]==1 || shape.type[axis]==4)) {
        int left[3]={p[0],p[1],p[2]};left[axis]=0;
        for(int c=0;c<11;++c) {right_base[c]=left_base[c];left_base[c]=0;qr[c]=ql[c];ql[c]=face_value(fields.q,left,c+1);}
        status=0;high_face(fields,mask,p,axis,p[axis]-1,shape,capturing,high,status);
        low_face(fields,p,axis,p[axis]-1,shape,low,status);if(status)atomicMax(failure,status);
        status=symmetric_face_trial(high,low,left_base,right_base,ql,qr,cdt,preserve_all,false,true,floor.value,flux,beta);
        if(status)atomicMax(failure,status);
        for(int c=0;c<11;++c)at(faces,left,c+1)=flux[c];at(faces,left,12)=beta;
    }
}
__device__ void record_failure(AstrAir5TransportFailure *report,FaceFields fields,AstrDeviceView base,
    const int p[3],int axis,int code,Shape shape,double cdt,const double state[11],const double flux[11],const double low[11])
{
    if(atomicCAS(report->failure,0,1)!=0)return;
    report->failure[1]=p[0];report->failure[2]=p[1];report->failure[3]=p[2];
    report->failure[4]=axis+1;report->failure[5]=code;report->failure[6]=shape.top?1:0;
    int opposite[3]={p[0],p[1],p[2]};opposite[axis]=shape.upper[axis];
    double opposite_low[11];int ignored=0;
    low_face(fields,opposite,axis,shape.upper[axis]-1,Shape{ {shape.upper[0],shape.upper[1],shape.upper[2]},
        {shape.type[0],shape.type[1],shape.type[2]},false,shape.count},opposite_low,ignored);
    for(int c=0;c<11;++c) {
        report->values[c]=state[c];report->values[11+c]=at(base,p,c+1);report->values[22+c]=flux[c];
        report->values[33+c]=low[c];report->values[44+c]=at(base,opposite,c+1);report->values[55+c]=opposite_low[c];
    }
    report->cdt=cdt;
}
__global__ void apply_kernel(FaceFields fields,AstrDeviceView base,AstrDeviceView faces,AstrDeviceView rhs,
    Shape shape,Floors floor,int axis,double cdt,int *failure,AstrAir5TransportFailure *report)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=shape.count)return;
    int p[3];point(id,shape,p);if(!active(p,shape))return;
    int left[3]={p[0],p[1],p[2]};--left[axis];
    double ll[11],lr[11],fl[11],fr[11],state[11];int status=0;
    low_face(fields,p,axis,p[axis]-1,shape,ll,status);low_face(fields,p,axis,p[axis],shape,lr,status);
    for(int c=0;c<11;++c) {fl[c]=at(faces,left,c+1);fr[c]=at(faces,p,c+1);state[c]=at(base,p,c+1)+6.*cdt*(fl[c]-ll[c]);}
    if(!state_admissible(state,floor.value)) {record_failure(report,fields,base,p,axis,202,shape,cdt,state,fl,ll);status=202;}
    for(int c=0;c<11;++c)state[c]=at(base,p,c+1)-6.*cdt*(fr[c]-lr[c]);
    if(!state_admissible(state,floor.value)) {record_failure(report,fields,base,p,axis,203,shape,cdt,state,fr,lr);status=203;}
    if(status)atomicMax(failure,status);
    for(int c=0;c<11;++c)at(rhs,p,c+1)=at(rhs,p,c+1)+fl[c]-fr[c];
}
__global__ void final_kernel(FaceFields fields,AstrDeviceView saved,AstrDeviceView rhs,Shape shape,Floors floor,
    double a,double b,double cdt,bool first,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=shape.count)return;
    int p[3];point(id,shape,p);if(!active(p,shape))return;
    const double jac=face_value(fields.jacobian,p);double state[11];
    for(int c=0;c<11;++c) {
        const double origin=first?face_value(fields.q,p,c+1)*jac:at(saved,p,c+1);
        state[c]=a*origin+b*face_value(fields.q,p,c+1)*jac+cdt*at(rhs,p,c+1);
    }
    if(!state_admissible(state,floor.value))atomicMax(failure,204);
}
__global__ void top_normal_kernel(FaceFields fields,AstrDeviceView base,AstrDeviceView rhs,AstrDeviceView high_plane,
    Shape shape,Floors floor,double cdt,bool preserve_all,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=static_cast<size_t>(shape.upper[0]+1)*(shape.upper[2]+1))return;
    const int p[3]={static_cast<int>(id%(shape.upper[0]+1)),shape.upper[1],static_cast<int>(id/(shape.upper[0]+1))};
    if(!active(p,shape))return;const int plane[3]={p[0],0,p[2]};
    double cell[11],high[11],low[11],q[11],flux[11],beta,state[11];
    for(int c=0;c<11;++c) {cell[c]=at(base,p,c+1);high[c]=-at(high_plane,plane,c+1);low[c]=-at(rhs,p,c+1);q[c]=face_value(fields.q,p,c+1);}
    int status=symmetric_face_trial(high,low,cell,cell,q,q,cdt,preserve_all,true,false,floor.value,flux,beta);
    if(status) {atomicMax(failure,status);return;}
    for(int c=0;c<11;++c)state[c]=cell[c]-6.*cdt*(flux[c]-low[c]);
    if(!state_admissible(state,floor.value)) {atomicMax(failure,205);return;}
    for(int c=0;c<11;++c)at(rhs,p,c+1)=-flux[c];
}
__global__ void plane_kernel(AstrDeviceView faces,double *planes,size_t capacity,Shape shape,int axis,bool unpack,
    bool receive_plus,bool receive_minus,int uaxis,int vaxis,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=count)return;
    const int n1=shape.upper[uaxis]+1,n2=shape.upper[vaxis]+1,c=id/(n1*n2)+1;
    int first[3]={},last[3]={};first[uaxis]=id%n1;first[vaxis]=(id/n1)%n2;
    for(int d=0;d<3;++d)last[d]=first[d];
    if(unpack) {
        first[axis]=shape.upper[axis];last[axis]=-1;
        if(receive_plus)at(faces,first,c)=planes[id+2*capacity];
        if(receive_minus)at(faces,last,c)=planes[id+3*capacity];
    } else {
        last[axis]=shape.upper[axis]-1;planes[id]=at(faces,first,c);planes[id+capacity]=at(faces,last,c);
    }
}
int shape_for(astr::State &s,const int *types,int top,Shape &shape)
{
    if(!types || top<0 || top>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    if(!s.air5_convection)return ASTR_DEVICE_NOT_ALLOCATED;
    shape.count=1;shape.top=top!=0;
    for(int a=0;a<3;++a) {
        if(types[a]<1 || types[a]>4)return ASTR_DEVICE_INVALID_ARGUMENT;
        shape.upper[a]=s.upper[a];shape.type[a]=types[a];shape.count*=s.upper[a]+1;
    }
    return 0;
}
int plane_shape(astr::State &s,int axis,Shape &shape,int &u,int &v,size_t &count)
{
    const int types[3]={3,3,3};const int status=shape_for(s,types,0,shape);if(status)return status;
    if(axis<0 || axis>2)return ASTR_DEVICE_INVALID_ARGUMENT;
    u=axis==0?1:0;v=axis==2?1:2;count=12*static_cast<size_t>(s.upper[u]+1)*(s.upper[v]+1);return 0;
}
}
extern "C" int astr_device_air5_symmetric_prepare(void *handle,const double floor[5],int with_top)
{
    if(!handle || !floor || with_top<0 || with_top>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fluid_ready || !s.fields[ASTR_DEVICE_INTERFACE_FLUX].view.data)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.air5_convection || s.halo<1)return ASTR_DEVICE_INVALID_ARGUMENT;
    s.air5_convection=new(std::nothrow) astr::Air5ConvectionPlan{};
    if(!s.air5_convection)return static_cast<int>(cudaErrorMemoryAllocation);
    auto &plan=*s.air5_convection;
    for(int i=0;i<5;++i)plan.floor.value[i]=floor[i];
    int status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_CONVECTION_BASE],s.upper,s.halo,4,11,1,sizeof(double));if(status)return status;
    if(with_top && !s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH].view.data) {
        const int upper[3]={s.upper[0],0,s.upper[2]};
        status=astr::allocate_field(s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH],upper,0,4,11,1,sizeof(double));if(status)return status;
    }
    status=static_cast<int>(cudaMalloc(&plan.failure,sizeof(AstrAir5TransportFailure)));if(status)return status;
    for(int a=0;a<3;++a) {
        const size_t count=12*static_cast<size_t>(s.upper[(a+1)%3]+1)*(s.upper[(a+2)%3]+1);
        if(count>plan.capacity)plan.capacity=count;
    }
    return static_cast<int>(cudaMalloc(&plan.planes,4*plan.capacity*sizeof(double)));
}
extern "C" int astr_device_air5_symmetric_base(void *handle,const int types[3],double a,double b,double cdt,int first,int top)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);Shape shape{};
    int status=shape_for(s,types,top,shape);if(status)return status;
    if(top && !s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH].view.data)return ASTR_DEVICE_NOT_ALLOCATED;
    status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    status=static_cast<int>(cudaMemsetAsync(s.air5_convection->failure,0,sizeof(AstrAir5TransportFailure),s.stream));if(status)return status;
    auto &base=s.fields[ASTR_DEVICE_AIR5_CONVECTION_BASE];status=static_cast<int>(cudaMemsetAsync(base.view.data,0,base.bytes,s.stream));if(status)return status;
    base_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),s.fields[ASTR_DEVICE_SAVE].view,
        s.fields[ASTR_DEVICE_RHS].view,base.view,shape,s.air5_convection->floor,a,b,cdt,first!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_symmetric_faces(void *handle,const int types[3],int axis,int capturing,double cdt,int top,int preserve)
{
    if(!handle || axis<1 || axis>3 || capturing<0 || capturing>1 || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);Shape shape{};int status=shape_for(s,types,top,shape);if(status)return status;
    if(capturing && !s.shock_mask)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.halo<(capturing?4:3) || ((types[axis-1]!=3) && s.upper[axis-1]<5))return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &faces=s.fields[ASTR_DEVICE_INTERFACE_FLUX];status=static_cast<int>(cudaMemsetAsync(faces.view.data,0,faces.bytes,s.stream));if(status)return status;
    owned_faces_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),s.fields[ASTR_DEVICE_AIR5_CONVECTION_BASE].view,
        faces.view,s.shock_mask,shape,s.air5_convection->floor,axis-1,capturing!=0,cdt,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_symmetric_apply(void *handle,const int types[3],int axis,double cdt,int top)
{
    if(!handle || axis<1 || axis>3)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);Shape shape{};
    const int status=shape_for(s,types,top,shape);if(status)return status;
    apply_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),s.fields[ASTR_DEVICE_AIR5_CONVECTION_BASE].view,
        s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,s.fields[ASTR_DEVICE_RHS].view,shape,s.air5_convection->floor,
        axis-1,cdt,s.air5_failure,s.air5_convection->failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_symmetric_final(void *handle,const int types[3],double a,double b,double cdt,int first,int top)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);Shape shape{};
    const int status=shape_for(s,types,top,shape);if(status)return status;
    final_kernel<<<(shape.count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),s.fields[ASTR_DEVICE_SAVE].view,
        s.fields[ASTR_DEVICE_RHS].view,shape,s.air5_convection->floor,a,b,cdt,first!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_symmetric_top_normal(void *handle,const int types[3],double cdt,int preserve)
{
    if(!handle || preserve<0 || preserve>1)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);Shape shape{};
    const int status=shape_for(s,types,1,shape);if(status)return status;
    if(!s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH].view.data)return ASTR_DEVICE_NOT_ALLOCATED;
    const size_t count=static_cast<size_t>(s.upper[0]+1)*(s.upper[2]+1);
    top_normal_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(face_fields(s),s.fields[ASTR_DEVICE_AIR5_CONVECTION_BASE].view,
        s.fields[ASTR_DEVICE_RHS].view,s.fields[ASTR_DEVICE_AIR5_TOP_NORMAL_HIGH].view,shape,s.air5_convection->floor,
        cdt,preserve!=0,s.air5_failure);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_symmetric_plane(void *handle,int axis,int unpack,int plus,int minus)
{
    if(!handle || unpack<0 || unpack>1 || plus<0 || plus>1 || minus<0 || minus>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);Shape shape{};int u,v;size_t count;
    const int status=plane_shape(s,axis-1,shape,u,v,count);if(status)return status;
    plane_kernel<<<(count+threads-1)/threads,threads,0,s.stream>>>(s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,s.air5_convection->planes,
        s.air5_convection->capacity,shape,axis-1,unpack!=0,plus!=0,minus!=0,u,v,count);
    return astr::finish_launch(s);
}
extern "C" int astr_device_air5_symmetric_buffers(void *handle,void *buffers[4],size_t *capacity)
{
    if(!handle || !buffers || !capacity)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_convection)return ASTR_DEVICE_NOT_ALLOCATED;
    auto &plan=*s.air5_convection;for(int slot=0;slot<4;++slot)buffers[slot]=plan.planes+slot*plan.capacity;
    *capacity=plan.capacity;return 0;
}
extern "C" int astr_device_air5_symmetric_copy_local_planes(void *handle,int axis)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);Shape shape{};int u,v;size_t count;
    int status=plane_shape(s,axis-1,shape,u,v,count);if(status)return status;auto &plan=*s.air5_convection;
    status=static_cast<int>(cudaMemcpyAsync(plan.planes+2*plan.capacity,plan.planes,count*sizeof(double),cudaMemcpyDeviceToDevice,s.stream));if(status)return status;
    return static_cast<int>(cudaMemcpyAsync(plan.planes+3*plan.capacity,plan.planes+plan.capacity,count*sizeof(double),cudaMemcpyDeviceToDevice,s.stream));
}
extern "C" int astr_device_air5_symmetric_failure(void *handle,AstrAir5TransportFailure *failure)
{
    if(!handle || !failure)return ASTR_DEVICE_INVALID_ARGUMENT;auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_convection)return ASTR_DEVICE_NOT_ALLOCATED;
    const int status=static_cast<int>(cudaMemcpyAsync(failure,s.air5_convection->failure,sizeof(*failure),cudaMemcpyDeviceToHost,s.stream));
    return status?status:static_cast<int>(cudaStreamSynchronize(s.stream));
}
