#include "chemistry_face_math.cuh"
#include "chemistry_transport.h"

namespace {
struct Box {int first[3],size[3],upper[3];size_t count;};
__global__ void reconstruct_kernel(astr::air5::FaceFields fields,AstrDeviceView faces,const int8_t *mask,
    Box box,int axis,int ntype,int kind,bool capturing,bool symmetric,int *failure)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(id>=box.count)return;
    const int p[3]={static_cast<int>(id%box.size[0])+box.first[0],
        static_cast<int>((id/box.size[0])%box.size[1])+box.first[1],
        static_cast<int>(id/(static_cast<size_t>(box.size[0])*box.size[1]))+box.first[2]};
    double flux[11];int status=0;
    if(kind<=1) {
        const bool shock=kind==0 && capturing && astr::air5::air5_shock_face(mask,box.upper,p,axis,ntype);
        astr::air5::ConvectiveLine line{fields,{p[0],p[1],p[2]},axis};
        astr::air5::convective_face(line,p[axis],box.upper[axis],ntype,shock,kind==1,symmetric,flux,status);
    } else {
        astr::air5::DiffusiveLine line{fields,{p[0],p[1],p[2]},axis,kind==3?1:0};
        astr::air5::centered_face(line,p[axis],box.upper[axis],ntype,flux);
    }
    if(status) {atomicMax(failure,status);return;}
    for(int c=0;c<11;++c)static_cast<double *>(faces.data)[astr::offset(faces,p[0],p[1],p[2],c+1)]=flux[c];
}
}
extern "C" int astr_device_air5_faces_prepare(void *handle)
{
    if(!handle)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);
    if(!s.air5_ready || !s.fluid_ready)return ASTR_DEVICE_NOT_ALLOCATED;
    if(s.fields[ASTR_DEVICE_INTERFACE_FLUX].view.data)return ASTR_DEVICE_INVALID_ARGUMENT;
    return astr::allocate_field(s.fields[ASTR_DEVICE_INTERFACE_FLUX],s.upper,1,4,12,1,sizeof(double));
}
extern "C" int astr_device_air5_reconstruct_faces(void *handle,int axis,int ntype,int kind,int capturing,int symmetric,
    const int first[3],const int last[3])
{
    if(!handle || !first || !last || axis<1 || axis>3 || ntype<1 || ntype>4 || kind<0 || kind>3 ||
        capturing<0 || capturing>1 || symmetric<0 || symmetric>1)return ASTR_DEVICE_INVALID_ARGUMENT;
    auto &s=*static_cast<astr::State *>(handle);--axis;
    if(!s.air5_ready || !s.fluid_ready || !s.fields[ASTR_DEVICE_INTERFACE_FLUX].view.data ||
        (kind>=2 && !s.viscous_ready) || (kind==3 && !s.fields[ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX].view.data) ||
        (kind==0 && capturing && !s.shock_mask))return ASTR_DEVICE_NOT_ALLOCATED;
    const int required_halo=kind==1?1:(kind==0 && capturing?4:3);
    if(s.halo<required_halo || (kind!=1 && (ntype==1 || ntype==2 || ntype==4) && s.upper[axis]<5))
        return ASTR_DEVICE_INVALID_ARGUMENT;
    Box box{};box.count=1;
    for(int a=0;a<3;++a) {
        const int h=a==axis?1:0;
        if(first[a]<-h || last[a]>s.upper[a] || first[a]>last[a])return ASTR_DEVICE_INVALID_ARGUMENT;
        box.first[a]=first[a];box.size[a]=last[a]-first[a]+1;box.upper[a]=s.upper[a];box.count*=box.size[a];
    }
    const auto fields=astr::air5::face_fields(s);
    const int status=static_cast<int>(cudaMemsetAsync(s.air5_failure,0,sizeof(int),s.stream));if(status)return status;
    reconstruct_kernel<<<(box.count+127)/128,128,0,s.stream>>>(fields,s.fields[ASTR_DEVICE_INTERFACE_FLUX].view,s.shock_mask,
        box,axis,ntype,kind,capturing!=0,symmetric!=0,s.air5_failure);
    return astr::finish_launch(s);
}
