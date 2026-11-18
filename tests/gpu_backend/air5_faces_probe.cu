#include "device_state_internal.cuh"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include "shock_sensor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {
void check(int code) {if(code){std::fprintf(stderr,"device status %d\n",code);std::exit(1);}}
size_t size(const AstrDeviceView &v) {return v.stride[4]*v.extent[4];}
size_t offset(const AstrDeviceView &v,int i,int j,int k,int c=1,int d=1)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3]+(d-1)*v.stride[4];}
void upload(void *h,int f,const std::vector<double> &v) {check(astr_device_state_upload(h,f,v.data(),v.size()*8));}
std::vector<double> download(void *h,int f,AstrDeviceView v)
{std::vector<double> x(size(v));check(astr_device_state_download(h,f,x.data(),x.size()*8));return x;}
}
int main(int argc,char **argv)
{
    if(argc!=4)return 2;FILE *file=std::fopen(argv[1],"rb");if(!file)return 2;
    int header[4];if(std::fread(header,4,4,file)!=4)return 2;
    const int im=header[0],jm=header[1],km=header[2],cases=header[3],halo=5;
    const size_t total=static_cast<size_t>(im+11)*(jm+11)*(km+11);
    const size_t physical=static_cast<size_t>(im+1)*(jm+1)*(km+1);
    std::vector<int> jobs(5*cases),mask(physical);std::vector<double> table(64*total);
    if(std::fread(jobs.data(),4,jobs.size(),file)!=jobs.size() || std::fread(table.data(),8,table.size(),file)!=table.size() ||
        std::fread(mask.data(),4,mask.size(),file)!=mask.size())return 2;
    std::fclose(file);void *handle=nullptr;
    bool viscous=false,capturing=false;
    for(int job=0;job<cases;++job) {viscous=viscous || jobs[5*job+2]>=2;capturing=capturing || jobs[5*job+3]!=0;}
    const bool gradients=viscous || capturing;
    check(astr_device_state_create(0,im,jm,km,halo,std::atoi(argv[3]),&handle));check(astr_device_air5_prepare(handle));
    check(astr_device_air5_geometry_prepare(handle));
    if(gradients)check(astr_device_air5_transport_prepare(handle,1));
    check(astr_device_air5_faces_prepare(handle));if(capturing)check(astr_device_shock_prepare(handle));
    const int fields[]={ASTR_DEVICE_Q,ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES,
        ASTR_DEVICE_METRIC,ASTR_DEVICE_JACOBIAN,ASTR_DEVICE_STRESS,ASTR_DEVICE_HEAT_FLUX,ASTR_DEVICE_AIR5_SPECIES_FLUX,
        ASTR_DEVICE_AIR5_EV_FLUX,ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX};
    const int components[]={11,1,3,1,5,3,1,6,3,5,3,6},directions[]={1,1,1,1,1,3,1,1,1,3,1,1};
    std::vector<std::vector<double>> original;std::vector<AstrDeviceView> views;int column=0;
    const int field_count=gradients?12:7;
    for(int f=0;f<field_count;++f) {
        AstrDeviceView v;check(astr_device_state_view(handle,fields[f],&v));views.push_back(v);
        std::vector<double> data(size(v));
        for(int d=1;d<=directions[f];++d)for(int c=1;c<=components[f];++c) {
            size_t id=0;
            for(int k=-halo;k<=km+halo;++k)for(int j=-halo;j<=jm+halo;++j)for(int i=-halo;i<=im+halo;++i,++id)
                data[offset(v,i,j,k,c,d)]=table[64*id+column];
            ++column;
        }
        upload(handle,fields[f],data);original.push_back(std::move(data));
    }
    if(column!=(gradients?64:31))return 3;
    if(!gradients) {
        AstrDeviceView unused;
        for(int field:{ASTR_DEVICE_DVELOCITY,ASTR_DEVICE_AIR5_DTV,ASTR_DEVICE_STRESS,ASTR_DEVICE_AIR5_SPECIES_FLUX})
            if(astr_device_state_view(handle,field,&unused)!=ASTR_DEVICE_NOT_ALLOCATED)return 10;
    }
    auto &state=*static_cast<astr::State *>(handle);std::vector<int8_t> markers(mask.begin(),mask.end());
    if(capturing)check(cudaMemcpyAsync(state.shock_mask,markers.data(),markers.size(),cudaMemcpyHostToDevice,state.stream));
    AstrDeviceView faces;check(astr_device_state_view(handle,ASTR_DEVICE_INTERFACE_FLUX,&faces));
    std::vector<double> untouched(size(faces),-123.);
    file=std::fopen(argv[2],"wb");if(!file)return 2;bool written=std::fwrite(&cases,4,1,file)==1;
    for(int job=0;job<cases;++job) {
        const int *config=jobs.data()+5*job;const int axis=config[0]-1;
        int first[]={0,0,0},last[]={im,jm,km};first[axis]=-1;
        upload(handle,ASTR_DEVICE_INTERFACE_FLUX,untouched);
        check(astr_device_air5_reconstruct_faces(handle,config[0],config[1],config[2],config[3],config[4],first,last));
        int status;check(astr_device_air5_check(handle,&status));if(status)return 4;
        const auto data=download(handle,ASTR_DEVICE_INTERFACE_FLUX,faces);
        const int n=(im+1+(axis==0))*(jm+1+(axis==1))*(km+1+(axis==2));
        std::vector<double> result(11*n);size_t id=0;
        for(int k=first[2];k<=last[2];++k)for(int j=first[1];j<=last[1];++j)for(int i=first[0];i<=last[0];++i,++id)
            for(int c=1;c<=11;++c)result[11*id+c-1]=data[offset(faces,i,j,k,c)];
        for(int k=-1;k<=km+1;++k)for(int j=-1;j<=jm+1;++j)for(int i=-1;i<=im+1;++i) {
            const bool selected=i>=first[0] && i<=last[0] && j>=first[1] && j<=last[1] && k>=first[2] && k<=last[2];
            for(int c=1;c<=12;++c)if((!selected || c==12) && data[offset(faces,i,j,k,c)]!=-123.)return 5;
        }
        written=written && std::fwrite(config,4,5,file)==5 && std::fwrite(&n,4,1,file)==1;
        written=written && std::fwrite(result.data(),8,result.size(),file)==result.size();
    }
    for(int f=0;f<field_count;++f) {
        const auto data=download(handle,fields[f],views[f]);
        if(std::memcmp(data.data(),original[f].data(),data.size()*8))return 6;
    }
    auto density=original[1];density[offset(views[1],0,0,0)]=-1.;upload(handle,ASTR_DEVICE_RHO,density);
    const int first[]={0,0,0},last[]={0,0,0};upload(handle,ASTR_DEVICE_INTERFACE_FLUX,untouched);
    check(astr_device_air5_reconstruct_faces(handle,1,3,1,0,1,first,last));
    int status;check(astr_device_air5_check(handle,&status));if(status!=3)return 7;
    const auto failed=download(handle,ASTR_DEVICE_INTERFACE_FLUX,faces);
    if(std::memcmp(failed.data(),untouched.data(),failed.size()*8))return 8;
    check(astr_device_state_destroy(&handle));
    return std::fclose(file)==0 && written?0:2;
}
