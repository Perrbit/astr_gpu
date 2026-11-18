#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
void check(int status) {if(status) {std::fprintf(stderr,"device status %d\n",status);std::exit(1);}}
size_t size(const AstrDeviceView &v) {return v.stride[4]*v.extent[4];}
size_t offset(const AstrDeviceView &v,int i,int j,int k,int c=1,int d=1)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3]+(d-1)*v.stride[4];}
std::vector<double> download(void *handle,int field,AstrDeviceView v)
{std::vector<double> data(size(v));check(astr_device_state_download(handle,field,data.data(),data.size()*8));return data;}
void upload(void *handle,int field,const std::vector<double> &data)
{check(astr_device_state_upload(handle,field,data.data(),data.size()*8));}
}
int main(int argc,char **argv)
{
    if(argc!=3) return 2;
    FILE *file=std::fopen(argv[1],"rb");if(!file)return 2;
    int header[11];if(std::fread(header,4,11,file)!=11)return 2;
    const int im=header[0],jm=header[1],km=header[2],halo=5,layered=header[9],zero_lower=header[10];
    const int *low=header+3,*high=header+6;
    const size_t total=static_cast<size_t>(im+11)*(jm+11)*(km+11),nodes=static_cast<size_t>(im+1)*(jm+1)*(km+1);
    std::vector<double> input(total*21);
    if(std::fread(input.data(),8,input.size(),file)!=input.size())return 2;
    std::fclose(file);
    void *handle=nullptr;check(astr_device_state_create(0,im,jm,km,halo,0,&handle));
    check(astr_device_air5_prepare(handle));check(astr_device_air5_geometry_prepare(handle));
    check(astr_device_air5_transport_prepare(handle,layered));
    const int input_fields[]={ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE,
        ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_PRESSURE,ASTR_DEVICE_AIR5_SPECIES,ASTR_DEVICE_METRIC};
    const int components[]={1,3,1,1,1,5,3},directions[]={1,1,1,1,1,1,3};
    int column=0;
    for(int f=0;f<7;++f) {
        AstrDeviceView v;check(astr_device_state_view(handle,input_fields[f],&v));std::vector<double> values(size(v));
        for(int d=1;d<=directions[f];++d) for(int c=1;c<=components[f];++c) {
            size_t id=0;
            for(int k=-halo;k<=km+halo;++k) for(int j=-halo;j<=jm+halo;++j) for(int i=-halo;i<=im+halo;++i,++id)
                values[offset(v,i,j,k,c,d)]=input[21*id+column];
            ++column;
        }
        upload(handle,input_fields[f],values);
    }
    AstrDeviceView q;check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));
    const std::vector<double> authority(size(q),-17.);upload(handle,ASTR_DEVICE_Q,authority);
    const int fields[]={ASTR_DEVICE_DVELOCITY,ASTR_DEVICE_DTEMPERATURE,ASTR_DEVICE_AIR5_DTV,
        ASTR_DEVICE_AIR5_DSPECIES,ASTR_DEVICE_STRESS,ASTR_DEVICE_HEAT_FLUX,
        ASTR_DEVICE_AIR5_SPECIES_FLUX,ASTR_DEVICE_AIR5_EV_FLUX,ASTR_DEVICE_AIR5_SPECIES_ENERGY_FLUX};
    const int cs[]={3,3,3,5,6,3,5,3,6},ds[]={3,1,1,3,1,1,3,1,1};
    for(int f=0;f<(layered?9:8);++f) {
        AstrDeviceView v;check(astr_device_state_view(handle,fields[f],&v));upload(handle,fields[f],std::vector<double>(size(v),-42.));
    }
    if(!layered) {
        AstrDeviceView unused;
        if(astr_device_state_view(handle,fields[8],&unused)!=ASTR_DEVICE_NOT_ALLOCATED)return 3;
    }
    check(astr_device_air5_gradients(handle,low,high,zero_lower));
    check(astr_device_air5_diffusive_flux(handle));int status;
    check(astr_device_air5_check(handle,&status));if(status)return 4;
    std::vector<double> output(nodes*63,0.);column=0;
    for(int f=0;f<(layered?9:8);++f) {
        AstrDeviceView v;check(astr_device_state_view(handle,fields[f],&v));const auto values=download(handle,fields[f],v);
        for(int d=1;d<=ds[f];++d)for(int c=1;c<=cs[f];++c) {
            size_t id=0;
            for(int k=0;k<=km;++k)for(int j=0;j<=jm;++j)for(int i=0;i<=im;++i,++id)
                output[63*id+column]=values[offset(v,i,j,k,c,d)];
            ++column;
        }
        if(f>=4) for(int k=-halo;k<=km+halo;++k)for(int j=-halo;j<=jm+halo;++j)for(int i=-halo;i<=im+halo;++i)
            if(i<0 || i>im || j<0 || j>jm || k<0 || k>km)
                for(int d=1;d<=ds[f];++d)for(int c=1;c<=cs[f];++c)if(values[offset(v,i,j,k,c,d)]!=-42.)return 5;
    }
    // A bad cell must report failure without committing its flux, while other
    // cells are still processed. No physical state repair is performed here.
    AstrDeviceView rho,stress;check(astr_device_state_view(handle,ASTR_DEVICE_RHO,&rho));
    auto density=download(handle,ASTR_DEVICE_RHO,rho);density[offset(rho,0,0,0)]=-1.;upload(handle,ASTR_DEVICE_RHO,density);
    check(astr_device_state_view(handle,ASTR_DEVICE_STRESS,&stress));upload(handle,ASTR_DEVICE_STRESS,std::vector<double>(size(stress),-42.));
    check(astr_device_air5_diffusive_flux(handle));check(astr_device_air5_check(handle,&status));if(!status)return 6;
    const auto stresses=download(handle,ASTR_DEVICE_STRESS,stress);
    for(int c=1;c<=6;++c)if(stresses[offset(stress,0,0,0,c)]!=-42.)return 7;
    if(stresses[offset(stress,1,0,0,1)]==-42.)return 8;
    const auto unchanged=download(handle,ASTR_DEVICE_Q,q);
    if(std::memcmp(unchanged.data(),authority.data(),authority.size()*8))return 9;
    check(astr_device_state_destroy(&handle));
    file=std::fopen(argv[2],"wb");if(!file)return 2;
    bool written=std::fwrite(output.data(),8,output.size(),file)==output.size();
    return std::fclose(file)==0 && written?0:2;
}
