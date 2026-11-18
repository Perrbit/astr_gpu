#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_coupling.h"
#include "chemistry_transport.h"
#include "fluid_explicit.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr int im=4,jm=3,km=2,halo=5,nodes=(im+1)*(jm+1)*(km+1);
void check(int code) {if(code) {std::fprintf(stderr,"device operation %d\n",code);std::exit(1);}}
size_t length(const AstrDeviceView &v) {return v.stride[4]*v.extent[4];}
size_t offset(const AstrDeviceView &v,int i,int j,int k,int c=1,int d=1)
{return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+(k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3]+(d-1)*v.stride[4];}
bool active(int i,int j,int k) {return i>=1 && i<=im-1 && j>=1 && j<=jm && k>=0 && k<=km-1;}
std::vector<double> download(void *handle,int field,const AstrDeviceView &v)
{
    std::vector<double> a(length(v));check(astr_device_state_download(handle,field,a.data(),a.size()*sizeof(double)));return a;
}
void upload(void *handle,int field,const std::vector<double> &a)
{check(astr_device_state_upload(handle,field,a.data(),a.size()*sizeof(double)));}
}
int main(int argc,char **argv)
{
    if(argc!=4) return 2;
    FILE *file=std::fopen(argv[1],"rb");if(!file) return 2;
    int n;double dt;
    if(std::fread(&n,sizeof(n),1,file)!=1 || std::fread(&dt,sizeof(dt),1,file)!=1 || n!=nodes) return 2;
    std::vector<double> input(65*n),out(104*n,0.);
    if(std::fread(input.data(),sizeof(double),input.size(),file)!=input.size()) return 2;
    std::fclose(file);
    const int comp=std::atoi(argv[3]);void *handle=nullptr;
    check(astr_device_state_create(0,im,jm,km,halo,0,&handle));check(astr_device_air5_prepare(handle));
    AstrAir5ChemistryOptions options{comp,0,3,1,1,32,0,1.e-9,1.e-13,0.};
    check(astr_device_air5_chemistry_prepare(handle,&options));check(astr_device_air5_geometry_prepare(handle));
    check(astr_device_air5_rk_prepare(handle));
    AstrDeviceView q,rhs,saved,jac,metric,carry,origin,origin_carry;
    check(astr_device_state_view(handle,ASTR_DEVICE_Q,&q));check(astr_device_state_view(handle,ASTR_DEVICE_RHS,&rhs));
    check(astr_device_state_view(handle,ASTR_DEVICE_SAVE,&saved));check(astr_device_state_view(handle,ASTR_DEVICE_JACOBIAN,&jac));
    check(astr_device_state_view(handle,ASTR_DEVICE_METRIC,&metric));
    if(comp) {
        check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_CARRY,&carry));
        check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_ORIGIN,&origin));
        check(astr_device_state_view(handle,ASTR_DEVICE_AIR5_ORIGIN_CARRY,&origin_carry));
    } else {
        AstrDeviceView unused;
        if(astr_device_state_view(handle,ASTR_DEVICE_AIR5_ORIGIN,&unused)!=ASTR_DEVICE_NOT_ALLOCATED) return 3;
    }
    std::vector<double> qdata(length(q),NAN),jdata(length(jac),-321.),mdata(length(metric),-321.),sdata(length(saved),-42.);
    std::vector<double> cdata=comp?std::vector<double>(length(carry)):std::vector<double>{};
    for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i) {
        const int id=i+(im+1)*(j+(jm+1)*k);const double *x=input.data()+65*id;
        for(int c=1;c<=11;++c) {qdata[offset(q,i,j,k,c)]=x[c-1];if(comp)cdata[offset(carry,i,j,k,c)]=x[54+c-1];}
        jdata[offset(jac,i,j,k)]=x[44];
        for(int d=1;d<=3;++d) for(int c=1;c<=3;++c) mdata[offset(metric,i,j,k,c,d)]=x[45+(c-1)+3*(d-1)];
    }
    upload(handle,ASTR_DEVICE_Q,qdata);upload(handle,ASTR_DEVICE_JACOBIAN,jdata);upload(handle,ASTR_DEVICE_METRIC,mdata);
    upload(handle,ASTR_DEVICE_SAVE,sdata);
    if(comp) {
        upload(handle,ASTR_DEVICE_AIR5_CARRY,cdata);
        upload(handle,ASTR_DEVICE_AIR5_ORIGIN,std::vector<double>(length(origin),-43.));
        upload(handle,ASTR_DEVICE_AIR5_ORIGIN_CARRY,std::vector<double>(length(origin_carry),-44.));
    }
    // Deliberately stale caches: AIR5 CFL must derive thermodynamics from q.
    for(int field:{ASTR_DEVICE_VELOCITY,ASTR_DEVICE_TEMPERATURE}) {
        AstrDeviceView v;check(astr_device_state_view(handle,field,&v));upload(handle,field,std::vector<double>(length(v),NAN));
    }
    const int global[]={20,18,12},start[]={5,4,3};double rates[4],basic[3];int64_t keys[4];
    std::vector<double> profile(4*(global[1]+1),0.);
    check(astr_device_fluid_cfl_diagnostics(handle,1,NAN,NAN,NAN,global,start,global[1]+1,rates,keys,profile.data()));
    check(astr_device_fluid_cfl(handle,1,NAN,NAN,NAN,basic));
    if(std::memcmp(basic,rates,sizeof(basic))) return 4;
    const int first_all[]={0,0,0},last_all[]={im,jm,km};
    check(astr_device_air5_primitives(handle,first_all,last_all,1));check(astr_device_fluid_rhs_clear(handle));
    check(astr_device_air5_chemistry_begin(handle));
    check(astr_device_air5_chemistry_region(handle,first_all,last_all,1.e-12,0));
    int status;check(astr_device_air5_chemistry_status(handle,&status));if(status) return 5;
    check(astr_device_fluid_rhs_clear(handle));
    const int first[]={1,1,0},last[]={im-1,jm,km-1};
    const double a[]={1.,.75,1./3.},b[]={0.,.25,2./3.},c[]={1.,.25,2./3.};
    for(int stage=0;stage<3;++stage) {
        std::vector<double> rdata(length(rhs));
        for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i) {
            const int id=i+(im+1)*(j+(jm+1)*k);
            for(int c=1;c<=11;++c) rdata[offset(rhs,i,j,k,c)]=input[65*id+11+11*stage+c-1];
        }
        upload(handle,ASTR_DEVICE_RHS,rdata);check(astr_device_air5_rk_update(handle,first,last,dt,a[stage],b[stage],c[stage],stage==0));
        const auto actual=download(handle,ASTR_DEVICE_Q,q);
        const auto ac=comp?download(handle,ASTR_DEVICE_AIR5_CARRY,carry):std::vector<double>{};
        for(int k=-halo;k<=km+halo;++k) for(int j=-halo;j<=jm+halo;++j) for(int i=-halo;i<=im+halo;++i)
            if(!active(i,j,k)) for(int c=1;c<=11;++c) {
                const size_t p=offset(q,i,j,k,c);if(std::memcmp(&actual[p],&qdata[p],sizeof(double)))return 6;
            }
        for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i) {
            const int id=i+(im+1)*(j+(jm+1)*k);
            for(int c=1;c<=11;++c) {
                out[104*id+11*stage+c-1]=actual[offset(q,i,j,k,c)];
                out[104*id+33+11*stage+c-1]=comp?ac[offset(carry,i,j,k,c)]:input[65*id+54+c-1];
            }
        }
    }
    const auto values=download(handle,ASTR_DEVICE_SAVE,saved);
    const auto oq=comp?download(handle,ASTR_DEVICE_AIR5_ORIGIN,origin):std::vector<double>{};
    const auto oc=comp?download(handle,ASTR_DEVICE_AIR5_ORIGIN_CARRY,origin_carry):std::vector<double>{};
    for(int k=0;k<=km;++k) for(int j=0;j<=jm;++j) for(int i=0;i<=im;++i) {
        const int id=i+(im+1)*(j+(jm+1)*k);
        for(int c=1;c<=11;++c) {
            const double s=values[offset(saved,i,j,k,c)];
            if(!active(i,j,k) && s!=-42.) return 7;
            out[104*id+66+c-1]=s;
            if(comp) {out[104*id+77+c-1]=oq[offset(origin,i,j,k,c)];out[104*id+88+c-1]=oc[offset(origin_carry,i,j,k,c)];}
        }
    }
    auto invalid=download(handle,ASTR_DEVICE_Q,q);invalid[offset(q,0,0,0,1)]=-1.;upload(handle,ASTR_DEVICE_Q,invalid);
    if(astr_device_fluid_cfl(handle,0,1.,1.4,287.,basic)!=ASTR_DEVICE_INVALID_SPECTRUM) return 8;
    check(astr_device_state_destroy(&handle));
    file=std::fopen(argv[2],"wb");if(!file) return 2;
    bool written=std::fwrite(&n,sizeof(n),1,file)==1;
    written=written && std::fwrite(out.data(),sizeof(double),out.size(),file)==out.size();
    written=written && std::fwrite(rates,sizeof(double),4,file)==4 && std::fwrite(keys,sizeof(int64_t),4,file)==4;
    written=written && std::fwrite(profile.data(),sizeof(double),profile.size(),file)==profile.size();
    return std::fclose(file)==0 && written?0:2;
}
