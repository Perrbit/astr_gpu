#include "device_state.h"
#include "chemistry_properties.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

static void check(int code) { if(code) { std::fprintf(stderr,"device status %d\n",code);std::exit(1); } }
static size_t elements(const AstrDeviceView &v) { return v.stride[4]*v.extent[4]; }
static size_t offset(const AstrDeviceView &v,int i,int j,int k,int c=1)
{
    return (i-v.lower[0])*v.stride[0]+(j-v.lower[1])*v.stride[1]+
           (k-v.lower[2])*v.stride[2]+(c-1)*v.stride[3];
}
int main(int argc,char **argv)
{
    if(argc!=4) return 2;
    FILE *f=std::fopen(argv[1],"rb");if(!f) return 2;
    int n=0;if(std::fread(&n,sizeof(n),1,f)!=1 || n<=0) return 2;
    std::vector<double> input(41*n);
    if(std::fread(input.data(),sizeof(double),input.size(),f)!=input.size()) return 2;
    std::fclose(f);
    constexpr int halo=2;
    const int im=n-1,jm=1,km=1;
    void *handle=nullptr;
    check(astr_device_state_create(0,im,jm,km,halo,1,&handle));
    AstrDeviceView unused;
    if(astr_device_state_view(handle,ASTR_DEVICE_AIR5_TV,&unused)!=ASTR_DEVICE_NOT_ALLOCATED) return 3;
    check(astr_device_air5_prepare(handle));
    int chemistry_status=-1;check(astr_device_air5_check(handle,&chemistry_status));
    if(chemistry_status) return 5;
    const int fields[]={ASTR_DEVICE_Q,ASTR_DEVICE_RHO,ASTR_DEVICE_VELOCITY,
        ASTR_DEVICE_TEMPERATURE,ASTR_DEVICE_AIR5_TV,ASTR_DEVICE_AIR5_SPECIES,ASTR_DEVICE_PRESSURE};
    const int columns[]={-1,0,1,4,10,5,-1};
    AstrDeviceView views[7];std::vector<double> buffers[7];
    for(int b=0;b<7;++b) {
        check(astr_device_state_view(handle,fields[b],&views[b]));
        buffers[b].resize(elements(views[b]),-321.);
        if(b>0 && b<6) for(int k=-halo;k<=km+halo;++k) for(int j=-halo;j<=jm+halo;++j)
            for(int i=-halo;i<=im+halo;++i) for(int c=1;c<=views[b].extent[3];++c) {
                const int row=(i%n+n)%n;
                buffers[b][offset(views[b],i,j,k,c)]=input[41*row+columns[b]+c-1];
            }
        check(astr_device_state_upload(handle,fields[b],buffers[b].data(),buffers[b].size()*sizeof(double)));
    }
    if(views[0].extent[3]!=11) return 4;
    const int first[]={-halo,-halo,-halo},last[]={im+halo,jm+halo,km+halo};
    check(astr_device_air5_conservative(handle,first,last));
    chemistry_status=-1;check(astr_device_air5_check(handle,&chemistry_status));
    if(chemistry_status) return 5;
    check(astr_device_state_download(handle,ASTR_DEVICE_Q,buffers[0].data(),buffers[0].size()*sizeof(double)));
    const std::vector<double> original_q=buffers[0];
    for(int b=1;b<7;++b) {
        std::fill(buffers[b].begin(),buffers[b].end(),NAN);
        check(astr_device_state_upload(handle,fields[b],buffers[b].data(),buffers[b].size()*sizeof(double)));
    }
    check(astr_device_air5_primitives(handle,first,last,std::strcmp(argv[3],"all")==0));
    check(astr_device_air5_check(handle,&chemistry_status));if(chemistry_status) return 6;
    for(int b=0;b<7;++b)
        check(astr_device_state_download(handle,fields[b],buffers[b].data(),buffers[b].size()*sizeof(double)));
    if(original_q!=buffers[0]) return 7;
    for(int k=-halo;k<=km+halo;++k) for(int j=-halo;j<=jm+halo;++j) for(int i=-halo;i<=im+halo;++i) {
        const bool corner=(i<0 || i>im)+(j<0 || j>jm)+(k<0 || k>km)>1;
        for(int b=1;b<7;++b) for(int c=1;c<=views[b].extent[3];++c) {
            const double v=buffers[b][offset(views[b],i,j,k,c)];
            if(corner?!std::isnan(v):!std::isfinite(v)) return 8;
        }
        if(corner) for(int c=1;c<=11;++c) if(buffers[0][offset(views[0],i,j,k,c)]!=-321.) return 9;
    }
    std::vector<double> output(114*n,0.);std::vector<int> statuses(7*n,0);
    for(int i=0;i<n;++i) {
        for(int c=1;c<=11;++c) output[114*i+c-1]=buffers[0][offset(views[0],i,0,0,c)];
        const int starts[]={0,11,12,15,21,16,22};
        for(int b=1;b<7;++b) for(int c=1;c<=views[b].extent[3];++c)
            output[114*i+starts[b]+c-1]=buffers[b][offset(views[b],i,0,0,c)];
        output[114*i+23]=output[114*i+22];
    }
    buffers[0][offset(views[0],0,0,0,1)]=-1.;
    check(astr_device_state_upload(handle,ASTR_DEVICE_Q,buffers[0].data(),buffers[0].size()*sizeof(double)));
    const int origin[]={0,0,0};
    check(astr_device_air5_primitives(handle,origin,origin,0));
    check(astr_device_air5_check(handle,&chemistry_status));if(chemistry_status!=2) return 10;
    const std::vector<double> invalid_q=buffers[0];
    check(astr_device_state_download(handle,ASTR_DEVICE_Q,buffers[0].data(),buffers[0].size()*sizeof(double)));
    if(buffers[0]!=invalid_q) return 11;
    check(astr_device_state_destroy(&handle));
    f=std::fopen(argv[2],"wb");if(!f) return 2;
    bool written=std::fwrite(&n,sizeof(n),1,f)==1;
    written=written && std::fwrite(statuses.data(),sizeof(int),statuses.size(),f)==statuses.size();
    written=written && std::fwrite(output.data(),sizeof(double),output.size(),f)==output.size();
    return std::fclose(f)==0 && written?0:2;
}
