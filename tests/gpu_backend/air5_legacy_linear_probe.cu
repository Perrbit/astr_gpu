#include "device_state.h"
#include "chemistry_properties.h"
#include "chemistry_transport.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void check(int status)
{
    if(status) {std::fprintf(stderr,"legacy linear test status %d\n",status);std::exit(1);}
}

int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    FILE *input=std::fopen(argv[1],"rb"),*ref=std::fopen(argv[2],"rb");
    if(!input || !ref)return 2;
    constexpr int im=12,jm=11,km=10,h=5,nc=11;
    constexpr size_t nx=im+1+2*h,ny=jm+1+2*h,nz=km+1+2*h;
    constexpr size_t padded=nx*ny*nz,real_count=(im+1)*(jm+1)*(km+1);
    std::vector<double> source(nc*padded),got(nc*padded),expected(nc*real_count);
    if(std::fread(source.data(),8,source.size(),input)!=source.size())return 2;
    check(astr_device_bind(0));
    size_t mismatches=0;
    for(int scalar=0;scalar<2;++scalar) {
        void *handle=nullptr;
        check(astr_device_state_create(0,im,jm,km,h,1,&handle));
        check(astr_device_air5_prepare(handle));
        check(astr_device_air5_legacy_linear_prepare(handle,scalar));
        for(int axis=0;axis<3;++axis) {
            const int field=axis==1 ? ASTR_DEVICE_Q : ASTR_DEVICE_FILTER_WORK;
            for(int ends=0;ends<6;++ends) {
                if(std::fread(expected.data(),8,expected.size(),ref)!=expected.size())return 2;
                check(astr_device_state_upload(handle,ASTR_DEVICE_Q,source.data(),source.size()*8));
                size_t changed=0,by_depth[6]{}; double error=0;
                for(int sel=0;sel<(scalar ? nc : 1);++sel) {
                    const size_t work_count=(scalar ? 1 : nc)*padded;
                    const double *work_base=source.data()+sel*padded;
                    check(astr_device_state_upload(handle,ASTR_DEVICE_FILTER_WORK,work_base,work_count*8));
                    if(ends>=4)check(astr_device_air5_legacy_linear_wall(handle,axis,scalar ? sel+1 : 0,1,1,ends-4));
                    else check(astr_device_air5_legacy_linear_axis(handle,axis,scalar ? sel+1 : 0,ends&1,(ends>>1)&1));
                    const size_t elements=axis==1 ? source.size() : work_count;
                    check(astr_device_state_download(handle,field,got.data(),elements*8));
                    const int first=axis==1 && scalar ? sel : 0;
                    const int last=scalar ? first+1 : nc;
                    for(int c=first;c<last;++c) {
                        size_t index=(scalar ? sel : c)*real_count;
                        for(int k=0;k<=km;++k)for(int j=0;j<=jm;++j)for(int i=0;i<=im;++i) {
                            const size_t at=c*padded+((k+h)*ny+j+h)*nx+i+h;
                            const double want=expected[index++];
                            if(std::memcmp(&got[at],&want,8)) {
                                ++changed; error=std::fmax(error,std::abs(got[at]-want));
                                const int node[]={i,j,k},dim[]={im,jm,km};
                                const int low=(ends>=4 || (ends&1)) ? node[axis] : 5;
                                const int high=(ends>=4 || (ends&2)) ? dim[axis]-node[axis] : 5;
                                const int depth=low<high ? low : high;
                                ++by_depth[depth<5 ? depth : 5];
                            }
                            got[at]=(axis==1 ? source.data() : work_base)[at];
                        }
                    }
                    // Non-target components and ghost nodes must be untouched.
                    const double *base=axis==1 ? source.data() : work_base;
                    if(std::memcmp(got.data(),base,elements*8))return 3;
                    if(axis==1 && scalar)check(astr_device_state_upload(handle,ASTR_DEVICE_Q,source.data(),source.size()*8));
                }
                mismatches+=changed;
                std::printf("scalar=%d axis=%d ends=%d words=%zu max_abs=%.17g\n",scalar,axis,ends,changed,error);
                if(changed) {
                    std::fputs("  mismatch by depth:",stdout);
                    for(const auto n:by_depth)std::printf(" %zu",n);
                    std::puts("");
                }
            }
        }
        check(astr_device_state_destroy(&handle));
    }
    if(std::fclose(input) || std::fclose(ref))return 2;
    return mismatches ? 4 : 0;
}
