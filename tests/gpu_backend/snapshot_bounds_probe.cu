#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "device_state.h"
#include "fluid_explicit.h"

namespace {
struct Bounds {int upper[3],low[3],high[3],axis;};
__host__ __device__ bool observed(int i,int j,int k,Bounds b)
{
    const int p[3]={i,j,k};
    for(int a=0;a<3;++a) if(a!=b.axis && (p[a]<0 || p[a]>b.upper[a])) return false;
    return !(p[b.axis]<0 && b.low[b.axis]) && !(p[b.axis]>b.upper[b.axis] && b.high[b.axis]);
}
__host__ __device__ double value(size_t id,int i,int j,int k,int component)
{
    return i==1 && j==2 && k==1 && component==2?-0.0:.125+.25*id;
}
__global__ void fill_defined(AstrDeviceView view,Bounds b,size_t count)
{
    const size_t id=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=count) return;
    const size_t node=id%view.stride[3];
    const int i=node%view.extent[0]+view.lower[0],j=(node/view.extent[0])%view.extent[1]+view.lower[1],
        k=node/(static_cast<size_t>(view.extent[0])*view.extent[1])+view.lower[2],component=id/view.stride[3]+1;
    if(observed(i,j,k,b)) static_cast<double *>(view.data)[id]=value(id,i,j,k,component);
    // Undefined physical exterior and corner storage is intentionally not written.
}
void check(int status,const char *label)
{
    if(status==0) return;
    std::fprintf(stderr,"SNAPSHOT_BOUNDS_FAIL %s status=%d\n",label,status);std::exit(1);
}
}

int main()
{
    int cases=0;
    for(int debug=0;debug<2;++debug) for(int axis=0;axis<3;++axis) for(int mode=0;mode<4;++mode) {
        void *state=nullptr;
        check(astr_device_state_create(0,6,5,4,3,debug,&state),"create");
        check(astr_device_fluid_prepare(state,1),"primitives");
        check(astr_device_fluid_rhs_prepare(state,1),"stress allocation");
        AstrDeviceView view{};check(astr_device_state_view(state,ASTR_DEVICE_STRESS,&view),"stress view");
        const size_t count=view.stride[3]*6;
        Bounds b{{6,5,4},{0,0,0},{0,0,0},axis};
        b.low[axis]=mode==1 || mode==2;b.high[axis]=mode==1 || mode==3;
        fill_defined<<<(count+255)/256,256>>>(view,b,count);
        check(cudaGetLastError(),"fill");check(cudaDeviceSynchronize(),"fill completion");
        std::vector<double> actual(count),expected(count);
        for(size_t id=0;id<count;++id) {
            const size_t node=id%view.stride[3];
            const int i=node%view.extent[0]+view.lower[0],j=(node/view.extent[0])%view.extent[1]+view.lower[1],
                k=node/(static_cast<size_t>(view.extent[0])*view.extent[1])+view.lower[2],component=id/view.stride[3]+1;
            expected[id]=observed(i,j,k,b)?value(id,i,j,k,component):0.0;
        }
        check(astr_device_state_snapshot_axis(state,ASTR_DEVICE_STRESS,axis,b.low,b.high,actual.data(),count*sizeof(double)),"copy");
        if(std::memcmp(actual.data(),expected.data(),count*sizeof(double))!=0) {
            std::fprintf(stderr,"SNAPSHOT_BOUNDS_FAIL axis=%d mode=%d debug=%d\n",axis,mode,debug);return 1;
        }
        check(astr_device_state_destroy(&state),"destroy");++cases;
    }
    std::printf("SNAPSHOT_BOUNDS_PASS cases=%d values_and_signed_zero_exact undefined_exterior_not_read\n",cases);
    return 0;
}
