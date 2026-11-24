#include <cuda_runtime.h>

#define ASTR_W1_FUNCTION __device__ __forceinline__
#include "keep_w1_tensor_math.h"

namespace {
__global__ void w1_normal_probe(const astr::w1::NormalTable *table,const double *f,const double *b,
    const double *jac,astr::w1::NormalResult *output,int nodes,int lines,double inv_h,double low,double high)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=nodes*lines) return;
    const int lane=id%lines,j=id/lines;
    output[id]=astr::w1::normal_point(*table,f+lane,b+lane,jac+lane,lines,nodes-1,j,inv_h,low,high);
}

__global__ void w1_tensor_flux(const astr::w1::TensorTable *table,astr::w1::TensorGrid grid,
    astr::w1::TensorState state,astr::w1::TensorFlux *flux,int count)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count) return;
    flux[id]=astr::w1::tensor_flux(*table,grid,state,id%grid.n[0],(id/grid.n[0])%grid.n[1],id/(grid.n[0]*grid.n[1]));
}

__global__ void w1_tensor_rhs(const astr::w1::TensorTable *table,astr::w1::TensorGrid grid,
    astr::w1::TensorState state,const astr::w1::TensorFlux *flux,astr::w1::TensorResult *out,int count)
{
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count) return;
    out[id]=astr::w1::tensor_rhs(*table,grid,state,astr::w1::PackedTensorFlux{flux},
                                id%grid.n[0],(id/grid.n[0])%grid.n[1],id/(grid.n[0]*grid.n[1]));
}
}

extern "C" int astr_w1_normal_probe_gpu(const astr::w1::NormalTable *table,const double *f,const double *b,
    const double *jac,astr::w1::NormalResult *output,int nodes,int lines,double inv_h,double low,double high)
{
    constexpr int block=128;
    w1_normal_probe<<<(nodes*lines+block-1)/block,block>>>(table,f,b,jac,output,nodes,lines,inv_h,low,high);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int astr_w1_tensor_probe_gpu(const astr::w1::TensorTable *table,astr::w1::TensorGrid grid,
    astr::w1::TensorState state,astr::w1::TensorFlux *flux,astr::w1::TensorResult *out)
{
    constexpr int block=128;
    const int count=grid.n[0]*grid.n[1]*grid.n[2],blocks=(count+block-1)/block;
    w1_tensor_flux<<<blocks,block>>>(table,grid,state,flux,count);
    const auto status=cudaDeviceSynchronize();
    if(status!=cudaSuccess) return static_cast<int>(status);
    w1_tensor_rhs<<<blocks,block>>>(table,grid,state,flux,out,count);
    return static_cast<int>(cudaGetLastError());
}
