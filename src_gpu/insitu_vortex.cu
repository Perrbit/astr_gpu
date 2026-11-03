#include "../src/insitu_vortex_math.h"
#include <cuda_runtime.h>

namespace {
__global__ void vortex_fields(int count,int stride,const astr::numerical_real* gradients,
    astr::numerical_real* output,int lambda2_slot,int ci_slot,int* invalid) {
  const int node=blockIdx.x*blockDim.x+threadIdx.x;
  if(node>=count) return;
  const auto value=astr_insitu::vortex_values(gradients+9*node,lambda2_slot>=0,ci_slot>=0);
  if(lambda2_slot>=0) {
    output[node+std::size_t(stride)*lambda2_slot]=value.lambda2;
    if(value.lambda2_status!=astr_insitu::VortexOk) atomicExch(invalid,1);
  }
  if(ci_slot>=0) {
    output[node+std::size_t(stride)*ci_slot]=value.lambda_ci;
    if(value.lambda_ci_status!=astr_insitu::VortexOk) atomicExch(invalid,1);
  }
}
}

extern "C" int astr_insitu_vortex_device(int count,int stride,const astr::numerical_real* gradients,
    astr::numerical_real* output,int lambda2_slot,int ci_slot,int* invalid) {
  if(count==0) return 0;
  vortex_fields<<<(count+127)/128,128>>>(count,stride,gradients,output,lambda2_slot,ci_slot,invalid);
  return int(cudaGetLastError());
}
