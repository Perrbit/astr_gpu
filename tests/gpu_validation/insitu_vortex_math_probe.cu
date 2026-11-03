#include "insitu_vortex_math.h"
#include <cuda_runtime.h>

namespace {
__global__ void evaluate_vortices(int count,const double* gradients,int requested,double* values,int* status,int* work) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=count) return;
  const auto result=astr_insitu::vortex_values(gradients+9*i,(requested&1)!=0,(requested&2)!=0);
  values[2*i]=result.lambda2;values[2*i+1]=result.lambda_ci;
  status[2*i]=result.lambda2_status;status[2*i+1]=result.lambda_ci_status;
  work[2*i]=result.jacobi_sweeps;work[2*i+1]=result.swirl_iterations;
}
struct Buffers {
  double *input=nullptr,*values=nullptr;
  int *status=nullptr,*work=nullptr;
  ~Buffers() {cudaFree(input);cudaFree(values);cudaFree(status);cudaFree(work);}
};
}

// Synthetic-matrix oracle only. Production will consume device gradients
// without either of these host transfers.
extern "C" int astr_vortex_math_gpu(int count,const double* gradients,int requested,double* values,int* status,int* work) {
  if(count==0) return 0;
  if(count<0) return 1;
  Buffers data;
  const std::size_t input_bytes=std::size_t(count)*9*sizeof(double),value_bytes=std::size_t(count)*2*sizeof(double);
  const std::size_t integer_bytes=std::size_t(count)*2*sizeof(int);
  if(cudaMalloc(&data.input,input_bytes)!=cudaSuccess || cudaMalloc(&data.values,value_bytes)!=cudaSuccess ||
     cudaMalloc(&data.status,integer_bytes)!=cudaSuccess || cudaMalloc(&data.work,integer_bytes)!=cudaSuccess) return 2;
  if(cudaMemcpy(data.input,gradients,input_bytes,cudaMemcpyHostToDevice)!=cudaSuccess) return 3;
  evaluate_vortices<<<(count+127)/128,128>>>(count,data.input,requested,data.values,data.status,data.work);
  if(cudaDeviceSynchronize()!=cudaSuccess) return 4;
  if(cudaMemcpy(values,data.values,value_bytes,cudaMemcpyDeviceToHost)!=cudaSuccess ||
     cudaMemcpy(status,data.status,integer_bytes,cudaMemcpyDeviceToHost)!=cudaSuccess ||
     cudaMemcpy(work,data.work,integer_bytes,cudaMemcpyDeviceToHost)!=cudaSuccess) return 5;
  return 0;
}
