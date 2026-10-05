#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

constexpr int count=4356;
void check(cudaError_t code) {
  if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
__global__ void fill(double* data,double offset) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<count) data[i]=offset+i/1024.;
}
__global__ void verify(const double* data,int* bad) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<count && data[i]!=101.+i/1024.) atomicAdd(bad,1);
}
int main() try {
  int devices=0,errors=0;
  check(cudaGetDeviceCount(&devices));
  if (devices<2) return 2;
  for (int source=0;source<2;++source) {
    const int destination=1-source;
    int accessible=0;
    check(cudaDeviceCanAccessPeer(&accessible,destination,source));
    if (!accessible) return 3;
    double* input=nullptr; double* output=nullptr; int* mismatch=nullptr;
    check(cudaSetDevice(source));
    check(cudaMalloc(&input,count*sizeof(double)));
    fill<<<(count+255)/256,256>>>(input,101.);
    check(cudaDeviceSynchronize());
    int* control=nullptr;
    check(cudaMalloc(&control,sizeof(int)));
    check(cudaMemset(control,0,sizeof(int)));
    verify<<<(count+255)/256,256>>>(input,control);
    check(cudaDeviceSynchronize());
    int source_errors=0;
    check(cudaMemcpy(&source_errors,control,sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaFree(control));
    if (source_errors) throw std::runtime_error("source generation/verification control failed");
    check(cudaSetDevice(destination));
    auto enabled=cudaDeviceEnablePeerAccess(source,0);
    if (enabled==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    else check(enabled);
    check(cudaMalloc(&output,count*sizeof(double)));
    check(cudaMalloc(&mismatch,sizeof(int)));
    fill<<<(count+255)/256,256>>>(output,-999.);
    check(cudaDeviceSynchronize());
    check(cudaMemcpyPeerAsync(output,destination,input,source,count*sizeof(double)));
    check(cudaDeviceSynchronize());
    check(cudaMemset(mismatch,0,sizeof(int)));
    verify<<<(count+255)/256,256>>>(output,mismatch);
    check(cudaDeviceSynchronize());
    int bad=0;
    double first=0.;
    check(cudaMemcpy(&bad,mismatch,sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&first,output,sizeof(double),cudaMemcpyDeviceToHost));
    std::printf("CUDA direct peer source=%d destination=%d bytes=%zu source_errors=%d mismatches=%d first=%.17g expected=101\n",
                source,destination,count*sizeof(double),source_errors,bad,first);
    errors+=bad;
    check(cudaFree(mismatch)); check(cudaFree(output));
    check(cudaSetDevice(source)); check(cudaFree(input));
  }
  return errors ? 1 : 0;
} catch (const std::exception& error) {
  std::fprintf(stderr,"CUDA direct peer diagnostic failed: %s\n",error.what());
  return 4;
}
