#include <cuda_runtime.h>

#include <cstdio>

__global__ void saxpy_probe(const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = 2.0 * x[i] + y[i];
  }
}

int main() {
  constexpr int n = 1 << 20;
  double *x = nullptr;
  double *y = nullptr;
  if (cudaMalloc(&x, n * sizeof(double)) != cudaSuccess ||
      cudaMalloc(&y, n * sizeof(double)) != cudaSuccess) {
    return 2;
  }
  cudaMemset(x, 0, n * sizeof(double));
  cudaMemset(y, 0, n * sizeof(double));
  saxpy_probe<<<(n + 255) / 256, 256>>>(x, y, n);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    return 3;
  }
  cudaFree(y);
  cudaFree(x);
  std::puts("A800 profiler probe complete");
  return 0;
}
