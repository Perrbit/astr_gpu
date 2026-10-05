#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void probe(double* result)
{
  const int i = threadIdx.x;
  result[i] = rsqrt(double(i + 1));
}

int main()
{
  double* result = nullptr;
  if (cudaMallocManaged(&result, 16 * sizeof(double)) != cudaSuccess) return 1;
  probe<<<1, 16>>>(result);
  if (cudaDeviceSynchronize() != cudaSuccess) return 2;
  std::vector<double> copy(result, result + 16);
  double error = 0.;
  for (int i = 0; i < 16; ++i)
    error = std::fmax(error, std::abs(copy[i] - 1. / std::sqrt(double(i + 1))));
  if (cudaFree(result) != cudaSuccess) return 3;
  std::printf("CUDA compiler probe FP64 max_error=%.17g\n", error);
  return std::isfinite(error) && error <= 2e-15 ? 0 : 4;
}
