#include <cuda_runtime.h>
#include "qualification_device.cuh"

__device__ double mb0_transform(double value)
{
    return 1.5 * value + 0.25;
}

__device__ double mb0_derivative(const double *field, int i, int n)
{
    const double inv60 = 1.0 / 60.0;
    const double d1 = field[(i + 1) % n] - field[(i + n - 1) % n];
    const double d2 = field[(i + 2) % n] - field[(i + n - 2) % n];
    const double d3 = field[(i + 3) % n] - field[(i + n - 3) % n];
    return (45.0 * d1 - 9.0 * d2 + d3) * inv60;
}
