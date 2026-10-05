#include "insitu_rk45_device.h"
#include <cuda_runtime.h>
#include <vtkFunctionSet.h>
#include <vtkRungeKutta45.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <cstdio>

struct Field {
  int mode;
  __host__ __device__ bool operator()(const double* x, double, double* f) const {
    if (mode == 2 && x[0] > 1.01) return false;
    if (mode == 1) {
      const double radius = std::sqrt(x[0]*x[0] + x[1]*x[1]);
      f[0] = -x[1]/radius; f[1] = x[0]/radius; f[2] = 0.;
    } else if (mode == 3) {
      f[0] = f[1] = f[2] = 0.;
    } else if (mode == 4) {
      f[0] = NAN; f[1] = f[2] = 0.;
    } else {
      f[0] = 1.; f[1] = f[2] = 0.;
    }
    return true;
  }
};

class ReferenceField : public vtkFunctionSet {
public:
  static ReferenceField* New();
  vtkTypeMacro(ReferenceField, vtkFunctionSet);
  using vtkFunctionSet::FunctionValues;
  Field evaluate{0};
  int FunctionValues(double* x, double* f) override { return evaluate(x, x[3], f); }
protected:
  ReferenceField() { NumFuncs = 3; NumIndepVars = 4; }
};
vtkStandardNewMacro(ReferenceField);

struct Result {
  double point[3], next[3], step, actual, error;
  int mode, status;
};

__global__ void run_step(Result* result, int count)
{
  const int i = threadIdx.x;
  if (i >= count) return;
  const double h = 2.*std::acos(-1.)/32.;
  result[i].status = astr_insitu::rk45_step(Field{result[i].mode}, result[i].point,
    result[i].next, 0., result[i].step, result[i].actual, .01*h, .5*h, 1e-8, result[i].error);
}

int main()
{
  constexpr int count = 32;
  Result* results = nullptr;
  if (cudaMallocManaged(&results, sizeof(Result)*count) != cudaSuccess) return 1;
  const double h = 2.*std::acos(-1.)/32.;
  for (int i = 0; i < count; ++i) {
    results[i] = Result{};
    results[i].mode = i%5;
    results[i].point[0] = 1.;
    results[i].point[1] = results[i].mode == 1 ? .01*(i/5) : 0.;
    results[i].step = (i%2 ? -1. : 1.) * (i<16 ? .1*h : .5*h);
  }
  run_step<<<1, 32>>>(results, count);
  if (cudaDeviceSynchronize() != cudaSuccess) return 2;
  double maximum = 0.;
  int matches = 0;
  for (int i = 0; i < count; ++i) {
    const auto& result = results[i];
    if (result.mode == 4) {
      if (result.status != astr_insitu::Nonfinite) return 3;
      continue;
    }
    vtkNew<ReferenceField> field;
    field->evaluate.mode = result.mode;
    vtkNew<vtkRungeKutta45> reference;
    reference->SetFunctionSet(field);
    double point[3] = {result.point[0],result.point[1],result.point[2]}, next[3] = {};
    double step = (i%2 ? -1. : 1.)*(i<16 ? .1*h : .5*h), actual = 0., error = 0.;
    int status = reference->ComputeNextStep(point, nullptr, next, 0., step, actual,
      .01*h, .5*h, 1e-8, error);
    if (status != result.status) return 4;
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(next[d]) || !std::isfinite(result.next[d])) return 7;
      maximum = std::fmax(maximum, std::abs(next[d]-result.next[d]));
    }
    if (!std::isfinite(step) || !std::isfinite(result.step) ||
        !std::isfinite(actual) || !std::isfinite(result.actual) ||
        !std::isfinite(error) || !std::isfinite(result.error)) return 8;
    maximum = std::fmax(maximum, std::abs(step-result.step));
    maximum = std::fmax(maximum, std::abs(actual-result.actual));
    maximum = std::fmax(maximum, std::abs(error-result.error));
    ++matches;
  }
  if (cudaFree(results) != cudaSuccess) return 5;
  std::printf("GPU RK45 versus original VTK matched=%d max_error=%.17g\n", matches, maximum);
  return std::isfinite(maximum) && maximum <= 2e-10 ? 0 : 6;
}
