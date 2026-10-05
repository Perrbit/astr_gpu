#include "insitu_streamline_worklet.h"
#include "insitu_device_array.h"
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/cont/cuda/DeviceAdapterCuda.h>
#include <viskores/filter/flow/worklet/Field.h>
#include <viskores/filter/flow/worklet/GridEvaluators.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <exception>
#include <vector>

__global__ void generate_velocity(viskores::Vec3f* velocity, int count, double h, int mode) {
  const int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= count) return;
  const double x = (i%33)*h, y = ((i/33)%33)*h;
  velocity[i] = mode == 0 ? viskores::Vec3f(1.,0.,0.) : viskores::Vec3f(1.+.02*y,.01*x,0.);
}

struct DeviceVelocity {
  viskores::Vec3f* pointer = nullptr;
  explicit DeviceVelocity(int count) {
    if (cudaMalloc(&pointer,sizeof(viskores::Vec3f)*count) != cudaSuccess)
      throw std::runtime_error("Cannot allocate synthetic device velocity");
  }
  ~DeviceVelocity() { if (pointer) cudaFree(pointer); }
  DeviceVelocity(const DeviceVelocity&) = delete;
  DeviceVelocity& operator=(const DeviceVelocity&) = delete;
};

struct Reference {
  int mode;
  __host__ __device__ bool operator()(const double* x, double, double* f) const {
    double u = mode == 0 ? 1. : 1.+.02*x[1], v = mode == 0 ? 0. : .01*x[0];
    double norm = std::sqrt(u*u+v*v);
    f[0] = u/norm; f[1] = v/norm; f[2] = 0.;
    return true;
  }
};

int main(int argc, char** argv)
try {
  viskores::cont::Initialize(argc, argv);
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  const double pi = std::acos(-1.), h = 2.*pi/32.;
  auto grid = viskores::cont::DataSetBuilderUniform().Create(viskores::Id3(33,33,33),
      viskores::Vec3f(0.,0.,0.), viskores::Vec3f(h,h,h));
  std::vector<viskores::Vec3f> seeds;
  for (int i = 0; i < 16; ++i) seeds.emplace_back(pi/2.,pi/8.+i*(6.*pi/8.)/15.,pi/4.);
  auto positions = viskores::cont::make_ArrayHandle(seeds,viskores::CopyFlag::On);
  viskores::cont::ArrayHandleConstant<double> steps(.1*h,16);
  double maximum = 0., adaptive_length_difference = 0.;
  int differing_adaptive_steps = 0;
  for (int mode = 0; mode < 2; ++mode) {
    constexpr int nodes = 33*33*33;
    DeviceVelocity allocation(nodes);
    generate_velocity<<<(nodes+255)/256,256>>>(allocation.pointer,nodes,h,mode);
    if (cudaDeviceSynchronize() != cudaSuccess) return 7;
    auto velocity = astr_insitu::borrow_cuda_array(allocation.pointer,nodes);
    astr_insitu::require_device_only(velocity);
    using Field = viskores::worklet::flow::VelocityField<decltype(velocity)>;
    using Grid = viskores::worklet::flow::GridEvaluator<Field>;
    astr_insitu::LengthGrid<Grid> evaluate(Grid(grid,Field(velocity)));
    viskores::cont::ArrayHandle<viskores::Vec3f> next;
    viskores::cont::ArrayHandle<double> suggested, actual, error;
    viskores::cont::ArrayHandle<viskores::Int32> status;
    invoke(astr_insitu::RK45StepWorklet(.01*h,.5*h,1e-8),positions,steps,evaluate,
           next,suggested,actual,error,status);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    astr_insitu::require_device_only(velocity);
    auto p = next.ReadPortal(); auto s = suggested.ReadPortal();
    auto a = actual.ReadPortal(); auto e = error.ReadPortal(); auto ok = status.ReadPortal();
    for (int i = 0; i < 16; ++i) {
      double input[3] = {seeds[i][0],seeds[i][1],seeds[i][2]}, output[3], step=.1*h, length=0., estimate=0.;
      int code = astr_insitu::rk45_step(Reference{mode},input,output,0.,step,length,
                                      .01*h,.5*h,1e-8,estimate);
      if (code != 0 || ok.Get(i) != code) return 2;
      if (!std::isfinite(s.Get(i)) || !std::isfinite(a.Get(i)) || !std::isfinite(e.Get(i))) return 4;
      if (mode == 1) {
        // Interpolated and analytic fields differ by rounding. Near zero error,
        // VTK's exact-zero branch can choose different accepted lengths. Compare
        // trajectories at the same accepted length, not unequal arc locations.
        adaptive_length_difference = std::fmax(adaptive_length_difference,std::abs(a.Get(i)-length));
        differing_adaptive_steps += std::abs(a.Get(i)-length)>2e-10;
        if (e.Get(i)>1e-8 || estimate>1e-8 || std::abs(s.Get(i))<.01*h || std::abs(s.Get(i))>.5*h)
          return 8;
        step=a.Get(i);
        code=astr_insitu::cash_karp_step(Reference{mode},input,output,0.,step,length,estimate);
        if (code != 0) return 9;
      }
      for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(p.Get(i)[d])) return 3;
        maximum = std::fmax(maximum,std::abs(p.Get(i)[d]-output[d]));
      }
      if (mode == 0) maximum = std::fmax(maximum,std::abs(s.Get(i)-step));
      maximum = std::fmax(maximum,std::abs(a.Get(i)-length));
      maximum = std::fmax(maximum,std::abs(e.Get(i)-estimate));
    }
    astr_insitu::require_device_only(velocity);
  }
  std::printf("Viskores CUDA FP64 grid/RK45 cases=32 same_arclength_max_error=%.17g input_host_mirror=0 affine_adaptive_different=%d length_difference=%.17g\n",
    maximum,differing_adaptive_steps,adaptive_length_difference);
  return std::isfinite(maximum) && maximum<=2e-10 ? 0 : 5;
} catch (const std::exception& error) {
  std::fprintf(stderr,"Viskores CUDA probe failed: %s\n",error.what());
  return 6;
}
