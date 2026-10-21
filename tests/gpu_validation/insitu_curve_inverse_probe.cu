#include "insitu_device_curve_trace.h"
#include <cstdio>

using Real=astr_insitu::Real;
using Vector=astr_insitu::NativeVector;

struct Coordinates {
  int mapping;
  __device__ Vector Get(viskores::Id id) const {
    const Real h=Real(2)*acos(Real(-1))/Real(32);
    const Real x=(id%33)*h,y=(id/33%33)*h,z=(id/(33*33))*h;
    if(mapping==1) return {x+Real(.15)*sin(x)*sin(y),y+Real(.15)*sin(y)*sin(z),z+Real(.15)*sin(z)*sin(x)};
    return {x,y+Real(.15)*sin(x)*sin(z),z};
  }
};

__global__ void check_cells(int mapping,int* counts,unsigned long long* maxima) {
  const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
  if(cell>=32*32*32) return;
  astr_insitu::ExecutionPhysicalTraceOwner<int,Coordinates> owner{
    0,Coordinates{mapping},viskores::Id3(32),viskores::Id3(0),viskores::Id3(0),viskores::Id3(32),32};
  const auto indices=owner.PointIds(cell);
  Real corners[8][3];
  for(int i=0;i<8;++i) for(int d=0;d<3;++d) corners[i][d]=owner.coordinates.Get(indices[i])[d];
  const auto points=lcl::makeFieldAccessorNestedSOA(corners,3);
  const Vector exact(Real(.23),Real(.37),Real(.61));
  Vector point,parametric;
  if(lcl::parametricToWorld(lcl::Hexahedron{},points,exact,point)!=lcl::ErrorCode::SUCCESS ||
      lcl::worldToParametric(lcl::Hexahedron{},points,point,parametric)!=lcl::ErrorCode::SUCCESS) {
    atomicAdd(counts,1);return;
  }
  int inverse=-1;
  const int result=owner.Refine(cell,point,parametric,&inverse);
  if(result) atomicAdd(counts+1,1);
  if(inverse>=0 && inverse<8) atomicAdd(counts+2+inverse,1);
  Vector reconstructed;
  lcl::parametricToWorld(lcl::Hexahedron{},points,parametric,reconstructed);
  double residual=0.,parameter_error=0.;
  for(int d=0;d<3;++d) {
    residual=fmax(residual,fabs(reconstructed[d]-point[d]));
    parameter_error=fmax(parameter_error,fabs(parametric[d]-exact[d]));
  }
  atomicMax(maxima,static_cast<unsigned long long>(__double_as_longlong(residual)));
  atomicMax(maxima+1,static_cast<unsigned long long>(__double_as_longlong(parameter_error)));
}

int main() {
  if(cudaSetDevice(0)!=cudaSuccess) return 3;
  for(int mapping=1;mapping<=2;++mapping) {
    int* counts=nullptr;
    unsigned long long* maxima=nullptr;
    if(cudaMalloc(&counts,10*sizeof(int))!=cudaSuccess || cudaMalloc(&maxima,2*sizeof(double))!=cudaSuccess ||
        cudaMemset(counts,0,10*sizeof(int))!=cudaSuccess || cudaMemset(maxima,0,2*sizeof(double))!=cudaSuccess) return 3;
    check_cells<<<128,256>>>(mapping,counts,maxima);
    int values[10];double errors[2];
    if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemcpy(values,counts,sizeof(values),cudaMemcpyDeviceToHost)!=cudaSuccess ||
        cudaMemcpy(errors,maxima,sizeof(errors),cudaMemcpyDeviceToHost)!=cudaSuccess) return 3;
    std::printf("CURVE inverse numerical_bytes=%zu mapping=%d coarse_failures=%d refined_failures=%d newton_ok=%d newton_nonconverged=%d residual=%.17g parameter_error=%.17g\n",
      sizeof(Real),mapping,values[0],values[1],values[2],values[7],errors[0],errors[1]);
    cudaFree(counts);cudaFree(maxima);
    if(values[0] || values[1] || !std::isfinite(errors[0]) || !std::isfinite(errors[1]) ||
        (sizeof(Real)==8 && (errors[0]>2e-10 || errors[1]>2e-10))) return 2;
  }
  return 0;
}
