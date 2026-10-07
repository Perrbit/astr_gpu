#include "insitu_device_wall.h"
#include <viskores/cont/Initialize.h>
#include <mpi.h>
#include <cstdio>
#include <vector>
namespace astr_insitu {std::vector<double> device_display_palette();}

__global__ void initialize(double* xyz,double* fields,int nodes) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=nodes) return;
  const double x=double(id%5)/4.,z=double(id/5%5)/4.;
  const int wall=id/25;
  xyz[id]=x;xyz[id+nodes]=wall+.05*x*z;xyz[id+2*nodes]=z;
  fields[id]=100.+20.*x;fields[id+nodes]=-.2+.4*z;
  fields[id+2*nodes]=15000.*x;fields[id+3*nodes]=wall?-1.:1.;
}
__global__ void inspect(const double* fields,const astr_insitu::DisplayColor* colors,
    int nodes,int component,double lower,double upper,int* bad) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=nodes) return;
  const double value=fmin(1.,fmax(0.,(fields[id+component*nodes]-lower)/(upper-lower)));
  const int expected[4]={int(255.*value+.5),int(255.*(1.-value)+.5),128,255};
  for(int d=0;d<4;++d) if(colors[id][d]!=expected[d]) atomicAdd(bad,1);
}
__global__ void initialize_air5(double* xyz,double* fields) {
  const int id=blockIdx.x*blockDim.x+threadIdx.x;
  if(id>=25) return;
  xyz[id]=.08*double(id%5)/4.;xyz[id+25]=0.;xyz[id+50]=.002*double(id/5)/4.;
  // Binary-exact bins isolate layout selection from 8-bit half-rounding ties.
  for(int c=0;c<17;++c) fields[id+c*25]=double(1u<<c)*double(id)/32.;
  fields[id+17*25]=1.;
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  try {
    if(cudaSetDevice(0)!=cudaSuccess) throw std::runtime_error("Cannot bind wall display probe device");
    viskores::cont::Initialize(argc,argv);
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
    viskores::cont::cuda::internal::CudaAllocator::ForceManagedMemoryOff();
    constexpr int nodes=50;
    double *xyz=nullptr,*fields=nullptr;int* bad=nullptr;
    if(cudaMalloc(&xyz,nodes*3*sizeof(double))!=cudaSuccess ||
       cudaMalloc(&fields,nodes*18*sizeof(double))!=cudaSuccess || cudaMalloc(&bad,sizeof(int))!=cudaSuccess)
      throw std::runtime_error("Cannot allocate wall display probe");
    initialize<<<1,64>>>(xyz,fields,nodes);
    astr_insitu::synchronize_device_stage("wall display initialization");
    std::vector<double> palette(4096*3);
    for(int i=0;i<4096;++i) {
      palette[3*i]=i/4095.;palette[3*i+1]=1.-i/4095.;palette[3*i+2]=.5;
    }
    initialize_air5<<<1,32>>>(xyz,fields);
    astr_insitu::synchronize_device_stage("AIR5 wall layout initialization");
    const int components[]={6,4,5,7,8,9,10,11,12,16};
    for(const int c:components) {
      const double upper=std::ldexp(1.,c);
      auto product=astr_insitu::extract_bc41_wall(xyz,fields,4,4,1,c,"air5_field",18,17);
      astr_insitu::DeviceGeometryOwner owner(product.data,1,1e-10,3,"air5_field");
      const auto colors=owner.color_display(palette,false,0.,upper);
      if(owner.get().points!=25 || owner.get().cells!=32 || product.audit[1]!=0. ||
         std::abs(product.audit[0]-.08*.002)>2e-17 || std::abs(product.measure[0]-.08*.002)>2e-17)
        throw std::runtime_error("AIR5 wall layout geometry differs");
      if(cudaMemset(bad,0,sizeof(int))!=cudaSuccess) throw std::runtime_error("AIR5 wall flag initialization");
      inspect<<<1,32>>>(fields,colors,25,c,0.,upper,bad);
      astr_insitu::synchronize_device_stage("AIR5 wall display inspection");
      int count=0;
      if(cudaMemcpy(&count,bad,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)
        throw std::runtime_error("AIR5 wall scalar reduction download");
      std::printf("ASTR_DEVICE_AIR5_WALL_LAYOUT component=%d points=25 triangles=32 mismatches=%d "
        "control_download_bytes=4 geometry_host_bytes=0\n",c,count);
      std::fflush(stdout);
      if(count) throw std::runtime_error("AIR5 wall scalar color differs");
    }
    auto empty=astr_insitu::extract_bc41_wall(nullptr,nullptr,4,4,0,6,"pressure",18,17);
    astr_insitu::DeviceGeometryOwner empty_owner(empty.data,1,1e-10,3,"pressure");
    if(empty_owner.get().points || empty_owner.get().cells || empty.audit[0]!=0.)
      throw std::runtime_error("AIR5 empty wall has geometry");
    bool rejected=false;
    try {astr_insitu::extract_bc41_wall(xyz,fields,4,4,1,17,"normal",18,17);}
    catch(const std::invalid_argument&) {rejected=true;}
    if(!rejected) throw std::runtime_error("Wall normal was admitted as a scalar product");
    std::printf("ASTR_DEVICE_AIR5_WALL_LAYOUT empty=1 normal_product_rejected=1\n");
    initialize<<<1,64>>>(xyz,fields,nodes);
    astr_insitu::synchronize_device_stage("channel wall layout reinitialization");
    const double ranges[3][2]={{100.,120.},{-.2,.2},{0.,15000.}};
    const char* names[3]={"wall_pressure","wall_shear_x","wall_heat_into_gas"};
    for(int c=0;c<3;++c) {
      auto product=astr_insitu::extract_bc41_wall(xyz,fields,4,4,2,c,names[c]);
      astr_insitu::DeviceGeometryOwner owner(product.data,1,.001,3,names[c]);
      const auto colors=owner.color_display(palette,false,ranges[c][0],ranges[c][1]);
      if(cudaMemset(bad,0,sizeof(int))!=cudaSuccess) throw std::runtime_error("Display probe flag initialization");
      inspect<<<1,64>>>(fields,colors,nodes,c,ranges[c][0],ranges[c][1],bad);
      astr_insitu::synchronize_device_stage("wall display inspection");
      int count=0;
      if(cudaMemcpy(&count,bad,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)
        throw std::runtime_error("Display probe reduction download");
      std::printf("ASTR_DEVICE_WALL_COLOR component=%s mismatches=%d control_download_bytes=4 geometry_host_bytes=0\n",
        names[c],count);
      if(count) throw std::runtime_error("Device scalar color differs from independent linear palette");
    }
    const auto actual=astr_insitu::device_display_palette();
    for(int i:{0,2048,4095}) std::printf("ASTR_DEVICE_WALL_PALETTE index=%d rgb=%.17g,%.17g,%.17g\n",
      i,actual[3*i],actual[3*i+1],actual[3*i+2]);
    if(actual[0]==actual[3*2048] && actual[1]==actual[3*2048+1] && actual[2]==actual[3*2048+2])
      throw std::runtime_error("Actual ParaView palette does not vary over the normalized range");
    cudaFree(bad);cudaFree(fields);cudaFree(xyz);
    MPI_Finalize();return 0;
  } catch(const std::exception& error) {
    std::fprintf(stderr,"Wall display probe: %s\n",error.what());MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
