#ifndef ASTR_INSITU_DEVICE_PRODUCT_AUDIT_H
#define ASTR_INSITU_DEVICE_PRODUCT_AUDIT_H

#include "insitu_device_geometry.h"
#ifdef ASTR_BUILD_TESTING
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nvtx3/nvToolsExt.h>
#endif

namespace astr_insitu {
struct ProductAudit { double field=0.,display=0.; };
__device__ inline void audit_max(double* destination,double value) {
  if(!isfinite(value)) value=INFINITY;
  atomicMax(reinterpret_cast<unsigned long long*>(destination),__double_as_longlong(value));
}

// Read-only oracle: interpolate the source arrays independently of Viskores.
__global__ void inspect_product(DeviceGeometryView view,const double* halo,
    const DeviceDiagnostics* diagnostics,int nx,int ny,int nz,int ox,int oy,int oz,
    int global_cells,int surface,int slice,int crossing,ProductAudit* audit,
    const viskores::Vec3f* computational,const viskores::Vec3f* physical_source,double iso) {
  const auto i=static_cast<viskores::Id>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=view.points) return;
  const double h=2.*acos(-1.)/global_cells;
  const int extent[3]={nx,ny,nz},offset[3]={ox,oy,oz};
  int base[3];double fraction[3];
  const auto xyz=view.coordinates[i];
  const auto sample=computational?computational[i]:xyz;
  double error=0.,display=0.;
  for(int d=0;d<3;++d) {
    const double position=sample[d]/h-offset[d];
    base[d]=max(-3,min(extent[d]+2,static_cast<int>(floor(position))));
    fraction[d]=position-base[d];
    display=fmax(display,fabs(double(view.display_coordinates[i][d])-double(float(xyz[d]))));
  }
  double expected[3]={0.,0.,0.},q=0.;
  const auto halo_nodes=static_cast<viskores::Id>(nx+7)*(ny+7)*(nz+7);
  for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
    const double weight=(x?fraction[0]:1.-fraction[0])*(y?fraction[1]:1.-fraction[1])*
      (z?fraction[2]:1.-fraction[2]);
    const auto index=static_cast<viskores::Id>(base[0]+x+3)+(nx+7)*
      (static_cast<viskores::Id>(base[1]+y+3)+(ny+7)*static_cast<viskores::Id>(base[2]+z+3));
    for(int d=0;d<3;++d) expected[d]+=weight*halo[index+d*halo_nodes];
  }
  for(int d=0;d<3;++d) {
    const double difference=fabs(view.velocity[i][d]-expected[d]);
    error=isfinite(difference)?fmax(error,difference):INFINITY;
  }
  error=fmax(error,fabs(view.u[i]-expected[0]));
  if(surface) {
    for(int d=0;d<3;++d) {
      const double position=sample[d]/h-offset[d];
      base[d]=max(0,min(extent[d]-1,static_cast<int>(floor(position))));
      fraction[d]=position-base[d];
    }
    viskores::Vec3f expected_xyz(0.);
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
      const double weight=(x?fraction[0]:1.-fraction[0])*(y?fraction[1]:1.-fraction[1])*
        (z?fraction[2]:1.-fraction[2]);
      const auto index=static_cast<viskores::Id>(base[0]+x)+(nx+1)*
        (static_cast<viskores::Id>(base[1]+y)+(ny+1)*static_cast<viskores::Id>(base[2]+z));
      q+=weight*diagnostics[index][9];
      if(physical_source) expected_xyz+=weight*physical_source[index];
    }
    if(physical_source) for(int d=0;d<3;++d) {
      const double difference=fabs(xyz[d]-expected_xyz[d]);
      error=isfinite(difference)?fmax(error,difference):INFINITY;
    }
    error=isfinite(q)?fmax(error,fabs(q-iso)):INFINITY;
    error=fmax(error,fabs(view.q[i]-iso));
  }
  if(slice) error=fmax(error,fabs(xyz[2]-acos(-1.)/4.));
  if(crossing) {
    const double pi=acos(-1.),seed=(xyz[1]-pi/8.)/(3.*pi/4./15.);
    error=fmax(error,fabs(xyz[1]-(pi/8.+round(seed)*(3.*pi/4./15.))));
    error=fmax(error,fabs(xyz[2]-pi/4.));
    error=fmax(error,fmax(0.,pi/2.-xyz[0]));
    error=fmax(error,fmax(0.,xyz[0]-3.*pi/2.));
  }
  const auto v=view.velocity[i];
  display=fmax(display,fabs(double(view.display_speed[i])-double(float(sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])))));
  audit_max(&audit->field,error);audit_max(&audit->display,display);
}

inline ProductAudit audit_device_product(const DeviceGeometryView& view,const double* halo,
    const DeviceDiagnostics* diagnostics,const viskores::Id3& extent,const viskores::Id3& offset,
    int global_cells,bool surface,bool slice,bool crossing,
    const viskores::Vec3f* computational=nullptr,const viskores::Vec3f* physical_source=nullptr,double iso=.25) {
  ProductAudit result,*device=nullptr;
  if(!view.points) return result;
  if(cudaMalloc(&device,sizeof(result))!=cudaSuccess)
    throw std::runtime_error("Device product audit allocation failed");
  try {
    if(cudaMemset(device,0,sizeof(result))!=cudaSuccess)
      throw std::runtime_error("Device product audit initialization failed");
    inspect_product<<<(view.points+255)/256,256>>>(view,halo,diagnostics,
      extent[0],extent[1],extent[2],offset[0],offset[1],offset[2],global_cells,surface,slice,crossing,device,
      computational,physical_source,iso);
    synchronize_device_stage("ReadOnlyProductAudit");
    if(cudaMemcpy(&result,device,sizeof(result),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Device product audit scalar readback failed");
  } catch(...) {cudaFree(device);throw;}
  cudaFree(device);
  if(!std::isfinite(result.field) || result.field>2e-10 || result.display!=0.)
    throw std::runtime_error("Device product numerical/display acceptance failed");
  return result;
}

#ifdef ASTR_BUILD_TESTING
inline void test_curve_trace_oracle(const viskores::cont::DataSet& data,const DeviceGeometryView& view,
    const char* name,int rank,int step,std::uint64_t host_budget) {
  const char* prefix=std::getenv("ASTR_INSITU_TEST_CURVE_TRACE_PREFIX");
  if(!prefix || !*prefix) return;
  if(view.arity!=2) throw std::runtime_error("CURVE trace oracle requires line geometry");
  using Accepted=viskores::Vec<double,5>;
  static_assert(sizeof(viskores::Id)==8 && sizeof(viskores::Vec3f)==24 && sizeof(Accepted)==40,
    "CURVE trace test oracle requires FP64 vectors and 64-bit connectivity");
  const std::uint64_t bytes=std::uint64_t(view.points)*12*sizeof(double)+
    std::uint64_t(view.cells)*2*sizeof(viskores::Id);
  if(bytes>host_budget || bytes>64*1024*1024)
    throw std::runtime_error("CURVE trace test oracle host budget exceeded");
  std::vector<viskores::Vec3f> xyz(view.points),velocity(view.points);
  std::vector<Accepted> accepted(view.points);
  std::vector<viskores::Id> particle(view.points),indices(view.cells*2);
  nvtxRangePushA("ASTR_X4_CURVE_TRACE_TEST_ORACLE");
  try {
    if(view.points) {
      auto lengths=data.GetPointField("accepted").GetData().AsArrayHandle<viskores::cont::ArrayHandle<Accepted>>();
      auto ids=data.GetPointField("particle").GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Id>>();
      require_device_only(lengths);require_device_only(ids);
      viskores::cont::Token token;
      const auto length_pointer=lengths.GetBuffers()[0].ReadPointerDevice(viskores::cont::DeviceAdapterTagCuda{},token);
      const auto id_pointer=ids.GetBuffers()[0].ReadPointerDevice(viskores::cont::DeviceAdapterTagCuda{},token);
      if(cudaMemcpy(xyz.data(),view.coordinates,xyz.size()*sizeof(xyz[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
         cudaMemcpy(velocity.data(),view.velocity,velocity.size()*sizeof(velocity[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
         cudaMemcpy(accepted.data(),length_pointer,accepted.size()*sizeof(accepted[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
         cudaMemcpy(particle.data(),id_pointer,particle.size()*sizeof(particle[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
         cudaMemcpy(indices.data(),view.connectivity,indices.size()*sizeof(indices[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)
        throw std::runtime_error("CURVE trace test oracle download failed");
    }
    std::ofstream file(std::string(prefix)+"."+name+".step"+std::to_string(step)+".rank"+std::to_string(rank)+".bin",
      std::ios::binary|std::ios::out);
    const std::uint64_t counts[2]={std::uint64_t(view.points),std::uint64_t(view.cells)};
    file.write(reinterpret_cast<const char*>(counts),sizeof(counts));
    for(const auto* values:{&xyz,&velocity}) if(!values->empty())
      file.write(reinterpret_cast<const char*>(values->data()),values->size()*sizeof((*values)[0]));
    if(!accepted.empty()) file.write(reinterpret_cast<const char*>(accepted.data()),accepted.size()*sizeof(accepted[0]));
    if(!particle.empty()) file.write(reinterpret_cast<const char*>(particle.data()),particle.size()*sizeof(particle[0]));
    if(!indices.empty()) file.write(reinterpret_cast<const char*>(indices.data()),indices.size()*sizeof(indices[0]));
    file.close();
    if(!file) throw std::runtime_error("CURVE trace test oracle publication failed");
  } catch(...) {nvtxRangePop();throw;}
  nvtxRangePop();
  std::printf("ASTR_INSITU_CURVE_TRACE_TEST_ORACLE rank=%d step=%d product=%s oracle_download_bytes=%llu "
    "runtime_zero_readback_evidence=0\n",rank,step,name,static_cast<unsigned long long>(bytes));
}

inline void test_curve_surface_oracle(const viskores::cont::DataSet& data,const DeviceGeometryView& view,
    int rank,int step,std::uint64_t host_budget) {
  const char* prefix=std::getenv("ASTR_INSITU_TEST_CURVE_Q_PREFIX");
  if(!prefix || !*prefix) return;
  const std::uint64_t bytes=std::uint64_t(view.points)*10*sizeof(double)+
    std::uint64_t(view.cells)*3*sizeof(viskores::Id);
  if(bytes>host_budget || bytes>64*1024*1024)
    throw std::runtime_error("CURVE Q test oracle host budget exceeded");
  std::vector<viskores::Vec3f> xyz(view.points),computational(view.points),velocity(view.points);
  std::vector<double> q(view.points);
  std::vector<viskores::Id> indices(view.cells*3);
  if(view.points) {
    const auto field=data.GetPointField("_astr_computational_xyz").GetData().AsArrayHandle<
      viskores::cont::ArrayHandle<viskores::Vec3f>>();
    require_device_only(field);
    viskores::cont::Token token;
    const auto pointer=field.GetBuffers()[0].ReadPointerDevice(viskores::cont::DeviceAdapterTagCuda{},token);
    if(cudaMemcpy(xyz.data(),view.coordinates,xyz.size()*sizeof(xyz[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(computational.data(),pointer,computational.size()*sizeof(computational[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(velocity.data(),view.velocity,velocity.size()*sizeof(velocity[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(q.data(),view.q,q.size()*sizeof(q[0]),cudaMemcpyDeviceToHost)!=cudaSuccess ||
       cudaMemcpy(indices.data(),view.connectivity,indices.size()*sizeof(indices[0]),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("CURVE Q test oracle download failed");
  }
  std::ofstream file(std::string(prefix)+".step"+std::to_string(step)+".rank"+std::to_string(rank)+".bin",
    std::ios::binary|std::ios::out);
  const std::uint64_t counts[2]={std::uint64_t(view.points),std::uint64_t(view.cells)};
  file.write(reinterpret_cast<const char*>(counts),sizeof(counts));
  for(const auto* values:{&xyz,&computational,&velocity}) if(!values->empty())
    file.write(reinterpret_cast<const char*>(values->data()),values->size()*sizeof((*values)[0]));
  if(!q.empty()) file.write(reinterpret_cast<const char*>(q.data()),q.size()*sizeof(q[0]));
  if(!indices.empty()) file.write(reinterpret_cast<const char*>(indices.data()),indices.size()*sizeof(indices[0]));
  file.close();
  if(!file) throw std::runtime_error("CURVE Q test oracle publication failed");
  std::printf("ASTR_INSITU_CURVE_Q_TEST_ORACLE rank=%d step=%d oracle_download_bytes=%llu "
    "runtime_zero_readback_evidence=0\n",rank,step,static_cast<unsigned long long>(bytes));
}
#endif
} // namespace astr_insitu
#endif
