#ifndef ASTR_INSITU_DEVICE_PRODUCT_AUDIT_H
#define ASTR_INSITU_DEVICE_PRODUCT_AUDIT_H

#include "insitu_device_geometry.h"

namespace astr_insitu {
struct ProductAudit { double field=0.,display=0.; };
__device__ inline void audit_max(double* destination,double value) {
  if(!isfinite(value)) value=INFINITY;
  atomicMax(reinterpret_cast<unsigned long long*>(destination),__double_as_longlong(value));
}

// Read-only oracle: interpolate the source arrays independently of Viskores.
__global__ void inspect_product(DeviceGeometryView view,const double* halo,
    const DeviceDiagnostics* diagnostics,int nx,int ny,int nz,int ox,int oy,int oz,
    int global_cells,int surface,int slice,int crossing,ProductAudit* audit) {
  const auto i=static_cast<viskores::Id>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i>=view.points) return;
  const double h=2.*acos(-1.)/global_cells;
  const int extent[3]={nx,ny,nz},offset[3]={ox,oy,oz};
  int base[3];double fraction[3];
  const auto xyz=view.coordinates[i];
  double error=0.,display=0.;
  for(int d=0;d<3;++d) {
    const double position=xyz[d]/h-offset[d];
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
      const double position=xyz[d]/h-offset[d];
      base[d]=max(0,min(extent[d]-1,static_cast<int>(floor(position))));
      fraction[d]=position-base[d];
    }
    for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
      const double weight=(x?fraction[0]:1.-fraction[0])*(y?fraction[1]:1.-fraction[1])*
        (z?fraction[2]:1.-fraction[2]);
      const auto index=static_cast<viskores::Id>(base[0]+x)+(nx+1)*
        (static_cast<viskores::Id>(base[1]+y)+(ny+1)*static_cast<viskores::Id>(base[2]+z));
      q+=weight*diagnostics[index][9];
    }
    error=isfinite(q)?fmax(error,fabs(q-.25)):INFINITY;
    error=fmax(error,fabs(view.q[i]-.25));
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
    int global_cells,bool surface,bool slice,bool crossing) {
  ProductAudit result,*device=nullptr;
  if(!view.points) return result;
  if(cudaMalloc(&device,sizeof(result))!=cudaSuccess)
    throw std::runtime_error("Device product audit allocation failed");
  try {
    if(cudaMemset(device,0,sizeof(result))!=cudaSuccess)
      throw std::runtime_error("Device product audit initialization failed");
    inspect_product<<<(view.points+255)/256,256>>>(view,halo,diagnostics,
      extent[0],extent[1],extent[2],offset[0],offset[1],offset[2],global_cells,surface,slice,crossing,device);
    synchronize_device_stage("ReadOnlyProductAudit");
    if(cudaMemcpy(&result,device,sizeof(result),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("Device product audit scalar readback failed");
  } catch(...) {cudaFree(device);throw;}
  cudaFree(device);
  if(!std::isfinite(result.field) || result.field>2e-10 || result.display!=0.)
    throw std::runtime_error("Device product numerical/display acceptance failed");
  return result;
}
} // namespace astr_insitu
#endif
