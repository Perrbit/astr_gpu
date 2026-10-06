#ifndef ASTR_INSITU_DEVICE_GEOMETRY_H
#define ASTR_INSITU_DEVICE_GEOMETRY_H

#include "insitu_device_array.h"
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/filter/contour/ContourFlyingEdges.h>
#include <viskores/worklet/WorkletMapField.h>
#include <cmath>
#include <cstdint>

namespace astr_insitu {
using DeviceDiagnostics = viskores::Vec<double,14>;
static_assert(sizeof(viskores::Vec3f)==3*sizeof(double), "IS8 requires FP64 coordinates/vectors");
static_assert(sizeof(DeviceDiagnostics)==14*sizeof(double), "IS8 diagnostic layout must be interleaved FP64");

struct SelectQAndU : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  VISKORES_EXEC void operator()(const DeviceDiagnostics& d,const viskores::Vec3f& v,
                                double& q,double& u) const { q=d[9]; u=v[0]; }
};

struct IndexPlane : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  viskores::Id nx,ny,plane;
  VISKORES_CONT IndexPlane(viskores::Id x,viskores::Id y,viskores::Id z):nx(x),ny(y),plane(z) {}
  template <class Portal>
  VISKORES_EXEC void operator()(viskores::Id i,const Portal& input,
                                viskores::Vec3f& velocity,double& u) const {
    velocity=input.Get(i+nx*ny*plane); u=velocity[0];
  }
};

struct DeviceGeometry {
  viskores::cont::DataSet slice,surface;
  bool has_slice=false,has_surface=false;
};

struct DeviceGeometryAudit {
  std::uintptr_t q=0,u=0;
  std::uint64_t scalar_bytes=0;
};

// Volume inputs remain borrowed and device-only. Only returned product arrays
// may acquire host mirrors when the compact geometry bridge consumes them.
inline DeviceGeometry extract_tgv_geometry(viskores::Vec3f* velocity_pointer,
    DeviceDiagnostics* diagnostics_pointer,const viskores::Id3& dimensions,
    const viskores::Id3& offset,bool slice_enabled,bool surface_enabled,double iso=.25,
    DeviceGeometryAudit* audit=nullptr,int global_cells=32) {
  if(global_cells!=32 && global_cells!=256)
    throw std::invalid_argument("Unsupported bounded TGV geometry resolution");
  for(int d=0;d<3;++d)
    if(dimensions[d]<2 || offset[d]<0 || offset[d]+dimensions[d]>global_cells+1)
      throw std::invalid_argument("IS8 geometry requires a bounded 32-cell TGV partition");
  if(!std::isfinite(iso)) throw std::invalid_argument("Nonfinite contour threshold");
  const auto nodes=dimensions[0]*dimensions[1]*dimensions[2];
  const double h=2.*std::acos(-1.)/global_cells;
  const viskores::Vec3f origin(offset[0]*h,offset[1]*h,offset[2]*h),spacing(h,h,h);
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  auto velocity=borrow_cuda_array(velocity_pointer,nodes);
  DeviceGeometry products;
  products.has_slice=slice_enabled;
  products.has_surface=surface_enabled;
  if(slice_enabled && offset[2]<=4 && offset[2]+dimensions[2]>4 &&
      !(offset[2]==4 && offset[2]>0)) {
    viskores::cont::ArrayHandle<viskores::Vec3f> values;
    viskores::cont::ArrayHandle<double> u;
    invoke(IndexPlane(dimensions[0],dimensions[1],4-offset[2]),
      viskores::cont::ArrayHandleIndex(dimensions[0]*dimensions[1]),velocity,values,u);
    synchronize_device_stage("IndexPlane");
    products.slice=viskores::cont::DataSetBuilderUniform().Create(
      viskores::Id3(dimensions[0],dimensions[1],1),
      viskores::Vec3f(origin[0],origin[1],4*h),spacing);
    products.slice.AddPointField("velocity",values);
    products.slice.AddPointField("u",u);
    require_device_only(values); require_device_only(u);
  }
  if(surface_enabled) {
    auto diagnostics=borrow_cuda_array(diagnostics_pointer,nodes);
    viskores::cont::ArrayHandle<double> q,u;
    invoke(SelectQAndU{},diagnostics,velocity,q,u);
    synchronize_device_stage("SelectQAndU");
    auto grid=viskores::cont::DataSetBuilderUniform().Create(dimensions,origin,spacing);
    grid.AddPointField("Q_rs",q); grid.AddPointField("u",u);
    grid.AddPointField("velocity",velocity);
    viskores::filter::contour::ContourFlyingEdges contour;
    contour.SetActiveField("Q_rs"); contour.SetIsoValue(iso);
    contour.SetGenerateNormals(false);
    contour.SetFieldsToPass({"Q_rs","u","velocity"});
    products.surface=contour.Execute(grid);
    synchronize_device_stage("FlyingEdges");
    require_device_only(diagnostics); require_device_only(q); require_device_only(u);
    if(audit) {
      viskores::cont::Token token;
      audit->q=reinterpret_cast<std::uintptr_t>(q.GetBuffers()[0].ReadPointerDevice(
        viskores::cont::DeviceAdapterTagCuda{},token));
      audit->u=reinterpret_cast<std::uintptr_t>(u.GetBuffers()[0].ReadPointerDevice(
        viskores::cont::DeviceAdapterTagCuda{},token));
      audit->scalar_bytes=static_cast<std::uint64_t>(nodes)*sizeof(double);
    }
  }
  if(cudaDeviceSynchronize()!=cudaSuccess)
    throw std::runtime_error("Device geometry extraction failed");
  require_device_only(velocity);
  return products;
}
} // namespace astr_insitu
#endif
