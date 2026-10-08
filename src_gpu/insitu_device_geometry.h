#ifndef ASTR_INSITU_DEVICE_GEOMETRY_H
#define ASTR_INSITU_DEVICE_GEOMETRY_H

#include "insitu_device_array.h"
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/filter/contour/ContourFlyingEdges.h>
#include <viskores/worklet/WorkletMapField.h>
#include <cmath>
#include <cstdint>
#include <vector>

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

struct CheckSurfaceArea : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  template<class X,class I>
  VISKORES_EXEC void operator()(viskores::Id cell,const X& xyz,const I& indices,viskores::Id& invalid) const {
    const auto a=xyz.Get(indices.Get(3*cell)),b=xyz.Get(indices.Get(3*cell+1)),c=xyz.Get(indices.Get(3*cell+2));
    const auto cross=viskores::Cross(b-a,c-a);
    const auto area2=viskores::Dot(cross,cross);
    invalid=viskores::IsFinite(area2) && area2>0.?0:1;
  }
};

// Both rendering bridges borrow this immutable view. The owner retains every
// array and its execution token until the graphics consumer has completed.
struct DeviceGeometryView {
  const viskores::Vec3f* coordinates=nullptr;
  const viskores::Vec3f* velocity=nullptr;
  const double* q=nullptr;
  const double* u=nullptr;
  const double* scalar=nullptr;
  const viskores::Id* connectivity=nullptr;
  const viskores::Vec3f_32* display_coordinates=nullptr;
  const float* display_speed=nullptr;
  const viskores::UInt32* display_connectivity=nullptr;
  viskores::Id points=0,cells=0;
  int arity=3;
  int device=-1,step=0;
  double time=0.;
  double bounds[6]={0.,0.,0.,0.,0.,0.};
  double speed_range[2]={0.,0.};
};

using DisplayMetadata=viskores::Vec<double,9>;
using DisplayColor=viskores::Vec<viskores::UInt8,4>;
struct MapDisplayColor : viskores::worklet::WorkletMapField {
  bool speed=false;
  double lower=-1.,inverse=0.5;
  VISKORES_CONT explicit MapDisplayColor(bool magnitude=false,double minimum=-1.,double maximum=1.)
    :speed(magnitude),lower(magnitude?0.:minimum),inverse(1./(maximum-(magnitude?0.:minimum))) {}
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  template<class Portal>
  VISKORES_EXEC void operator()(double value,const Portal& palette,DisplayColor& color) const {
    const double position=viskores::Min(1.,viskores::Max(0.,(value-lower)*inverse))*
      static_cast<double>(palette.GetNumberOfValues()-1);
    const auto lower=static_cast<viskores::Id>(position);
    const auto upper=viskores::Min(lower+1,palette.GetNumberOfValues()-1);
    const double weight=position-lower;
    const auto a=palette.Get(lower),b=palette.Get(upper);
    for(int d=0;d<3;++d) color[d]=static_cast<viskores::UInt8>(
      viskores::Min(255.,viskores::Max(0.,255.*((1.-weight)*a[d]+weight*b[d])+.5)));
    color[3]=255;
  }
  template<class Portal>
  VISKORES_EXEC void operator()(const viskores::Vec3f& value,const Portal& palette,DisplayColor& color) const {
    operator()(viskores::Sqrt(viskores::Dot(value,value)),palette,color);
  }
};
struct PackScalarDisplay : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  VISKORES_EXEC void operator()(const viskores::Vec3f& p,double scalar,
      viskores::Vec3f_32& display,DisplayMetadata& m) const {
    bool finite=viskores::IsFinite(scalar);
    for(int d=0;d<3;++d) {
      display[d]=static_cast<float>(p[d]);
      finite=finite && viskores::IsFinite(p[d]) && viskores::IsFinite(display[d]);
      m[2*d]=m[2*d+1]=p[d];
    }
    m[6]=m[7]=scalar;m[8]=finite?0.:1.;
  }
};
struct PackSurfaceDisplay : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldIn,FieldIn,FieldOut,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,_7);
  VISKORES_EXEC void operator()(const viskores::Vec3f& p,const viskores::Vec3f& v,
      double q,double u,viskores::Vec3f_32& display,float& speed,DisplayMetadata& m) const {
    bool finite=viskores::IsFinite(q) && viskores::IsFinite(u);
    for(int d=0;d<3;++d) {
      display[d]=static_cast<float>(p[d]);
      finite=finite && viskores::IsFinite(p[d]) && viskores::IsFinite(v[d]) &&
        viskores::IsFinite(display[d]);
      m[2*d]=m[2*d+1]=p[d];
    }
    const double magnitude=viskores::Sqrt(viskores::Dot(v,v));
    speed=static_cast<float>(magnitude);
    m[6]=m[7]=magnitude;
    m[8]=finite && viskores::IsFinite(magnitude) && viskores::IsFinite(speed)?0.:1.;
  }
};
struct MergeDisplayMetadata {
  VISKORES_EXEC_CONT DisplayMetadata operator()(const DisplayMetadata& a,const DisplayMetadata& b) const {
    DisplayMetadata result;
    for(int d=0;d<4;++d) {
      result[2*d]=viskores::Min(a[2*d],b[2*d]);
      result[2*d+1]=viskores::Max(a[2*d+1],b[2*d+1]);
    }
    result[8]=a[8]+b[8];
    return result;
  }
};
struct PackSurfaceIndex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id points;
  VISKORES_CONT explicit PackSurfaceIndex(viskores::Id n):points(n) {}
  VISKORES_EXEC void operator()(viskores::Id i,viskores::UInt32& display,viskores::Id& bad) const {
    bad=i<0 || i>=points || static_cast<viskores::UInt64>(i)>
      static_cast<viskores::UInt64>(0xffffffffu)?1:0;
    display=bad?0:static_cast<viskores::UInt32>(i);
  }
};

class DeviceGeometryOwner {
  viskores::cont::ArrayHandle<viskores::Vec3f> coordinates,velocity;
  viskores::cont::ArrayHandle<double> q,u,scalar;
  viskores::cont::ArrayHandle<viskores::Id> connectivity;
  viskores::cont::ArrayHandle<viskores::Vec3f_32> display_coordinates;
  viskores::cont::ArrayHandle<float> display_speed;
  viskores::cont::ArrayHandle<viskores::UInt32> display_connectivity;
  viskores::cont::ArrayHandle<DisplayColor> display_colors;
  viskores::cont::Token token;
  DeviceGeometryView view;

  template<class T> const T* pin(const viskores::cont::ArrayHandle<T>& array) {
    require_device_only(array);
    if(array.GetBuffers().size()!=1) throw std::runtime_error("Noncontiguous device surface array");
    return static_cast<const T*>(array.GetBuffers()[0].ReadPointerDevice(
      viskores::cont::DeviceAdapterTagCuda{},token));
  }
public:
  DeviceGeometryOwner(const viskores::cont::DataSet& data,int step,double time,int empty_arity=3,
      const char* scalar_field=nullptr) {
    if(empty_arity!=2 && empty_arity!=3) throw std::invalid_argument("Invalid device product arity");
    view.arity=empty_arity;
    if(step<0 || !std::isfinite(time) || cudaGetDevice(&view.device)!=cudaSuccess)
      throw std::invalid_argument("Invalid device surface identity");
    view.step=step;view.time=time;
    if(!data.GetNumberOfCoordinateSystems()) return;
    view.points=data.GetCoordinateSystem().GetNumberOfPoints();
    view.cells=data.GetCellSet().GetNumberOfCells();
    if(!view.points) {
      if(view.cells) throw std::runtime_error("Empty product has nonempty connectivity");
      return;
    }
    if(static_cast<viskores::UInt64>(view.points)>0xffffffffu ||
       view.cells>std::numeric_limits<viskores::Id>::max()/3)
      throw std::overflow_error("Display surface extent exceeds index capacity");
    coordinates=data.GetCoordinateSystem().GetData().AsArrayHandle<decltype(coordinates)>();
    if(scalar_field) scalar=data.GetPointField(scalar_field).GetData().AsArrayHandle<decltype(scalar)>();
    else {
      velocity=data.GetPointField("velocity").GetData().AsArrayHandle<decltype(velocity)>();
      q=data.GetPointField("Q_rs").GetData().AsArrayHandle<decltype(q)>();
      u=data.GetPointField("u").GetData().AsArrayHandle<decltype(u)>();
    }
    const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
    const auto shape=cells.GetCellShape(0);
    if(shape!=viskores::CELL_SHAPE_TRIANGLE && shape!=viskores::CELL_SHAPE_LINE)
      throw std::runtime_error("Device product requires triangle or line connectivity");
    view.arity=shape==viskores::CELL_SHAPE_TRIANGLE?3:2;
    connectivity=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{});
    if((scalar_field?scalar.GetNumberOfValues()!=view.points:
       velocity.GetNumberOfValues()!=view.points || q.GetNumberOfValues()!=view.points ||
       u.GetNumberOfValues()!=view.points) || connectivity.GetNumberOfValues()!=view.arity*view.cells)
      throw std::runtime_error("Device surface fields/connectivity have inconsistent extents");
    require_device_only(coordinates);require_device_only(connectivity);
    if(scalar_field) require_device_only(scalar);
    else {require_device_only(velocity);require_device_only(q);require_device_only(u);}
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    viskores::cont::ArrayHandle<DisplayMetadata> metadata;
    if(scalar_field) invoke(PackScalarDisplay{},coordinates,scalar,display_coordinates,metadata);
    else invoke(PackSurfaceDisplay{},coordinates,velocity,q,u,display_coordinates,display_speed,metadata);
    synchronize_device_stage("PackSurfaceDisplay");
    DisplayMetadata initial;
    for(int d=0;d<4;++d) {
      initial[2*d]=std::numeric_limits<double>::infinity();
      initial[2*d+1]=-std::numeric_limits<double>::infinity();
    }
    initial[8]=0.;
    const auto summary=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},
      metadata,initial,MergeDisplayMetadata{});
    if(summary[8]!=0.) throw std::runtime_error("Nonfinite device surface or display conversion");
    for(int d=0;d<6;++d) view.bounds[d]=summary[d];
    view.speed_range[0]=summary[6];view.speed_range[1]=summary[7];
    viskores::cont::ArrayHandle<viskores::Id> bad_indices;
    invoke(PackSurfaceIndex(view.points),connectivity,display_connectivity,bad_indices);
    synchronize_device_stage("PackSurfaceIndex");
    if(viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},bad_indices,viskores::Id(0))!=0)
      throw std::runtime_error("Out-of-range device surface connectivity");
    view.coordinates=pin(coordinates);view.connectivity=pin(connectivity);
    if(scalar_field) view.scalar=pin(scalar);
    else {view.velocity=pin(velocity);view.q=pin(q);view.u=pin(u);view.display_speed=pin(display_speed);}
    view.display_coordinates=pin(display_coordinates);
    view.display_connectivity=pin(display_connectivity);
  }
  DeviceGeometryOwner(const DeviceGeometryOwner&)=delete;
  DeviceGeometryOwner& operator=(const DeviceGeometryOwner&)=delete;
  const DeviceGeometryView& get() const {return view;}
  const DisplayColor* color_display(const std::vector<double>& values,bool speed=false,
      double minimum=-1.,double maximum=1.) {
    if(values.size()!=4096*3) throw std::invalid_argument("Invalid bounded display palette");
    if(!std::isfinite(minimum) || !std::isfinite(maximum) || minimum>=maximum || (speed && view.scalar))
      throw std::invalid_argument("Invalid device scalar display range");
    if(!view.points) return nullptr;
    std::vector<viskores::Vec3f> table(4096);
    for(std::size_t i=0;i<table.size();++i) for(int d=0;d<3;++d) {
      const double value=values[3*i+d];
      if(!std::isfinite(value) || value<0. || value>1.)
        throw std::invalid_argument("Nonfinite or out-of-range display palette");
      table[i][d]=value;
    }
    const auto palette=viskores::cont::make_ArrayHandle(table,viskores::CopyFlag::On);
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    if(view.scalar) invoke(MapDisplayColor{false,minimum,maximum},scalar,palette,display_colors);
    else if(speed) invoke(MapDisplayColor{true,minimum,maximum},velocity,palette,display_colors);
    else invoke(MapDisplayColor{},u,palette,display_colors);
    synchronize_device_stage("MapDisplayColor");
    return pin(display_colors);
  }
};

struct PackSliceCoordinate : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id nx;
  viskores::Vec3f origin;
  double h;
  VISKORES_CONT PackSliceCoordinate(viskores::Id x,viskores::Vec3f o,double spacing):nx(x),origin(o),h(spacing) {}
  VISKORES_EXEC void operator()(viskores::Id i,viskores::Vec3f& p,double& q) const {
    p=origin+viskores::Vec3f((i%nx)*h,(i/nx)*h,0.);q=0.;
  }
};
struct PackSliceTriangle : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  viskores::Id nx;
  VISKORES_CONT explicit PackSliceTriangle(viskores::Id x):nx(x) {}
  VISKORES_EXEC void operator()(viskores::Id i,viskores::Id& index) const {
    const auto cell=i/6,corner=i%6,base=(cell/(nx-1))*nx+cell%(nx-1);
    const viskores::Id offsets[6]={0,1,nx+1,0,nx+1,nx};
    index=base+offsets[corner];
  }
};
inline viskores::cont::DataSet triangulate_device_slice(const viskores::cont::DataSet& input,
    const viskores::Id3& dimensions,const viskores::Id3& offset,int global_cells) {
  viskores::cont::DataSet result;
  if(!input.GetNumberOfCoordinateSystems()) return result;
  const auto count=input.GetCoordinateSystem().GetNumberOfPoints();
  if(!count) return result;
  const double h=2.*std::acos(-1.)/global_cells;
  viskores::cont::ArrayHandle<viskores::Vec3f> xyz;
  viskores::cont::ArrayHandle<double> q;
  viskores::cont::ArrayHandle<viskores::Id> indices;
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  invoke(PackSliceCoordinate(dimensions[0],viskores::Vec3f(offset[0]*h,offset[1]*h,std::acos(-1.)/4.),h),
    viskores::cont::ArrayHandleIndex(count),xyz,q);
  synchronize_device_stage("PackSliceCoordinate");
  invoke(PackSliceTriangle(dimensions[0]),
    viskores::cont::ArrayHandleIndex(6*(dimensions[0]-1)*(dimensions[1]-1)),indices);
  synchronize_device_stage("PackSliceTriangle");
  viskores::cont::CellSetSingleType<> cells;
  cells.Fill(count,viskores::CELL_SHAPE_TRIANGLE,3,indices);
  result.SetCellSet(cells);
  result.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",xyz));
  result.AddPointField("velocity",input.GetPointField("velocity").GetData());
  result.AddPointField("u",input.GetPointField("u").GetData());result.AddPointField("Q_rs",q);
  return result;
}

// Volume inputs remain borrowed and device-only. Only returned product arrays
// may acquire host mirrors when the compact geometry bridge consumes them.
inline DeviceGeometry extract_tgv_geometry(viskores::Vec3f* velocity_pointer,
    DeviceDiagnostics* diagnostics_pointer,const viskores::Id3& dimensions,
    const viskores::Id3& offset,bool slice_enabled,bool surface_enabled,double iso=.25,
    DeviceGeometryAudit* audit=nullptr,int global_cells=32,viskores::Vec3f* physical_pointer=nullptr,
    viskores::Id3 global_dimensions=viskores::Id3(0)) {
  if(global_dimensions==viskores::Id3(0)) global_dimensions=viskores::Id3(global_cells);
  if(global_cells!=32 && global_cells!=256 && !(physical_pointer && (global_cells==64 || global_cells==128)))
    throw std::invalid_argument("Unsupported bounded TGV geometry resolution");
  for(int d=0;d<3;++d)
    if(dimensions[d]<2 || offset[d]<0 || offset[d]+dimensions[d]>global_dimensions[d]+1)
      throw std::invalid_argument("IS8 geometry requires a bounded 32-cell TGV partition");
  if(!std::isfinite(iso)) throw std::invalid_argument("Nonfinite contour threshold");
  if(physical_pointer && ((global_cells!=32 && global_cells!=64 && global_cells!=128 && global_cells!=256) ||
      slice_enabled || !surface_enabled))
    throw std::invalid_argument("Physical contour coordinates require bounded CURVE Q-only extraction");
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
    if(physical_pointer) grid.AddPointField("_astr_physical_xyz",borrow_cuda_array(physical_pointer,nodes));
    viskores::filter::contour::ContourFlyingEdges contour;
    contour.SetActiveField("Q_rs"); contour.SetIsoValue(iso);
    contour.SetGenerateNormals(false);
    if(physical_pointer) contour.SetFieldsToPass({"Q_rs","u","velocity","_astr_physical_xyz"});
    else contour.SetFieldsToPass({"Q_rs","u","velocity"});
    products.surface=contour.Execute(grid);
    synchronize_device_stage("FlyingEdges");
    if(physical_pointer) {
      // The contour's edge fractions interpolate physical coordinates and fields
      // together. Its temporary uniform coordinates are never used for rendering.
      auto physical=products.surface.GetPointField("_astr_physical_xyz").GetData().AsArrayHandle<
        viskores::cont::ArrayHandle<viskores::Vec3f>>();
      auto computational=products.surface.GetCoordinateSystem().GetData().AsArrayHandle<decltype(physical)>();
      viskores::cont::DataSet mapped;
      mapped.SetCellSet(products.surface.GetCellSet());
      mapped.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",physical));
      for(const char* name:{"Q_rs","u","velocity"}) mapped.AddField(products.surface.GetPointField(name));
      mapped.AddPointField("_astr_computational_xyz",computational);
      products.surface=std::move(mapped);
      const auto cells=products.surface.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
      const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{});
      viskores::cont::ArrayHandle<viskores::Id> invalid;
      invoke(CheckSurfaceArea{},viskores::cont::ArrayHandleIndex(cells.GetNumberOfCells()),physical,indices,invalid);
      synchronize_device_stage("CheckPhysicalContourArea");
      if(viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},invalid,viskores::Id(0))!=0)
        throw std::runtime_error("CURVE contour has nonfinite or zero-area triangles");
      synchronize_device_stage("ReducePhysicalContourArea");
      require_device_only(physical);require_device_only(computational);require_device_only(invalid);
    }
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
