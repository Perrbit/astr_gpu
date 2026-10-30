#ifndef ASTR_INSITU_DEVICE_GEOMETRY_H
#define ASTR_INSITU_DEVICE_GEOMETRY_H

#include "insitu_device_array.h"
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/ArrayHandleConstant.h>
#include <viskores/cont/ArrayHandleTransform.h>
#include <viskores/cont/ArrayHandleZip.h>
#include <viskores/cont/ArrayHandlePermutation.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/filter/contour/ContourFlyingEdges.h>
#include <viskores/worklet/WorkletMapField.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <mpi.h>

namespace astr_insitu {
inline MPI_Datatype numerical_mpi_type() {
#ifdef ASTR_REAL32
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}
using DeviceDiagnostics = viskores::Vec<Real,14>;
static_assert(sizeof(astr_insitu::NativeVector)==3*sizeof(Real), "Coordinates/vectors must match the native kind");
static_assert(sizeof(DeviceDiagnostics)==14*sizeof(Real), "Diagnostic layout must be interleaved native reals");

inline bool m12_hd_grid(const viskores::Id3& dimensions) {
  return dimensions==viskores::Id3(64,32,24) || dimensions==viskores::Id3(476,130,96) ||
    dimensions==viskores::Id3(2250,260,480);
}

struct SelectQAndU : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  VISKORES_EXEC void operator()(const DeviceDiagnostics& d,const astr_insitu::NativeVector& v,
                                Real& q,Real& u) const { q=d[9]; u=v[0]; }
};

struct SelectU : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& v,Real& u) const {u=v[0];}
};
struct SelectQ : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(const DeviceDiagnostics& d,Real& q) const {q=d[9];}
};
struct VelocityComponentZero {
  VISKORES_EXEC_CONT Real operator()(const astr_insitu::NativeVector& v) const {return v[0];}
};
struct IndexPlaneVector : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id nx,ny,plane;
  VISKORES_CONT IndexPlaneVector(viskores::Id x,viskores::Id y,viskores::Id z):nx(x),ny(y),plane(z) {}
  template<class Portal>
  VISKORES_EXEC void operator()(viskores::Id i,const Portal& input,astr_insitu::NativeVector& v) const {
    v=input.Get(i+nx*ny*plane);
  }
};

struct IndexPlane : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  viskores::Id nx,ny,plane;
  VISKORES_CONT IndexPlane(viskores::Id x,viskores::Id y,viskores::Id z):nx(x),ny(y),plane(z) {}
  template <class Portal>
  VISKORES_EXEC void operator()(viskores::Id i,const Portal& input,
                                astr_insitu::NativeVector& velocity,Real& u) const {
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
    invalid=!viskores::IsFinite(area2)?2:(area2>Real(0.)?0:1);
  }
};

struct SurfaceTriangleNumber {
  VISKORES_EXEC_CONT viskores::Id operator()(viskores::Id index) const {return index/3;}
};
struct ValidSurfaceTriangle {
  VISKORES_EXEC_CONT bool operator()(viskores::Id status) const {return status==0;}
};
struct MaximumSurfaceStatus {
  VISKORES_EXEC_CONT viskores::Id operator()(viskores::Id a,viskores::Id b) const {return a>b?a:b;}
};

inline void remove_zero_area_surface_cells(viskores::cont::DataSet& surface) {
  const auto device=viskores::cont::DeviceAdapterTagCuda{};
  const auto cells=surface.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
  const auto xyz=surface.GetCoordinateSystem().GetData().AsArrayHandle<viskores::cont::ArrayHandle<NativeVector>>();
  const auto indices=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{});
  viskores::cont::ArrayHandle<viskores::Id> status,kept;
  viskores::cont::Invoker invoke(device);
  invoke(CheckSurfaceArea{},viskores::cont::ArrayHandleIndex(cells.GetNumberOfCells()),xyz,indices,status);
  synchronize_device_stage("ClassifyPhysicalSurfaceCells");
  if(viskores::cont::Algorithm::Reduce(device,status,viskores::Id(0),MaximumSurfaceStatus{})==2)
    throw std::runtime_error("Physical contour has nonfinite triangles");
  const auto removed=viskores::cont::Algorithm::Reduce(device,status,viskores::Id(0));
  if(!removed) return;
  const auto triangle=viskores::cont::make_ArrayHandleTransform(
    viskores::cont::ArrayHandleIndex(indices.GetNumberOfValues()),SurfaceTriangleNumber{});
  const auto mask=viskores::cont::make_ArrayHandlePermutation(triangle,status);
  viskores::cont::Algorithm::CopyIf(device,indices,mask,kept,ValidSurfaceTriangle{});
  synchronize_device_stage("RemoveZeroAreaPhysicalSurfaceCells");
  if(kept.GetNumberOfValues()==0) surface=viskores::cont::DataSet{};
  else {
    viskores::cont::CellSetSingleType<> clean;
    clean.Fill(xyz.GetNumberOfValues(),viskores::CELL_SHAPE_TRIANGLE,3,kept);
    surface.SetCellSet(clean);
  }
  std::printf("ASTR_INSITU_ZERO_AREA_REMOVED cells=%lld retained=%lld nonfinite=0 tolerance=0 geometry_host_bytes=0\n",
    static_cast<long long>(removed),static_cast<long long>(kept.GetNumberOfValues()/3));
  require_device_only(xyz);require_device_only(status);require_device_only(kept);
}

// Both rendering bridges borrow this immutable view. The owner retains every
// array and its execution token until the graphics consumer has completed.
struct DeviceGeometryView {
  const astr_insitu::NativeVector* coordinates=nullptr;
  const astr_insitu::NativeVector* velocity=nullptr;
  const Real* q=nullptr;
  const Real* u=nullptr;
  const Real* scalar=nullptr;
  const viskores::Id* connectivity=nullptr;
  const viskores::Vec3f_32* display_coordinates=nullptr;
  const float* display_speed=nullptr;
  const viskores::UInt32* display_connectivity=nullptr;
  viskores::Id points=0,cells=0;
  int arity=3;
  int device=-1,step=0;
  Real time=Real(0.);
  Real bounds[6]={Real(0.),Real(0.),Real(0.),Real(0.),Real(0.),Real(0.)};
  Real speed_range[2]={Real(0.),Real(0.)};
};

using DisplayMetadata=viskores::Vec<Real,9>;
using DisplayColor=viskores::Vec<viskores::UInt8,4>;
using ColorRange=viskores::Vec<Real,2>;
struct SelectColorRange : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  bool speed;
  int component;
  VISKORES_CONT SelectColorRange(bool magnitude,int selected):speed(magnitude),component(selected) {}
  VISKORES_EXEC void operator()(Real value,ColorRange& range) const {range=ColorRange(value);}
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& v,ColorRange& range) const {
    range=ColorRange(speed?viskores::Sqrt(viskores::Dot(v,v)):v[component]);
  }
};
struct MergeColorRange {
  VISKORES_EXEC_CONT ColorRange operator()(const ColorRange& a,const ColorRange& b) const {
    return ColorRange(viskores::Min(a[0],b[0]),viskores::Max(a[1],b[1]));
  }
};
struct MapDisplayColor : viskores::worklet::WorkletMapField {
  bool speed=false;
  int component=0;
  Real lower=-Real(1.),inverse=Real(0.5);
  VISKORES_CONT explicit MapDisplayColor(bool magnitude=false,Real minimum=-Real(1.),Real maximum=Real(1.),int selected=0)
    :speed(magnitude),component(selected),lower(minimum),inverse(Real(1.)/(maximum-minimum)) {}
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  template<class Portal>
  VISKORES_EXEC void operator()(Real value,const Portal& palette,DisplayColor& color) const {
    const Real position=viskores::Min(Real(1.),viskores::Max(Real(0.),(value-lower)*inverse))*
      static_cast<Real>(palette.GetNumberOfValues()-1);
    const auto lower=static_cast<viskores::Id>(position);
    const auto upper=viskores::Min(lower+1,palette.GetNumberOfValues()-1);
    const Real weight=position-lower;
    const auto a=palette.Get(lower),b=palette.Get(upper);
    for(int d=0;d<3;++d) color[d]=static_cast<viskores::UInt8>(
      viskores::Min(Real(255.),viskores::Max(Real(0.),Real(255.)*((Real(1.)-weight)*a[d]+weight*b[d])+Real(.5))));
    color[3]=255;
  }
  template<class Portal>
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& value,const Portal& palette,DisplayColor& color) const {
    operator()(speed?viskores::Sqrt(viskores::Dot(value,value)):value[component],palette,color);
  }
};
struct PackScalarDisplay : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& p,viskores::Vec3f_32& display) const {
    for(int d=0;d<3;++d) display[d]=static_cast<float>(p[d]);
  }
};
struct PackSurfaceDisplay : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& p,const astr_insitu::NativeVector& v,
      viskores::Vec3f_32& display,float& speed) const {
    for(int d=0;d<3;++d) display[d]=static_cast<float>(p[d]);
    speed=static_cast<float>(viskores::Sqrt(viskores::Dot(v,v)));
  }
};
struct ScalarDisplayMetadata {
  VISKORES_EXEC_CONT DisplayMetadata operator()(const viskores::Pair<astr_insitu::NativeVector,Real>& input) const {
    DisplayMetadata m;
    bool finite=viskores::IsFinite(input.second);
    for(int d=0;d<3;++d) {
      const auto p=input.first[d];
      finite=finite && viskores::IsFinite(p) && viskores::IsFinite(static_cast<float>(p));
      m[2*d]=m[2*d+1]=p;
    }
    m[6]=m[7]=input.second;m[8]=finite?Real(0.):Real(1.);
    return m;
  }
};
struct SurfaceDisplayMetadata {
  using Entry=viskores::Pair<viskores::Pair<astr_insitu::NativeVector,astr_insitu::NativeVector>,viskores::Pair<Real,Real>>;
  VISKORES_EXEC_CONT DisplayMetadata operator()(const Entry& input) const {
    const auto& p=input.first.first;const auto& v=input.first.second;
    auto m=ScalarDisplayMetadata{}(viskores::make_Pair(p,input.second.first));
    bool finite=m[8]==Real(0.) && viskores::IsFinite(input.second.second);
    for(int d=0;d<3;++d) finite=finite && viskores::IsFinite(v[d]);
    const Real magnitude=viskores::Sqrt(viskores::Dot(v,v));
    m[6]=m[7]=magnitude;
    m[8]=finite && viskores::IsFinite(magnitude) && viskores::IsFinite(static_cast<float>(magnitude))?Real(0.):Real(1.);
    return m;
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
struct InvalidSurfaceIndex {
  viskores::Id points;
  VISKORES_EXEC_CONT viskores::Id operator()(viskores::Id i) const {
    return i<0 || i>=points || static_cast<viskores::UInt64>(i)>static_cast<viskores::UInt64>(0xffffffffu)?1:0;
  }
};
struct PackSurfaceIndex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  viskores::Id points;
  VISKORES_CONT explicit PackSurfaceIndex(viskores::Id n):points(n) {}
  VISKORES_EXEC void operator()(viskores::Id i,viskores::UInt32& display) const {
    display=InvalidSurfaceIndex{points}(i)?0:static_cast<viskores::UInt32>(i);
  }
};

class DeviceGeometryOwner {
  viskores::cont::ArrayHandle<astr_insitu::NativeVector> coordinates,velocity;
  viskores::cont::ArrayHandle<Real> q,u,scalar;
  viskores::cont::ArrayHandle<viskores::Id> connectivity;
  viskores::cont::ArrayHandle<viskores::Vec3f_32> display_coordinates;
  viskores::cont::ArrayHandle<float> display_speed;
  viskores::cont::ArrayHandle<viskores::UInt32> display_connectivity;
  viskores::cont::ArrayHandle<DisplayColor> display_colors;
  viskores::cont::Token token;
  viskores::cont::Token color_token;
  DeviceGeometryView view;
  ColorRange selected_color_range;
  std::map<std::string,viskores::cont::ArrayHandle<Real>> native_point_arrays;
  std::map<std::string,viskores::cont::ArrayHandle<viskores::Id>> native_cell_arrays;
  std::map<std::string,const Real*> native_point_views;
  std::map<std::string,const viskores::Id*> native_cell_views;

  template<class T> const T* pin(const viskores::cont::ArrayHandle<T>& array) {
    require_device_only(array);
    if(array.GetBuffers().size()!=1) throw std::runtime_error("Noncontiguous device surface array");
    return static_cast<const T*>(array.GetBuffers()[0].ReadPointerDevice(
      viskores::cont::DeviceAdapterTagCuda{},token));
  }
public:
  DeviceGeometryOwner(const viskores::cont::DataSet& data,int step,Real time,int empty_arity=3,
      const char* scalar_field=nullptr,bool image_only=false,
      const std::vector<std::string>& point_fields={},const std::vector<std::string>& cell_fields={},bool native_only=false) {
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
    if((!native_only && static_cast<viskores::UInt64>(view.points)>0xffffffffu) ||
       view.cells>std::numeric_limits<viskores::Id>::max()/3)
      throw std::overflow_error("Display surface extent exceeds index capacity");
    coordinates=data.GetCoordinateSystem().GetData().AsArrayHandle<decltype(coordinates)>();
    if(scalar_field) scalar=data.GetPointField(scalar_field).GetData().AsArrayHandle<decltype(scalar)>();
    else {
      velocity=data.GetPointField("velocity").GetData().AsArrayHandle<decltype(velocity)>();
      if(!image_only || data.HasPointField("Q_rs")) q=data.GetPointField("Q_rs").GetData().AsArrayHandle<decltype(q)>();
      if(!image_only) u=data.GetPointField("u").GetData().AsArrayHandle<decltype(u)>();
    }
    const auto cells=data.GetCellSet().AsCellSet<viskores::cont::CellSetSingleType<>>();
    const auto shape=cells.GetCellShape(0);
    if(shape!=viskores::CELL_SHAPE_TRIANGLE && shape!=viskores::CELL_SHAPE_LINE)
      throw std::runtime_error("Device product requires triangle or line connectivity");
    view.arity=shape==viskores::CELL_SHAPE_TRIANGLE?3:2;
    connectivity=cells.GetConnectivityArray(viskores::TopologyElementTagCell{},viskores::TopologyElementTagPoint{});
    if((scalar_field?scalar.GetNumberOfValues()!=view.points:
       velocity.GetNumberOfValues()!=view.points ||
       (q.GetNumberOfValues()!=view.points && (!image_only || view.arity!=2 || q.GetNumberOfValues()!=0)) ||
       (!image_only && u.GetNumberOfValues()!=view.points)) || connectivity.GetNumberOfValues()!=view.arity*view.cells)
      throw std::runtime_error("Device surface fields/connectivity have inconsistent extents");
    require_device_only(coordinates);require_device_only(connectivity);
    if(scalar_field) require_device_only(scalar);
    else {require_device_only(velocity);require_device_only(q);require_device_only(u);}
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    if(!native_only) {
      if(scalar_field || image_only) invoke(PackScalarDisplay{},coordinates,display_coordinates);
      else invoke(PackSurfaceDisplay{},coordinates,velocity,display_coordinates,display_speed);
      synchronize_device_stage("PackSurfaceDisplay");
    }
    DisplayMetadata initial;
    for(int d=0;d<4;++d) {
      initial[2*d]=std::numeric_limits<Real>::infinity();
      initial[2*d+1]=-std::numeric_limits<Real>::infinity();
    }
    initial[8]=Real(0.);
    DisplayMetadata summary;
    if(scalar_field) {
      const auto metadata=viskores::cont::make_ArrayHandleTransform(
        viskores::cont::make_ArrayHandleZip(coordinates,scalar),ScalarDisplayMetadata{});
      summary=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},metadata,initial,MergeDisplayMetadata{});
    } else if(image_only) {
      const auto component=viskores::cont::make_ArrayHandleTransform(velocity,VelocityComponentZero{});
      if(q.GetNumberOfValues()) {
        const auto metadata=viskores::cont::make_ArrayHandleTransform(viskores::cont::make_ArrayHandleZip(
          viskores::cont::make_ArrayHandleZip(coordinates,velocity),viskores::cont::make_ArrayHandleZip(q,component)),
          SurfaceDisplayMetadata{});
        summary=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},metadata,initial,MergeDisplayMetadata{});
      } else {
        const auto metadata=viskores::cont::make_ArrayHandleTransform(viskores::cont::make_ArrayHandleZip(
          viskores::cont::make_ArrayHandleZip(coordinates,velocity),viskores::cont::make_ArrayHandleZip(
            viskores::cont::make_ArrayHandleConstant(Real(0.),view.points),component)),SurfaceDisplayMetadata{});
        summary=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},metadata,initial,MergeDisplayMetadata{});
      }
    } else {
      const auto metadata=viskores::cont::make_ArrayHandleTransform(viskores::cont::make_ArrayHandleZip(
        viskores::cont::make_ArrayHandleZip(coordinates,velocity),viskores::cont::make_ArrayHandleZip(q,u)),
        SurfaceDisplayMetadata{});
      summary=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},metadata,initial,MergeDisplayMetadata{});
    }
    trace_device_memory("display-metadata",0);
    if(summary[8]!=Real(0.)) throw std::runtime_error("Nonfinite device surface or display conversion");
    for(int d=0;d<6;++d) view.bounds[d]=summary[d];
    view.speed_range[0]=summary[6];view.speed_range[1]=summary[7];
    const auto bad_indices=viskores::cont::make_ArrayHandleTransform(connectivity,InvalidSurfaceIndex{view.points});
    if(!native_only) {
      invoke(PackSurfaceIndex(view.points),connectivity,display_connectivity);
      synchronize_device_stage("PackSurfaceIndex");
    }
    if(viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},bad_indices,viskores::Id(0))!=0)
      throw std::runtime_error("Out-of-range device surface connectivity");
    view.coordinates=pin(coordinates);view.connectivity=pin(connectivity);
    if(scalar_field) view.scalar=pin(scalar);
    else {
      view.velocity=pin(velocity);
      if(q.GetNumberOfValues()) view.q=pin(q);
      if(!image_only) {view.u=pin(u);view.display_speed=pin(display_speed);}
    }
    if(!native_only) {
      view.display_coordinates=pin(display_coordinates);
      view.display_connectivity=pin(display_connectivity);
    }
    for(const auto& name:point_fields) {
      auto field=data.GetPointField(name).GetData().AsArrayHandle<viskores::cont::ArrayHandle<Real>>();
      if(field.GetNumberOfValues()!=view.points) throw std::runtime_error("Native point field extent differs");
      native_point_arrays.emplace(name,field);native_point_views[name]=pin(field);
    }
    for(const auto& name:cell_fields) {
      auto field=data.GetCellField(name).GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Id>>();
      if(field.GetNumberOfValues()!=view.cells) throw std::runtime_error("Native cell field extent differs");
      native_cell_arrays.emplace(name,field);native_cell_views[name]=pin(field);
    }
  }
  DeviceGeometryOwner(const DeviceGeometryOwner&)=delete;
  DeviceGeometryOwner& operator=(const DeviceGeometryOwner&)=delete;
  const DeviceGeometryView& get() const {return view;}
  const std::map<std::string,const Real*>& point_fields() const {return native_point_views;}
  const std::map<std::string,const viskores::Id*>& cell_fields() const {return native_cell_views;}
  void retain_display_only() {
    // Final color/audit kernels have completed. Other datasets keep their own
    // references; do not ReleaseResources() on buffers shared with those datasets.
    if(!view.points) return;
    token.DetachFromAll();
    coordinates={};velocity={};q={};u={};scalar={};connectivity={};display_speed={};
    native_point_arrays.clear();native_cell_arrays.clear();native_point_views.clear();native_cell_views.clear();
    view.coordinates=nullptr;view.velocity=nullptr;view.q=nullptr;view.u=nullptr;
    view.scalar=nullptr;view.connectivity=nullptr;view.display_speed=nullptr;
    view.display_coordinates=pin(display_coordinates);view.display_connectivity=pin(display_connectivity);
    trace_device_memory("image-only-owner",std::uint64_t(view.points)*16+std::uint64_t(view.cells)*view.arity*4);
  }
  ColorRange color_range() const {return selected_color_range;}
  const DisplayColor* color_display(const std::vector<double>& values,bool speed=false,
      Real minimum=-Real(1.),Real maximum=Real(1.),int component=0,bool audit_range=false) {
    if(values.size()!=4096*3) throw std::invalid_argument("Invalid bounded display palette");
    if(!std::isfinite(minimum) || !std::isfinite(maximum) || minimum>=maximum || (speed && view.scalar) ||
        component<0 || component>2)
      throw std::invalid_argument("Invalid device scalar display range");
    selected_color_range=ColorRange(std::numeric_limits<Real>::infinity(),
      -std::numeric_limits<Real>::infinity());
    if(!view.points) return nullptr;
    if(!view.scalar && !view.velocity) throw std::logic_error("Product fields retired before a color consumer");
    // Previous synchronous render completed; only color storage may be rewritten.
    color_token.DetachFromAll();
    std::vector<astr_insitu::NativeVector> table(4096);
    for(std::size_t i=0;i<table.size();++i) for(int d=0;d<3;++d) {
      const Real value=values[3*i+d];
      if(!std::isfinite(value) || value<Real(0.) || value>Real(1.))
        throw std::invalid_argument("Nonfinite or out-of-range display palette");
      table[i][d]=value;
    }
    const auto palette=viskores::cont::make_ArrayHandle(table,viskores::CopyFlag::On);
    viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
    if(view.scalar) invoke(MapDisplayColor{false,minimum,maximum},scalar,palette,display_colors);
    else if(speed) invoke(MapDisplayColor{true,minimum,maximum},velocity,palette,display_colors);
    else invoke(MapDisplayColor{false,minimum,maximum,component},velocity,palette,display_colors);
    synchronize_device_stage("MapDisplayColor");
    if(audit_range) {
      viskores::cont::ArrayHandle<ColorRange> ranges;
      if(view.scalar) invoke(SelectColorRange{false,0},scalar,ranges);
      else invoke(SelectColorRange{speed,component},velocity,ranges);
      synchronize_device_stage("SelectColorRange");
      selected_color_range=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},
        ranges,selected_color_range,MergeColorRange{});
      synchronize_device_stage("MergeColorRange");
    }
    require_device_only(display_colors);
    return static_cast<const DisplayColor*>(display_colors.GetBuffers()[0].ReadPointerDevice(
      viskores::cont::DeviceAdapterTagCuda{},color_token));
  }
};

struct PackSliceCoordinate : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  viskores::Id nx;
  astr_insitu::NativeVector origin;
  Real h;
  VISKORES_CONT PackSliceCoordinate(viskores::Id x,astr_insitu::NativeVector o,Real spacing):nx(x),origin(o),h(spacing) {}
  VISKORES_EXEC void operator()(viskores::Id i,astr_insitu::NativeVector& p,Real& q) const {
    p=origin+astr_insitu::NativeVector((i%nx)*h,(i/nx)*h,Real(0.));q=Real(0.);
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
  const Real h=Real(2.)*std::acos(-Real(1.))/global_cells;
  viskores::cont::ArrayHandle<astr_insitu::NativeVector> xyz;
  viskores::cont::ArrayHandle<Real> q;
  viskores::cont::ArrayHandle<viskores::Id> indices;
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  invoke(PackSliceCoordinate(dimensions[0],astr_insitu::NativeVector(offset[0]*h,offset[1]*h,std::acos(-Real(1.))/Real(4.)),h),
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
  if(input.HasPointField("u")) result.AddPointField("u",input.GetPointField("u").GetData());
  result.AddPointField("Q_rs",q);
  return result;
}

// Volume inputs remain borrowed and device-only. Only returned product arrays
// may acquire host mirrors when the compact geometry bridge consumes them.
inline DeviceGeometry extract_tgv_geometry(astr_insitu::NativeVector* velocity_pointer,
    void* diagnostics_pointer,const viskores::Id3& dimensions,
    const viskores::Id3& offset,bool slice_enabled,bool surface_enabled,Real iso=Real(.25),
    DeviceGeometryAudit* audit=nullptr,int global_cells=32,astr_insitu::NativeVector* physical_pointer=nullptr,
    viskores::Id3 global_dimensions=viskores::Id3(0),int diagnostic_components=14,bool image_only=false,
    bool remove_zero_area=false) {
  if(global_dimensions==viskores::Id3(0)) global_dimensions=viskores::Id3(global_cells);
  if(!physical_pointer && global_cells!=32 && global_cells!=256)
    throw std::invalid_argument("Unsupported bounded device geometry resolution");
  for(int d=0;d<3;++d)
    if(dimensions[d]<2 || offset[d]<0 || offset[d]+dimensions[d]>global_dimensions[d]+1)
      throw std::invalid_argument("Device geometry partition exceeds the admitted global grid");
  if(!std::isfinite(iso)) throw std::invalid_argument("Nonfinite contour threshold");
  if(physical_pointer && (slice_enabled || !surface_enabled))
    throw std::invalid_argument("Physical contour coordinates require bounded CURVE Q-only extraction");
  const auto nodes=dimensions[0]*dimensions[1]*dimensions[2];
  const Real h=Real(2.)*std::acos(-Real(1.))/global_cells;
  const astr_insitu::NativeVector origin(offset[0]*h,offset[1]*h,offset[2]*h),spacing(h,h,h);
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  auto velocity=borrow_cuda_array(velocity_pointer,nodes);
  DeviceGeometry products;
  products.has_slice=slice_enabled;
  products.has_surface=surface_enabled;
  if(slice_enabled && offset[2]<=4 && offset[2]+dimensions[2]>4 &&
      !(offset[2]==4 && offset[2]>0)) {
    viskores::cont::ArrayHandle<astr_insitu::NativeVector> values;
    viskores::cont::ArrayHandle<Real> u;
    if(image_only) invoke(IndexPlaneVector(dimensions[0],dimensions[1],4-offset[2]),
      viskores::cont::ArrayHandleIndex(dimensions[0]*dimensions[1]),velocity,values);
    else invoke(IndexPlane(dimensions[0],dimensions[1],4-offset[2]),
      viskores::cont::ArrayHandleIndex(dimensions[0]*dimensions[1]),velocity,values,u);
    synchronize_device_stage("IndexPlane");
    products.slice=viskores::cont::DataSetBuilderUniform().Create(
      viskores::Id3(dimensions[0],dimensions[1],1),
      astr_insitu::NativeVector(origin[0],origin[1],4*h),spacing);
    products.slice.AddPointField("velocity",values);
    if(!image_only) products.slice.AddPointField("u",u);
    require_device_only(values); require_device_only(u);
  }
  if(surface_enabled) {
    viskores::cont::ArrayHandle<Real> q,u;
    if(diagnostic_components==1) {
      q=borrow_cuda_array(static_cast<Real*>(diagnostics_pointer),nodes);
      if(!image_only) invoke(SelectU{},velocity,u);
    } else if(diagnostic_components==14) {
      const auto diagnostics=borrow_cuda_array(static_cast<DeviceDiagnostics*>(diagnostics_pointer),nodes);
      if(image_only) invoke(SelectQ{},diagnostics,q);
      else invoke(SelectQAndU{},diagnostics,velocity,q,u);
      require_device_only(diagnostics);
    } else throw std::invalid_argument("Unsupported explicit device diagnostic layout");
    synchronize_device_stage("SelectQAndU");
    trace_device_memory("selected-q-and-u",nodes*((diagnostic_components==1?0:1)+(image_only?0:1))*sizeof(Real));
    auto grid=viskores::cont::DataSetBuilderUniform().Create(dimensions,origin,spacing);
    grid.AddPointField("Q_rs",q);
    if(!image_only) grid.AddPointField("u",u);
    grid.AddPointField("velocity",velocity);
    if(physical_pointer) grid.AddPointField("_astr_physical_xyz",borrow_cuda_array(physical_pointer,nodes));
    viskores::filter::contour::ContourFlyingEdges contour;
    contour.SetActiveField("Q_rs"); contour.SetIsoValue(iso);
    contour.SetGenerateNormals(false);
    if(image_only) {
      if(physical_pointer) contour.SetFieldsToPass({"Q_rs","velocity","_astr_physical_xyz"});
      else contour.SetFieldsToPass({"Q_rs","velocity"});
    } else {
      if(physical_pointer) contour.SetFieldsToPass({"Q_rs","u","velocity","_astr_physical_xyz"});
      else contour.SetFieldsToPass({"Q_rs","u","velocity"});
    }
    products.surface=contour.Execute(grid);
    synchronize_device_stage("FlyingEdges");
    if(physical_pointer) {
      // The contour's edge fractions interpolate physical coordinates and fields
      // together. Its temporary uniform coordinates are never used for rendering.
      auto physical=products.surface.GetPointField("_astr_physical_xyz").GetData().AsArrayHandle<
        viskores::cont::ArrayHandle<astr_insitu::NativeVector>>();
      auto computational=products.surface.GetCoordinateSystem().GetData().AsArrayHandle<decltype(physical)>();
      viskores::cont::DataSet mapped;
      mapped.SetCellSet(products.surface.GetCellSet());
      mapped.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",physical));
      mapped.AddField(products.surface.GetPointField("Q_rs"));
      mapped.AddField(products.surface.GetPointField("velocity"));
      if(!image_only) {
        mapped.AddField(products.surface.GetPointField("u"));
        mapped.AddPointField("_astr_computational_xyz",computational);
      }
      products.surface=std::move(mapped);
      if(remove_zero_area) remove_zero_area_surface_cells(products.surface);
      else {
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
    }
    require_device_only(q); require_device_only(u);
    if(audit) {
      viskores::cont::Token token;
      audit->q=reinterpret_cast<std::uintptr_t>(q.GetBuffers()[0].ReadPointerDevice(
        viskores::cont::DeviceAdapterTagCuda{},token));
      if(!image_only) audit->u=reinterpret_cast<std::uintptr_t>(u.GetBuffers()[0].ReadPointerDevice(
        viskores::cont::DeviceAdapterTagCuda{},token));
      audit->scalar_bytes=static_cast<std::uint64_t>(nodes)*sizeof(Real);
    }
  }
  if(cudaDeviceSynchronize()!=cudaSuccess)
    throw std::runtime_error("Device geometry extraction failed");
  require_device_only(velocity);
  return products;
}
} // namespace astr_insitu
#endif
