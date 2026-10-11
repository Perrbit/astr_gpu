#ifndef ASTR_INSITU_M12_PRODUCTS_H
#define ASTR_INSITU_M12_PRODUCTS_H
#include "insitu_device_structured.h"
#include <viskores/cont/ArrayHandleExtractComponent.h>
#include <cstdlib>

namespace astr_insitu {
// For this orthogonal Cartesian fixture, extract only the two coordinate
// layers bracketing the requested plane, not every tetrahedron in the volume.
struct M12Slab : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6);
  viskores::Id3 full,slab;
  int axis,first;
  VISKORES_CONT M12Slab(viskores::Id3 f,viskores::Id3 s,int a,int i):full(f),slab(s),axis(a),first(i) {}
  template<class X,class V,class T>
  VISKORES_EXEC void operator()(viskores::Id id,const X& xyz,const V& velocity,const T& temperature,
      viskores::Vec3f& position,viskores::Vec3f& fields) const {
    viskores::Id3 p(id%slab[0],id/slab[0]%slab[1],id/(slab[0]*slab[1]));
    p[axis]+=first;
    const auto source=p[0]+full[0]*(p[1]+full[1]*p[2]);
    position=xyz.Get(source);
    const auto v=velocity.Get(source);
    // Private interpolation payload; never published as a physical velocity.
    fields=viskores::Vec3f(v[0],temperature.Get(source),v[2]);
  }
};
struct M12Component : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  int component;
  VISKORES_CONT explicit M12Component(int c):component(c) {}
  VISKORES_EXEC void operator()(const viskores::Vec3f& fields,double& value) const {value=fields[component];}
};
using M12Range=viskores::Vec<double,2>;
struct M12WindowRange : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  double xmin,xmax,ymax;
  VISKORES_CONT M12WindowRange(double lo,double hi,double y):xmin(lo),xmax(hi),ymax(y) {}
  VISKORES_EXEC void operator()(const viskores::Vec3f& p,double value,M12Range& range) const {
    range=p[0]>=xmin && p[0]<=xmax && p[1]<=ymax?M12Range(value,value):M12Range(INFINITY,-INFINITY);
  }
};
struct M12MergeRange {
  VISKORES_EXEC_CONT M12Range operator()(const M12Range& a,const M12Range& b) const {
    return {viskores::Min(a[0],b[0]),viskores::Max(a[1],b[1])};
  }
};
inline double m12_weak_half_range(const M12Range& actual,const M12Range& main) {
  const double amplitude=std::max(std::abs(actual[0]),std::abs(actual[1]));
  if(amplitude==0. || amplitude>=.01*(main[1]-main[0])*.5) return 0.;
  const double decade=std::pow(10.,std::floor(std::log10(amplitude)));
  for(double factor:{1.,2.,5.,10.}) {
    const double bound=factor*decade;
    if(bound>=amplitude && std::isfinite(bound) && std::isfinite(1./(2.*bound))) return bound;
  }
  throw std::runtime_error("M12 weak display range is not representable");
}
struct M12AuxiliaryColors {
  std::shared_ptr<DeviceGeometryOwner> geometry;
  viskores::cont::ArrayHandle<DisplayColor> colors;
  viskores::cont::Token token;
};
inline void append_m12_planes(std::vector<DeviceMesh>& meshes,MPI_Comm comm,int step,double time,
    const viskores::Id3& global,const viskores::Id3& extent,const viskores::Id3& offset,
    double* xyz_pointer,double* velocity_pointer,double* temperature_pointer,int plane_j,
    std::uint64_t budget,std::uint64_t retained) {
  const auto dims=extent+viskores::Id3(1);
  const auto nodes=dims[0]*dims[1]*dims[2];
  auto xyz=borrow_cuda_array(reinterpret_cast<viskores::Vec3f*>(xyz_pointer),nodes);
  auto velocity=borrow_cuda_array(reinterpret_cast<viskores::Vec3f*>(velocity_pointer),nodes);
  auto temperature=borrow_cuda_array(temperature_pointer,nodes);
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  static std::map<std::string,M12Range> frozen_ranges;
  for(int axis:{2,1}) {
    const int first=axis==2?int(extent[2]/2-1):plane_j;
    auto cells=extent,start=offset;cells[axis]=1;start[axis]+=first;
    const auto slab=cells+viskores::Id3(1);
    const auto count=slab[0]*slab[1]*slab[2];
    viskores::cont::ArrayHandle<viskores::Vec3f> positions,fields;
    invoke(M12Slab{dims,slab,axis,first},viskores::cont::ArrayHandleIndex(count),
      xyz,velocity,temperature,positions,fields);
    synchronize_device_stage("M12PlaneSlab");
    viskores::cont::Token token;
    const auto px=positions.PrepareForInput(viskores::cont::DeviceAdapterTagCuda{},token);
    const auto pf=fields.PrepareForInput(viskores::cont::DeviceAdapterTagCuda{},token);
    plane_geometry::Point origin{{0.,0.,axis==2?45.:0.}},normal{{0.,0.,0.}};
    if(axis==1) origin[1]=1.;
    normal[axis]=1.;
    const structured_geometry::Partition partition(global,cells,start);
    auto product=structured_geometry::extract_plane(const_cast<viskores::Vec3f*>(px.GetIteratorBegin()),
      const_cast<viskores::Vec3f*>(pf.GetIteratorBegin()),nullptr,partition,origin,normal,budget,retained+count*48,true);
    for(int component:axis==2?std::vector<int>{0,1}:std::vector<int>{2}) {
      const std::string name=component==0?"velocity_slice":(component==1?"temperature_slice":"spanwise_slice");
      const std::string field=component==0?"u_Uinf":(component==1?"T_Tinf":"w_Uinf");
      auto data=product.data;
      std::vector<M12Range> windows;
      viskores::cont::ArrayHandle<double> scalar;
      viskores::cont::ArrayHandle<viskores::Vec3f> coordinates;
      if(data.GetNumberOfCoordinateSystems()) {
        const auto payload=data.GetPointField("velocity").GetData().AsArrayHandle<
          viskores::cont::ArrayHandle<viskores::Vec3f>>();
        invoke(M12Component{component},payload,scalar);
        synchronize_device_stage("M12PlaneScalar");
        data.AddPointField(field,scalar);
        coordinates=data.GetCoordinateSystem().GetData().AsArrayHandle<decltype(coordinates)>();
      }
      for(int window=0;window<(axis==2?4:1);++window) {
        M12Range range(INFINITY,-INFINITY);
        if(data.GetNumberOfCoordinateSystems()) {
          viskores::cont::ArrayHandle<M12Range> ranges;
          const double xmin=window==0?0.:(window-1)*100.;
          const double xmax=window==0?(axis==2?1100.:300.):window*100.;
          invoke(M12WindowRange{xmin,xmax,window==0?150.:20.},coordinates,scalar,ranges);
          synchronize_device_stage("M12PlaneWindowRange");
          range=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},ranges,
            M12Range(INFINITY,-INFINITY),M12MergeRange{});
          synchronize_device_stage("M12PlaneRangeReduction");
          require_device_only(ranges);
        }
        double low=range[0],high=range[1];
        trace_mpi(MPI_Allreduce(MPI_IN_PLACE,&low,1,MPI_DOUBLE,MPI_MIN,comm));
        trace_mpi(MPI_Allreduce(MPI_IN_PLACE,&high,1,MPI_DOUBLE,MPI_MAX,comm));
        if(!std::isfinite(low) || !std::isfinite(high) || low>high)
          throw std::runtime_error("M12 physical slice window has no finite samples");
        windows.push_back(M12Range(low,high));
      }
      if(!frozen_ranges.count(field)) {
        const auto range=windows[0];
        if(component==2) {
          double w=std::max(1.e-6,1.1*std::max(std::abs(range[0]),std::abs(range[1])));
          if(const char* configured=std::getenv("ASTR_INSITU_M12_W_COLOR_MAX")) {
            std::size_t end=0;
            w=std::stod(configured,&end);
            if(configured[end]!='\0' || !std::isfinite(w) || w<=0.)
              throw std::invalid_argument("ASTR_INSITU_M12_W_COLOR_MAX must be finite and positive");
          }
          frozen_ranges[field]=M12Range(-w,w);
        } else {
          const double pad=.05*std::max(1.,range[1]-range[0]);
          frozen_ranges[field]=M12Range(range[0]-pad,range[1]+pad);
        }
      }
      const auto range=frozen_ranges.at(field);
      auto owner=std::make_shared<DeviceGeometryOwner>(data,step,time,3,field.c_str());
      const auto colors=owner->color_display(device_display_palette(),false,range[0],range[1]);
      const auto& view=owner->get();
      DeviceMesh mesh;mesh.name=name;mesh.owner=owner;
      mesh.draw.positions=reinterpret_cast<const float*>(view.display_coordinates);
      mesh.draw.colors=reinterpret_cast<const unsigned char*>(colors);mesh.draw.indices=view.display_connectivity;
      mesh.draw.points=view.points;mesh.draw.cells=view.cells;mesh.draw.device=view.device;mesh.draw.arity=3;
      std::copy(view.bounds,view.bounds+6,mesh.draw.bounds);
      mesh.controls["color_min"]=range[0];mesh.controls["color_max"]=range[1];
      for(std::size_t w=0;w<windows.size();++w) {
        mesh.controls["range"+std::to_string(w)+"_min"]=windows[w][0];
        mesh.controls["range"+std::to_string(w)+"_max"]=windows[w][1];
      }
      for(int d=0;d<3;++d) {
        mesh.controls["plane_origin"+std::to_string(d)]=origin[d];
        mesh.controls["plane_normal"+std::to_string(d)]=normal[d];
      }
      if(component==2) {
        const double weak=m12_weak_half_range(windows[0],range);
        mesh.controls["weak_half_range"]=weak;
        if(weak>0.) {
          // Share coordinates/connectivity; retain only an independent GPU color array.
          auto extra=std::make_shared<M12AuxiliaryColors>();
          extra->geometry=owner;
          auto auxiliary=mesh;
          auxiliary.name="spanwise_slice_weak";
          auxiliary.owner=extra;
          auxiliary.controls["color_min"]=-weak;
          auxiliary.controls["color_max"]=weak;
          auxiliary.controls["shared_device_geometry"]=
            auxiliary.draw.positions==mesh.draw.positions && auxiliary.draw.indices==mesh.draw.indices?1.:0.;
          if(view.points) {
            const auto values=device_display_palette();
            std::vector<viskores::Vec3f> table(4096);
            for(std::size_t i=0;i<table.size();++i)
              table[i]=viskores::Vec3f(values[3*i],values[3*i+1],values[3*i+2]);
            const auto palette=viskores::cont::make_ArrayHandle(table,viskores::CopyFlag::On);
            invoke(MapDisplayColor{false,-weak,weak},scalar,palette,extra->colors);
            synchronize_device_stage("M12WeakDisplayColor");
            require_device_only(extra->colors);
            auxiliary.draw.colors=static_cast<const unsigned char*>(extra->colors.GetBuffers()[0].ReadPointerDevice(
              viskores::cont::DeviceAdapterTagCuda{},extra->token));
          }
          meshes.push_back(std::move(auxiliary));
        }
      }
      meshes.push_back(std::move(mesh));
    }
    require_device_only(positions);require_device_only(fields);
  }
  require_device_only(xyz);require_device_only(velocity);require_device_only(temperature);
}
} // namespace astr_insitu
#endif
