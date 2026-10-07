#ifndef ASTR_INSITU_DEVICE_STRUCTURED_H
#define ASTR_INSITU_DEVICE_STRUCTURED_H

#include "insitu_device_geometry.h"
#include "insitu_device_plane.h"
#include <viskores/cont/ArrayCopy.h>

namespace astr_insitu {
namespace structured_geometry {

using PlaneKey=viskores::Vec<viskores::Id,2>;
using PlaneCounts=viskores::Vec<viskores::Id,2>;
struct KeyLess {
  VISKORES_EXEC_CONT bool operator()(const PlaneKey& a,const PlaneKey& b) const {
    return a[0]<b[0] || (a[0]==b[0] && a[1]<b[1]);
  }
};
struct Partition {
  plane_geometry::Grid grid;
  viskores::Id3 cells,offset;
  VISKORES_CONT Partition(const viskores::Id3& global,const viskores::Id3& local,const viskores::Id3& start)
    :grid(plane_geometry::make_grid(global[0],global[1],global[2])),cells(local),offset(start) {
    for(int d=0;d<3;++d)
      if(cells[d]<1 || offset[d]<0 || cells[d]>global[d] || offset[d]>global[d]-cells[d])
        throw std::invalid_argument("Invalid structured device partition");
  }
  VISKORES_EXEC_CONT viskores::Id local_nodes() const {
    return (cells[0]+1)*(cells[1]+1)*(cells[2]+1);
  }
  VISKORES_EXEC_CONT viskores::Id tetrahedra() const {return 6*cells[0]*cells[1]*cells[2];}
  template<class Portal>
  VISKORES_EXEC plane_geometry::Cut cut(const plane_geometry::Plane& plane,viskores::Id id,
      const Portal& coordinates,plane_geometry::Vertex* hex,viskores::Id* local_ids) const {
    const auto cell=id/6;
    const viskores::Id3 base(cell%cells[0],cell/cells[0]%cells[1],cell/(cells[0]*cells[1]));
    for(int c=0;c<8;++c) {
      const viskores::Id3 node(base[0]+(c&1),base[1]+((c>>1)&1),base[2]+((c>>2)&1));
      local_ids[c]=node[0]+(cells[0]+1)*(node[1]+(cells[1]+1)*node[2]);
      const auto position=coordinates.Get(local_ids[c]);
      for(int d=0;d<3;++d) hex[c].position[d]=position[d];
      hex[c].id=plane_geometry::node_id(grid,node[0]+offset[0],node[1]+offset[1],node[2]+offset[2]);
    }
    const auto global=plane_geometry::cell_id(grid,base[0]+offset[0],base[1]+offset[1],base[2]+offset[2]);
    return plane_geometry::cut_structured_tetrahedron(plane,grid,hex,global,int(id%6));
  }
};

struct CountPlane : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4);
  Partition partition;
  plane_geometry::Plane plane;
  VISKORES_CONT CountPlane(Partition p,plane_geometry::Plane s):partition(p),plane(s) {}
  template<class Portal>
  VISKORES_EXEC void operator()(viskores::Id id,const Portal& coordinates,PlaneCounts& count,viskores::Id& bad) const {
    plane_geometry::Vertex hex[8];viskores::Id local[8];
    const auto cut=partition.cut(plane,id,coordinates,hex,local);
    bad=cut.status!=plane_geometry::Status::ok?1:0;
    count=bad?PlaneCounts(0):PlaneCounts(cut.triangles?cut.points:0,3*cut.triangles);
  }
};

struct WritePlane : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,
    WholeArrayOut,WholeArrayOut,WholeArrayOut,WholeArrayOut,WholeArrayOut,WholeArrayOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12);
  Partition partition;
  plane_geometry::Plane plane;
  bool has_q;
  VISKORES_CONT WritePlane(Partition p,plane_geometry::Plane s,bool q=true):partition(p),plane(s),has_q(q) {}
  template<class XYZ,class V,class D,class O,class PX,class PV,class PQ,class PU,class PK,class PI>
  VISKORES_EXEC void operator()(viskores::Id id,const XYZ& xyz,const V& input_v,const D& input_d,
      const O& offsets,const PX& points,const PV& vectors,const PQ& q,const PU& u,const PK& keys,
      const PI& indices,viskores::Id& bad) const {
    plane_geometry::Vertex hex[8];viskores::Id local[8];
    const auto cut=partition.cut(plane,id,xyz,hex,local);
    bad=cut.status!=plane_geometry::Status::ok?1:0;
    if(bad || !cut.triangles) return;
    const auto start=offsets.Get(id);
    for(int p=0;p<cut.points;++p) {
      const auto vertex=cut.vertices[p];
      int lo=-1,hi=-1;
      for(int c=0;c<8;++c) {
        if(hex[c].id==vertex.key.first) lo=c;
        if(hex[c].id==vertex.key.second) hi=c;
      }
      if(lo<0 || hi<0) {bad=1;return;}
      const auto a=input_v.Get(local[lo]),b=input_v.Get(local[hi]);
      const double qa=has_q?input_d.Get(local[lo])[9]:0.,qb=has_q?input_d.Get(local[hi])[9]:0.;
      viskores::Vec3f position,velocity;
      for(int d=0;d<3;++d) {
        position[d]=vertex.position[d];
        velocity[d]=a[d]+vertex.fraction*(b[d]-a[d]);
        if(!viskores::IsFinite(position[d]) || !viskores::IsFinite(velocity[d]) ||
           !viskores::IsFinite(a[d]) || !viskores::IsFinite(b[d])) bad=1;
      }
      const double scalar=qa+vertex.fraction*(qb-qa);
      if(!viskores::IsFinite(qa) || !viskores::IsFinite(qb) || !viskores::IsFinite(scalar)) bad=1;
      const auto node=start[0]+p;
      points.Set(node,position);vectors.Set(node,velocity);q.Set(node,scalar);u.Set(node,velocity[0]);
      keys.Set(node,PlaneKey(vertex.key.first,vertex.key.second));
    }
    for(int i=0;i<3*cut.triangles;++i) indices.Set(start[1]+i,start[0]+cut.indices[i]);
  }
};
struct CopyIndex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,FieldOut);
  using ExecutionSignature=void(_1,_2);
  VISKORES_EXEC void operator()(viskores::Id id,viskores::Id& value) const {value=id;}
};
struct CheckSharedPoint : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,_7);
  template<class K,class I,class X,class V,class Q>
  VISKORES_EXEC void operator()(viskores::Id id,const K& keys,const I& original,const X& xyz,
      const V& velocity,const Q& scalar,viskores::Id& bad) const {
    bad=0;
    if(id==0 || keys.Get(id)!=keys.Get(id-1)) return;
    const auto a=original.Get(id),b=original.Get(id-1);
    if(xyz.Get(a)!=xyz.Get(b) || velocity.Get(a)!=velocity.Get(b) || scalar.Get(a)!=scalar.Get(b)) bad=1;
  }
};
struct SelectUniquePoint : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,
    FieldOut,FieldOut,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,_7,_8,_9);
  template<class I,class X,class V,class Q>
  VISKORES_EXEC void operator()(viskores::Id first,const I& original,const X& xyz,const V& velocity,
      const Q& scalar,viskores::Vec3f& p,viskores::Vec3f& v,double& q,double& u) const {
    const auto source=original.Get(first);
    p=xyz.Get(source);v=velocity.Get(source);q=scalar.Get(source);u=v[0];
  }
};
struct RemapPlaneIndex : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  template<class P>
  VISKORES_EXEC void operator()(viskores::Id raw,const P& unique,viskores::Id& index) const {
    index=unique.Get(raw);
  }
};
struct PlaneProduct {
  viskores::cont::DataSet data;
  viskores::cont::ArrayHandle<PlaneKey> keys;
  std::uint64_t controlled_bound=0;
};
inline void require_no_errors(const viskores::cont::ArrayHandle<viskores::Id>& bad,const char* message) {
  const auto count=viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},bad,viskores::Id(0));
  synchronize_device_stage("ValidateStructuredPlane");
  if(count) throw std::runtime_error(message);
}

// No host mirrors of fields, keys, intersections or connectivity. Device scan
// totals and validation reductions are small control metadata only. The caller
// retains source arrays and accounts for third-party scratch/resource peaks.
inline PlaneProduct extract_plane(viskores::Vec3f* coordinates_pointer,viskores::Vec3f* velocity_pointer,
    DeviceDiagnostics* diagnostics_pointer,const Partition& partition,const plane_geometry::Point& origin,
    const plane_geometry::Point& normal,std::uint64_t budget,std::uint64_t retained=0,bool canonical=false) {
  const auto plane=canonical?plane_geometry::Plane{origin,normal}:plane_geometry::make_plane(origin,normal);
  if(canonical) {
    double squared=0.;
    for(int d=0;d<3;++d) {
      if(!std::isfinite(origin[d]) || !std::isfinite(normal[d]))
        throw std::invalid_argument("Nonfinite canonical plane");
      squared+=normal[d]*normal[d];
    }
    if(std::abs(squared-1.)>8.*DBL_EPSILON || normal[plane_geometry::dominant_axis(plane)]<=0.)
      throw std::invalid_argument("Invalid canonical plane normal");
  }
  const auto tetrahedra=partition.tetrahedra(),nodes=partition.local_nodes();
  if(tetrahedra>std::numeric_limits<viskores::Id>::max()/6 ||
     static_cast<std::uint64_t>(tetrahedra)>std::numeric_limits<std::uint64_t>::max()/1024)
    throw std::overflow_error("Structured plane array extent overflow");
  PlaneProduct result;
  result.controlled_bound=static_cast<std::uint64_t>(tetrahedra)*1024;
  if(retained>budget || result.controlled_bound>budget-retained)
    throw std::runtime_error("Structured plane controlled-array budget exceeded");
  std::size_t free_bytes=0,total_bytes=0;
  if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess || free_bytes<1073741824ULL ||
      result.controlled_bound>free_bytes-1073741824ULL)
    throw std::runtime_error("Structured plane device reserve exceeded");
  const auto device=viskores::cont::DeviceAdapterTagCuda{};
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(device);
  auto xyz=borrow_cuda_array(coordinates_pointer,nodes);
  auto velocity=borrow_cuda_array(velocity_pointer,nodes);
  viskores::cont::ArrayHandle<DeviceDiagnostics> diagnostics;
  if(diagnostics_pointer) diagnostics=borrow_cuda_array(diagnostics_pointer,nodes);
  viskores::cont::Invoker invoke(device);
  const auto cells=viskores::cont::ArrayHandleIndex(tetrahedra);
  viskores::cont::ArrayHandle<PlaneCounts> counts,offsets;
  viskores::cont::ArrayHandle<viskores::Id> bad;
  invoke(CountPlane(partition,plane),cells,xyz,counts,bad);
  synchronize_device_stage("CountStructuredPlane");
  require_no_errors(bad,"Invalid structured plane geometry/range");
  const auto total=viskores::cont::Algorithm::ScanExclusive(device,counts,offsets);
  synchronize_device_stage("ScanStructuredPlane");
  if(!total[1]) {
    require_device_only(xyz);require_device_only(velocity);
    if(diagnostics_pointer) require_device_only(diagnostics);
    require_device_only(counts);require_device_only(offsets);require_device_only(bad);
    return result;
  }
  viskores::cont::ArrayHandle<viskores::Vec3f> raw_xyz,raw_velocity;
  viskores::cont::ArrayHandle<double> raw_q,raw_u;
  viskores::cont::ArrayHandle<PlaneKey> raw_keys,sorted_keys;
  viskores::cont::ArrayHandle<viskores::Id> raw_indices,original,first,unique,indices;
  raw_xyz.Allocate(total[0]);raw_velocity.Allocate(total[0]);raw_q.Allocate(total[0]);raw_u.Allocate(total[0]);
  raw_keys.Allocate(total[0]);raw_indices.Allocate(total[1]);
  invoke(WritePlane(partition,plane,diagnostics_pointer!=nullptr),cells,xyz,velocity,diagnostics,offsets,
    raw_xyz,raw_velocity,raw_q,raw_u,raw_keys,raw_indices,bad);
  synchronize_device_stage("WriteStructuredPlane");
  require_no_errors(bad,"Nonfinite structured plane field/intersection");
  if(!viskores::cont::Algorithm::Copy(device,raw_keys,sorted_keys))
    throw std::runtime_error("Cannot copy device plane keys");
  synchronize_device_stage("CopyPlaneKeys");
  invoke(CopyIndex{},viskores::cont::ArrayHandleIndex(total[0]),original);
  synchronize_device_stage("InitializePlaneKeys");
  viskores::cont::Algorithm::SortByKey(device,sorted_keys,original,KeyLess{});
  synchronize_device_stage("SortPlaneKeys");
  invoke(CheckSharedPoint{},viskores::cont::ArrayHandleIndex(total[0]),sorted_keys,original,
    raw_xyz,raw_velocity,raw_q,bad);
  synchronize_device_stage("CheckSharedPlanePoint");
  require_no_errors(bad,"Shared plane key has inconsistent coordinates/fields");
  if(!viskores::cont::Algorithm::Copy(device,sorted_keys,result.keys))
    throw std::runtime_error("Cannot copy unique device plane keys");
  synchronize_device_stage("CopySortedPlaneKeys");
  viskores::cont::Algorithm::Unique(device,result.keys);
  synchronize_device_stage("UniquePlaneKeys");
  viskores::cont::Algorithm::LowerBounds(device,sorted_keys,result.keys,first,KeyLess{});
  synchronize_device_stage("SelectUniquePlaneKey");
  viskores::cont::Algorithm::LowerBounds(device,result.keys,raw_keys,unique,KeyLess{});
  synchronize_device_stage("MapPlaneKeys");
  viskores::cont::ArrayHandle<viskores::Vec3f> product_xyz,product_velocity;
  viskores::cont::ArrayHandle<double> product_q,product_u;
  invoke(SelectUniquePoint{},first,original,raw_xyz,raw_velocity,raw_q,
    product_xyz,product_velocity,product_q,product_u);
  synchronize_device_stage("PackUniquePlanePoints");
  invoke(RemapPlaneIndex{},raw_indices,unique,indices);
  synchronize_device_stage("PackUniquePlaneProduct");
  viskores::cont::CellSetSingleType<> product_cells;
  product_cells.Fill(result.keys.GetNumberOfValues(),viskores::CELL_SHAPE_TRIANGLE,3,indices);
  result.data.SetCellSet(product_cells);
  result.data.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",product_xyz));
  result.data.AddPointField("velocity",product_velocity);
  if(diagnostics_pointer) result.data.AddPointField("Q_rs",product_q);
  result.data.AddPointField("u",product_u);
  require_device_only(xyz);require_device_only(velocity);
  if(diagnostics_pointer) require_device_only(diagnostics);
  require_device_only(counts);require_device_only(offsets);require_device_only(bad);
  require_device_only(raw_xyz);require_device_only(raw_velocity);require_device_only(raw_q);require_device_only(raw_u);
  require_device_only(raw_keys);require_device_only(raw_indices);require_device_only(sorted_keys);
  require_device_only(original);require_device_only(first);require_device_only(unique);require_device_only(result.keys);
  require_device_only(product_xyz);require_device_only(product_velocity);
  require_device_only(product_q);require_device_only(product_u);require_device_only(indices);
  return result;
}

} // namespace structured_geometry
} // namespace astr_insitu
#endif
