#ifndef ASTR_INSITU_DEVICE_CURVE_TRACE_H
#define ASTR_INSITU_DEVICE_CURVE_TRACE_H

#include "insitu_device_array.h"
#include "insitu_rk45_device.h"
#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/CellLocatorTwoLevel.h>
#include <viskores/cont/CellSetStructured.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/Invoker.h>
#include <viskores/worklet/WorkletMapField.h>
#include <viskores/filter/flow/worklet/Field.h>
#include <viskores/filter/flow/worklet/GridEvaluatorStatus.h>
#include <lcl/lcl.h>
#include <algorithm>
#include <limits>

namespace astr_insitu {
#ifdef __CUDACC__
#define ASTR_CURVE_EXEC_NOINLINE VISKORES_EXEC __noinline__
#else
#define ASTR_CURVE_EXEC_NOINLINE VISKORES_EXEC
#endif
struct CropPhysicalTraceHalo : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,WholeArrayIn,WholeArrayIn,WholeArrayIn,
                              FieldOut,FieldOut,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,_7,_8);
  viskores::Id3 input_dimensions,dimensions,start;
  template<class Coordinates,class Trace,class Color>
  VISKORES_EXEC void operator()(viskores::Id id,const Coordinates& coordinates,
      const Trace& trace,const Color& color,viskores::Vec3f& p,viskores::Vec3f& v,
      viskores::Vec3f& c,viskores::Id& bad) const {
    const viskores::Id3 ijk(id%dimensions[0],id/dimensions[0]%dimensions[1],
                          id/(dimensions[0]*dimensions[1]));
    const auto index=ijk+start;
    const auto source=index[0]+input_dimensions[0]*(index[1]+input_dimensions[1]*index[2]);
    p=coordinates.Get(source);v=trace.Get(source);c=color.Get(source);bad=0;
    for(int d=0;d<3;++d) bad+=!viskores::IsFinite(p[d]) ||
      !viskores::IsFinite(v[d]) || !viskores::IsFinite(c[d]);
  }
};

struct PhysicalTraceData {
  viskores::cont::DataSet grid;
  viskores::cont::ArrayHandle<viskores::Vec3f> coordinates,trace,color;
  viskores::Id3 begin,dimensions;
};

inline PhysicalTraceData crop_physical_trace_halo(
    const viskores::cont::ArrayHandle<viskores::Vec3f>& coordinates,
    const viskores::cont::ArrayHandle<viskores::Vec3f>& trace,
    const viskores::cont::ArrayHandle<viskores::Vec3f>& color,
    const viskores::Id3& extent,const viskores::Id3& offset,const viskores::Id3& global_cells) {
  PhysicalTraceData data;
  const viskores::Id3 input_dimensions=extent+viskores::Id3(7);
  const auto nodes=input_dimensions[0]*input_dimensions[1]*input_dimensions[2];
  if(coordinates.GetNumberOfValues()!=nodes ||
      trace.GetNumberOfValues()!=nodes || color.GetNumberOfValues()!=nodes)
    throw std::invalid_argument("Invalid bounded physical streamline halo extent");
  for(int d=0;d<3;++d) {
    if(global_cells[d]<1 || extent[d]<1 || offset[d]<0 || offset[d]+extent[d]>global_cells[d])
      throw std::invalid_argument("Invalid physical streamline partition");
    data.begin[d]=std::max<viskores::Id>(0,offset[d]-3);
    const auto end=std::min<viskores::Id>(global_cells[d],offset[d]+extent[d]+3);
    data.dimensions[d]=end-data.begin[d]+1;
  }
  // Only interior-rank halos are cells. No periodic wrap or invented outer geometry.
  const auto start=data.begin-offset+viskores::Id3(3);
  const auto count=data.dimensions[0]*data.dimensions[1]*data.dimensions[2];
  viskores::cont::ArrayHandle<viskores::Id> bad;
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  CropPhysicalTraceHalo crop;
  crop.input_dimensions=input_dimensions;crop.dimensions=data.dimensions;crop.start=start;
  invoke(crop,viskores::cont::ArrayHandleIndex(count),coordinates,trace,color,
    data.coordinates,data.trace,data.color,bad);
  synchronize_device_stage("CropPhysicalTraceHalo");
  if(viskores::cont::Algorithm::Reduce(viskores::cont::DeviceAdapterTagCuda{},bad,viskores::Id(0)))
    throw std::invalid_argument("Nonfinite physical streamline coordinates or fields");
  viskores::cont::CellSetStructured<3> cells;
  cells.SetPointDimensions(data.dimensions);
  data.grid.SetCellSet(cells);
  data.grid.AddCoordinateSystem(viskores::cont::CoordinateSystem("coords",data.coordinates));
  require_device_only(coordinates);require_device_only(trace);require_device_only(color);
  require_device_only(data.coordinates);require_device_only(data.trace);require_device_only(data.color);
  return data;
}

inline PhysicalTraceData crop_physical_trace_halo(
    const viskores::cont::ArrayHandle<viskores::Vec3f>& coordinates,
    const viskores::cont::ArrayHandle<viskores::Vec3f>& trace,
    const viskores::cont::ArrayHandle<viskores::Vec3f>& color,
    const viskores::Id3& extent,const viskores::Id3& offset,int global_cells) {
  return crop_physical_trace_halo(coordinates,trace,color,extent,offset,viskores::Id3(global_cells));
}

template<class Locator,class Coordinates>
struct ExecutionPhysicalTraceOwner {
  Locator locator;
  Coordinates coordinates;
  viskores::Id3 cells,begin,offset,extent;
  viskores::Id3 global_cells;
  VISKORES_EXEC viskores::Vec<viskores::Id,8> PointIds(viskores::Id id) const {
    const viskores::Id3 index(id%cells[0],id/cells[0]%cells[1],id/(cells[0]*cells[1]));
    const auto dims=cells+viskores::Id3(1);
    const auto p=index[0]+dims[0]*(index[1]+dims[1]*index[2]);
    const auto plane=dims[0]*dims[1];
    return {p,p+1,p+1+dims[0],p+dims[0],p+plane,p+plane+1,
      p+plane+1+dims[0],p+plane+dims[0]};
  }
  VISKORES_EXEC int Refine(viskores::Id id,const viskores::Vec3f& point,
      viskores::Vec3f& parametric,int* inverse_status=nullptr) const {
    const auto indices=PointIds(id);
    double vertices[8][3],scale=1.;
    for(int i=0;i<8;++i) for(int d=0;d<3;++d) {
      vertices[i][d]=coordinates.Get(indices[i])[d];
      scale=viskores::Max(scale,viskores::Abs(vertices[i][d]));
    }
    const auto points=lcl::makeFieldAccessorNestedSOA(vertices,3);
    const lcl::Hexahedron shape;
    using Vector=lcl::internal::Vector<double,3>;
    auto jacobian=[&](const Vector& p,lcl::internal::Matrix<double,3,3>& matrix) {
      lcl::internal::jacobian3D(shape,points,p,matrix);
      const double determinant=matrix(0,0)*(matrix(1,1)*matrix(2,2)-matrix(1,2)*matrix(2,1))-
        matrix(0,1)*(matrix(1,0)*matrix(2,2)-matrix(1,2)*matrix(2,0))+
        matrix(0,2)*(matrix(1,0)*matrix(2,1)-matrix(1,1)*matrix(2,0));
      if(!viskores::IsFinite(determinant) || determinant<=0.)
        return lcl::ErrorCode::DEGENERATE_CELL_DETECTED;
      return lcl::ErrorCode::SUCCESS;
    };
    auto world=[&](const Vector& p,Vector& result) {
      return lcl::parametricToWorld(shape,points,p,result);
    };
    Vector p{parametric[0],parametric[1],parametric[2]};
    const Vector target{point[0],point[1],point[2]};
    constexpr double epsilon=std::numeric_limits<double>::epsilon();
    const auto status=lcl::internal::newtonsMethod(jacobian,world,target,p,32.*epsilon,32);
    if(inverse_status) *inverse_status=static_cast<int>(status);
    for(int d=0;d<3;++d) parametric[d]=p[d];
    if(status!=lcl::ErrorCode::SUCCESS && status!=lcl::ErrorCode::SOLUTION_DID_NOT_CONVERGE) return -2;
    // An increment may oscillate at FP64 roundoff. The physical residual is
    // authoritative for either Newton return before containment is considered.
    Vector reconstructed;
    if(world(p,reconstructed)!=lcl::ErrorCode::SUCCESS) return -2;
    for(int d=0;d<3;++d) if(!viskores::IsFinite(p[d]) || !viskores::IsFinite(reconstructed[d]) ||
        viskores::Abs(reconstructed[d]-point[d])>128.*epsilon*scale) return -2;
    // Roundoff-only face closure; the library's 1e-6 inside tolerance is not used.
    for(int d=0;d<3;++d) {
      if(!viskores::IsFinite(p[d])) return -2;
      if(p[d]<-64.*epsilon || p[d]>1.+64.*epsilon) return -1;
      p[d]=viskores::Max(0.,viskores::Min(1.,p[d]));
    }
    if(world(p,reconstructed)!=lcl::ErrorCode::SUCCESS) return -2;
    for(int d=0;d<3;++d) {
      if(!viskores::IsFinite(reconstructed[d]) ||
          viskores::Abs(reconstructed[d]-point[d])>128.*epsilon*scale) return -2;
      parametric[d]=p[d];
    }
    return 0;
  }
  ASTR_CURVE_EXEC_NOINLINE viskores::Id Find(const viskores::Vec3f& point,
      viskores::Id& selected,viskores::Vec3f& pcoords) const {
    selected=-1;
    const auto count=locator.CountAllCells(point);
    if(count<0 || count>8) return -2;
    if(!count) return -1;
    viskores::VecVariable<viskores::Id,8> candidates;
    viskores::VecVariable<viskores::Vec3f,8> parametric;
    for(int i=0;i<count;++i) {candidates.Append(-1);parametric.Append(viskores::Vec3f(0.));}
    if(locator.FindAllCells(point,candidates,parametric)!=viskores::ErrorCode::Success) return -2;
    viskores::Id best=std::numeric_limits<viskores::Id>::max();
    for(int i=0;i<count;++i) {
      const auto id=candidates[i];
      if(id<0 || id>=cells[0]*cells[1]*cells[2]) return -2;
      auto refined=parametric[i];
      const int valid=Refine(id,point,refined);
      if(valid==-2) return -2;
      if(valid==-1) continue;
      const viskores::Id3 index(id%cells[0],id/cells[0]%cells[1],id/(cells[0]*cells[1]));
      const auto global=index+begin;
      const auto candidate=global[0]+global_cells[0]*(global[1]+global_cells[1]*global[2]);
      if(candidate<best) {best=candidate;selected=id;pcoords=refined;}
    }
    return selected<0?-1:best;
  }
  VISKORES_EXEC viskores::Id GlobalCell(const viskores::Vec3f& point) const {
    viskores::Id selected;
    viskores::Vec3f pcoords;
    return Find(point,selected,pcoords);
  }
  ASTR_CURVE_EXEC_NOINLINE viskores::Id FindCached(const viskores::Vec3f& point,
      viskores::Id& selected,viskores::Vec3f& pcoords) const {
    if(selected>=0 && selected<cells[0]*cells[1]*cells[2]) {
      auto refined=pcoords;
      if(Refine(selected,point,refined)==0) {
        constexpr double face=64.*std::numeric_limits<double>::epsilon();
        bool interior=true;
        for(int d=0;d<3;++d) interior=interior && refined[d]>face && refined[d]<1.-face;
        if(interior) {
          pcoords=refined;
          const viskores::Id3 index(selected%cells[0],selected/cells[0]%cells[1],
            selected/(cells[0]*cells[1]));
          const auto global=index+begin;
          return global[0]+global_cells[0]*(global[1]+global_cells[1]*global[2]);
        }
      }
    }
    // Face ties and failed speculative inverses retain the canonical all-cell search.
    return Find(point,selected,pcoords);
  }
  VISKORES_EXEC bool Owns(viskores::Id id) const {
    if(id<0) return false;
    const viskores::Id3 index(id%global_cells[0],id/global_cells[0]%global_cells[1],
      id/(global_cells[0]*global_cells[1]));
    for(int d=0;d<3;++d) if(index[d]<offset[d] || index[d]>=offset[d]+extent[d]) return false;
    return true;
  }
  VISKORES_EXEC bool operator()(const double* point) const {
    return Owns(GlobalCell(viskores::Vec3f(point[0],point[1],point[2])));
  }
};

class PhysicalTraceOwner : public viskores::cont::ExecutionObjectBase {
  viskores::cont::CellLocatorTwoLevel locator;
  viskores::cont::ArrayHandle<viskores::Vec3f> coordinates;
  viskores::Id3 cells,begin,offset,extent;
  viskores::Id3 global_cells;
public:
  PhysicalTraceOwner(const PhysicalTraceData& data,const viskores::Id3& extent,
      const viskores::Id3& offset,const viskores::Id3& global_cells)
    : coordinates(data.coordinates),cells(data.dimensions-viskores::Id3(1)),begin(data.begin),offset(offset),
      extent(extent),global_cells(global_cells) {
    locator.SetCoordinates(data.grid.GetCoordinateSystem());locator.SetCellSet(data.grid.GetCellSet());
    locator.Update();
    synchronize_device_stage("PhysicalTraceOwnerLocator");
    require_device_only(data.coordinates);
  }
  PhysicalTraceOwner(const PhysicalTraceData& data,const viskores::Id3& extent,
      const viskores::Id3& offset,int global_cells)
    :PhysicalTraceOwner(data,extent,offset,viskores::Id3(global_cells)) {}
  auto PrepareForExecution(viskores::cont::DeviceAdapterId device,viskores::cont::Token& token) const {
    auto execution=locator.PrepareForExecution(device,token);
    auto points=coordinates.PrepareForInput(device,token);
    return ExecutionPhysicalTraceOwner<decltype(execution),decltype(points)>{
      execution,points,cells,begin,offset,extent,global_cells};
  }
};

using PhysicalTraceField=viskores::worklet::flow::VelocityField<
  viskores::cont::ArrayHandle<viskores::Vec3f>>;
template<class Grid>
struct CachedPhysicalTraceGrid {
  Grid grid;
  mutable viskores::Id cell=-1;
  mutable viskores::Vec3f parametric=viskores::Vec3f(.5);
  VISKORES_EXEC viskores::worklet::flow::GridEvaluatorStatus Evaluate(
      const viskores::Vec3f& point,double,viskores::VecVariable<viskores::Vec3f,2>& values) const {
    const auto result=grid.owner.FindCached(point,cell,parametric);
    return grid.Interpolate(result,cell,parametric,values);
  }
};
template<class Owner>
struct ExecutionPhysicalTraceGrid {
  Owner owner;
  PhysicalTraceField::ExecutionType field;
  VISKORES_EXEC auto WithCache() const {
    return CachedPhysicalTraceGrid<ExecutionPhysicalTraceGrid>{*this};
  }
  VISKORES_EXEC viskores::worklet::flow::GridEvaluatorStatus Evaluate(
      const viskores::Vec3f& point,double,viskores::VecVariable<viskores::Vec3f,2>& values) const {
    viskores::Id cell;
    viskores::Vec3f parametric;
    const auto result=owner.Find(point,cell,parametric);
    return Interpolate(result,cell,parametric,values);
  }
  VISKORES_EXEC viskores::worklet::flow::GridEvaluatorStatus Interpolate(viskores::Id result,
      viskores::Id cell,const viskores::Vec3f& parametric,
      viskores::VecVariable<viskores::Vec3f,2>& values) const {
    viskores::worklet::flow::GridEvaluatorStatus status;
    if(result<0) {
      status.SetFail();
      if(result==-1) status.SetSpatialBounds();
      return status;
    }
    const auto ids=owner.PointIds(cell);
    viskores::VecVariable<viskores::Id,8> indices;
    for(int i=0;i<8;++i) indices.Append(ids[i]);
    field.GetValue(indices,8,parametric,viskores::CELL_SHAPE_HEXAHEDRON,values);
    status.SetOk();
    return status;
  }
};
class PhysicalTraceGrid : public viskores::cont::ExecutionObjectBase {
  PhysicalTraceOwner owner;
  PhysicalTraceField field;
public:
  PhysicalTraceGrid(const PhysicalTraceOwner& owner,const PhysicalTraceField& field)
    : owner(owner),field(field) {}
  auto PrepareForExecution(viskores::cont::DeviceAdapterId device,viskores::cont::Token& token) const {
    auto location=owner.PrepareForExecution(device,token);
    return ExecutionPhysicalTraceGrid<decltype(location)>{location,field.PrepareForExecution(device,token)};
  }
};

struct LocatePhysicalTraceOwner : viskores::worklet::WorkletMapField {
  using ControlSignature=void(FieldIn,ExecObject,FieldOut);
  using ExecutionSignature=void(_1,_2,_3);
  template<class State,class Owner>
  VISKORES_EXEC void operator()(const State& state,const Owner& owner,viskores::Id& result) const {
    if(state[7]!=TraceActive && state[7]!=TraceTransfer) {result=-1;return;}
    const auto id=owner.GlobalCell(viskores::Vec3f(state[0],state[1],state[2]));
    result=id==-2?-2:(owner.Owns(id)?id:-1);
  }
};
#undef ASTR_CURVE_EXEC_NOINLINE
} // namespace astr_insitu
#endif
