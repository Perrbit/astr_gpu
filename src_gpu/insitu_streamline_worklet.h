#ifndef ASTR_INSITU_STREAMLINE_WORKLET_H
#define ASTR_INSITU_STREAMLINE_WORKLET_H

#include "insitu_rk45_device.h"
#include <viskores/Types.h>
#include <viskores/VecVariable.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/worklet/WorkletMapField.h>
#include <type_traits>
#include <utility>
#include <limits>

namespace astr_insitu {
static_assert(std::is_same<viskores::FloatDefault, double>::value,
              "Device in-situ streamlines require FP64 Viskores default types");

template <class Grid>
struct NormalizedGrid {
  Grid evaluate;
  viskores::Vec3f lower,upper;
  VISKORES_EXEC auto WithCache() const {
    return NormalizedGrid<decltype(evaluate.WithCache())>{evaluate.WithCache(),lower,upper};
  }
  VISKORES_EXEC bool operator()(const double* point, double parameter, double* tangent) const {
    for(int d=0;d<3;++d) if(point[d]<lower[d] || point[d]>upper[d]) return false;
    viskores::VecVariable<viskores::Vec3f, 2> values;
    const auto status = evaluate.Evaluate(viskores::Vec3f(point[0],point[1],point[2]), parameter, values);
    if (status.CheckFail()) {
      if(status.CheckSpatialBounds() || status.CheckTemporalBounds()) return false;
      tangent[0]=tangent[1]=tangent[2]=NAN;
      return true;
    }
    const auto velocity = values[0];
    const double norm = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]);
    if (!std::isfinite(norm)) {
      tangent[0] = tangent[1] = tangent[2] = NAN;
    } else {
      for (int d = 0; d < 3; ++d) tangent[d] = norm == 0. ? 0. : velocity[d]/norm;
    }
    return true;
  }
};

template <class Grid>
class LengthGrid : public viskores::cont::ExecutionObjectBase {
public:
  explicit LengthGrid(const Grid& grid,
      viskores::Vec3f lower=viskores::Vec3f(-std::numeric_limits<double>::infinity()),
      viskores::Vec3f upper=viskores::Vec3f(std::numeric_limits<double>::infinity()))
    : evaluate(grid),lower(lower),upper(upper) {}
  auto PrepareForExecution(viskores::cont::DeviceAdapterId device, viskores::cont::Token& token) const
    -> NormalizedGrid<decltype(std::declval<Grid>().PrepareForExecution(device,token))> {
    return {evaluate.PrepareForExecution(device,token),lower,upper};
  }
private:
  Grid evaluate;
  viskores::Vec3f lower,upper;
};

// One accepted RK45 step. MPI ownership/continuation is handled separately.
class RK45StepWorklet : public viskores::worklet::WorkletMapField {
public:
  using ControlSignature = void(FieldIn, FieldIn, ExecObject, FieldOut, FieldOut, FieldOut, FieldOut, FieldOut);
  using ExecutionSignature = void(_1,_2,_3,_4,_5,_6,_7,_8);
  using InputDomain = _1;
  RK45StepWorklet(double minimum, double maximum, double tolerance)
    : minimum(minimum), maximum(maximum), tolerance(tolerance) {}

  template <class Evaluate>
  VISKORES_EXEC void operator()(const viskores::Vec3f& previous, double initial,
      const Evaluate& evaluate, viskores::Vec3f& next, double& suggested,
      double& actual, double& error, viskores::Int32& status) const {
    double point[3] = {previous[0],previous[1],previous[2]};
    double output[3] = {point[0],point[1],point[2]};
    suggested = initial;
    status = rk45_step(evaluate, point, output, 0., suggested, actual,
                      minimum, maximum, tolerance, error);
    next = viskores::Vec3f(output[0],output[1],output[2]);
  }
private:
  double minimum, maximum, tolerance;
};

// Bounded output geometry plus the small continuation state. No field portal
// is read by the control environment when a particle changes MPI ownership.
class RK45TraceWorklet : public viskores::worklet::WorkletMapField {
  struct Owner {
    viskores::Vec3f lower,upper;
    viskores::Vec<bool,3> last;
    VISKORES_EXEC bool operator()(const double* point) const {
      for(int d=0;d<3;++d)
        if(!(point[d]>=lower[d] && (point[d]<upper[d] || (last[d] && point[d]==upper[d]))))
          return false;
      return true;
    }
  };
  template <class Portal> struct Writer {
    Portal output;
    viskores::Id offset;
    VISKORES_EXEC void operator()(int index,const double* point,double length,double actual) const {
      output.Set(offset+index,viskores::Vec<double,5>(point[0],point[1],point[2],length,actual));
    }
  };
public:
  using State = viskores::Vec<double,8>;
  using ControlSignature = void(FieldIn,ExecObject,WholeArrayInOut,FieldOut,FieldOut);
  using ExecutionSignature = void(_1,_2,_3,_4,_5,WorkIndex);
  using InputDomain = _1;
  RK45TraceWorklet(int limit,double minimum,double maximum,double tolerance,
                   double target,int steps,double lower,double upper,bool last)
    : RK45TraceWorklet(limit,minimum,maximum,tolerance,target,steps,
        viskores::Vec3f(lower,-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()),
        viskores::Vec3f(upper,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()),
        viskores::Vec<bool,3>(last,true,true)) {}
  RK45TraceWorklet(int limit,double minimum,double maximum,double tolerance,
                   double target,int steps,viskores::Vec3f lower,viskores::Vec3f upper,
                   viskores::Vec<bool,3> last)
    : limit(limit),steps(steps),minimum(minimum),maximum(maximum),tolerance(tolerance),
      target(target),lower(lower),upper(upper),last(last) {}
  template <class Evaluate,class Portal>
  VISKORES_EXEC void operator()(const State& input,const Evaluate& evaluate,
      const Portal& geometry,State& output,viskores::Int32& count,viskores::Id index) const {
    Execute(input,evaluate,Owner{lower,upper,last},geometry,output,count,index);
  }
  template <class Evaluate,class Ownership,class Portal>
  VISKORES_EXEC void Execute(const State& input,const Evaluate& evaluate,const Ownership& owner,
      const Portal& geometry,State& output,viskores::Int32& count,viskores::Id index) const {
    TraceState state{{input[0],input[1],input[2]},input[3],input[4],input[5],
                     static_cast<int>(input[6]),static_cast<int>(input[7])};
    count=trace_segment(evaluate,owner,Writer<Portal>{geometry,index*(limit+1)},
                        state,limit,minimum,maximum,tolerance,target,steps);
    output=State(state.point[0],state.point[1],state.point[2],state.suggested,state.length,
                 state.error,static_cast<double>(state.steps),static_cast<double>(state.status));
  }
private:
  int limit,steps;
  double minimum,maximum,tolerance,target;
  viskores::Vec3f lower,upper;
  viskores::Vec<bool,3> last;
};

class RK45PhysicalTraceWorklet : public RK45TraceWorklet {
public:
  using ControlSignature=void(FieldIn,ExecObject,ExecObject,WholeArrayInOut,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,WorkIndex);
  using InputDomain=_1;
  RK45PhysicalTraceWorklet(int limit,double minimum,double maximum,double tolerance,double target,int steps)
    : RK45TraceWorklet(limit,minimum,maximum,tolerance,target,steps,0.,0.,false) {}
  template<class Evaluate,class Owner,class Portal>
  VISKORES_EXEC void operator()(const State& input,const Evaluate& evaluate,const Owner& owner,
      const Portal& geometry,State& output,viskores::Int32& count,viskores::Id index) const {
    // A cursor belongs to this particle invocation, never to the shared input object.
    const auto particle_evaluate=evaluate.WithCache();
    this->Execute(input,particle_evaluate,owner,geometry,output,count,index);
  }
};
} // namespace astr_insitu
#endif
