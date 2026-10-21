#ifndef ASTR_INSITU_STREAMLINE_WORKLET_H
#define ASTR_INSITU_STREAMLINE_WORKLET_H

#include "insitu_rk45_device.h"
#include "insitu_device_array.h"
#include <viskores/Types.h>
#include <viskores/VecVariable.h>
#include <viskores/cont/ExecutionObjectBase.h>
#include <viskores/worklet/WorkletMapField.h>
#include <type_traits>
#include <utility>
#include <limits>

namespace astr_insitu {
static_assert(std::is_same<viskores::FloatDefault, Real>::value,
              "Device streamlines require matching Viskores numerical types");

struct ParticleState {
  viskores::Vec<Real,6> values;
  viskores::Int32 steps=0,status=0;
  VISKORES_EXEC_CONT ParticleState()=default;
  VISKORES_EXEC_CONT ParticleState(Real x,Real y,Real z,Real h,Real length,Real error,
      viskores::Int32 count,viskores::Int32 code):values(x,y,z,h,length,error),steps(count),status(code) {}
  VISKORES_EXEC_CONT Real& operator[](int component) {return values[component];}
  VISKORES_EXEC_CONT Real operator[](int component) const {return values[component];}
};

template <class Grid>
struct NormalizedGrid {
  Grid evaluate;
  astr_insitu::NativeVector lower,upper;
  VISKORES_EXEC auto WithCache() const {
    return NormalizedGrid<decltype(evaluate.WithCache())>{evaluate.WithCache(),lower,upper};
  }
  VISKORES_EXEC bool operator()(const Real* point, Real parameter, Real* tangent) const {
    for(int d=0;d<3;++d) if(point[d]<lower[d] || point[d]>upper[d]) return false;
    viskores::VecVariable<astr_insitu::NativeVector, 2> values;
    const auto status = evaluate.Evaluate(astr_insitu::NativeVector(point[0],point[1],point[2]), parameter, values);
    if (status.CheckFail()) {
      if(status.CheckSpatialBounds() || status.CheckTemporalBounds()) return false;
      tangent[0]=tangent[1]=tangent[2]=NAN;
      return true;
    }
    const auto velocity = values[0];
    const Real norm = std::sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]);
    if (!std::isfinite(norm)) {
      tangent[0] = tangent[1] = tangent[2] = NAN;
    } else {
      for (int d = 0; d < 3; ++d) tangent[d] = norm == Real(0.) ? Real(0.) : velocity[d]/norm;
    }
    return true;
  }
};

template <class Grid>
class LengthGrid : public viskores::cont::ExecutionObjectBase {
public:
  explicit LengthGrid(const Grid& grid,
      astr_insitu::NativeVector lower=astr_insitu::NativeVector(-std::numeric_limits<Real>::infinity()),
      astr_insitu::NativeVector upper=astr_insitu::NativeVector(std::numeric_limits<Real>::infinity()))
    : evaluate(grid),lower(lower),upper(upper) {}
  auto PrepareForExecution(viskores::cont::DeviceAdapterId device, viskores::cont::Token& token) const
    -> NormalizedGrid<decltype(std::declval<Grid>().PrepareForExecution(device,token))> {
    return {evaluate.PrepareForExecution(device,token),lower,upper};
  }
private:
  Grid evaluate;
  astr_insitu::NativeVector lower,upper;
};

// One accepted RK45 step. MPI ownership/continuation is handled separately.
class RK45StepWorklet : public viskores::worklet::WorkletMapField {
public:
  using ControlSignature = void(FieldIn, FieldIn, ExecObject, FieldOut, FieldOut, FieldOut, FieldOut, FieldOut);
  using ExecutionSignature = void(_1,_2,_3,_4,_5,_6,_7,_8);
  using InputDomain = _1;
  RK45StepWorklet(Real minimum, Real maximum, Real tolerance)
    : minimum(minimum), maximum(maximum), tolerance(tolerance) {}

  template <class Evaluate>
  VISKORES_EXEC void operator()(const astr_insitu::NativeVector& previous, Real initial,
      const Evaluate& evaluate, astr_insitu::NativeVector& next, Real& suggested,
      Real& actual, Real& error, viskores::Int32& status) const {
    Real point[3] = {previous[0],previous[1],previous[2]};
    Real output[3] = {point[0],point[1],point[2]};
    suggested = initial;
    status = rk45_step(evaluate, point, output, Real(0.), suggested, actual,
                      minimum, maximum, tolerance, error);
    next = astr_insitu::NativeVector(output[0],output[1],output[2]);
  }
private:
  Real minimum, maximum, tolerance;
};

// Bounded output geometry plus the small continuation state. No field portal
// is read by the control environment when a particle changes MPI ownership.
class RK45TraceWorklet : public viskores::worklet::WorkletMapField {
  struct Owner {
    astr_insitu::NativeVector lower,upper;
    viskores::Vec<bool,3> last;
    VISKORES_EXEC bool operator()(const Real* point) const {
      for(int d=0;d<3;++d)
        if(!(point[d]>=lower[d] && (point[d]<upper[d] || (last[d] && point[d]==upper[d]))))
          return false;
      return true;
    }
  };
  template <class Portal> struct Writer {
    Portal output;
    viskores::Id offset;
    VISKORES_EXEC void operator()(int index,const Real* point,Real length,Real actual) const {
      output.Set(offset+index,viskores::Vec<Real,5>(point[0],point[1],point[2],length,actual));
    }
  };
public:
  using State = ParticleState;
  using ControlSignature = void(FieldIn,ExecObject,WholeArrayInOut,FieldOut,FieldOut);
  using ExecutionSignature = void(_1,_2,_3,_4,_5,WorkIndex);
  using InputDomain = _1;
  RK45TraceWorklet(int limit,Real minimum,Real maximum,Real tolerance,
                   Real target,int steps,Real lower,Real upper,bool last)
    : RK45TraceWorklet(limit,minimum,maximum,tolerance,target,steps,
        astr_insitu::NativeVector(lower,-std::numeric_limits<Real>::infinity(),-std::numeric_limits<Real>::infinity()),
        astr_insitu::NativeVector(upper,std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::infinity()),
        viskores::Vec<bool,3>(last,true,true)) {}
  RK45TraceWorklet(int limit,Real minimum,Real maximum,Real tolerance,
                   Real target,int steps,astr_insitu::NativeVector lower,astr_insitu::NativeVector upper,
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
                     input.steps,input.status};
    count=trace_segment(evaluate,owner,Writer<Portal>{geometry,index*(limit+1)},
                        state,limit,minimum,maximum,tolerance,target,steps);
    output=State(state.point[0],state.point[1],state.point[2],state.suggested,state.length,
                 state.error,state.steps,state.status);
  }
private:
  int limit,steps;
  Real minimum,maximum,tolerance,target;
  astr_insitu::NativeVector lower,upper;
  viskores::Vec<bool,3> last;
};

class RK45PhysicalTraceWorklet : public RK45TraceWorklet {
public:
  using ControlSignature=void(FieldIn,ExecObject,ExecObject,WholeArrayInOut,FieldOut,FieldOut);
  using ExecutionSignature=void(_1,_2,_3,_4,_5,_6,WorkIndex);
  using InputDomain=_1;
  RK45PhysicalTraceWorklet(int limit,Real minimum,Real maximum,Real tolerance,Real target,int steps)
    : RK45TraceWorklet(limit,minimum,maximum,tolerance,target,steps,Real(0.),Real(0.),false) {}
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
