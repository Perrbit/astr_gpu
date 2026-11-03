#ifndef ASTR_INSITU_WALL_TRACE_H
#define ASTR_INSITU_WALL_TRACE_H

#include "insitu_rk45_device.h"
#include <cstdint>

#ifdef __CUDACC__
#define ASTR_WALL_EXEC __host__ __device__
#else
#define ASTR_WALL_EXEC
#endif

namespace astr_insitu {
enum WallTraceStatus {
  WallZero = 20, WallStageZero = 21, WallAccuracyLimited = 22,
  WallPhysicalExit = 23, WallRegionExit = 24, WallSupportExit = 25
};
enum class WallSample { valid, zero, physical_exit, region_exit, support_exit, nonfinite };

ASTR_WALL_EXEC inline int wall_sample_status(WallSample status,bool stage=false) {
  switch(status) {
    case WallSample::valid:return StepOk;
    case WallSample::zero:return stage?WallStageZero:WallZero;
    case WallSample::physical_exit:return WallPhysicalExit;
    case WallSample::region_exit:return WallRegionExit;
    case WallSample::support_exit:return WallSupportExit;
    default:return Nonfinite;
  }
}

// The portals expose the actual rectangular wall coordinates and shear, x-fast.
// Axis ordering and node finiteness are checked once by the field provider.
template<class X,class Z,class TX,class TZ>
struct WallBilinearGrid {
  X x;
  Z z;
  TX tx;
  TZ tz;
  int nx,nz;
  IntegrationReal wall_y,physical[4],region[4]; // xmin,xmax,zmin,zmax

  template<class Axis>
  ASTR_WALL_EXEC static int interval(const Axis& axis,int nodes,IntegrationReal value) {
    if(nodes<2 || value<axis.Get(0) || value>axis.Get(nodes-1)) return -1;
    int lower=0,upper=nodes-1;
    while(upper-lower>1) {
      const int middle=lower+(upper-lower)/2;
      if(value<axis.Get(middle)) upper=middle;
      else lower=middle;
    }
    return lower==nodes-1?lower-1:lower;
  }

  ASTR_WALL_EXEC WallSample sample(const IntegrationReal* point,IntegrationReal* shear) const {
    using R=IntegrationReal;
    for(int d=0;d<3;++d) if(!std::isfinite(point[d])) return WallSample::nonfinite;
    const R px=point[0],pz=point[2];
    if(px<physical[0] || px>physical[1] || pz<physical[2] || pz>physical[3])
      return WallSample::physical_exit;
    if(px<region[0] || px>region[1] || pz<region[2] || pz>region[3])
      return WallSample::region_exit;
    if(point[1]!=wall_y) return WallSample::support_exit;
    const int i=interval(x,nx,px),k=interval(z,nz,pz);
    if(i<0 || k<0) return WallSample::support_exit;
    const R dx=x.Get(i+1)-x.Get(i),dz=z.Get(k+1)-z.Get(k);
    if(!(dx>R(0)) || !(dz>R(0)) || !std::isfinite(dx) || !std::isfinite(dz))
      return WallSample::nonfinite;
    const R a=(px-x.Get(i))/dx,b=(pz-z.Get(k))/dz;
    const std::int64_t j=std::int64_t(k)*nx+i;
    const R sx0=(R(1)-a)*tx.Get(j)+a*tx.Get(j+1);
    const R sx1=(R(1)-a)*tx.Get(j+nx)+a*tx.Get(j+nx+1);
    const R sz0=(R(1)-a)*tz.Get(j)+a*tz.Get(j+1);
    const R sz1=(R(1)-a)*tz.Get(j+nx)+a*tz.Get(j+nx+1);
    shear[0]=(R(1)-b)*sx0+b*sx1;
    shear[1]=R(0);
    shear[2]=(R(1)-b)*sz0+b*sz1;
    if(!std::isfinite(shear[0]) || !std::isfinite(shear[2])) return WallSample::nonfinite;
    return shear[0]==R(0) && shear[2]==R(0)?WallSample::zero:WallSample::valid;
  }

  ASTR_WALL_EXEC WallSample tangent(const IntegrationReal* point,IntegrationReal* output) const {
    using R=IntegrationReal;
    const WallSample status=sample(point,output);
    if(status!=WallSample::valid) return status;
    const R scale=std::fmax(std::abs(output[0]),std::abs(output[2]));
    const R u=output[0]/scale,w=output[2]/scale,norm=std::sqrt(u*u+w*w);
    output[0]=u/norm;output[1]=R(0);output[2]=w/norm;
    return WallSample::valid;
  }
};

template<class Grid>
struct WallStageEvaluator {
  const Grid& grid;
  WallSample* failure;
  ASTR_WALL_EXEC bool operator()(const IntegrationReal* point,IntegrationReal,IntegrationReal* tangent) const {
    *failure=grid.tangent(point,tangent);
    return *failure==WallSample::valid;
  }
};

// Reuse the shared Cash-Karp tableau, not the fluid tracer's permissive min step.
template<class Grid>
ASTR_WALL_EXEC int wall_rk45_step(const Grid& grid,const IntegrationReal* previous,
    IntegrationReal* next,IntegrationReal& request,IntegrationReal& actual,
    IntegrationReal minimum,IntegrationReal maximum,IntegrationReal tolerance,IntegrationReal& error) {
  using R=IntegrationReal;
  error=RK45ErrorSentinel;actual=R(0);
  if(!std::isfinite(request) || request==R(0) || !std::isfinite(minimum) ||
     !std::isfinite(maximum) || !std::isfinite(tolerance) ||
     minimum<=R(0) || maximum<minimum || tolerance<=R(0)) return Nonfinite;
  R tangent[3];
  const auto initial=grid.tangent(previous,tangent);
  if(initial!=WallSample::valid) return wall_sample_status(initial);
  R magnitude=std::fmax(minimum,std::fmin(std::abs(request),maximum));
  const R direction=std::copysign(R(1),request);
  for(;;) {
    const R trial=direction*magnitude;
    WallSample failure=WallSample::valid;
    const int status=cash_karp_step(WallStageEvaluator<Grid>{grid,&failure},previous,
      next,R(0),trial,actual,error);
    if(status!=StepOk) return failure==WallSample::valid?status:wall_sample_status(failure,true);
    if(error<=tolerance) {
      const R factor=error==R(0)?R(5):R(.9)*std::pow(error/tolerance,R(-.2));
      request=direction*std::fmax(minimum,std::fmin(maximum,magnitude*factor));
      return StepOk;
    }
    if(magnitude==minimum) return WallAccuracyLimited;
    const R reduced=R(.9)*magnitude*std::pow(error/tolerance,R(-.25));
    if(!std::isfinite(reduced)) return Nonfinite;
    magnitude=std::fmax(minimum,reduced);
  }
}

template<class Grid,class Owner,class Writer>
ASTR_WALL_EXEC int wall_trace_segment(const Grid& grid,const Owner& owner,const Writer& write,
    TraceState& state,int limit,IntegrationReal minimum,IntegrationReal maximum,
    IntegrationReal tolerance,IntegrationReal target,int maximum_steps) {
  using R=IntegrationReal;
  if(state.status!=TraceActive) return 0;
  if(!std::isfinite(target) || target<=R(0) || !std::isfinite(state.length) ||
     state.length<R(0) || state.steps<0 || limit<=0 || maximum_steps<=0 ||
     !std::isfinite(minimum) || !std::isfinite(maximum) || minimum<=R(0) || maximum<minimum) {
    state.status=Nonfinite;return 0;
  }
  for(int d=0;d<3;++d) if(!std::isfinite(state.point[d])) {state.status=Nonfinite;return 0;}
  int vertices=1;
  write(0,state.point,state.length,R(0));
#ifdef __CUDACC__
#pragma unroll 1
#endif
  for(int iteration=0;iteration<limit;++iteration) {
    if(state.length>=target) {state.status=TraceLength;break;}
    if(state.steps>=maximum_steps) {state.status=TraceSteps;break;}
    if(!owner(state.point)) {state.status=TraceTransfer;break;}
    const R remaining=target-state.length;
    const R bound=std::fmin(maximum,remaining);
    // A final interval shorter than min is tested once; it is never accepted above tolerance.
    const R floor=std::fmin(minimum,bound);
    R next[3]={state.point[0],state.point[1],state.point[2]},actual=R(0),request=state.suggested;
    const int status=wall_rk45_step(grid,state.point,next,request,actual,floor,bound,tolerance,state.error);
    state.suggested=request;
    if(status!=StepOk) {state.status=status;break;}
    R tangent[3];
    const auto final=grid.tangent(next,tangent);
    if(final!=WallSample::valid) {state.status=wall_sample_status(final,true);break;}
    state.length+=std::abs(actual);
    if(!std::isfinite(state.length)) {state.status=Nonfinite;break;}
    for(int d=0;d<3;++d) state.point[d]=next[d];
    ++state.steps;
    write(vertices++,state.point,state.length,actual);
  }
  if(state.status==TraceActive) {
    if(state.length>=target) state.status=TraceLength;
    else if(state.steps>=maximum_steps) state.status=TraceSteps;
    else if(!owner(state.point)) state.status=TraceTransfer;
  }
  return vertices;
}
} // namespace astr_insitu
#undef ASTR_WALL_EXEC
#endif
