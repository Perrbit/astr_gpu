#include "insitu_wall_trace.h"
#include <algorithm>

namespace {
using R=astr_insitu::IntegrationReal;
static_assert(sizeof(R)==8,"This numerical oracle is FP64 only");
struct Portal {
  const R* values;
  R Get(std::int64_t i) const {return values[i];}
};
struct Owner {bool operator()(const R*) const {return true;}};
using Grid=astr_insitu::WallBilinearGrid<Portal,Portal,Portal,Portal>;
struct RangeWriter {
  const Grid& grid;
  R* range;
  bool* valid;
  void operator()(int,const R* point,R,R) const {
    R shear[3];
    const auto status=grid.sample(point,shear);
    if(status!=astr_insitu::WallSample::valid && status!=astr_insitu::WallSample::zero) {*valid=false;return;}
    const R scale=std::fmax(std::abs(shear[0]),std::abs(shear[2]));
    const R norm=scale==R(0)?R(0):scale*std::hypot(shear[0]/scale,shear[2]/scale);
    range[0]=std::fmin(range[0],norm);range[1]=std::fmax(range[1],norm);
  }
};
}

// Test-only C ABI: HDF5/NumPy supply a native ASTR CPU wall, not a Python
// flow solver. The exact same wall interpolation/tableau runs independently.
extern "C" int astr_wall_trace_reference(int nx,int nz,const double* x,const double* z,
    const double* tx,const double* tz,const double* seeds,int count,const double* settings,
    int maximum_steps,int direction,double* output,int* terminal,double* ranges) {
  if(nx<2 || nz<2 || count<1 || maximum_steps<1 || direction<-1 || direction>1) return 1;
  const R y=seeds[1];
  const Grid grid{
    {x},{z},{tx},{tz},nx,nz,y,{x[0],x[nx-1],z[0],z[nz-1]},{x[0],x[nx-1],z[0],z[nz-1]}};
  const int particles=count*(direction==0?2:1);
  for(int i=0;i<particles;++i) {
    const int sign=direction<0?-1:(i<count?1:-1),seed=i%count;
    astr_insitu::TraceState state{{seeds[3*seed],seeds[3*seed+1],seeds[3*seed+2]},
      sign*settings[0],R(0),R(0),0,astr_insitu::TraceActive};
    ranges[2*i]=INFINITY;ranges[2*i+1]=-INFINITY;bool valid=true;
    astr_insitu::wall_trace_segment(grid,Owner{},RangeWriter{grid,ranges+2*i,&valid},state,maximum_steps,settings[1],settings[2],
      settings[3],settings[4],maximum_steps);
    if(!valid) return 2;
    for(int d=0;d<3;++d) output[4*i+d]=state.point[d];
    output[4*i+3]=state.length;terminal[2*i]=state.steps;terminal[2*i+1]=state.status;
  }
  return 0;
}
