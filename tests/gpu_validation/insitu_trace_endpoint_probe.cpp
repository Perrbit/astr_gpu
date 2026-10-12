#include "insitu_rk45_device.h"
#include <cstdio>

struct Field {
  bool curved;
  bool operator()(const double* p,double,double* tangent) const {
    if(p[1]<0.) return false;
    tangent[0]=1.;tangent[1]=curved?-p[0]*p[0]*p[0]:0.;tangent[2]=0.;
    return true;
  }
};
struct Owner {
  bool handoff;
  bool operator()(const double* p) const {return p[1]>=0. && (!handoff || p[0]<.5);}
};
struct Writer {
  double* minimum_y;
  void operator()(int,const double* p,double,double) const {
    *minimum_y=std::fmin(*minimum_y,p[1]);
  }
};

int main() {
  // All six Cash-Karp stage points are inside y>=0; the combination is y=-.01.
  astr_insitu::TraceState state{{0.,.24,0.},1.,0.,0.,0,astr_insitu::TraceActive};
  double minimum_y=state.point[1];
  const int vertices=astr_insitu::trace_segment(Field{true},Owner{false},Writer{&minimum_y},
    state,1,1.,1.,1e-8,2.,8);
  if(state.status!=astr_insitu::Outside || vertices!=1 || state.steps!=0 ||
      state.length!=0. || state.point[1]!=.24 || minimum_y<0.) {
    std::printf("FAIL endpoint vertices=%d status=%d y=%.17g\n",vertices,state.status,state.point[1]);
    return 1;
  }
  state={{0.,.24,0.},1.,0.,0.,0,astr_insitu::TraceActive};minimum_y=state.point[1];
  const int transferred=astr_insitu::trace_segment(Field{false},Owner{true},Writer{&minimum_y},
    state,1,1.,1.,1e-8,2.,8);
  if(state.status!=astr_insitu::TraceTransfer || transferred!=2 || state.point[0]!=1. || state.steps!=1)
    return 2;
  std::puts("PASS endpoint domain guard and valid MPI handoff");
}
