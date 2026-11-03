#include "insitu_wall_trace.h"
#include <algorithm>
#include <cstdio>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define WALL_PROBE_EXEC __host__ __device__
#else
#define WALL_PROBE_EXEC
#endif

namespace {
using R=astr_insitu::IntegrationReal;
constexpr int nodes=17,cases=18,capacity=4097;
struct Portal {
  const R* data;
  WALL_PROBE_EXEC R Get(int i) const {return data[i];}
};
using Grid=astr_insitu::WallBilinearGrid<Portal,Portal,Portal,Portal>;
struct Input {
  R x[nodes],z[nodes],tx[nodes*nodes],tz[nodes*nodes];
  R minimum,maximum,tolerance,target,initial,seed[3],region_upper,support_upper;
  int maximum_steps;
};
struct Result {
  astr_insitu::TraceState state;
  R points[capacity][5];
  int count,endpoint_guard;
};
struct Owner {
  WALL_PROBE_EXEC bool operator()(const R*) const {return true;}
};
struct Writer {
  Result* result;
  WALL_PROBE_EXEC void operator()(int i,const R* p,R arc,R actual) const {
    for(int d=0;d<3;++d) result->points[i][d]=p[d];
    result->points[i][3]=arc;result->points[i][4]=actual;
  }
};

struct FinalZeroCircle {
  R zero[3];
  bool enabled;
  WALL_PROBE_EXEC astr_insitu::WallSample tangent(const R* p,R* output) const {
    if(enabled && p[0]==zero[0] && p[1]==zero[1] && p[2]==zero[2]) return astr_insitu::WallSample::zero;
    const R x=p[0]-R(.5),z=p[2]-R(.5),norm=std::sqrt(x*x+z*z);
    output[0]=-z/norm;output[1]=R(0);output[2]=x/norm;
    return astr_insitu::WallSample::valid;
  }
};
struct DropWriter {
  WALL_PROBE_EXEC void operator()(int,const R*,R,R) const {}
};

WALL_PROBE_EXEC bool final_zero_guard() {
  using namespace astr_insitu;
  FinalZeroCircle circle{{R(0),R(0),R(0)},false};
  const R previous[3]={R(.75),R(.125),R(.5)};
  R next[3],request=R(.00625),actual=R(0),error=R(0);
  if(wall_rk45_step(circle,previous,next,request,actual,R(.00625),R(.00625),R(1e-8),error)!=StepOk) return false;
  for(int d=0;d<3;++d) circle.zero[d]=next[d];
  circle.enabled=true;
  WallSample failure=WallSample::valid;
  if(cash_karp_step(WallStageEvaluator<FinalZeroCircle>{circle,&failure},previous,next,R(0),
      R(.00625),actual,error)!=StepOk) return false; // All six stages are nonzero.
  TraceState state{{previous[0],previous[1],previous[2]},R(.00625),R(0),R(0),0,TraceActive};
  const int count=wall_trace_segment(circle,Owner{},DropWriter{},state,1,R(.00625),R(.00625),R(1e-8),R(1),10);
  return count==1 && state.status==WallStageZero && state.steps==0 && state.length==R(0) &&
    state.point[0]==previous[0] && state.point[1]==previous[1] && state.point[2]==previous[2];
}

WALL_PROBE_EXEC Grid grid_for(const Input& in) {
  return Grid{Portal{in.x},Portal{in.z},Portal{in.tx},Portal{in.tz},nodes,nodes,R(.125),
    {R(0),R(1),R(0),R(1)},{R(0),in.region_upper,R(0),R(1)}};
}

WALL_PROBE_EXEC void run_one(const Input& in,Result& out) {
  out.state={{in.seed[0],in.seed[1],in.seed[2]},in.initial,R(0),R(0),0,astr_insitu::TraceActive};
  out.count=astr_insitu::wall_trace_segment(grid_for(in),Owner{},Writer{&out},out.state,capacity-1,
    in.minimum,in.maximum,in.tolerance,in.target,in.maximum_steps);
  out.endpoint_guard=final_zero_guard();
}

#ifdef __CUDACC__
__global__ void run_gpu(const Input* input,Result* result) {
  const int i=threadIdx.x;
  if(i<cases) run_one(input[i],result[i]);
}
#endif

void initialize(Input& in,int mode) {
  in={};
  in.minimum=R(.01)/16;in.maximum=R(.5)/16;in.initial=R(.1)/16;in.tolerance=R(1e-8);
  in.target=R(.6);in.maximum_steps=100000;in.region_upper=R(1);in.support_upper=R(1);
  in.seed[0]=R(.25);in.seed[1]=R(.125);in.seed[2]=R(.5);
  if(mode==1) in.seed[0]=R(.75);
  if(mode==2 || mode==3) {in.seed[0]=R(.5);in.seed[2]=mode==2?R(.25):R(.75);}
  if(mode==4 || mode==5 || mode==16) {in.seed[0]=in.seed[2]=R(.2);in.target=R(.5);}
  if(mode==7 || mode==9 || mode==17) {
    in.seed[0]=R(.75);in.target=R(.25);in.tolerance=R(1e-12);
  }
  if(mode==8) {in.initial=in.minimum=in.maximum=R(.3125);in.target=R(.5);}
  if(mode==9) {in.initial=in.minimum=in.maximum=R(.2);in.tolerance=R(1e-30);}
  if(mode==10) in.seed[0]=R(.99);
  if(mode==11) {in.seed[0]=R(.5);in.region_upper=R(.6);}
  if(mode==12) {in.seed[0]=R(.49);in.support_upper=R(.5);}
  if(mode==13) in.maximum_steps=2;
  if(mode==15) in.target=R(.2501);
  if(mode==16) {in.initial=-in.initial;in.seed[0]=in.seed[2]=R(.8);}
  for(int i=0;i<nodes;++i) {
    const R t=R(i)/16;
    in.x[i]=mode==17?t*t:in.support_upper*t;
    in.z[i]=mode==17?t*t*t:t;
  }
  for(int k=0;k<nodes;++k) for(int i=0;i<nodes;++i) {
    const int j=k*nodes+i;
    in.tx[j]=R(1);in.tz[j]=R(0);
    if(mode==1) in.tx[j]=R(-1);
    if(mode==2 || mode==3) {in.tx[j]=R(0);in.tz[j]=mode==2?R(1):R(-1);}
    if(mode==4 || mode==5 || mode==16) {
      const R scale=mode==5?R(1e-300):R(1);
      in.tx[j]=R(3)*scale;in.tz[j]=R(4)*scale;
    }
    if(mode==6) in.tx[j]=R(0);
    if(mode==7 || mode==9 || mode==17) {in.tx[j]=R(.5)-in.z[k];in.tz[j]=in.x[i]-R(.5);}
    if(mode==8 && i==5) in.tx[j]=R(0);
    if(mode==14) in.tx[j]=NAN;
  }
}

#ifdef __CUDACC__
bool compare(const Result& cpu,const Result& gpu,R& worst) {
  if(cpu.count!=gpu.count || cpu.state.steps!=gpu.state.steps || cpu.state.status!=gpu.state.status ||
     cpu.endpoint_guard!=1 || gpu.endpoint_guard!=1) return false;
  const auto check=[&](R a,R b) {
    if(!std::isfinite(a) || !std::isfinite(b)) return false;
    worst=std::fmax(worst,std::abs(a-b)/std::fmax(R(1),std::fmax(std::abs(a),std::abs(b))));
    return true;
  };
  for(int d=0;d<3;++d) if(!check(cpu.state.point[d],gpu.state.point[d])) return false;
  if(!check(cpu.state.length,gpu.state.length)) return false;
  for(int i=0;i<cpu.count;++i) for(int d=0;d<5;++d)
    if(!check(cpu.points[i][d],gpu.points[i][d])) return false;
  return worst<=R(2e-10);
}
#endif

bool analytic(const Input& in,const Result& out,int mode,R& worst) {
  using namespace astr_insitu;
  int expected=TraceLength;
  switch(mode) {
    case 6:expected=WallZero;break;
    case 8:expected=WallStageZero;break;
    case 9:expected=WallAccuracyLimited;break;
    case 10:expected=WallPhysicalExit;break;
    case 11:expected=WallRegionExit;break;
    case 12:expected=WallSupportExit;break;
    case 13:expected=TraceSteps;break;
    case 14:expected=Nonfinite;break;
  }
  if(out.state.status!=expected || out.count!=out.state.steps+1) return false;
  if(mode==6 || mode==8 || mode==9 || mode==14)
    return out.state.steps==0 && out.state.length==R(0) && out.state.point[0]==in.seed[0];
  if(mode==10 || mode==11 || mode==12 || mode==13) {
    for(int i=0;i<out.count;++i)
      if(out.points[i][0]>in.region_upper || out.points[i][0]>in.support_upper || out.points[i][0]<R(0)) return false;
    return mode!=13 || out.state.steps==2;
  }
  R expected_x=in.seed[0],expected_z=in.seed[2];
  if(mode==7 || mode==17) {expected_x=R(.5)+R(.25)*std::cos(R(1));expected_z=R(.5)+R(.25)*std::sin(R(1));}
  else if(mode==2 || mode==3) expected_z+=(mode==2?R(1):R(-1))*in.target;
  else if(mode==4 || mode==5 || mode==16) {
    const R direction=mode==16?R(-1):R(1);
    expected_x+=direction*R(.6)*in.target;expected_z+=direction*R(.8)*in.target;
  } else expected_x+=(mode==1?R(-1):R(1))*in.target;
  worst=std::fmax(worst,std::abs(out.state.point[0]-expected_x));
  worst=std::fmax(worst,std::abs(out.state.point[2]-expected_z));
  worst=std::fmax(worst,std::abs(out.state.length-in.target));
  for(int i=0;i<out.count;++i) {
    if(out.points[i][1]!=in.seed[1]) return false;
    if(mode==7 || mode==17) worst=std::fmax(worst,std::abs(std::hypot(out.points[i][0]-R(.5),out.points[i][2]-R(.5))-R(.25)));
    else {
      const R s=out.points[i][3];
      const R dx=mode==2 || mode==3?R(0):(mode==4 || mode==5?R(.6):(mode==16?R(-.6):(mode==1?R(-1):R(1))));
      const R dz=mode==2?R(1):(mode==3?R(-1):(mode==4 || mode==5?R(.8):(mode==16?R(-.8):R(0))));
      worst=std::fmax(worst,std::abs(out.points[i][0]-in.seed[0]-dx*s));
      worst=std::fmax(worst,std::abs(out.points[i][2]-in.seed[2]-dz*s));
    }
  }
  return worst<=R(2e-10);
}
}

#ifndef ASTR_WALL_TRACE_NO_MAIN
int main() {
  std::vector<Input> input(cases);
  std::vector<Result> cpu(cases);
  for(int i=0;i<cases;++i) {initialize(input[i],i);run_one(input[i],cpu[i]);}
  R numerical=R(0),reference=R(0);
  for(int i=0;i<cases;++i) {
    if(cpu[i].endpoint_guard!=1) return 3;
    if(!analytic(input[i],cpu[i],i,reference)) {
      std::fprintf(stderr,"wall trace analytic failure case=%d status=%d steps=%d error=%.17g\n",
        i,cpu[i].state.status,cpu[i].state.steps,double(reference));return 2;
    }
  }
#ifdef __CUDACC__
  Input* device_input=nullptr;Result* device_output=nullptr;
  if(cudaMalloc(&device_input,sizeof(Input)*cases)!=cudaSuccess ||
     cudaMalloc(&device_output,sizeof(Result)*cases)!=cudaSuccess) return 4;
  if(cudaMemcpy(device_input,input.data(),sizeof(Input)*cases,cudaMemcpyHostToDevice)!=cudaSuccess) return 5;
  run_gpu<<<1,32>>>(device_input,device_output);
  if(cudaDeviceSynchronize()!=cudaSuccess) return 6;
  std::vector<Result> gpu(cases);
  if(cudaMemcpy(gpu.data(),device_output,sizeof(Result)*cases,cudaMemcpyDeviceToHost)!=cudaSuccess) return 7;
  if(cudaFree(device_input)!=cudaSuccess || cudaFree(device_output)!=cudaSuccess) return 8;
  for(int i=0;i<cases;++i) if(!compare(cpu[i],gpu[i],numerical)) return 9;
#endif
  Input test;initialize(test,0);
  for(int k=0;k<nodes;++k) {
    test.tx[k*nodes+4]=R(1);test.tz[k*nodes+4]=R(0);
    test.tx[k*nodes+5]=R(0);test.tz[k*nodes+5]=R(100);
  }
  R point[3]={R(.28125),R(.125),R(.5)},shear[3],tangent[3];
  const auto grid=grid_for(test);
  if(grid.sample(point,shear)!=astr_insitu::WallSample::valid || shear[0]!=R(.5) || shear[2]!=R(50) ||
     grid.tangent(point,tangent)!=astr_insitu::WallSample::valid ||
     std::abs(tangent[0]-R(1)/std::sqrt(R(10001)))>R(2e-10)) return 10;
  std::printf("PASS wall trace cases=%d analytic_error=%.17g cpu_gpu_error=%.17g interpolation=raw-bilinear endpoint_zero_retained=1\n",
    cases,double(reference),double(numerical));
  return 0;
}
#endif
