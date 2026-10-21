#include "insitu_rk45_device.h"
#include <cstdio>
#include <type_traits>
#ifdef ASTR_NATIVE_RK45_GPU_TEST
#include <cuda_runtime.h>
#define PROBE_EXEC __host__ __device__
#else
#define PROBE_EXEC
#endif

using R=astr_insitu::IntegrationReal;
static_assert(std::is_same<R,astr::numerical_real>::value,"RK45 numerical type differs");

struct Field {
  int mode;
  PROBE_EXEC bool operator()(const R* x,R,R* f) const {
    if(mode==2 && x[0]>R(1)) return false;
    if(mode==1) {
      const R norm=std::sqrt(x[0]*x[0]+x[1]*x[1]);
      f[0]=-x[1]/norm;f[1]=x[0]/norm;f[2]=0;
    } else if(mode==3) {
      f[0]=f[1]=f[2]=0;
    } else if(mode==4) {
      f[0]=R(NAN);f[1]=f[2]=0;
    } else {f[0]=1;f[1]=f[2]=0;}
    return true;
  }
};
struct Owner {PROBE_EXEC bool operator()(const R*) const {return true;}};
struct Writer {
  R* largest;
  PROBE_EXEC void operator()(int,const R*,R length,R) const {if(length>*largest) *largest=length;}
};

PROBE_EXEC int check_core(R* result) {
  using namespace astr_insitu;
  const R weight_sum=R(37)/R(378)+R(250)/R(621)+R(125)/R(594)+R(512)/R(1771);
  for(int sign=-1;sign<=1;sign+=2) {
    R point[3]={0,0,0},next[3]={},step=R(sign)*R(.125),actual=0,error=0;
    if(rk45_step(Field{0},point,next,R(0),step,actual,R(.01),R(.5),R(1.e-8),error)!=StepOk) return 1;
    if(!std::isfinite(error) || next[0]!=actual*weight_sum || next[1]!=0 || next[2]!=0) return 2;
  }
  R point[3]={0,0,0},next[3]={},step=R(.1),actual=0,error=0;
  if(rk45_step(Field{3},point,next,R(0),step,actual,R(.01),R(.5),R(1.e-8),error)!=Unexpected) return 3;
  if(rk45_step(Field{4},point,next,R(0),step,actual,R(.01),R(.5),R(1.e-8),error)!=Nonfinite) return 4;
  point[0]=R(2);
  if(rk45_step(Field{2},point,next,R(0),step,actual,R(.01),R(.5),R(1.e-8),error)!=Outside) return 5;
  R largest=0;
  TraceState trace{{0,0,0},R(.125),0,0,0,TraceActive};
  trace_segment(Field{0},Owner{},Writer{&largest},trace,32,R(.125),R(.125),R(1.e-8),R(1),100);
  if(trace.status!=TraceLength || trace.length!=R(1) || largest>R(1) || !std::isfinite(trace.point[0])) return 6;
  trace=TraceState{{0,0,0},R(.1),R(.9995),0,0,TraceActive};
  trace_segment(Field{0},Owner{},Writer{&largest},trace,1,R(.001),R(.25),R(1.e-8),R(1),100);
  if(!trace_subminimum_stop(trace,R(1),R(.001)) || !std::isfinite(trace.error)) return 7;
  point[0]=1;point[1]=0;step=R(.02);
  const int status=rk45_step(Field{1},point,next,R(0),step,actual,R(.001),R(.1),R(1.e-8),error);
  if(status!=StepOk || !std::isfinite(actual) || !std::isfinite(error)) return 8;
  result[0]=next[0];result[1]=next[1];result[2]=actual;result[3]=error;
  return 0;
}

#ifdef ASTR_NATIVE_RK45_GPU_TEST
__global__ void check_kernel(R* result,int* status) {*status=check_core(result);}
#endif

int main() {
  R values[4]={};
#ifdef ASTR_NATIVE_RK45_GPU_TEST
  R* device=nullptr;int* status=nullptr;
  if(cudaMallocManaged(&device,4*sizeof(R))!=cudaSuccess || cudaMallocManaged(&status,sizeof(int))!=cudaSuccess) return 10;
  check_kernel<<<1,1>>>(device,status);
  if(cudaDeviceSynchronize()!=cudaSuccess) return 11;
  if(*status) return *status;
  for(int i=0;i<4;++i) values[i]=device[i];
  if(cudaFree(device)!=cudaSuccess || cudaFree(status)!=cudaSuccess) return 12;
#else
  const int status=check_core(values);
  if(status) return status;
#endif
  const double analytic_error=std::fmax(std::abs(double(values[0])-std::cos(double(values[2]))),
                                       std::abs(double(values[1])-std::sin(double(values[2]))));
  std::printf("PASS native RK45 bytes=%zu circle_error=%.17g estimate=%.17g\n",sizeof(R),analytic_error,double(values[3]));
}
#undef PROBE_EXEC
