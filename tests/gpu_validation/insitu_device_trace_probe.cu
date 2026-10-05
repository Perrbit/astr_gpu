#include "insitu_rk45_device.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

constexpr int particles=16,capacity=129;
struct Vertex { double point[3],length,actual; };
struct Tangent {
  int mode;
  __host__ __device__ bool operator()(const double* point,double,double* value) const {
    const double pi=std::acos(-1.);
    if (mode) {
      const double x=point[0]-pi,y=point[1]-pi,r=std::sqrt(x*x+y*y);
      value[0]=-y/r; value[1]=x/r; value[2]=0.;
    } else { value[0]=1.; value[1]=value[2]=0.; }
    return true;
  }
};
struct Owner {
  bool lower;
  __host__ __device__ bool operator()(const double* point) const {
    return !lower || point[0]<std::acos(-1.);
  }
};
struct Writer {
  Vertex* vertices;
  __host__ __device__ void operator()(int i,const double* point,double length,double actual) const {
    for (int d=0;d<3;++d) vertices[i].point[d]=point[d];
    vertices[i].length=length; vertices[i].actual=actual;
  }
};
__global__ void segment(astr_insitu::TraceState* state,Vertex* vertices,int* counts,
                       int mode,int limit,bool lower) {
  const int i=threadIdx.x;
  if (i>=particles) return;
  const double pi=std::acos(-1.),h=2.*pi/32.;
  counts[i]=astr_insitu::trace_segment(Tangent{mode},Owner{lower},Writer{vertices+i*capacity},
    state[i],limit,.01*h,.5*h,1e-8,pi,100000);
}
bool equal(const astr_insitu::TraceState& a,const astr_insitu::TraceState& b) {
  for (int d=0;d<3;++d) if (a.point[d]!=b.point[d]) return false;
  return a.suggested==b.suggested && a.length==b.length && a.error==b.error &&
         a.steps==b.steps && a.status==b.status;
}
int main() {
  astr_insitu::TraceState* state=nullptr;
  Vertex* vertices=nullptr;
  int* counts=nullptr;
  if (cudaMallocManaged(&state,particles*sizeof(*state))!=cudaSuccess ||
      cudaMallocManaged(&vertices,particles*capacity*sizeof(*vertices))!=cudaSuccess ||
      cudaMallocManaged(&counts,particles*sizeof(*counts))!=cudaSuccess) return 1;
  const double pi=std::acos(-1.),h=2.*pi/32.;
  astr_insitu::TraceState initial[particles],baseline[particles],saved[particles];
  double endpoint_error=0.,counter_error=0.,circle_gap=0.;
  for (int mode=0;mode<2;++mode) {
    for (int i=0;i<particles;++i) {
      initial[i]={};
      initial[i].point[0]=mode ? pi+.2 : pi/2.;
      initial[i].point[1]=mode ? pi : pi/8.+i*(6.*pi/8.)/15.;
      initial[i].point[2]=pi/4.;
      initial[i].suggested=.1*h;
      state[i]=initial[i];
    }
    segment<<<1,32>>>(state,vertices,counts,mode,128,false);
    if (cudaDeviceSynchronize()!=cudaSuccess) return 2;
    for (int i=0;i<particles;++i) {
      baseline[i]=state[i];
      if (counts[i]<2 || counts[i]>capacity || state[i].status==astr_insitu::Nonfinite) return 3;
      double accepted=0.;
      for (int v=1;v<counts[i];++v) {
        accepted+=std::abs(vertices[i*capacity+v].actual);
        for (int d=0;d<3;++d) if (!std::isfinite(vertices[i*capacity+v].point[d])) return 4;
      }
      counter_error=std::fmax(counter_error,std::abs(accepted-state[i].length));
      if (!mode) {
        endpoint_error=std::fmax(endpoint_error,std::abs(state[i].point[0]-3.*pi/2.));
        for (int d=1;d<3;++d)
          endpoint_error=std::fmax(endpoint_error,std::abs(state[i].point[d]-initial[i].point[d]));
      } else circle_gap=std::fmax(circle_gap,std::abs(pi-state[i].length));
      state[i]=initial[i];
    }
    // Pause after eight accepted steps, preserve only the small particle state.
    segment<<<1,32>>>(state,vertices,counts,mode,8,false);
    if (cudaDeviceSynchronize()!=cudaSuccess) return 5;
    for (int i=0;i<particles;++i) {
      if (state[i].steps!=8 || state[i].status!=astr_insitu::TraceActive) return 6;
      saved[i]=state[i]; state[i]={};
    }
    for (int i=0;i<particles;++i) state[i]=saved[i];
    segment<<<1,32>>>(state,vertices,counts,mode,128,false);
    if (cudaDeviceSynchronize()!=cudaSuccess) return 7;
    for (int i=0;i<particles;++i) if (!equal(state[i],baseline[i])) return 8;
  }
  // Ownership changes after an accepted step, not during an RK substage.
  for (int i=0;i<particles;++i) {
    initial[i]={}; initial[i].point[0]=pi/2.; initial[i].point[1]=pi/2.;
    initial[i].point[2]=pi/4.; initial[i].suggested=.1*h; state[i]=initial[i];
  }
  segment<<<1,32>>>(state,vertices,counts,0,128,false);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 9;
  for (int i=0;i<particles;++i) { baseline[i]=state[i]; state[i]=initial[i]; }
  segment<<<1,32>>>(state,vertices,counts,0,128,true);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 10;
  for (int i=0;i<particles;++i) {
    if (state[i].status!=astr_insitu::TraceTransfer || state[i].point[0]<pi) return 11;
    state[i].status=astr_insitu::TraceActive;
  }
  segment<<<1,32>>>(state,vertices,counts,0,128,false);
  if (cudaDeviceSynchronize()!=cudaSuccess) return 12;
  for (int i=0;i<particles;++i) if (!equal(state[i],baseline[i])) return 13;
  // Forced final remainder: retain the last accepted vertex and raw VTK status.
  for(int i=0;i<particles;++i) {
    state[i]=initial[i];state[i].length=pi-.005*h;state[i].steps=9;
    saved[i]=state[i];
  }
  segment<<<1,32>>>(state,vertices,counts,0,128,false);
  if(cudaDeviceSynchronize()!=cudaSuccess) return 16;
  for(int i=0;i<particles;++i) {
    if(!astr_insitu::trace_subminimum_stop(state[i],pi,.01*h) || counts[i]!=1 ||
       state[i].length!=saved[i].length || state[i].steps!=saved[i].steps) return 17;
    for(int d=0;d<3;++d) if(state[i].point[d]!=saved[i].point[d]) return 18;
    auto invalid=state[i];invalid.error=0.;
    if(astr_insitu::trace_subminimum_stop(invalid,pi,.01*h)) return 19;
    invalid=state[i];invalid.status=astr_insitu::Nonfinite;
    if(astr_insitu::trace_subminimum_stop(invalid,pi,.01*h)) return 20;
    invalid=state[i];invalid.suggested=.1*h;
    if(astr_insitu::trace_subminimum_stop(invalid,pi,.01*h)) return 21;
    if(astr_insitu::trace_subminimum_stop(state[i],pi,.001*h)) return 22;
  }
  if (cudaFree(counts)!=cudaSuccess || cudaFree(vertices)!=cudaSuccess || cudaFree(state)!=cudaSuccess) return 14;
  std::printf("GPU accepted-length trace particles=16 endpoint_error=%.17g counter_error=%.17g circle_gap=%.17g exact_state_resume=1 ownership_resume=1 subminimum_stop=1 unrelated_errors_rejected=1\n",
    endpoint_error,counter_error,circle_gap);
  return std::isfinite(endpoint_error) && std::isfinite(counter_error) && std::isfinite(circle_gap) &&
         endpoint_error<=2e-10 && counter_error<=2e-10 && circle_gap<=.01*h ? 0 : 15;
}
