#include "insitu_streamline_worklet.h"
#include "insitu_device_array.h"
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/Invoker.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/cont/cuda/DeviceAdapterCuda.h>
#include <viskores/filter/flow/worklet/Field.h>
#include <viskores/filter/flow/worklet/GridEvaluators.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

constexpr int chunk=128;
using State=astr_insitu::RK45TraceWorklet::State;
using Vertex=viskores::Vec<double,5>;
__global__ void constant_field(viskores::Vec3f* velocity,int nodes,int axis) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<nodes) {velocity[i]=viskores::Vec3f(0.); velocity[i][axis]=1.;}
}
struct Allocation {
  viskores::Vec3f* pointer=nullptr;
  explicit Allocation(int nodes) {
    if (cudaMalloc(&pointer,nodes*sizeof(*pointer))!=cudaSuccess)
      throw std::runtime_error("device field allocation failed");
  }
  ~Allocation() { if (pointer) cudaFree(pointer); }
};

int run(int argc,char** argv,int rank,int ranks) {
  if (ranks!=1 && ranks!=2) throw std::runtime_error("bounded trace supports NP=1/2 only");
  const int axis=argc>1 ? std::atoi(argv[1]) : 0;
  const bool bidirectional=argc>2 && std::string(argv[2])=="bidirectional";
  if(axis<0 || axis>2) throw std::runtime_error("invalid trace axis");
  const int particles=bidirectional?32:16;
  int devices=0;
  if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices<ranks ||
      cudaSetDevice(rank)!=cudaSuccess) throw std::runtime_error("one GPU per rank required");
  viskores::cont::Initialize(argc,argv);
  viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  viskores::cont::Invoker invoke(viskores::cont::DeviceAdapterTagCuda{});
  const double pi=std::acos(-1.),h=2.*pi/32.;
  // Three synthetic halo cells permit RK substages beyond the ownership face.
  // This checks distributed interpolation/particles, not solver halo exchange.
  const int lower=ranks==1 ? 0 : (rank==0 ? 0 : 13);
  const int upper=ranks==1 ? 32 : (rank==0 ? 19 : 32);
  viskores::Id3 dimensions(33,33,33); dimensions[axis]=upper-lower+1;
  viskores::Vec3f origin(0.); origin[axis]=lower*h;
  const int nodes=dimensions[0]*dimensions[1]*dimensions[2];
  auto grid=viskores::cont::DataSetBuilderUniform().Create(dimensions,origin,viskores::Vec3f(h,h,h));
  Allocation allocation(nodes);
  constant_field<<<(nodes+255)/256,256>>>(allocation.pointer,nodes,axis);
  if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("field generation failed");
  auto velocity=astr_insitu::borrow_cuda_array(allocation.pointer,nodes);
  using Field=viskores::worklet::flow::VelocityField<decltype(velocity)>;
  using Grid=viskores::worklet::flow::GridEvaluator<Field>;
  astr_insitu::LengthGrid<Grid> evaluate(Grid(grid,Field(velocity)));
  std::vector<State> state(particles),local(particles);
  std::vector<double> wire(particles*8),received(particles*8*ranks);
  std::vector<Vertex> previous(particles);
  std::vector<viskores::Vec3f> seeds(particles);
  std::vector<bool> have_previous(particles,false);
  for (int i=0;i<particles;++i) {
    const bool reverse=i>=16;
    seeds[i]=viskores::Vec3f(pi/2.,pi/8.+(i%16)*(6.*pi/8.)/15.,pi/4.);
    if(axis!=0) seeds[i][0]=pi/8.+(i%16)*(6.*pi/8.)/15.;
    seeds[i][axis]=reverse?3.*pi/2.:pi/2.;
    state[i]=State(seeds[i][0],seeds[i][1],seeds[i][2],(reverse?-.1:.1)*h,0.,0.,0.,0.);
  }
  double error=0.,seam=0.;
  int transfers=0,total_vertices=0,rounds=0;
  for (;rounds<1000;++rounds) {
    int active=0;
    for (int i=0;i<particles;++i) {
      local[i]=state[i];
      const bool running=state[i][7]==astr_insitu::TraceActive || state[i][7]==astr_insitu::TraceTransfer;
      const int owner=ranks==1 ? 0 : (state[i][axis]<pi ? 0 : 1);
      active+=running;
      local[i][7]=running && owner==rank ? astr_insitu::TraceActive : astr_insitu::TraceTransfer;
    }
    if (!active) break;
    auto input=viskores::cont::make_ArrayHandle(local,viskores::CopyFlag::On);
    viskores::cont::ArrayHandle<State> output;
    viskores::cont::ArrayHandle<viskores::Int32> counts;
    viskores::cont::ArrayHandle<Vertex> geometry;
    geometry.Allocate(particles*(chunk+1));
    viskores::Vec3f owned_lower(0.),owned_upper(2.*pi);
    owned_lower[axis]=rank*2.*pi/ranks; owned_upper[axis]=(rank+1)*2.*pi/ranks;
    viskores::Vec<bool,3> last(true); last[axis]=rank==ranks-1;
    invoke(astr_insitu::RK45TraceWorklet(chunk,.01*h,.5*h,1e-8,pi,100000,
       owned_lower,owned_upper,last),input,evaluate,geometry,output,counts);
    if (cudaDeviceSynchronize()!=cudaSuccess) throw std::runtime_error("trace dispatch failed");
    astr_insitu::require_device_only(velocity);
    // Host access is limited to particle state and extracted trajectory geometry.
    const auto result=output.ReadPortal();
    const auto sizes=counts.ReadPortal();
    const auto points=geometry.ReadPortal();
    for (int i=0;i<particles;++i) {
      local[i]=result.Get(i);
      for (int d=0;d<8;++d) wire[i*8+d]=local[i][d];
      const int count=sizes.Get(i);
      if (count<0 || count>chunk+1) throw std::runtime_error("invalid segment capacity");
      total_vertices+=count;
      if (count && have_previous[i])
        for (int d=0;d<4;++d) seam=std::fmax(seam,std::abs(points.Get(i*(chunk+1))[d]-previous[i][d]));
      for (int j=0;j<count;++j) {
        const auto point=points.Get(i*(chunk+1)+j);
        for (int d=0;d<5;++d) if (!std::isfinite(point[d])) throw std::runtime_error("nonfinite geometry");
        for(int d=0;d<3;++d)
          error=std::fmax(error,std::abs(point[d]-seeds[i][d]-(d==axis?(i>=16?-point[3]:point[3]):0.)));
      }
      if (count) { previous[i]=points.Get(i*(chunk+1)+count-1); have_previous[i]=true; }
    }
    if (MPI_Allgather(wire.data(),particles*8,MPI_DOUBLE,received.data(),particles*8,
                      MPI_DOUBLE,MPI_COMM_WORLD)!=MPI_SUCCESS)
      throw std::runtime_error("small-state exchange failed");
    for (int i=0;i<particles;++i) {
      const bool running=state[i][7]==astr_insitu::TraceActive || state[i][7]==astr_insitu::TraceTransfer;
      if (!running) continue;
      const int owner=ranks==1 ? 0 : (state[i][axis]<pi ? 0 : 1);
      for (int d=0;d<8;++d) state[i][d]=received[owner*particles*8+i*8+d];
      if (state[i][7]==astr_insitu::TraceTransfer) {
        if (rank==owner) ++transfers;
        // Every rank retains only the last extracted vertex as seam metadata.
        for (int d=0;d<4;++d) previous[i][d]=state[i][d==3 ? 4 : d];
        have_previous[i]=true;
      } else if (state[i][7]!=astr_insitu::TraceActive && state[i][7]!=astr_insitu::TraceLength)
        throw std::runtime_error("unexpected trajectory termination");
    }
    astr_insitu::require_device_only(velocity);
  }
  if (rounds==1000) throw std::runtime_error("bounded round limit exceeded");
  for (int i=0;i<particles;++i) {
    const auto& particle=state[i];
    if (particle[7]!=astr_insitu::TraceLength) throw std::runtime_error("unfinished trajectory");
    error=std::fmax(error,std::abs(particle[axis]-(i>=16?pi/2.:3.*pi/2.)));
    error=std::fmax(error,std::abs(particle[4]-pi));
  }
  double errors[2]={error,seam},maximum[2];
  int totals[2]={transfers,total_vertices},global[2];
  MPI_Allreduce(errors,maximum,2,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(totals,global,2,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (!rank) std::printf("Viskores CUDA trace NP=%d particles=%d max_error=%.17g seam_error=%.17g transfers=%d vertices=%d rounds=%d input_host_mirror=0 axis=%d bidirectional=%d\n",
                        ranks,particles,maximum[0],maximum[1],global[0],global[1],rounds,axis,bidirectional);
  return std::isfinite(maximum[0]) && std::isfinite(maximum[1]) && maximum[0]<=2e-10 &&
         maximum[1]<=2e-10 && global[0]==(ranks==2 ? particles : 0) ? 0 : 2;
}
int main(int argc,char** argv) {
  int provided=0,rank=0,ranks=0;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  int status=0;
  try { status=run(argc,argv,rank,ranks); }
  catch (const std::exception& error) {
    std::fprintf(stderr,"rank %d device trace failed: %s\n",rank,error.what());
    MPI_Abort(MPI_COMM_WORLD,3);
  }
  MPI_Finalize();
  return status;
}
