#include <mpi.h>
#include <cuda_runtime_api.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" int astr_insitu_resource_begin(int,std::int64_t,std::int64_t,std::int64_t,int,const char*);
extern "C" int astr_insitu_resource_finish();
extern "C" int astr_insitu_allocation_guard_active();
extern "C" void astr_insitu_device_allocation_preflight(std::size_t,const char*);
extern "C" void astr_insitu_device_allocation_commit(const void*,std::size_t,const char*);
extern "C" int viskores_astr_allocation_lifecycle_version();
extern "C" int vtk_astr_buffer_allocation_lifecycle_version();

void require(bool condition,const char* message) {
  if(!condition) throw std::runtime_error(message);
}
int main(int argc,char** argv) try {
  require(MPI_Init(&argc,&argv)==MPI_SUCCESS,"MPI initialization");
  int rank=0,ranks=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  require(argc==3 && ranks==4,"expected output-directory, success/refuse, and four ranks");
  require(cudaSetDevice(rank%2)==cudaSuccess,"CUDA device selection");
  using Allocator=viskores::cont::cuda::internal::CudaAllocator;
  viskores::cont::GetRuntimeDeviceTracker().SetThreadFriendlyMemAlloc(false);
  auto warmup=Allocator::Allocate(8);Allocator::Free(warmup);
  require(viskores_astr_allocation_lifecycle_version()==1 &&
    vtk_astr_buffer_allocation_lifecycle_version()==1,"lifecycle capability");
  MPI_Barrier(MPI_COMM_WORLD);
  const bool refuse=std::string(argv[2])=="refuse";
  require(refuse || std::string(argv[2])=="success","invalid probe mode");
  const long long limit=(refuse?16:256)*1024LL*1024;
  require(astr_insitu_resource_begin(MPI_Comm_c2f(MPI_COMM_WORLD),4LL*1024*1024*1024,
    limit,1024LL*1024*1024,1,argv[1])==0,"resource observer");
  require(astr_insitu_allocation_guard_active()==1,"shared guard inactive");
  if(refuse) {
    std::printf("shared_probe rank=%d preflight_only=1 no_device_allocation=1\n",rank);std::fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    astr_insitu_device_allocation_preflight(12*1024*1024,"Viskores array");
    char identity;
    astr_insitu_device_allocation_commit(&identity,12*1024*1024,"Viskores array");
    MPI_Barrier(MPI_COMM_WORLD);
    throw std::runtime_error("concurrent over-budget reservation was admitted");
  }
  Allocator::SetAllocationPreflight(astr_insitu_device_allocation_preflight);
  auto immediate=Allocator::Allocate(4*1024*1024);
  MPI_Barrier(MPI_COMM_WORLD);Allocator::Free(immediate);
  auto deferred=Allocator::Allocate(4*1024*1024);
  Allocator::FreeDeferred(deferred,4*1024*1024);
  MPI_Barrier(MPI_COMM_WORLD);
  auto flush=Allocator::Allocate(32*1024*1024);
  Allocator::FreeDeferred(flush,32*1024*1024);
  MPI_Barrier(MPI_COMM_WORLD);
  Allocator::SetAllocationPreflight(nullptr);
  require(astr_insitu_resource_finish()==0,"resource observer finish");
  std::printf("shared_probe rank=%d immediate_and_deferred_release=1\n",rank);
  MPI_Finalize();return 0;
} catch(const std::exception& error) {
  std::fprintf(stderr,"shared_probe: %s\n",error.what());
  int started=0;MPI_Initialized(&started);
  if(started) MPI_Abort(MPI_COMM_WORLD,1);
  return 1;
}
