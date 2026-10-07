#include <viskores/cont/Algorithm.h>
#include <viskores/cont/ArrayHandleCounting.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/ErrorBadValue.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/cont/cuda/internal/CudaAllocator.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
int temporary_calls=0,rejected=0;
bool reject_temporary=false;
void admission(std::size_t bytes,const char* source) {
  const bool temporary=std::strstr(source,"Thrust")!=nullptr;
  if(temporary) ++temporary_calls;
  if(bytes>=64*1024*1024 || (temporary && reject_temporary)) {
    ++rejected;
    // Viskores suppresses generic backend exceptions. Budget rejection is a
    // device-independent fatal condition, matching production's MPI abort.
    throw viskores::cont::ErrorBadValue("allocation rejected by probe");
  }
}
void require(bool value,const char* message) {
  if(!value) throw std::runtime_error(message);
}
}

int main(int argc,char** argv) try {
#ifndef ASTR_VISKORES_ALLOCATION_PREFLIGHT
  throw std::runtime_error("allocation-preflight dependency patch missing");
#else
  viskores::cont::Initialize(argc,argv);
  auto& tracker=viskores::cont::GetRuntimeDeviceTracker();
  tracker.ForceDevice(viskores::cont::DeviceAdapterTagCuda{});
  tracker.SetThreadFriendlyMemAlloc(false);
  using Allocator=viskores::cont::cuda::internal::CudaAllocator;
  Allocator::SetAllocationPreflight(admission);
  std::size_t before=0,after=0,total=0;
  require(cudaMemGetInfo(&before,&total)==cudaSuccess,"memory query failed");
  bool caught=false;
  try { Allocator::Allocate(1024*1024*1024); }
  catch(const std::exception&) { caught=true; }
  require(cudaMemGetInfo(&after,&total)==cudaSuccess,"memory query failed");
  require(caught && rejected==1 && after>=before,"large allocation was not rejected before malloc");

  viskores::cont::ArrayHandle<viskores::Id> values;
  viskores::cont::Algorithm::Copy(viskores::cont::DeviceAdapterTagCuda{},
    viskores::cont::ArrayHandleCounting<viskores::Id>(16383,-1,16384),values);
  temporary_calls=0;
  viskores::cont::Algorithm::Sort(viskores::cont::DeviceAdapterTagCuda{},values);
  require(temporary_calls>0,"sort temporary did not enter admission hook");
  auto sorted=values.ReadPortal();
  for(viskores::Id i=0;i<16384;++i) require(sorted.Get(i)==i,"guard changed sort result");
  viskores::cont::ArrayHandle<viskores::Id> blocked;
  viskores::cont::Algorithm::Copy(viskores::cont::DeviceAdapterTagCuda{},
    viskores::cont::ArrayHandleCounting<viskores::Id>(16383,-1,16384),blocked);
  reject_temporary=true;caught=false;
  try { viskores::cont::Algorithm::Sort(viskores::cont::DeviceAdapterTagCuda{},blocked); }
  catch(const std::exception&) { caught=true; }
  require(caught && rejected>1,"sort scratch rejection did not stop dispatch");
  Allocator::SetAllocationPreflight(nullptr);
  std::printf("allocation_guard array_rejected_before_malloc=1 sort_temporary_calls=%d scratch_rejection=1\n",
    temporary_calls);
  return 0;
#endif
} catch(const std::exception& error) {
  std::fprintf(stderr,"allocation_guard: %s\n",error.what());return 1;
}
