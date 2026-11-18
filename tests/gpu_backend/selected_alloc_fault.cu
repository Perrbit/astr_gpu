#include <cuda_runtime.h>

namespace {
int remaining=0,hits=0;
}
extern "C" cudaError_t __real_cudaMalloc(void **pointer,size_t bytes);
extern "C" cudaError_t __wrap_cudaMalloc(void **pointer,size_t bytes)
{
    if(remaining>0 && --remaining==0) {++hits;return cudaErrorMemoryAllocation;}
    return __real_cudaMalloc(pointer,bytes);
}
extern "C" void astr_selected_test_fail_allocation(int after) {remaining=after;hits=0;}
extern "C" int astr_selected_test_allocation_failures() {return hits;}
