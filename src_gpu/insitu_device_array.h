#ifndef ASTR_INSITU_DEVICE_ARRAY_H
#define ASTR_INSITU_DEVICE_ARRAY_H

#include <cuda_runtime.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/cuda/DeviceAdapterCuda.h>
#include <viskores/cont/internal/Buffer.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace astr_insitu {
inline void synchronize_device_stage(const char* stage) {
  const auto status=cudaDeviceSynchronize();
  if(status!=cudaSuccess) throw std::runtime_error(std::string(stage)+": "+cudaGetErrorString(status));
}
// Borrow only. The owner must retain an unchanged allocation until all consumers
// and their execution tokens have been destroyed and CUDA work has completed.
template <class T>
viskores::cont::ArrayHandle<T> borrow_cuda_array(T* pointer, viskores::Id count) {
  if (!pointer || count <= 0 ||
      static_cast<unsigned long long>(count) >
          static_cast<unsigned long long>(std::numeric_limits<viskores::BufferSizeType>::max())/sizeof(T))
    throw std::invalid_argument("Invalid borrowed CUDA array extent");
  cudaPointerAttributes attributes{};
  int device = -1;
  if (cudaPointerGetAttributes(&attributes,pointer) != cudaSuccess ||
      cudaGetDevice(&device) != cudaSuccess)
    throw std::runtime_error("Cannot inspect borrowed CUDA allocation");
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
    throw std::invalid_argument("Borrowed array must reside on the current CUDA device");
  return viskores::cont::ArrayHandle<T>(std::vector<viskores::cont::internal::Buffer>{
    viskores::cont::internal::MakeBuffer(viskores::cont::DeviceAdapterTagCuda{},pointer,pointer,
      static_cast<viskores::BufferSizeType>(count)*sizeof(T), [](void*) {},
      viskores::cont::internal::InvalidRealloc)});
}

template <class T>
void require_device_only(const viskores::cont::ArrayHandle<T>& array) {
  for (const auto& buffer : array.GetBuffers())
    if (buffer.IsAllocatedOnHost() ||
        !buffer.IsAllocatedOnDevice(viskores::cont::DeviceAdapterTagCuda{}))
      throw std::runtime_error("Device input acquired a host mirror or lost CUDA residency");
}
} // namespace astr_insitu
#endif
