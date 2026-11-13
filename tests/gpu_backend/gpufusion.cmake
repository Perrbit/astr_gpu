# Source the selected DTK and CUDA environments before configuring the root project.
set(CMAKE_CUDA_COMPILER "$ENV{ROCM_PATH}/cuda/cuda-12/bin/nvcc" CACHE FILEPATH "GPUfusion driver")
set(CMAKE_CUDA_ARCHITECTURES OFF CACHE STRING "Use the explicit DCU target, not NVIDIA CC")
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "GPUfusion runtime linkage")
# Device linking must be completed by GPUfusion, including the final host link.
set(CMAKE_CUDA_LINK_EXECUTABLE
  "<CMAKE_CUDA_COMPILER> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
