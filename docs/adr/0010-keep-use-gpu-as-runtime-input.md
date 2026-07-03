# Keep use_gpu as a runtime input

`use_gpu` is a runtime option read from ASTR input files, not a compile-time CMake option. A CUDA-capable binary may include both CPU and GPU paths; the input file decides which path is taken, while any compile-time switch must be named for CUDA code inclusion rather than execution selection.
