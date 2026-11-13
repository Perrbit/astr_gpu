add_library(astr_device_state STATIC EXCLUDE_FROM_ALL
  device_state.cu device_state.F90 device_halo.cu device_halo.F90
  halo_transport.cu halo_transport.F90
  fluid_explicit.cu fluid_explicit.F90 fluid_step.F90 fluid_diagnostics.cu)
set(ASTR_DEVICE_STATE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/state_modules")
file(MAKE_DIRECTORY "${ASTR_DEVICE_STATE_MODULES}")
set_target_properties(astr_device_state PROPERTIES CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED YES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE YES
  Fortran_MODULE_DIRECTORY "${ASTR_DEVICE_STATE_MODULES}")
target_include_directories(astr_device_state PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${ASTR_DEVICE_STATE_MODULES}")
target_compile_options(astr_device_state PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
target_link_libraries(astr_device_state PUBLIC MPI::MPI_C)
