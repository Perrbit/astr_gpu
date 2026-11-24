option(ASTR_MB8_TEST_NUMERICS "Use the MB8 NR1/NR2/NR4 candidates in local CUDA validation" OFF)
set(air5_candidate_sources chemistry_convection.cu chemistry_filter.cu chemistry_boundary.cu)
if(ASTR_MB8_TEST_NUMERICS)
  if(NOT BUILD_TESTING OR NOT ASTR_BUILD_GPU_BACKEND_PROBES OR NOT ASTR_WITH_AIR5_CHEMISTRY
      OR NOT ASTR_GPU_CPP_BACKEND STREQUAL "cuda" OR NOT ASTR_NUMERICAL_PRECISION STREQUAL "fp64")
    message(FATAL_ERROR "ASTR_MB8_TEST_NUMERICS requires CUDA C++ FP64 AIR5 and the backend tests")
  endif()
  include("${CMAKE_SOURCE_DIR}/tests/gpu_backend/air5_mb8_test_kernels.cmake")
endif()

add_library(astr_device_state STATIC EXCLUDE_FROM_ALL
  device_state.cu device_state.F90 device_halo.cu device_halo.F90
  halo_transport.cu halo_transport.F90
  fluid_explicit.cu fluid_explicit.F90 fluid_step.F90 fluid_diagnostics.cu
  chemistry_properties.cu chemistry_properties.F90
  chemistry_coupling.cu chemistry_coupling.F90
  chemistry_transport.cu chemistry_transport.F90 chemistry_faces.cu chemistry_diffusion.cu ${air5_candidate_sources}
  chemistry_convection_budget.cu
  chemistry_boundary.F90 chemistry_step.F90
  chemistry_statistics.cu chemistry_statistics.F90
  shock_sensor.cu shock_sensor.F90
  sponge_operator.cu sponge_operator.F90
  production_statistics.cu production_statistics.F90
  crashfix_operator.cu crashfix_operator.F90
  upwind_operator.cu upwind_operator.F90
  mp_positivity_operator.cu mp_positivity_operator.F90
  compact_line_solver.cu compact_operator.cu keep_operator.cu me6opti_operator.cu keep_w1_operator.cu
  boundary_state.cu boundary_state.F90 insitu_sampling.cu insitu_sampling.F90 insitu_index.cu insitu_index.F90
  insitu_moments.cu insitu_moments.F90 insitu_wall.cu insitu_wall.F90)
set(ASTR_DEVICE_STATE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/state_modules")
file(MAKE_DIRECTORY "${ASTR_DEVICE_STATE_MODULES}")
set_target_properties(astr_device_state PROPERTIES CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED YES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE YES
  Fortran_MODULE_DIRECTORY "${ASTR_DEVICE_STATE_MODULES}")
target_include_directories(astr_device_state PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/src" "${ASTR_DEVICE_STATE_MODULES}")
target_compile_options(astr_device_state PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
if(ASTR_GPU_CPP_BACKEND STREQUAL "gpufusion" AND ASTR_NUMERICAL_PRECISION STREQUAL "fp64")
  set_property(SOURCE fluid_explicit.cu APPEND PROPERTY COMPILE_DEFINITIONS ASTR_GPUFUSION_PERIODIC_CONVECTION)
endif()
set_source_files_properties(
  chemistry_properties.cu chemistry_coupling.cu chemistry_transport.cu chemistry_faces.cu
  ${air5_candidate_sources} chemistry_diffusion.cu chemistry_convection_budget.cu
  chemistry_statistics.cu halo_transport.cu
  PROPERTIES COMPILE_OPTIONS "${ASTR_AIR5_FP_OPTIONS}")
if(BUILD_TESTING)
  set_property(SOURCE chemistry_step.F90 APPEND PROPERTY COMPILE_DEFINITIONS ASTR_BUILD_TESTING)
endif()
target_link_libraries(astr_device_state PUBLIC MPI::MPI_C MPI::MPI_Fortran)
