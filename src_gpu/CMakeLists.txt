add_library(astr_device_state STATIC EXCLUDE_FROM_ALL
  device_state.cu device_state.F90 device_halo.cu device_halo.F90
  halo_transport.cu halo_transport.F90
  fluid_explicit.cu fluid_explicit.F90 fluid_step.F90 fluid_diagnostics.cu
  chemistry_properties.cu chemistry_properties.F90
  chemistry_coupling.cu chemistry_coupling.F90
  chemistry_transport.cu chemistry_transport.F90 chemistry_faces.cu chemistry_convection.cu chemistry_diffusion.cu chemistry_filter.cu
  chemistry_convection_budget.cu
  chemistry_boundary.cu chemistry_boundary.F90 chemistry_step.F90
  chemistry_statistics.cu chemistry_statistics.F90
  shock_sensor.cu shock_sensor.F90
  sponge_operator.cu sponge_operator.F90
  production_statistics.cu production_statistics.F90
  crashfix_operator.cu crashfix_operator.F90
  upwind_operator.cu upwind_operator.F90
  mp_positivity_operator.cu mp_positivity_operator.F90
  compact_line_solver.cu compact_operator.cu keep_operator.cu me6opti_operator.cu
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
set_source_files_properties(chemistry_properties.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_coupling.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_transport.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_faces.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_convection.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_diffusion.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_convection_budget.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_filter.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_boundary.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(chemistry_statistics.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
set_source_files_properties(halo_transport.cu PROPERTIES COMPILE_OPTIONS "--ftz=false")
target_link_libraries(astr_device_state PUBLIC MPI::MPI_C MPI::MPI_Fortran)
