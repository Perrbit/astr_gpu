enable_language(CUDA)
add_executable(astr_backend_air5_primitives_regions EXCLUDE_FROM_ALL air5_primitives_regions_probe.cu)
set_target_properties(astr_backend_air5_primitives_regions PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_primitives_regions PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_primitives_regions PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
if(NOT ASTR_NUMERICAL_PRECISION STREQUAL "fp32")
  add_executable(astr_backend_air5_convection_budget EXCLUDE_FROM_ALL air5_convection_budget_probe.F90
    ${CMAKE_SOURCE_DIR}/src/chemistry_air5_data.F90)
  set_target_properties(astr_backend_air5_convection_budget PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_convection_budget PRIVATE astr_device_state MPI::MPI_Fortran)
  add_executable(astr_backend_air5_transport_rk3 EXCLUDE_FROM_ALL air5_transport_rk3_probe.F90)
  set_target_properties(astr_backend_air5_transport_rk3 PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_transport_rk3 PRIVATE astr_device_state MPI::MPI_Fortran)
  add_executable(astr_backend_air5_diffusion EXCLUDE_FROM_ALL air5_diffusion_probe.F90)
  set_target_properties(astr_backend_air5_diffusion PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_diffusion PRIVATE astr_device_state MPI::MPI_Fortran)
  add_executable(astr_backend_air5_monitor EXCLUDE_FROM_ALL air5_monitor_storage.F90
    ${CMAKE_SOURCE_DIR}/src/chemistry_air5_data.F90 ${CMAKE_SOURCE_DIR}/src/chemistry_core.F90
    ${CMAKE_SOURCE_DIR}/src/chemistry_properties.F90 ${CMAKE_SOURCE_DIR}/src/chemistry_monitor.F90 air5_monitor_probe.F90)
  set_target_properties(astr_backend_air5_monitor PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/air5_monitor_modules")
  target_include_directories(astr_backend_air5_monitor BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/air5_monitor_modules")
  target_compile_definitions(astr_backend_air5_monitor PRIVATE ASTR_GPU_CPP)
  target_link_libraries(astr_backend_air5_monitor PRIVATE astr_device_state MPI::MPI_Fortran)
  add_executable(astr_backend_air5_mean_runtime EXCLUDE_FROM_ALL air5_mean_runtime_storage.F90
    ${CMAKE_SOURCE_DIR}/src/checkpoint_state_io.F90 ${CMAKE_SOURCE_DIR}/src_gpu/chemistry_statistics_runtime.F90
    air5_mean_runtime_probe.F90)
  set_target_properties(astr_backend_air5_mean_runtime PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/air5_mean_runtime_modules")
  target_include_directories(astr_backend_air5_mean_runtime BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/air5_mean_runtime_modules" ${HDF5_INCLUDE_DIRS})
  target_link_libraries(astr_backend_air5_mean_runtime PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran ${HDF5_Fortran_LIBRARIES})
  add_executable(astr_backend_air5_mean EXCLUDE_FROM_ALL air5_mean_probe.F90)
  set_target_properties(astr_backend_air5_mean PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_mean PRIVATE astr_device_state MPI::MPI_Fortran)
  add_executable(astr_backend_air5_chemistry_phase EXCLUDE_FROM_ALL air5_chemistry_phase_probe.F90)
  set_target_properties(astr_backend_air5_chemistry_phase PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_chemistry_phase PRIVATE astr_device_state MPI::MPI_Fortran)
  add_executable(astr_backend_air5_compensation_exchange EXCLUDE_FROM_ALL air5_compensation_exchange_probe.cu
    air5_compensation_oracle.F90 ${CMAKE_SOURCE_DIR}/src/chemistry_compensation.F90
    ${CMAKE_SOURCE_DIR}/src/chemistry_air5_data.F90 ${CMAKE_SOURCE_DIR}/src/chemistry_core.F90 ${CMAKE_SOURCE_DIR}/src/chemistry_properties.F90)
  set_target_properties(astr_backend_air5_compensation_exchange PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA POSITION_INDEPENDENT_CODE YES
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}"
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/air5_compensation_oracle_modules")
  target_include_directories(astr_backend_air5_compensation_exchange BEFORE PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/air5_compensation_oracle_modules")
  target_link_libraries(astr_backend_air5_compensation_exchange PRIVATE astr_device_state MPI::MPI_Fortran)
  target_compile_options(astr_backend_air5_compensation_exchange PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
  add_executable(astr_backend_air5_boundary_stage EXCLUDE_FROM_ALL air5_boundary_stage_probe.F90)
  set_target_properties(astr_backend_air5_boundary_stage PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_boundary_stage PRIVATE astr_device_state MPI::MPI_Fortran)
endif()
add_executable(astr_backend_air5_characteristic_field EXCLUDE_FROM_ALL air5_characteristic_field_probe.cu)
set_target_properties(astr_backend_air5_characteristic_field PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_characteristic_field PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_characteristic_field PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_characteristic EXCLUDE_FROM_ALL air5_characteristic_probe.cu)
set_target_properties(astr_backend_air5_characteristic PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_characteristic PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_characteristic PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_boundary_targets EXCLUDE_FROM_ALL air5_boundary_targets_probe.cu)
set_target_properties(astr_backend_air5_boundary_targets PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_boundary_targets PRIVATE astr_device_state MPI::MPI_C)
target_compile_options(astr_backend_air5_boundary_targets PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_boundary EXCLUDE_FROM_ALL air5_boundary_probe.cu)
set_target_properties(astr_backend_air5_boundary PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_boundary PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_boundary PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_filter_field EXCLUDE_FROM_ALL air5_filter_field_probe.cu)
target_sources(astr_backend_air5_filter_field PRIVATE air5_rk_stage_probe.F90)
set_target_properties(astr_backend_air5_filter_field PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_filter_field PRIVATE astr_device_state)
set_target_properties(astr_backend_air5_filter_field PROPERTIES BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}")
add_executable(astr_backend_air5_legacy_filter_repair EXCLUDE_FROM_ALL air5_legacy_filter_repair_probe.cu)
set_target_properties(astr_backend_air5_legacy_filter_repair PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_legacy_filter_repair PRIVATE astr_device_state)
add_executable(astr_backend_air5_legacy_linear EXCLUDE_FROM_ALL air5_legacy_linear_probe.cu)
set_target_properties(astr_backend_air5_legacy_linear PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_legacy_linear PRIVATE astr_device_state)
add_executable(astr_backend_air5_filter_math EXCLUDE_FROM_ALL air5_filter_math_probe.cu)
set_target_properties(astr_backend_air5_filter_math PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_filter_math PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_filter_math PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_flux_exchange EXCLUDE_FROM_ALL air5_flux_exchange_probe.cu)
set_target_properties(astr_backend_air5_flux_exchange PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_flux_exchange PRIVATE astr_device_state)
add_executable(astr_backend_air5_face_exchange EXCLUDE_FROM_ALL air5_face_exchange_probe.cu)
set_target_properties(astr_backend_air5_face_exchange PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_face_exchange PRIVATE astr_device_state)
add_executable(astr_backend_air5_convection EXCLUDE_FROM_ALL air5_convection_probe.cu)
set_target_properties(astr_backend_air5_convection PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_convection PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_convection PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
if(NOT ASTR_NUMERICAL_PRECISION STREQUAL "fp32")
  add_executable(astr_backend_air5_convection_abi EXCLUDE_FROM_ALL air5_convection_abi.F90)
  set_target_properties(astr_backend_air5_convection_abi PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_convection_abi PRIVATE astr_device_state MPI::MPI_Fortran)
endif()
add_executable(astr_backend_air5_faces EXCLUDE_FROM_ALL air5_faces_probe.cu)
set_target_properties(astr_backend_air5_faces PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_faces PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_faces PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_flux_math EXCLUDE_FROM_ALL air5_flux_math_probe.cu)
set_target_properties(astr_backend_air5_flux_math PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_flux_math PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_flux_math PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_nr1_witness EXCLUDE_FROM_ALL air5_nr1_witness_probe.cu)
set_target_properties(astr_backend_air5_nr1_witness PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_nr1_witness PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_nr1_witness PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_spatial EXCLUDE_FROM_ALL air5_spatial_probe.cu)
set_target_properties(astr_backend_air5_spatial PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_spatial PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_spatial PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_transport EXCLUDE_FROM_ALL air5_transport_probe.cu)
set_target_properties(astr_backend_air5_transport PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_transport PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_transport PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_ros2 EXCLUDE_FROM_ALL air5_ros2_probe.cu)
set_target_properties(astr_backend_air5_ros2 PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_ros2 PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_ros2 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
if(NOT ASTR_NUMERICAL_PRECISION STREQUAL "fp32")
  add_executable(astr_backend_air5_coupling EXCLUDE_FROM_ALL air5_coupling_probe.F90)
  set_target_properties(astr_backend_air5_coupling PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_air5_coupling PRIVATE astr_device_state MPI::MPI_Fortran)
endif()
add_executable(astr_backend_air5_kinetics EXCLUDE_FROM_ALL air5_kinetics_probe.cu)
set_target_properties(astr_backend_air5_kinetics PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_kinetics PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_kinetics PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_properties EXCLUDE_FROM_ALL air5_properties_probe.cu)
set_target_properties(astr_backend_air5_properties PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_air5_properties PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_air5_properties PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_air5_state EXCLUDE_FROM_ALL air5_state_probe.cu)
set_target_properties(astr_backend_air5_state PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_state PRIVATE astr_device_state)
add_executable(astr_backend_air5_coupling_state EXCLUDE_FROM_ALL air5_coupling_state_probe.cu)
set_target_properties(astr_backend_air5_coupling_state PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_air5_coupling_state PRIVATE astr_device_state)
target_compile_options(astr_backend_air5_coupling_state PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--ftz=false>")
add_executable(astr_backend_mpi_cuda_context EXCLUDE_FROM_ALL mpi_cuda_context_probe.c)
target_link_libraries(astr_backend_mpi_cuda_context PRIVATE MPI::MPI_C)
add_executable(astr_backend_sparse_probe EXCLUDE_FROM_ALL sparse_probe.F90)
set_target_properties(astr_backend_sparse_probe PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_sparse_probe PRIVATE astr_device_state MPI::MPI_Fortran)
add_executable(astr_backend_profile_wall EXCLUDE_FROM_ALL profile_wall_probe.F90)
set_target_properties(astr_backend_profile_wall PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_profile_wall PRIVATE astr_device_state MPI::MPI_Fortran)
add_executable(astr_backend_adaptive_energy EXCLUDE_FROM_ALL adaptive_energy_probe.cu)
set_target_properties(astr_backend_adaptive_energy PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_adaptive_energy PRIVATE astr_device_state)
add_executable(astr_backend_mp_positivity_stage EXCLUDE_FROM_ALL mp_positivity_stage_probe.cu)
set_target_properties(astr_backend_mp_positivity_stage PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_mp_positivity_stage PRIVATE astr_device_state MPI::MPI_C)
target_compile_options(astr_backend_mp_positivity_stage PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
add_executable(astr_backend_mp_positivity_math EXCLUDE_FROM_ALL mp_positivity_math_probe.cu)
set_target_properties(astr_backend_mp_positivity_math PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_mp_positivity_math PRIVATE "${CMAKE_SOURCE_DIR}/src_gpu" "${CMAKE_SOURCE_DIR}/src")
target_compile_options(astr_backend_mp_positivity_math PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
add_executable(astr_backend_high_open_math EXCLUDE_FROM_ALL high_open_probe.cu)
set_target_properties(astr_backend_high_open_math PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_high_open_math PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_high_open_math PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
add_executable(astr_backend_incident_math EXCLUDE_FROM_ALL nscbc_incident_probe.cu)
set_target_properties(astr_backend_incident_math PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_incident_math PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_incident_math PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
add_library(astr_backend_precision STATIC EXCLUDE_FROM_ALL ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90)
set(ASTR_BACKEND_PRECISION_MODULES "${CMAKE_CURRENT_BINARY_DIR}/precision_modules")
file(MAKE_DIRECTORY "${ASTR_BACKEND_PRECISION_MODULES}")
set_target_properties(astr_backend_precision PROPERTIES Fortran_MODULE_DIRECTORY "${ASTR_BACKEND_PRECISION_MODULES}"
  POSITION_INDEPENDENT_CODE YES)
target_include_directories(astr_backend_precision PUBLIC "${ASTR_BACKEND_PRECISION_MODULES}")
target_link_libraries(astr_backend_precision PUBLIC MPI::MPI_Fortran)
add_executable(astr_backend_trace_minimum EXCLUDE_FROM_ALL trace_minimum_probe.cpp)
add_executable(astr_backend_critical_history EXCLUDE_FROM_ALL critical_history_probe.F90)
set_target_properties(astr_backend_critical_history PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_critical_history PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
add_executable(astr_backend_native_explicit32 EXCLUDE_FROM_ALL native_explicit_probe.F90)
set_target_properties(astr_backend_native_explicit32 PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_native_explicit32 PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
add_executable(astr_backend_native_diagnostics EXCLUDE_FROM_ALL native_diagnostics_probe.F90)
add_executable(astr_backend_native_boundary EXCLUDE_FROM_ALL native_boundary_probe.F90)
set_target_properties(astr_backend_native_boundary PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_native_boundary PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
add_executable(astr_backend_native_curve_gradient EXCLUDE_FROM_ALL native_curve_gradient_probe.F90)
set_target_properties(astr_backend_native_curve_gradient PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_native_curve_gradient PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
add_executable(astr_backend_boundary_filter EXCLUDE_FROM_ALL filter_boundary_probe.F90)
set_target_properties(astr_backend_boundary_filter PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_boundary_filter PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
set_target_properties(astr_backend_native_diagnostics PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_native_diagnostics PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
target_compile_features(astr_backend_trace_minimum PRIVATE cxx_std_17)
target_include_directories(astr_backend_trace_minimum PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src_gpu")
add_executable(astr_backend_rk45_gpu EXCLUDE_FROM_ALL ../gpu_validation/native_precision_rk45_probe.cu)
set_target_properties(astr_backend_rk45_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
target_include_directories(astr_backend_rk45_gpu PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/src_gpu")
target_compile_options(astr_backend_rk45_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
add_executable(astr_backend_mb0 EXCLUDE_FROM_ALL
  qualification.F90 qualification.cu qualification_device.cu)
set_target_properties(astr_backend_mb0 PROPERTIES
  CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE YES
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}"
  LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_mb0 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(astr_backend_mb0 PRIVATE MPI::MPI_Fortran
  ${HDF5_Fortran_LIBRARIES})
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  find_file(ASTR_MB0_FORTRAN_MAIN f90main.o
    PATHS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH REQUIRED)
  target_sources(astr_backend_mb0 PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  if(TARGET astr_backend_air5_coupling)
    target_sources(astr_backend_air5_coupling PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_convection_abi)
    target_sources(astr_backend_air5_convection_abi PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_boundary_stage)
    target_sources(astr_backend_air5_boundary_stage PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_diffusion)
    target_sources(astr_backend_air5_diffusion PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_convection_budget)
    target_sources(astr_backend_air5_convection_budget PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_transport_rk3)
    target_sources(astr_backend_air5_transport_rk3 PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_chemistry_phase)
    target_sources(astr_backend_air5_chemistry_phase PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_mean)
    target_sources(astr_backend_air5_mean PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_mean_runtime)
    target_sources(astr_backend_air5_mean_runtime PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  if(TARGET astr_backend_air5_monitor)
    target_sources(astr_backend_air5_monitor PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  target_sources(astr_backend_sparse_probe PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_profile_wall PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_native_explicit32 PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_native_diagnostics PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_native_boundary PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_native_curve_gradient PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_boundary_filter PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  target_sources(astr_backend_critical_history PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
# This target tests the final CUDA device link, not CUDA Fortran compilation.
add_executable(astr_backend_mp1_replay EXCLUDE_FROM_ALL mp1_replay_probe.cpp)
set_target_properties(astr_backend_mp1_replay PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES
  POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_mp1_replay PRIVATE astr_device_state)
add_executable(astr_backend_mixed_viscous EXCLUDE_FROM_ALL mixed_viscous_probe.F90)
set_target_properties(astr_backend_mixed_viscous PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_mixed_viscous PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_mixed_viscous PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_mixed_derivative EXCLUDE_FROM_ALL mixed_derivative_probe.F90)
set_target_properties(astr_backend_mixed_derivative PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_mixed_derivative PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_mixed_derivative PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_snapshot_bounds EXCLUDE_FROM_ALL snapshot_bounds_probe.cu)
set_target_properties(astr_backend_snapshot_bounds PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_snapshot_bounds PRIVATE astr_device_state)
target_compile_options(astr_backend_snapshot_bounds PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
add_executable(astr_backend_production_statistics EXCLUDE_FROM_ALL production_statistics_probe.F90)
set_target_properties(astr_backend_production_statistics PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_production_statistics PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_production_statistics PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_sponge EXCLUDE_FROM_ALL sponge_probe.F90)
set_target_properties(astr_backend_sponge PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_sponge PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_sponge PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_wall_forcing EXCLUDE_FROM_ALL wall_forcing_probe.F90)
set_target_properties(astr_backend_wall_forcing PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_wall_forcing PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_wall_forcing PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_dynamic_inflow EXCLUDE_FROM_ALL dynamic_inflow_probe.F90)
set_target_properties(astr_backend_dynamic_inflow PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_dynamic_inflow PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_dynamic_inflow PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_crashfix EXCLUDE_FROM_ALL crashfix_probe.F90 ../../src_gpu/crashfix_runtime.F90)
set(ASTR_CRASHFIX_PROBE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/crashfix_modules")
file(MAKE_DIRECTORY "${ASTR_CRASHFIX_PROBE_MODULES}")
target_include_directories(astr_backend_crashfix PRIVATE "${ASTR_CRASHFIX_PROBE_MODULES}")
set_target_properties(astr_backend_crashfix PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  Fortran_MODULE_DIRECTORY "${ASTR_CRASHFIX_PROBE_MODULES}"
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_crashfix PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_crashfix PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_critical_flux EXCLUDE_FROM_ALL critical_flux_probe.F90)
add_executable(astr_backend_critical_shared_flux EXCLUDE_FROM_ALL critical_shared_flux_probe.F90 ../../src_gpu/crashfix_runtime.F90)
set(ASTR_CRITICAL_SHARED_MODULES "${CMAKE_CURRENT_BINARY_DIR}/critical_shared_modules")
file(MAKE_DIRECTORY "${ASTR_CRITICAL_SHARED_MODULES}")
target_include_directories(astr_backend_critical_shared_flux PRIVATE "${ASTR_CRITICAL_SHARED_MODULES}")
set_target_properties(astr_backend_critical_shared_flux PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  Fortran_MODULE_DIRECTORY "${ASTR_CRITICAL_SHARED_MODULES}"
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_critical_shared_flux PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_critical_shared_flux PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
set_target_properties(astr_backend_critical_flux PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_critical_flux PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_critical_flux PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
# Both compilers accept this flag; it keeps the reference arithmetic explicit.
target_compile_options(astr_backend_mb0 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

add_executable(astr_backend_mb1 EXCLUDE_FROM_ALL state_probe.F90 state_consumer.cu)
set_target_properties(astr_backend_mb1 PROPERTIES
  CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE YES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_mb1 PRIVATE astr_device_state MPI::MPI_Fortran)
target_link_libraries(astr_backend_mb1 PRIVATE astr_backend_precision)
target_include_directories(astr_backend_mb1 PRIVATE "${CMAKE_SOURCE_DIR}/src")
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_mb1 PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_fluid EXCLUDE_FROM_ALL fluid_probe.F90
  ../../src/astr_precision.F90 ../../src/cfl_spectrum.F90)
set(ASTR_FLUID_PROBE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/fluid_modules")
file(MAKE_DIRECTORY "${ASTR_FLUID_PROBE_MODULES}")
set_target_properties(astr_backend_fluid PROPERTIES
  CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE YES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  Fortran_MODULE_DIRECTORY "${ASTR_FLUID_PROBE_MODULES}"
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_fluid PRIVATE "${ASTR_FLUID_PROBE_MODULES}")
target_link_libraries(astr_backend_fluid PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_fluid PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_rhs EXCLUDE_FROM_ALL rhs_probe.F90)
set_target_properties(astr_backend_rhs PROPERTIES
  POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_rhs PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(astr_backend_rhs PRIVATE astr_device_state MPI::MPI_Fortran ${HDF5_Fortran_LIBRARIES})
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_rhs PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_halo EXCLUDE_FROM_ALL halo_probe.F90 halo_wire_probe.c)
set_target_properties(astr_backend_halo PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_halo PRIVATE astr_device_state MPI::MPI_Fortran)
target_link_libraries(astr_backend_halo PRIVATE astr_backend_precision)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_halo PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_compare EXCLUDE_FROM_ALL field_compare.cpp)
target_compile_features(astr_backend_compare PRIVATE cxx_std_17)
target_include_directories(astr_backend_compare PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(astr_backend_compare PRIVATE MPI::MPI_C ${HDF5_Fortran_LIBRARIES})

add_executable(astr_backend_exterior_initialization EXCLUDE_FROM_ALL exterior_initialization_probe.F90)
set_target_properties(astr_backend_exterior_initialization PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_exterior_initialization PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_exterior_initialization PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_shock_sensor EXCLUDE_FROM_ALL shock_sensor_probe.F90)
set_target_properties(astr_backend_shock_sensor PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_shock_sensor PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_shock_sensor PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_shock_mask_halo EXCLUDE_FROM_ALL shock_mask_halo_probe.F90 shock_mask_interface_probe.cu)
set_target_properties(astr_backend_shock_mask_halo PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_shock_mask_halo PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_shock_mask_halo PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_upwind EXCLUDE_FROM_ALL upwind_probe.F90)
set_target_properties(astr_backend_upwind PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_upwind PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_upwind PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_flat_statistics EXCLUDE_FROM_ALL flat_statistics_probe.F90)
set_target_properties(astr_backend_flat_statistics PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_flat_statistics PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_flat_statistics PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_profile_boundary EXCLUDE_FROM_ALL profile_boundary_probe.F90)
set_target_properties(astr_backend_profile_boundary PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_profile_boundary PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_profile_boundary PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_high_wall_state EXCLUDE_FROM_ALL high_wall_state_probe.F90)
set_target_properties(astr_backend_high_wall_state PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_high_wall_state PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_high_wall_state PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_rti EXCLUDE_FROM_ALL rti_probe.F90)
set_target_properties(astr_backend_rti PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_rti PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_rti PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_slip_wall EXCLUDE_FROM_ALL slip_wall_probe.F90)
set_target_properties(astr_backend_slip_wall PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_slip_wall PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_slip_wall PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_nscbc_outlet EXCLUDE_FROM_ALL nscbc_outlet_probe.F90)
set_target_properties(astr_backend_nscbc_outlet PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_nscbc_outlet PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_nscbc_outlet PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_nscbc_inlet EXCLUDE_FROM_ALL nscbc_outlet_probe.F90)
target_compile_definitions(astr_backend_nscbc_inlet PRIVATE TEST_INLET)
set_target_properties(astr_backend_nscbc_inlet PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_nscbc_inlet PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_nscbc_inlet PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_nscbc_top EXCLUDE_FROM_ALL nscbc_top_probe.F90)
set_target_properties(astr_backend_nscbc_top PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_nscbc_top PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_nscbc_top PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_nscbc_top_viscous EXCLUDE_FROM_ALL nscbc_top_probe.F90)
target_compile_definitions(astr_backend_nscbc_top_viscous PRIVATE TEST_TOP_VISCOUS)
set_target_properties(astr_backend_nscbc_top_viscous PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_nscbc_top_viscous PRIVATE astr_device_state astr_backend_precision MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_nscbc_top_viscous PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_nscbc_filter EXCLUDE_FROM_ALL nscbc_filter_probe.F90)
set_target_properties(astr_backend_nscbc_filter PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_nscbc_filter PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_nscbc_filter PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_nscbc_top_filter EXCLUDE_FROM_ALL nscbc_filter_probe.F90)
target_compile_definitions(astr_backend_nscbc_top_filter PRIVATE TEST_TOP_FILTER)
set_target_properties(astr_backend_nscbc_top_filter PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_nscbc_top_filter PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_nscbc_top_filter PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_wall_header_probe EXCLUDE_FROM_ALL wall_header_probe.F90)
add_executable(astr_backend_sbli_wall EXCLUDE_FROM_ALL sbli_wall_probe.F90)
set_target_properties(astr_backend_sbli_wall PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_sbli_wall PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_sbli_wall PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_insitu_wall EXCLUDE_FROM_ALL insitu_wall_probe.F90)
set_target_properties(astr_backend_insitu_wall PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_insitu_wall PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_insitu_wall PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

if(ASTR_NUMERICAL_PRECISION STREQUAL "fp64")
  add_executable(astr_backend_insitu_selected EXCLUDE_FROM_ALL insitu_selected_probe.F90
    ../../src/astr_precision.F90 ../../src/insitu_time_integral.F90 ../../src/insitu_velocity_statistics.F90
    ../../src/insitu_sample_moments.F90)
  set(ASTR_SELECTED_PROBE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/selected_modules")
  file(MAKE_DIRECTORY "${ASTR_SELECTED_PROBE_MODULES}")
  set_target_properties(astr_backend_insitu_selected PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    Fortran_MODULE_DIRECTORY "${ASTR_SELECTED_PROBE_MODULES}"
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_include_directories(astr_backend_insitu_selected PRIVATE "${ASTR_SELECTED_PROBE_MODULES}")
  target_link_libraries(astr_backend_insitu_selected PRIVATE astr_device_state MPI::MPI_Fortran)
  if(ASTR_GPU_CPP_BACKEND STREQUAL "cuda")
    target_sources(astr_backend_insitu_selected PRIVATE selected_alloc_fault.cu)
    target_compile_definitions(astr_backend_insitu_selected PRIVATE ASTR_SELECTED_ALLOCATION_TEST)
    target_link_options(astr_backend_insitu_selected PRIVATE "LINKER:--wrap=cudaMalloc")
  endif()
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_insitu_selected PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_insitu_moments EXCLUDE_FROM_ALL insitu_moments_probe.F90
    ../../src/astr_precision.F90 ../../src/insitu_time_integral.F90 ../../src/insitu_velocity_statistics.F90)
  set(ASTR_MOMENTS_PROBE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/moments_modules")
  file(MAKE_DIRECTORY "${ASTR_MOMENTS_PROBE_MODULES}")
  set_target_properties(astr_backend_insitu_moments PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    Fortran_MODULE_DIRECTORY "${ASTR_MOMENTS_PROBE_MODULES}"
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_include_directories(astr_backend_insitu_moments PRIVATE "${ASTR_MOMENTS_PROBE_MODULES}")
  target_link_libraries(astr_backend_insitu_moments PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_insitu_moments PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_insitu_vectors EXCLUDE_FROM_ALL insitu_vectors_probe.F90)
  add_executable(astr_backend_owned_vector EXCLUDE_FROM_ALL owned_vector_probe.F90)
  set_target_properties(astr_backend_owned_vector PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_owned_vector PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_owned_vector PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  set_target_properties(astr_backend_insitu_vectors PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_insitu_vectors PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_insitu_vectors PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_insitu_index EXCLUDE_FROM_ALL insitu_index_probe.F90
    ../../src/astr_precision.F90 ../../src/index_slice_stencil.F90 ../../src/insitu_vortex.cpp)
  set(ASTR_INDEX_PROBE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/index_modules")
  file(MAKE_DIRECTORY "${ASTR_INDEX_PROBE_MODULES}")
  set_target_properties(astr_backend_insitu_index PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    Fortran_MODULE_DIRECTORY "${ASTR_INDEX_PROBE_MODULES}"
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_include_directories(astr_backend_insitu_index PRIVATE "${ASTR_INDEX_PROBE_MODULES}")
  target_link_libraries(astr_backend_insitu_index PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_insitu_index PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_insitu_sampling EXCLUDE_FROM_ALL insitu_sampling_probe.F90)
  set_target_properties(astr_backend_insitu_sampling PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_insitu_sampling PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_insitu_sampling PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_channel EXCLUDE_FROM_ALL channel_probe.F90)
  set_target_properties(astr_backend_channel PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_channel PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_channel PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_wall41_step EXCLUDE_FROM_ALL wall41_step_probe.F90
    ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90 ../../src/compact_line_solver.F90
    ../../src/compact_operator.F90 ../../src_gpu/compact_line_solver.F90 ../../src/keep_operator.F90
    ../../src/me6opti_operator.F90)
  set(ASTR_WALL_STEP_MODULES "${CMAKE_CURRENT_BINARY_DIR}/wall_step_modules")
  file(MAKE_DIRECTORY "${ASTR_WALL_STEP_MODULES}")
  set_target_properties(astr_backend_wall41_step PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    Fortran_MODULE_DIRECTORY "${ASTR_WALL_STEP_MODULES}"
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_include_directories(astr_backend_wall41_step PRIVATE "${ASTR_WALL_STEP_MODULES}")
  target_link_libraries(astr_backend_wall41_step PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_wall41_step PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_boundary_gradient EXCLUDE_FROM_ALL gradient_boundary_probe.F90
    ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90 ../../src/compact_line_solver.F90
    ../../src/compact_operator.F90 ../../src_gpu/compact_line_solver.F90)
  set(ASTR_BOUNDARY_GRADIENT_MODULES "${CMAKE_CURRENT_BINARY_DIR}/boundary_gradient_modules")
  file(MAKE_DIRECTORY "${ASTR_BOUNDARY_GRADIENT_MODULES}")
  set_target_properties(astr_backend_boundary_gradient PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    Fortran_MODULE_DIRECTORY "${ASTR_BOUNDARY_GRADIENT_MODULES}"
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_include_directories(astr_backend_boundary_gradient PRIVATE "${ASTR_BOUNDARY_GRADIENT_MODULES}")
  target_link_libraries(astr_backend_boundary_gradient PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_boundary_gradient PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
  add_executable(astr_backend_wall41 EXCLUDE_FROM_ALL wall41_probe.F90)
  set_target_properties(astr_backend_wall41 PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_link_libraries(astr_backend_wall41 PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_wall41 PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()
endif()

  add_executable(astr_backend_me6 EXCLUDE_FROM_ALL me6_probe.F90 halo_wire_probe.c
    ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90 ../../src/me6opti_operator.F90)
  set(ASTR_ME6_CPP_MODULES "${CMAKE_CURRENT_BINARY_DIR}/me6_modules")
  file(MAKE_DIRECTORY "${ASTR_ME6_CPP_MODULES}")
  set_target_properties(astr_backend_me6 PROPERTIES POSITION_INDEPENDENT_CODE YES
    Fortran_MODULE_DIRECTORY "${ASTR_ME6_CPP_MODULES}" CUDA_RESOLVE_DEVICE_SYMBOLS ON
    BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
  target_include_directories(astr_backend_me6 PRIVATE "${ASTR_ME6_CPP_MODULES}")
  target_link_libraries(astr_backend_me6 PRIVATE astr_device_state MPI::MPI_Fortran)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
    target_sources(astr_backend_me6 PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
  endif()

add_executable(astr_backend_compact_filter EXCLUDE_FROM_ALL compact_filter_probe.F90
  ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90 ../../src/compact_line_solver.F90
  ../../src/compact_operator.F90 ../../src_gpu/compact_line_solver.F90)
set(ASTR_COMPACT_FILTER_MODULES "${CMAKE_CURRENT_BINARY_DIR}/compact_filter_modules")
file(MAKE_DIRECTORY "${ASTR_COMPACT_FILTER_MODULES}")
set_target_properties(astr_backend_compact_filter PROPERTIES POSITION_INDEPENDENT_CODE YES
  Fortran_MODULE_DIRECTORY "${ASTR_COMPACT_FILTER_MODULES}" CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_compact_filter PRIVATE "${ASTR_COMPACT_FILTER_MODULES}")
target_link_libraries(astr_backend_compact_filter PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_compact_filter PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_keep_host_guard EXCLUDE_FROM_ALL keep_host_guard_probe.F90
  ../../src/astr_precision.F90 ../../src/keep_operator.F90)
set(ASTR_KEEP_HOST_MODULES "${CMAKE_CURRENT_BINARY_DIR}/keep_host_modules")
file(MAKE_DIRECTORY "${ASTR_KEEP_HOST_MODULES}")
set_target_properties(astr_backend_keep_host_guard PROPERTIES Fortran_MODULE_DIRECTORY "${ASTR_KEEP_HOST_MODULES}")
target_include_directories(astr_backend_keep_host_guard PRIVATE "${ASTR_KEEP_HOST_MODULES}")
add_executable(astr_backend_keep EXCLUDE_FROM_ALL keep_probe.F90
  ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90 ../../src/keep_operator.F90
  ../../src/compact_line_solver.F90 ../../src/compact_operator.F90 ../../src_gpu/compact_line_solver.F90)
set(ASTR_KEEP_CPP_MODULES "${CMAKE_CURRENT_BINARY_DIR}/keep_modules")
file(MAKE_DIRECTORY "${ASTR_KEEP_CPP_MODULES}")
set_target_properties(astr_backend_keep PROPERTIES POSITION_INDEPENDENT_CODE YES
  Fortran_MODULE_DIRECTORY "${ASTR_KEEP_CPP_MODULES}" CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_keep PRIVATE "${ASTR_KEEP_CPP_MODULES}")
target_link_libraries(astr_backend_keep PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_keep PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

# Compact components follow the build kind; ordinary FP32 main remains separate.
add_executable(astr_backend_boundary_flux EXCLUDE_FROM_ALL fluid_boundary_probe.F90
  ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90 ../../src/compact_line_solver.F90
  ../../src/compact_operator.F90 ../../src_gpu/compact_line_solver.F90)
set(ASTR_BOUNDARY_FLUX_MODULES "${CMAKE_CURRENT_BINARY_DIR}/boundary_flux_modules")
file(MAKE_DIRECTORY "${ASTR_BOUNDARY_FLUX_MODULES}")
set_target_properties(astr_backend_boundary_flux PROPERTIES POSITION_INDEPENDENT_CODE YES CUDA_RESOLVE_DEVICE_SYMBOLS ON
  Fortran_MODULE_DIRECTORY "${ASTR_BOUNDARY_FLUX_MODULES}"
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_boundary_flux PRIVATE "${ASTR_BOUNDARY_FLUX_MODULES}")
target_link_libraries(astr_backend_boundary_flux PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_boundary_flux PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_compact_line EXCLUDE_FROM_ALL compact_line_probe.F90
  ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90
  ../../src/compact_line_solver.F90 ../../src/compact_operator.F90
  ../../src_gpu/compact_line_solver.F90)
set(ASTR_COMPACT_PROBE_MODULES "${CMAKE_CURRENT_BINARY_DIR}/compact_modules")
file(MAKE_DIRECTORY "${ASTR_COMPACT_PROBE_MODULES}")
set_target_properties(astr_backend_compact_line PROPERTIES POSITION_INDEPENDENT_CODE YES
  Fortran_MODULE_DIRECTORY "${ASTR_COMPACT_PROBE_MODULES}" CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_compact_line PRIVATE "${ASTR_COMPACT_PROBE_MODULES}")
target_link_libraries(astr_backend_compact_line PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_compact_line PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
add_executable(astr_backend_compact_distributed EXCLUDE_FROM_ALL compact_distributed_probe.F90 halo_wire_probe.c
  ../../src/astr_precision.F90 ../../src/astr_precision_mpi.F90
  ../../src/compact_line_solver.F90 ../../src/compact_operator.F90
  ../../src_gpu/compact_line_solver.F90)
set(ASTR_COMPACT_DISTRIBUTED_MODULES "${CMAKE_CURRENT_BINARY_DIR}/compact_distributed_modules")
file(MAKE_DIRECTORY "${ASTR_COMPACT_DISTRIBUTED_MODULES}")
set_target_properties(astr_backend_compact_distributed PROPERTIES POSITION_INDEPENDENT_CODE YES
  Fortran_MODULE_DIRECTORY "${ASTR_COMPACT_DISTRIBUTED_MODULES}" CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_include_directories(astr_backend_compact_distributed PRIVATE "${ASTR_COMPACT_DISTRIBUTED_MODULES}")
target_link_libraries(astr_backend_compact_distributed PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_compact_distributed PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()
