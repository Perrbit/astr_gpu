message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

set(ASTR_SOURCES
  astr.F90
  bc.F90
  CMakeLists.txt
  cmdefne.F90
  commarray.F90
  comsolver.F90
  commcal.F90
  commfunc.F90
  commtype.F90
  commvar.F90
  constdef.F90
  derivative.F90
  fdnn.F90
  filter.F90
  fludyna.F90
  flux.F90
  geom.F90
  gridgeneration.F90
  hdf5io.F90
  ibmethod.F90
  initialisation.F90
  interp.F90
  mainloop.F90
  models.F90
  mpiio.F90
  parallel.F90
  pp.F90
  readwrite.F90
  riemann.F90
  singleton.F90
  solver.F90
  perfect_gas_boundary.F90
  conservative_boundary_config.F90
  conservative_boundary_runtime.F90
  rectilinear_metric_halo.F90
  conservative_boundary_faces.F90
  perfect_gas_transport.F90
  sponge_layer.F90
  statistic.F90
  stlaio.F90
  strings.F90
  tecio.F90
  test.F90
  ../tests/gpu_validation/boundary_rhs_manufactured.F90
  thermchem.F90
  ../user_define_module/userdefine.F90
  utility.F90
  validation_io.F90
  vtkio.F90)

if(ASTR_WITH_CUDA)
  set(ASTR_GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_check.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/device_runtime_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/inflow_interpolation_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/inflow_timeseries_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/commarray_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/production_statistics_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/commvar_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/qswap_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/halo_transport_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/halo_exchange_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/nscbc_characteristic_policy_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/boundary_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/perfect_gas_boundary_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/conservative_boundary_faces_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/conservative_boundary_stage_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/sponge_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/case_capability_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/statistic_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gradcal_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/shock_sensor_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/solver_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/mainloop_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_runtime.cuf)
endif()

add_executable(astr ${ASTR_SOURCES} ${ASTR_GPU_SOURCES})

add_executable(boundary_contract_cpu_probe EXCLUDE_FROM_ALL
  perfect_gas_boundary.F90 ../tests/gpu_validation/boundary_contract_probe.F90)
set_target_properties(boundary_contract_cpu_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/boundary_contract_cpu_modules")
target_include_directories(boundary_contract_cpu_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/boundary_contract_cpu_modules")

add_executable(conservative_boundary_config_probe EXCLUDE_FROM_ALL
  perfect_gas_boundary.F90 conservative_boundary_config.F90
  ../tests/gpu_validation/conservative_boundary_config_probe.F90)
set_target_properties(conservative_boundary_config_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/boundary_config_modules")
target_include_directories(conservative_boundary_config_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/boundary_config_modules")
add_executable(boundary_faces_cpu_probe EXCLUDE_FROM_ALL
  perfect_gas_boundary.F90 conservative_boundary_faces.F90
  ../tests/gpu_validation/boundary_faces_probe.F90)
set_target_properties(boundary_faces_cpu_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/boundary_faces_cpu_modules")
target_include_directories(boundary_faces_cpu_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/boundary_faces_cpu_modules")

if(ASTR_WITH_CUDA)
  add_executable(boundary_faces_gpu_probe EXCLUDE_FROM_ALL
    ../src_gpu/perfect_gas_boundary_gpu.cuf ../src_gpu/conservative_boundary_faces_gpu.cuf
    ../tests/gpu_validation/boundary_faces_probe.F90)
  target_compile_definitions(boundary_faces_gpu_probe PRIVATE TEST_CUDA)
  set_target_properties(boundary_faces_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/boundary_faces_gpu_modules")
  target_include_directories(boundary_faces_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/boundary_faces_gpu_modules")
  add_executable(boundary_contract_gpu_probe EXCLUDE_FROM_ALL
    ../src_gpu/perfect_gas_boundary_gpu.cuf ../tests/gpu_validation/boundary_contract_probe.F90)
  target_compile_definitions(boundary_contract_gpu_probe PRIVATE TEST_CUDA)
  set_target_properties(boundary_contract_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/boundary_contract_gpu_modules")
  target_include_directories(boundary_contract_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/boundary_contract_gpu_modules")
  add_executable(halo_transport_test EXCLUDE_FROM_ALL
    ../src_gpu/halo_transport_gpu.cuf
    ../tests/gpu_validation/halo_transport_test.F90)
  set_target_properties(halo_transport_test PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/halo_test_modules")
  target_include_directories(halo_transport_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/halo_test_modules")
  target_link_libraries(halo_transport_test PRIVATE MPI::MPI_Fortran)
  add_executable(halo_pinned_probe EXCLUDE_FROM_ALL
    ../tests/gpu_validation/halo_pinned_probe.cuf)
  add_executable(dynamic_inflow_interpolation_probe EXCLUDE_FROM_ALL
    ../src_gpu/inflow_interpolation_gpu.cuf
    ../tests/gpu_validation/dynamic_inflow_interpolation_probe.cuf)
  add_executable(compact_statistics_probe EXCLUDE_FROM_ALL
    ../src_gpu/production_statistics_gpu.cuf
    ../tests/gpu_validation/compact_statistics_probe.cuf)
  target_compile_definitions(compact_statistics_probe PRIVATE TEST_COMPACT_STATS)
  set_target_properties(compact_statistics_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compact_statistics_probe_modules")
  target_include_directories(compact_statistics_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/compact_statistics_probe_modules")
  add_executable(nscbc_characteristic_policy_probe EXCLUDE_FROM_ALL
    ../src_gpu/nscbc_characteristic_policy_gpu.cuf
    ../tests/gpu_validation/nscbc_characteristic_policy_probe.cuf)
  set_target_properties(nscbc_characteristic_policy_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nscbc_policy_probe_modules")
  target_include_directories(nscbc_characteristic_policy_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/nscbc_policy_probe_modules")
  add_executable(halo_transport_setup_test EXCLUDE_FROM_ALL
    ../src_gpu/halo_transport_gpu.cuf
    ../tests/gpu_validation/halo_transport_setup_test.cuf)
  set_target_properties(halo_transport_setup_test PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/halo_setup_test_modules")
  target_include_directories(halo_transport_setup_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/halo_setup_test_modules")
  target_link_libraries(halo_transport_setup_test PRIVATE MPI::MPI_Fortran)
  add_executable(halo_overlap_probe EXCLUDE_FROM_ALL
    constdef.F90 ../tests/gpu_validation/halo_overlap_probe.cuf)
  set_target_properties(halo_overlap_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/overlap_probe_modules")
  target_include_directories(halo_overlap_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/overlap_probe_modules")
  target_link_libraries(halo_overlap_probe PRIVATE MPI::MPI_Fortran)
  include(CheckFortranSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_Fortran)
  check_fortran_source_compiles("
      program cuda_query_link_test
      use iso_c_binding, only: c_int
      interface
        function query() bind(C,name='MPIX_Query_cuda_support')
          import c_int
          integer(c_int) :: query
        end function query
      end interface
      print *, query()
      end program cuda_query_link_test
  " ASTR_HAS_MPI_CUDA_QUERY)
  cmake_pop_check_state()
  if(ASTR_HAS_MPI_CUDA_QUERY)
    add_executable(halo_cuda_aware_probe EXCLUDE_FROM_ALL
      ../tests/gpu_validation/halo_cuda_aware_probe.cuf)
    target_link_libraries(halo_cuda_aware_probe PRIVATE MPI::MPI_Fortran)
  else()
    message(STATUS "CUDA-aware query probe unavailable; host transport remains supported")
  endif()
endif()

if (ASTR_WITH_CUDA AND (Fortran_COMPILER_NAME MATCHES "PGI" OR Fortran_COMPILER_NAME MATCHES "NVHPC"))
  set(ASTR_S0A_MAXREGCOUNT "128" CACHE STRING
      "NVHPC register cap for the controlled S0-A characteristic kernels")
  # The selective Roe kernel must retain the prescribed 512-thread x block.
  # Cap its register allocation so the launch is legal on Ada and later GPUs.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/solver_gpu.cuf
    PROPERTIES COMPILE_OPTIONS "-gpu=maxregcount:${ASTR_S0A_MAXREGCOUNT}")
endif()
                    
target_link_libraries(astr)
if (MPI_FOUND)
  target_link_libraries(astr PRIVATE MPI::MPI_Fortran)
endif (MPI_FOUND)
if (CHEMISTRY)
  message(STATUS "CANTERA library: ${CTRDIR}/lib")
  target_link_libraries(astr PRIVATE -L${CTRDIR}/lib -lcantera_fortran -lcantera -lstdc++ -pthread)
endif (CHEMISTRY)

if(ASTR_WITH_CUDA)
  # Recompile the production modules in isolation, without the ASTR main program.
  set(ASTR_HALO_CONTRACT_SOURCES ${ASTR_SOURCES})
  list(REMOVE_ITEM ASTR_HALO_CONTRACT_SOURCES astr.F90 CMakeLists.txt)
  add_executable(halo_exchange_contract_test EXCLUDE_FROM_ALL
    ${ASTR_HALO_CONTRACT_SOURCES} ${ASTR_GPU_SOURCES}
    ../tests/gpu_validation/halo_exchange_contract_test.cuf)
  set_target_properties(halo_exchange_contract_test PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/halo_contract_modules")
  target_include_directories(halo_exchange_contract_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/halo_contract_modules")
  get_target_property(ASTR_HALO_CONTRACT_LIBRARIES astr LINK_LIBRARIES)
  target_link_libraries(halo_exchange_contract_test PRIVATE ${ASTR_HALO_CONTRACT_LIBRARIES})
endif()

option(ASTR_BUILD_MPI_COMPLETION_TRACE "Build optional MPI completion profiling interposer" OFF)
if(ASTR_WITH_CUDA AND ASTR_BUILD_MPI_COMPLETION_TRACE)
  enable_language(C)
  get_filename_component(ASTR_COMPILER_BIN "${CMAKE_Fortran_COMPILER}" DIRECTORY)
  find_program(ASTR_NVFORTRAN nvfortran)
  get_filename_component(ASTR_NVFORTRAN_BIN "${ASTR_NVFORTRAN}" DIRECTORY)
  find_path(ASTR_NVTX3_INCLUDE nvtx3/nvToolsExt.h
    HINTS "${ASTR_COMPILER_BIN}/../include-stdpar"
          "${ASTR_NVFORTRAN_BIN}/../include-stdpar")
  if(NOT ASTR_NVTX3_INCLUDE)
    message(FATAL_ERROR "MPI completion tracing requires NVTX3 headers")
  endif()
  add_library(mpi_completion_trace SHARED EXCLUDE_FROM_ALL
    ../tests/gpu_validation/mpi_completion_trace.F90
    ../tests/gpu_validation/mpi_completion_trace.c)
  target_include_directories(mpi_completion_trace PRIVATE "${ASTR_NVTX3_INCLUDE}")
  target_link_libraries(mpi_completion_trace PRIVATE MPI::MPI_Fortran ${CMAKE_DL_LIBS})
  add_executable(mpi_completion_probe EXCLUDE_FROM_ALL
    ../tests/gpu_validation/mpi_completion_probe.F90)
  target_link_libraries(mpi_completion_probe PRIVATE MPI::MPI_Fortran)
endif()

install(TARGETS astr
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

add_executable(rectilinear_metric_halo_probe EXCLUDE_FROM_ALL
  rectilinear_metric_halo.F90 ../tests/gpu_validation/rectilinear_metric_halo_probe.F90)
set_target_properties(rectilinear_metric_halo_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metric_halo_probe_modules")
target_include_directories(rectilinear_metric_halo_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/metric_halo_probe_modules")
