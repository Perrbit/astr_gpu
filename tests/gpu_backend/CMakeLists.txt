enable_language(CUDA)
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
endif()
# This target tests the final CUDA device link, not CUDA Fortran compilation.
# Both compilers accept this flag; it keeps the reference arithmetic explicit.
target_compile_options(astr_backend_mb0 PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

add_executable(astr_backend_mb1 EXCLUDE_FROM_ALL state_probe.F90 state_consumer.cu)
set_target_properties(astr_backend_mb1 PROPERTIES
  CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES POSITION_INDEPENDENT_CODE YES
  CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON
  BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_mb1 PRIVATE astr_device_state MPI::MPI_Fortran)
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

add_executable(astr_backend_halo EXCLUDE_FROM_ALL halo_probe.F90)
set_target_properties(astr_backend_halo PROPERTIES POSITION_INDEPENDENT_CODE YES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON BUILD_RPATH "${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}" LINKER_LANGUAGE CUDA)
target_link_libraries(astr_backend_halo PRIVATE astr_device_state MPI::MPI_Fortran)
if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC|PGI")
  target_sources(astr_backend_halo PRIVATE ${ASTR_MB0_FORTRAN_MAIN})
endif()

add_executable(astr_backend_compare EXCLUDE_FROM_ALL field_compare.cpp)
target_compile_features(astr_backend_compare PRIVATE cxx_std_17)
target_include_directories(astr_backend_compare PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(astr_backend_compare PRIVATE MPI::MPI_C ${HDF5_Fortran_LIBRARIES})
