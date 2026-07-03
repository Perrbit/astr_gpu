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
  sponge_layer.F90
  statistic.F90
  stlaio.F90
  strings.F90
  tecio.F90
  test.F90
  thermchem.F90
  ../user_define_module/userdefine.F90
  utility.F90
  vtkio.F90)

if(ASTR_WITH_CUDA)
  set(ASTR_GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_check.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/device_runtime_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/commarray_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/commvar_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/qswap_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/halo_exchange_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/statistic_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gradcal_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/solver_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/mainloop_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_runtime.cuf)
endif()

add_executable(astr ${ASTR_SOURCES} ${ASTR_GPU_SOURCES})
                    
target_link_libraries(astr)
if (MPI_FOUND)
  target_link_libraries(astr PRIVATE MPI::MPI_Fortran)
endif (MPI_FOUND)
if (CHEMISTRY)
  message(STATUS "CANTERA library: ${CTRDIR}/lib")
  target_link_libraries(astr PRIVATE -L${CTRDIR}/lib -lcantera_fortran -lcantera -lstdc++ -pthread)
endif (CHEMISTRY)

install(TARGETS astr
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
