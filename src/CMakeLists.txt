message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

enable_language(C)
# Parallel HDF5 headers include mpi.h; use the selected Fortran MPI installation.
if(NOT MPI_C_COMPILER)
  get_filename_component(ASTR_MPI_BINDIR "${MPI_Fortran_COMPILER}" DIRECTORY)
  if(EXISTS "${ASTR_MPI_BINDIR}/mpicc")
    set(MPI_C_COMPILER "${ASTR_MPI_BINDIR}/mpicc" CACHE FILEPATH "Matching MPI C wrapper")
  endif()
endif()
find_package(MPI REQUIRED COMPONENTS C)
add_library(astr_checkpoint_filesystem STATIC checkpoint_filesystem.c checkpoint_hdf5_resources.c)
target_include_directories(astr_checkpoint_filesystem PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(astr_checkpoint_filesystem PRIVATE ${HDF5_LIBRARIES} MPI::MPI_C)
set_target_properties(astr_checkpoint_filesystem PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)

# New checkpoint publication must not emulate no-replace with a racy existence check.
include(CheckFortranSourceCompiles)
check_fortran_source_compiles("
      program check_renameat2
      use iso_c_binding
      interface
        function ren(a,b,c,d,e) bind(C,name='renameat2') result(status)
          import c_int,c_char
          integer(c_int),value :: a,c,e
          character(c_char) :: b(*),d(*)
          integer(c_int) :: status
        end function
      end interface
      print *,ren(-100_c_int,'a'//c_null_char,-100_c_int,'b'//c_null_char,1_c_int)
      end program
" ASTR_HAS_RENAMEAT2)
if(ASTR_HAS_RENAMEAT2 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_source_files_properties(checkpoint_bundle.F90 PROPERTIES COMPILE_DEFINITIONS ASTR_HAS_RENAMEAT2)
else()
  message(STATUS "New checkpoint publication unavailable: no exclusive directory rename; legacy I/O unchanged")
endif()

set(ASTR_SOURCES
  astr.F90
  bc.F90
  CMakeLists.txt
  cmdefne.F90
  commarray.F90
  comsolver.F90
  commcal.F90
  cfl_spectrum.F90
  commfunc.F90
  commtype.F90
  commvar.F90
  benchmark_runtime.F90
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
  insitu_runtime.F90
  insitu_resource_budget.F90
  insitu_run_config.F90
  insitu_config_collective.F90
  insitu_fields.F90
  insitu_geometry.F90
  insitu_session.F90
  insitu_schedule.F90
  adaptive_output.F90
  insitu_product_schedule.F90
  insitu_checkpoint_batch.F90
  insitu_time_integral.F90
  insitu_velocity_statistics.F90
  insitu_wall_statistics.F90
  insitu_wall_separation.F90
  insitu_air5_statistics.F90
  insitu_spatial_statistics.F90
  interp.F90
  mainloop.F90
  output_config.F90
  output_input_resources.F90
  output_fields.F90
  output_lineage.F90
  output_archive.F90
  output_runtime.F90
  checkpoint_state_io.F90
  checkpoint_bundle.F90
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
  thermchem.F90
  ../user_define_module/userdefine.F90
  utility.F90
  validation_io.F90
  vtkio.F90)

if(BUILD_TESTING)
  list(APPEND ASTR_SOURCES ../tests/gpu_validation/boundary_rhs_manufactured.F90)
endif()

if(ASTR_WITH_CUDA)
  set(ASTR_GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_check.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_phase_timing.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/device_runtime_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/inflow_interpolation_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/inflow_timeseries_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/precision_mode_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/mixed_candidate_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/upwind_flux_mode_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/commarray_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/checkpoint_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/checkpoint_state_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/output_fields_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/production_statistics_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/commvar_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/qswap_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/halo_transport_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/device_mpi_transport_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/halo_exchange_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/nscbc_characteristic_policy_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/boundary_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/perfect_gas_boundary_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/conservative_boundary_faces_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/conservative_boundary_stage_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/sponge_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/case_capability_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/statistic_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/cfl_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gradcal_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/shock_sensor_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/solver_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/mainloop_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/gpu_runtime.cuf)
  list(APPEND ASTR_GPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_sample_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_products_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/output_derivatives_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_device_velocity.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_wall_fields_gpu.cuf)
endif()

if(ASTR_WITH_CUDA)
  list(APPEND ASTR_GPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_statistics_gpu.cuf)
  list(APPEND ASTR_GPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_wall_statistics_gpu.cuf)
  list(APPEND ASTR_GPU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/insitu_air5_statistics_gpu.cuf)
endif()

if(ASTR_WITH_AIR5_CHEMISTRY)
  list(APPEND ASTR_SOURCES
    chemistry_air5_data.F90
    chemistry_core.F90
    chemistry_properties.F90
    chemistry_kinetics.F90
    chemistry_characteristic.F90
    chemistry_boundary_state.F90
    chemistry_boundary.F90
    chemistry_runtime.F90
    chemistry_compensation.F90
    chemistry_monitor.F90
    chemistry_solver.F90)
endif()

if(ASTR_WITH_CUDA AND ASTR_WITH_AIR5_CHEMISTRY)
  list(APPEND ASTR_GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_compensation_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_monitor_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_core_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_flow_state_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_transport_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_relaxation_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_kinetics_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_boundary_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_solver_gpu.cuf
    ${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_coupling_gpu.cuf)
endif()

add_executable(astr ${ASTR_SOURCES} ${ASTR_GPU_SOURCES})
target_link_libraries(astr PRIVATE astr_checkpoint_filesystem)
set_target_properties(astr PROPERTIES LINKER_LANGUAGE Fortran)

if(ASTR_WITH_CATALYST)
  find_package(MPI REQUIRED COMPONENTS C)
  set_target_properties(astr PROPERTIES ENABLE_EXPORTS YES)
  if(ASTR_WITH_INSITU_DEVICE OR (BUILD_TESTING AND ASTR_BUILD_INSITU_DEVICE_PROBES))
    if(CMAKE_VERSION VERSION_LESS 3.24)
      message(FATAL_ERROR "Device in-situ products/probes require CMake 3.24 or later")
    endif()
    enable_language(CUDA)
    if(ASTR_WITH_INSITU_DEVICE OR Viskores_DIR)
      find_package(Viskores CONFIG REQUIRED)
      if(ASTR_INSITU_FLYING_EDGES_SOURCE)
        if(NOT EXISTS "${ASTR_INSITU_FLYING_EDGES_SOURCE}")
          message(FATAL_ERROR "ASTR_INSITU_FLYING_EDGES_SOURCE does not exist")
        endif()
        add_library(astr_insitu_flying_edges STATIC "${ASTR_INSITU_FLYING_EDGES_SOURCE}")
        set_source_files_properties("${ASTR_INSITU_FLYING_EDGES_SOURCE}" PROPERTIES LANGUAGE CUDA)
        set_target_properties(astr_insitu_flying_edges PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_compile_definitions(astr_insitu_flying_edges PRIVATE viskores_filter_contour_EXPORTS)
        target_compile_options(astr_insitu_flying_edges PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        target_link_libraries(astr_insitu_flying_edges PRIVATE viskores::cont viskores::worklet
          viskores::cuda viskores::filter_core viskores::filter_vector_analysis)
      endif()
    endif()
  endif()
  add_library(astr_catalyst_adapter STATIC catalyst_adapter.cpp insitu_mesh_adapter.cpp)
  if(ASTR_WITH_CUDA)
    find_path(ASTR_EGL_INCLUDE_DIR EGL/eglext.h REQUIRED)
    get_filename_component(ASTR_COMPILER_BIN "${CMAKE_Fortran_COMPILER}" DIRECTORY)
    find_path(ASTR_NVML_INCLUDE_DIR nvml.h HINTS "${ASTR_COMPILER_BIN}/../../cuda/include" REQUIRED)
    target_sources(astr_catalyst_adapter PRIVATE insitu_device_map.cpp insitu_resource_observer.cpp)
    target_compile_definitions(astr_catalyst_adapter PRIVATE ASTR_CUDA_RENDERER)
    target_include_directories(astr_catalyst_adapter PRIVATE ${ASTR_EGL_INCLUDE_DIR})
    target_include_directories(astr_catalyst_adapter PRIVATE ${ASTR_NVML_INCLUDE_DIR})
  endif()
  set_target_properties(astr_catalyst_adapter PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES)
  target_compile_definitions(astr_catalyst_adapter INTERFACE ASTR_WITH_CATALYST)
  target_link_libraries(astr_catalyst_adapter PRIVATE catalyst::catalyst MPI::MPI_C ${CMAKE_DL_LIBS})
  target_link_libraries(astr PRIVATE astr_catalyst_adapter)
  if(ASTR_WITH_INSITU_DEVICE_RENDERING)
    find_package(CUDAToolkit REQUIRED)
    find_package(ParaView 6.1 REQUIRED COMPONENTS RemotingViews)
    find_package(VTK REQUIRED COMPONENTS RenderingOpenGL2 AcceleratorsVTKmCore IOCatalystConduit)
    set_target_properties(astr_catalyst_adapter PROPERTIES CXX_STANDARD 17)
    target_compile_definitions(astr_catalyst_adapter PRIVATE ASTR_INSITU_DEVICE_RENDERING)
    target_include_directories(astr_catalyst_adapter PRIVATE ../src_gpu ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(astr_catalyst_adapter PRIVATE VTK::RenderingOpenGL2
      VTK::AcceleratorsVTKmCore VTK::IOCatalystConduit ParaView::RemotingViews CUDA::cudart)
    vtk_module_autoinit(TARGETS astr_catalyst_adapter MODULES VTK::RenderingOpenGL2)
    # Python's narrow control bridge resolves only these exported local entry points.
    set_target_properties(astr PROPERTIES ENABLE_EXPORTS YES)
  endif()
  if(ASTR_WITH_INSITU_DEVICE)
    target_compile_definitions(astr_catalyst_adapter PRIVATE ASTR_INSITU_DEVICE_PRODUCTS)
    target_include_directories(astr_catalyst_adapter PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
    add_library(astr_insitu_device_products STATIC ../src_gpu/insitu_device_products.cu
      ../src_gpu/insitu_plane_products.cu)
    if(BUILD_TESTING)
      set_source_files_properties(../src_gpu/insitu_plane_products.cu ../src_gpu/insitu_device_products.cu PROPERTIES
        COMPILE_DEFINITIONS ASTR_BUILD_TESTING)
    endif()
    set_target_properties(astr_insitu_device_products PROPERTIES CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
    target_include_directories(astr_insitu_device_products PRIVATE . ../src_gpu)
    target_compile_options(astr_insitu_device_products PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
    target_link_libraries(astr_insitu_device_products PRIVATE astr_catalyst_adapter
      viskores::cont viskores::worklet viskores::cuda viskores::filter_core MPI::MPI_C)
    if(TARGET astr_insitu_flying_edges)
      target_link_libraries(astr_insitu_device_products PRIVATE astr_insitu_flying_edges)
    else()
      target_link_libraries(astr_insitu_device_products PRIVATE viskores::filter_contour)
    endif()
    target_compile_definitions(astr PRIVATE ASTR_INSITU_DEVICE_PRODUCTS)
    if(ASTR_WITH_INSITU_DEVICE_RENDERING)
      target_compile_definitions(astr_insitu_device_products PRIVATE ASTR_INSITU_DEVICE_RENDERING)
    endif()
    target_link_libraries(astr PRIVATE astr_insitu_device_products)
  endif()
  if(BUILD_TESTING)
    if(ASTR_BUILD_INSITU_DEVICE_PROBES)
      find_package(VTK REQUIRED COMPONENTS CommonCore CommonMath)
      find_package(VTK REQUIRED COMPONENTS RenderingOpenGL2)
      add_executable(insitu_graphics_interop_probe EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_graphics_interop_probe.cu insitu_device_map.cpp)
      set_target_properties(insitu_graphics_interop_probe PROPERTIES CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
      target_include_directories(insitu_graphics_interop_probe PRIVATE ../src_gpu ${ASTR_EGL_INCLUDE_DIR})
      target_link_libraries(insitu_graphics_interop_probe PRIVATE VTK::RenderingOpenGL2 ${CMAKE_DL_LIBS})
      if(ASTR_BUILD_INSITU_RENDER_PROBES)
        find_package(VTK REQUIRED COMPONENTS AcceleratorsVTKmCore IOCatalystConduit)
        add_executable(insitu_standard_device_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_standard_device_probe.cu insitu_device_map.cpp)
        set_target_properties(insitu_standard_device_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_standard_device_probe PRIVATE ../src_gpu ${ASTR_EGL_INCLUDE_DIR})
        target_compile_options(insitu_standard_device_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        target_link_libraries(insitu_standard_device_probe PRIVATE VTK::RenderingOpenGL2
          VTK::AcceleratorsVTKmCore VTK::IOCatalystConduit catalyst::catalyst
          viskores::cont viskores::cuda ${CMAKE_DL_LIBS})
        vtk_module_autoinit(TARGETS insitu_standard_device_probe MODULES VTK::RenderingOpenGL2)
      endif()
      add_executable(insitu_device_compiler_probe EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_device_compiler_probe.cu)
      add_executable(insitu_device_rk45_probe EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_device_rk45_probe.cu)
      add_executable(insitu_device_trace_probe EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_device_trace_probe.cu)
      add_executable(insitu_device_peer_probe EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_device_peer_probe.cu)
      add_executable(insitu_device_plane_probe EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_device_plane_probe.cu)
      target_include_directories(insitu_device_plane_probe PRIVATE ../src_gpu)
      target_link_libraries(insitu_device_plane_probe PRIVATE MPI::MPI_C)
      target_include_directories(insitu_device_trace_probe PRIVATE ../src_gpu)
      foreach(probe insitu_device_compiler_probe insitu_device_rk45_probe
          insitu_device_trace_probe insitu_device_peer_probe insitu_device_plane_probe)
        set_target_properties(${probe} PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_compile_options(${probe} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
      endforeach()
      target_include_directories(insitu_device_rk45_probe PRIVATE ../src_gpu)
      target_link_libraries(insitu_device_rk45_probe PRIVATE VTK::CommonMath VTK::CommonCore VTK::vtksys)
      add_executable(insitu_vtk_length_audit EXCLUDE_FROM_ALL
        ../tests/gpu_validation/insitu_vtk_length_audit.cpp)
      target_link_libraries(insitu_vtk_length_audit PRIVATE VTK::CommonMath VTK::CommonCore VTK::vtksys)
      if(ASTR_INSITU_PATCHED_STREAMTRACER_SOURCE)
        find_package(VTK REQUIRED COMPONENTS FiltersFlowPaths)
        add_executable(insitu_streamtracer_length_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_streamtracer_length_probe.cpp)
        add_executable(insitu_streamtracer_length_fixed_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_streamtracer_length_probe.cpp
          "${ASTR_INSITU_PATCHED_STREAMTRACER_SOURCE}")
        foreach(probe insitu_streamtracer_length_probe insitu_streamtracer_length_fixed_probe)
          target_link_libraries(${probe} PRIVATE VTK::FiltersFlowPaths)
        endforeach()
        # Link the patched translation unit only into the private reference probe.
        # It does not overwrite or install any validated VTK library.
        target_compile_definitions(insitu_streamtracer_length_fixed_probe PRIVATE vtkFiltersFlowPaths_EXPORTS)
      endif()
      if(Viskores_DIR)
        add_executable(insitu_allocation_guard_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_allocation_guard_probe.cu)
        set_target_properties(insitu_allocation_guard_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_link_libraries(insitu_allocation_guard_probe PRIVATE viskores::cont viskores::cuda)
        add_executable(insitu_viskores_rk45_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_viskores_rk45_probe.cu)
        set_target_properties(insitu_viskores_rk45_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_viskores_rk45_probe PRIVATE ../src_gpu)
        target_compile_options(insitu_viskores_rk45_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        target_link_libraries(insitu_viskores_rk45_probe PRIVATE viskores::cont viskores::worklet viskores::cuda)
        add_executable(insitu_viskores_trace_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_viskores_trace_probe.cu)
        set_target_properties(insitu_viskores_trace_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_viskores_trace_probe PRIVATE ../src_gpu)
        target_compile_options(insitu_viskores_trace_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        target_link_libraries(insitu_viskores_trace_probe PRIVATE
          viskores::cont viskores::worklet viskores::cuda MPI::MPI_C)
        add_executable(insitu_device_streamlines_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_device_streamlines_probe.cu)
        set_target_properties(insitu_device_streamlines_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_device_streamlines_probe PRIVATE ../src_gpu)
        # Numerical component probe only; production device products keep their normal optimization.
        target_compile_options(insitu_device_streamlines_probe PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O1>")
        target_link_libraries(insitu_device_streamlines_probe PRIVATE
          viskores::cont viskores::worklet viskores::cuda MPI::MPI_C)
        add_executable(insitu_curve_inverse_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_curve_inverse_probe.cu)
        set_target_properties(insitu_curve_inverse_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_curve_inverse_probe PRIVATE ../src_gpu)
        target_compile_options(insitu_curve_inverse_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        target_link_libraries(insitu_curve_inverse_probe PRIVATE viskores::cont viskores::worklet viskores::cuda)
        add_executable(insitu_device_geometry_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_device_geometry_probe.cu)
        set_target_properties(insitu_device_geometry_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_device_geometry_probe PRIVATE ../src_gpu)
        target_compile_options(insitu_device_geometry_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        if(ASTR_INSITU_FLYING_EDGES_SOURCE)
          target_link_libraries(insitu_device_geometry_probe PRIVATE astr_insitu_flying_edges)
        else()
          target_link_libraries(insitu_device_geometry_probe PRIVATE viskores::filter_contour)
        endif()
        target_link_libraries(insitu_device_geometry_probe PRIVATE
          viskores::cont viskores::worklet viskores::cuda viskores::filter_core MPI::MPI_C)
        find_package(VTK REQUIRED COMPONENTS CommonDataModel IOXML)
        target_link_libraries(insitu_device_geometry_probe PRIVATE VTK::CommonDataModel VTK::IOXML)
        add_executable(insitu_device_structured_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_device_structured_probe.cu)
        set_target_properties(insitu_device_structured_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_device_structured_probe PRIVATE ../src_gpu)
        target_compile_options(insitu_device_structured_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        target_link_libraries(insitu_device_structured_probe PRIVATE
          viskores::cont viskores::worklet viskores::cuda MPI::MPI_C)
        if(ASTR_INSITU_FLYING_EDGES_SOURCE)
          target_link_libraries(insitu_device_structured_probe PRIVATE astr_insitu_flying_edges)
        else()
          target_link_libraries(insitu_device_structured_probe PRIVATE viskores::filter_contour)
        endif()
        add_executable(insitu_device_wall_probe EXCLUDE_FROM_ALL
          ../tests/gpu_validation/insitu_device_wall_probe.cu)
        set_target_properties(insitu_device_wall_probe PROPERTIES CUDA_STANDARD 17
          CUDA_STANDARD_REQUIRED YES CUDA_ARCHITECTURES native)
        target_include_directories(insitu_device_wall_probe PRIVATE ../src_gpu)
        target_compile_options(insitu_device_wall_probe PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
        find_package(Python3 REQUIRED COMPONENTS Development.Embed)
        target_link_libraries(insitu_device_wall_probe PRIVATE
          viskores::cont viskores::worklet viskores::cuda MPI::MPI_C astr_catalyst_adapter Python3::Python)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
          # ParaView's Python module DSOs resolve through the embedding process.
          target_link_options(insitu_device_wall_probe PRIVATE "LINKER:--no-as-needed")
        endif()
      endif()
    endif()
    add_library(insitu_allocation_fault SHARED EXCLUDE_FROM_ALL ../tests/gpu_validation/insitu_allocation_fault.cpp)
    target_link_libraries(insitu_allocation_fault PRIVATE ${CMAKE_DL_LIBS})
    if(ASTR_WITH_CUDA)
      add_executable(insitu_device_map_probe EXCLUDE_FROM_ALL ../tests/gpu_validation/insitu_device_map_probe.cuf)
      set_target_properties(insitu_device_map_probe PROPERTIES LINKER_LANGUAGE Fortran)
      target_link_libraries(insitu_device_map_probe PRIVATE astr_catalyst_adapter MPI::MPI_Fortran)
    endif()
  endif()
  # Keep NVHPC's CUDA Fortran device-link path authoritative.
  set_target_properties(astr PROPERTIES LINKER_LANGUAGE Fortran)
endif()

if(ASTR_WITH_AIR5_CHEMISTRY)
  target_compile_definitions(astr PRIVATE ASTR_AIR5_CHEMISTRY)
endif()

# This capability is needed by the production transport, not only its probes.
if(ASTR_WITH_CUDA)
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
    target_compile_definitions(astr PRIVATE ASTR_MPI_CUDA_AWARE_QUERY)
  else()
    message(STATUS "CUDA-aware query unavailable; host transport remains supported")
  endif()
  if(BUILD_TESTING AND ASTR_WITH_CATALYST)
    add_executable(insitu_device_velocity_probe EXCLUDE_FROM_ALL
      constdef.F90 ../src_gpu/output_derivatives_gpu.cuf
      ../src_gpu/insitu_device_velocity.cuf ../tests/gpu_validation/insitu_device_velocity_probe.cuf)
    set_target_properties(insitu_device_velocity_probe PROPERTIES LINKER_LANGUAGE Fortran
      Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_device_velocity_modules")
    target_include_directories(insitu_device_velocity_probe PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/insitu_device_velocity_modules")
    target_link_libraries(insitu_device_velocity_probe PRIVATE MPI::MPI_Fortran)
    add_executable(insitu_device_fields_probe EXCLUDE_FROM_ALL
      constdef.F90 ../src_gpu/output_derivatives_gpu.cuf ../src_gpu/insitu_device_velocity.cuf
      ../tests/gpu_validation/insitu_device_fields_probe.cuf)
    set_target_properties(insitu_device_fields_probe PROPERTIES LINKER_LANGUAGE Fortran
      Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_device_fields_modules")
    target_include_directories(insitu_device_fields_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_device_fields_modules")
    target_link_libraries(insitu_device_fields_probe PRIVATE MPI::MPI_Fortran)
    add_executable(insitu_device_boundary_probe EXCLUDE_FROM_ALL
      constdef.F90 ../src_gpu/output_derivatives_gpu.cuf ../src_gpu/insitu_device_velocity.cuf
      ../tests/gpu_validation/insitu_device_boundary_probe.cuf)
    set_target_properties(insitu_device_boundary_probe PROPERTIES LINKER_LANGUAGE Fortran
      Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_device_boundary_modules")
    target_include_directories(insitu_device_boundary_probe PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/insitu_device_boundary_modules")
    target_link_libraries(insitu_device_boundary_probe PRIVATE MPI::MPI_Fortran)
    if(ASTR_INSITU_VELOCITY_DEBUG)
      target_compile_definitions(insitu_device_velocity_probe PRIVATE ASTR_INSITU_VELOCITY_DEBUG)
    endif()
    if(ASTR_HAS_MPI_CUDA_QUERY)
      target_compile_definitions(insitu_device_velocity_probe PRIVATE ASTR_MPI_CUDA_AWARE_QUERY)
      target_compile_definitions(insitu_device_fields_probe PRIVATE ASTR_MPI_CUDA_AWARE_QUERY)
      target_compile_definitions(insitu_device_boundary_probe PRIVATE ASTR_MPI_CUDA_AWARE_QUERY)
    endif()
  endif()
endif()

if(BUILD_TESTING)
  target_compile_definitions(astr PRIVATE ASTR_BUILD_TESTING)

  add_executable(checkpoint_bundle_probe EXCLUDE_FROM_ALL
    insitu_checkpoint_batch.F90 checkpoint_bundle.F90 ../tests/gpu_validation/checkpoint_bundle_probe.F90)
  set_target_properties(checkpoint_bundle_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/checkpoint_bundle_modules")
  target_include_directories(checkpoint_bundle_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/checkpoint_bundle_modules")
  target_link_libraries(checkpoint_bundle_probe PRIVATE MPI::MPI_Fortran astr_checkpoint_filesystem)

  add_executable(checkpoint_state_probe EXCLUDE_FROM_ALL
    checkpoint_state_io.F90 insitu_time_integral.F90 insitu_velocity_statistics.F90
    ../tests/gpu_validation/checkpoint_state_probe.F90)
  set_target_properties(checkpoint_state_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/checkpoint_state_modules")
  target_include_directories(checkpoint_state_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/checkpoint_state_modules")
  target_link_libraries(checkpoint_state_probe PRIVATE MPI::MPI_Fortran)

  add_executable(output_config_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 adaptive_output.F90 output_config.F90 ../tests/gpu_validation/output_config_probe.F90)
  set_target_properties(output_config_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output_config_modules")
  target_include_directories(output_config_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/output_config_modules")
  target_link_libraries(output_config_probe PRIVATE MPI::MPI_Fortran)

  add_executable(output_config_collective_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 adaptive_output.F90 output_config.F90 ../tests/gpu_validation/output_config_collective_probe.F90)
  set_target_properties(output_config_collective_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/output_collective_modules")
  target_include_directories(output_config_collective_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/output_collective_modules")
  target_link_libraries(output_config_collective_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_checkpoint_batch_probe EXCLUDE_FROM_ALL
    insitu_checkpoint_batch.F90 ../tests/gpu_validation/insitu_checkpoint_batch_probe.F90)
  set_target_properties(insitu_checkpoint_batch_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_batch_modules")
  target_include_directories(insitu_checkpoint_batch_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_batch_modules")
  target_link_libraries(insitu_checkpoint_batch_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_spatial_statistics_probe EXCLUDE_FROM_ALL
    insitu_time_integral.F90 insitu_velocity_statistics.F90 insitu_spatial_statistics.F90
    ../tests/gpu_validation/insitu_spatial_statistics_probe.F90)
  set_target_properties(insitu_spatial_statistics_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_spatial_modules")
  target_include_directories(insitu_spatial_statistics_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_spatial_modules")
  target_link_libraries(insitu_spatial_statistics_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_geometry_probe EXCLUDE_FROM_ALL
    insitu_geometry.F90 ../tests/gpu_validation/insitu_geometry_probe.F90)
  set_target_properties(insitu_geometry_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_geometry_modules")
  target_include_directories(insitu_geometry_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_geometry_modules")

  add_executable(insitu_velocity_statistics_probe EXCLUDE_FROM_ALL
    insitu_time_integral.F90 insitu_velocity_statistics.F90
    ../tests/gpu_validation/insitu_velocity_statistics_probe.F90)
  set_target_properties(insitu_velocity_statistics_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_velocity_modules")
  target_include_directories(insitu_velocity_statistics_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_velocity_modules")

  add_executable(insitu_time_integral_probe EXCLUDE_FROM_ALL
    insitu_time_integral.F90 ../tests/gpu_validation/insitu_time_integral_probe.F90)
  set_target_properties(insitu_time_integral_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_integral_modules")
  target_include_directories(insitu_time_integral_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_integral_modules")

  add_executable(insitu_wall_separation_probe EXCLUDE_FROM_ALL
    insitu_wall_separation.F90 ../tests/gpu_validation/insitu_wall_separation_probe.F90)
  set_target_properties(insitu_wall_separation_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_separation_modules")
  target_include_directories(insitu_wall_separation_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_separation_modules")

  add_executable(insitu_schedule_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 ../tests/gpu_validation/insitu_schedule_probe.F90)
  set_target_properties(insitu_schedule_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_schedule_modules")
  target_include_directories(insitu_schedule_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_schedule_modules")

  add_executable(adaptive_output_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 adaptive_output.F90 ../tests/gpu_validation/adaptive_output_probe.F90)
  set_target_properties(adaptive_output_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/adaptive_output_modules")
  target_include_directories(adaptive_output_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/adaptive_output_modules")
  target_link_libraries(adaptive_output_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_product_schedule_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 adaptive_output.F90 insitu_run_config.F90 insitu_product_schedule.F90
    ../tests/gpu_validation/insitu_product_schedule_probe.F90)
  set_target_properties(insitu_product_schedule_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_product_schedule_modules")
  target_include_directories(insitu_product_schedule_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_product_schedule_modules")
  target_link_libraries(insitu_product_schedule_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_resource_budget_probe EXCLUDE_FROM_ALL
    insitu_resource_budget.F90 ../tests/gpu_validation/insitu_resource_budget_probe.F90)
  set_target_properties(insitu_resource_budget_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_resource_modules")
  target_include_directories(insitu_resource_budget_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_resource_modules")

  add_executable(insitu_run_config_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 adaptive_output.F90 insitu_run_config.F90 ../tests/gpu_validation/insitu_run_config_probe.F90)
  set_target_properties(insitu_run_config_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_config_modules")
  target_include_directories(insitu_run_config_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_config_modules")
  target_link_libraries(insitu_run_config_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_config_collective_probe EXCLUDE_FROM_ALL
    insitu_schedule.F90 adaptive_output.F90 insitu_run_config.F90 insitu_config_collective.F90
    ../tests/gpu_validation/insitu_config_collective_probe.F90)
  set_target_properties(insitu_config_collective_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_collective_modules")
  target_include_directories(insitu_config_collective_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_collective_modules")
  target_link_libraries(insitu_config_collective_probe PRIVATE MPI::MPI_Fortran)

  add_executable(insitu_runtime_probe EXCLUDE_FROM_ALL
    insitu_runtime.F90 ../tests/gpu_validation/insitu_runtime_probe.F90)
  set_target_properties(insitu_runtime_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/insitu_probe_modules"
    LINKER_LANGUAGE Fortran)
  target_include_directories(insitu_runtime_probe PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/insitu_probe_modules")
  target_link_libraries(insitu_runtime_probe PRIVATE MPI::MPI_Fortran)
  if(ASTR_WITH_CATALYST)
    target_link_libraries(insitu_runtime_probe PRIVATE astr_catalyst_adapter)
  endif()

if(ASTR_WITH_AIR5_CHEMISTRY)
  add_executable(air5_frozen_source_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90 chemistry_properties.F90
    chemistry_kinetics.F90
    ../tests/gpu_validation/air5_frozen_source_probe.F90)
  if(ASTR_WITH_CUDA)
    target_sources(air5_frozen_source_probe PRIVATE
      ../src_gpu/gpu_check.cuf ../src_gpu/chemistry_core_gpu.cuf
      ../src_gpu/chemistry_relaxation_gpu.cuf ../src_gpu/chemistry_kinetics_gpu.cuf)
  endif()
  set_target_properties(air5_frozen_source_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/frozen_source_mod")
  target_include_directories(air5_frozen_source_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/frozen_source_mod")
  target_link_libraries(air5_frozen_source_probe PRIVATE MPI::MPI_Fortran)
  add_executable(chemistry_compensation_lifecycle_probe EXCLUDE_FROM_ALL
    chemistry_compensation.F90
    ../tests/gpu_validation/chemistry_compensation_lifecycle_probe.F90)
  set_target_properties(chemistry_compensation_lifecycle_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compensation_lifecycle_modules")
  target_include_directories(chemistry_compensation_lifecycle_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/compensation_lifecycle_modules")
  target_link_libraries(chemistry_compensation_lifecycle_probe PRIVATE MPI::MPI_Fortran)
  add_executable(chemistry_ros2_roundoff_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90 chemistry_properties.F90
    chemistry_kinetics.F90
    ../tests/gpu_validation/chemistry_ros2_roundoff_probe.F90)
  set_target_properties(chemistry_ros2_roundoff_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/chemistry_ros2_roundoff_probe_modules")
  target_include_directories(chemistry_ros2_roundoff_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/chemistry_ros2_roundoff_probe_modules")
endif()

if(ASTR_WITH_AIR5_CHEMISTRY)
  add_executable(air5_characteristic_thermo_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90 chemistry_properties.F90
    chemistry_characteristic.F90
    ../tests/gpu_validation/air5_characteristic_thermo_probe.F90)
  set_target_properties(air5_characteristic_thermo_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/characteristic_thermo_mod")
  target_include_directories(air5_characteristic_thermo_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/characteristic_thermo_mod")
  add_executable(air5_pressure_outlet_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90 chemistry_properties.F90
    chemistry_boundary_state.F90
    ../tests/gpu_validation/air5_pressure_outlet_probe.F90)
  set_target_properties(air5_pressure_outlet_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pressure_outlet_mod")
  target_include_directories(air5_pressure_outlet_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/pressure_outlet_mod")
  add_executable(air5_transport_roundoff_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90
    ../tests/gpu_validation/air5_transport_roundoff_probe.F90)
  set_target_properties(air5_transport_roundoff_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/transport_roundoff_mod")
  target_include_directories(air5_transport_roundoff_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/transport_roundoff_mod")
endif()

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

add_executable(benchmark_runtime_probe EXCLUDE_FROM_ALL
  benchmark_runtime.F90 ../tests/gpu_validation/benchmark_runtime_probe.F90)
set_target_properties(benchmark_runtime_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/benchmark_runtime_probe_modules")
target_include_directories(benchmark_runtime_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/benchmark_runtime_probe_modules")
target_link_libraries(benchmark_runtime_probe PRIVATE MPI::MPI_Fortran)

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
  add_executable(precision_mode_setup_test EXCLUDE_FROM_ALL
    ../src_gpu/precision_mode_gpu.cuf
    ../tests/gpu_validation/precision_mode_setup_test.cuf)
  set_target_properties(precision_mode_setup_test PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/precision_mode_probe_modules")
  target_include_directories(precision_mode_setup_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/precision_mode_probe_modules")
  target_link_libraries(precision_mode_setup_test PRIVATE MPI::MPI_Fortran)
  add_executable(mixed_candidate_setup_test EXCLUDE_FROM_ALL
    ../src_gpu/precision_mode_gpu.cuf
    ../src_gpu/mixed_candidate_gpu.cuf
    ../tests/gpu_validation/mixed_candidate_setup_test.cuf)
  set_target_properties(mixed_candidate_setup_test PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mixed_candidate_probe_modules")
  target_include_directories(mixed_candidate_setup_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/mixed_candidate_probe_modules")
  target_link_libraries(mixed_candidate_setup_test PRIVATE MPI::MPI_Fortran)
  add_executable(upwind_flux_mode_setup_test EXCLUDE_FROM_ALL
    ../src_gpu/upwind_flux_mode_gpu.cuf
    ../tests/gpu_validation/upwind_flux_mode_setup_test.cuf)
  set_target_properties(upwind_flux_mode_setup_test PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/upwind_flux_mode_probe_modules")
  target_include_directories(upwind_flux_mode_setup_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/upwind_flux_mode_probe_modules")
  target_link_libraries(upwind_flux_mode_setup_test PRIVATE MPI::MPI_Fortran)
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
  if(ASTR_HAS_MPI_CUDA_QUERY)
    add_executable(halo_cuda_aware_probe EXCLUDE_FROM_ALL
      ../tests/gpu_validation/halo_cuda_aware_probe.cuf)
    target_link_libraries(halo_cuda_aware_probe PRIVATE MPI::MPI_Fortran)
  endif()
endif()

if(ASTR_WITH_CUDA AND ASTR_WITH_AIR5_CHEMISTRY)
  add_executable(air5_flux_gpu_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90
    ../src_gpu/chemistry_core_gpu.cuf
    ../tests/gpu_validation/air5_flux_gpu_probe_wrappers.cuf
    ../tests/gpu_validation/air5_transport_roundoff_probe.F90)
  set_target_properties(air5_flux_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/air5_flux_gpu_probe_modules")
  target_include_directories(air5_flux_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/air5_flux_gpu_probe_modules")
  target_compile_definitions(air5_flux_gpu_probe PRIVATE AIR5_FLUX_GPU_PROBE)
  add_executable(chemistry_flow_gpu_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90
    chemistry_core.F90
    chemistry_properties.F90
    ../src_gpu/gpu_check.cuf
    ../src_gpu/chemistry_core_gpu.cuf
    ../src_gpu/chemistry_flow_state_gpu.cuf
    ../src_gpu/chemistry_transport_gpu.cuf
    ../tests/gpu_validation/chemistry_flow_gpu_probe.cuf)
  set_target_properties(chemistry_flow_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/chemistry_flow_gpu_probe_modules")
  target_include_directories(chemistry_flow_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/chemistry_flow_gpu_probe_modules")
  target_link_libraries(chemistry_flow_gpu_probe PRIVATE MPI::MPI_Fortran)
  add_executable(chemistry_source_gpu_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90
    chemistry_core.F90
    chemistry_properties.F90
    chemistry_kinetics.F90
    ../src_gpu/gpu_check.cuf
    ../src_gpu/chemistry_core_gpu.cuf
    ../src_gpu/chemistry_relaxation_gpu.cuf
    ../src_gpu/chemistry_kinetics_gpu.cuf
    ../tests/gpu_validation/chemistry_source_gpu_probe.cuf)
  set_target_properties(chemistry_source_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/chemistry_source_gpu_probe_modules")
  target_include_directories(chemistry_source_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/chemistry_source_gpu_probe_modules")
  target_link_libraries(chemistry_source_gpu_probe PRIVATE MPI::MPI_Fortran)
  add_executable(chemistry_ros2_gpu_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90
    chemistry_core.F90
    chemistry_properties.F90
    chemistry_kinetics.F90
    ../src_gpu/gpu_check.cuf
    ../src_gpu/chemistry_core_gpu.cuf
    ../src_gpu/chemistry_relaxation_gpu.cuf
    ../src_gpu/chemistry_kinetics_gpu.cuf
    ../tests/gpu_validation/chemistry_ros2_gpu_probe.cuf)
  set_target_properties(chemistry_ros2_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/chemistry_ros2_gpu_probe_modules")
  target_include_directories(chemistry_ros2_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/chemistry_ros2_gpu_probe_modules")
  target_link_libraries(chemistry_ros2_gpu_probe PRIVATE MPI::MPI_Fortran)
  add_executable(chemistry_ros2_compensation_gpu_probe EXCLUDE_FROM_ALL
    chemistry_air5_data.F90 chemistry_core.F90 chemistry_properties.F90
    chemistry_kinetics.F90
    ../src_gpu/gpu_check.cuf ../src_gpu/chemistry_core_gpu.cuf
    ../src_gpu/chemistry_relaxation_gpu.cuf ../src_gpu/chemistry_kinetics_gpu.cuf
    ../tests/gpu_validation/chemistry_ros2_compensation_gpu_probe.cuf)
  set_target_properties(chemistry_ros2_compensation_gpu_probe PROPERTIES
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/chemistry_ros2_compensation_gpu_probe_modules")
  target_include_directories(chemistry_ros2_compensation_gpu_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/chemistry_ros2_compensation_gpu_probe_modules")
  target_link_libraries(chemistry_ros2_compensation_gpu_probe PRIVATE MPI::MPI_Fortran)
endif()

endif() # BUILD_TESTING

if (ASTR_WITH_CUDA AND (Fortran_COMPILER_NAME MATCHES "PGI" OR Fortran_COMPILER_NAME MATCHES "NVHPC"))
  set(ASTR_S0A_MAXREGCOUNT "128" CACHE STRING
      "NVHPC register cap for the controlled S0-A characteristic kernels")
  # The selective Roe kernel must retain the prescribed 512-thread x block.
  # Cap its register allocation so the launch is legal on Ada and later GPUs.
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/solver_gpu.cuf
    PROPERTIES COMPILE_OPTIONS "-gpu=maxregcount:${ASTR_S0A_MAXREGCOUNT}")
  if(ASTR_WITH_AIR5_CHEMISTRY)
    set(ASTR_AIR5_MAXREGCOUNT "128" CACHE STRING
        "NVHPC register cap for fixed air5 C4 transport kernels")
    # Keep the prescribed 512-thread directional blocks legal for the
    # higher-register 11-variable transport kernels.
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_solver_gpu.cuf
      PROPERTIES COMPILE_OPTIONS "-gpu=maxregcount:${ASTR_AIR5_MAXREGCOUNT}")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_transport_gpu.cuf
      PROPERTIES COMPILE_OPTIONS "-gpu=maxregcount:${ASTR_AIR5_MAXREGCOUNT}")
    # Device callees must respect the launcher's register cap for RDC linking.
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../src_gpu/chemistry_core_gpu.cuf
      PROPERTIES COMPILE_OPTIONS "-gpu=maxregcount:${ASTR_AIR5_MAXREGCOUNT}")
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/chemistry_characteristic.F90
      PROPERTIES COMPILE_OPTIONS "-gpu=maxregcount:${ASTR_AIR5_MAXREGCOUNT}")
  endif()
endif()
                    
target_link_libraries(astr)
if (MPI_FOUND)
  target_link_libraries(astr PRIVATE MPI::MPI_Fortran)
endif (MPI_FOUND)
if (CHEMISTRY)
  message(STATUS "CANTERA library: ${CTRDIR}/lib")
  target_link_libraries(astr PRIVATE -L${CTRDIR}/lib -lcantera_fortran -lcantera -lstdc++ -pthread)
endif (CHEMISTRY)

if(ASTR_WITH_CUDA AND BUILD_TESTING)
  # Recompile the production modules in isolation, without the ASTR main program.
  set(ASTR_HALO_CONTRACT_SOURCES ${ASTR_SOURCES})
  list(REMOVE_ITEM ASTR_HALO_CONTRACT_SOURCES astr.F90 CMakeLists.txt)
  add_executable(halo_exchange_contract_test EXCLUDE_FROM_ALL
    ${ASTR_HALO_CONTRACT_SOURCES} ${ASTR_GPU_SOURCES}
    ../tests/gpu_validation/halo_exchange_contract_test.cuf)
  set_target_properties(halo_exchange_contract_test PROPERTIES
    LINKER_LANGUAGE Fortran
    Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/halo_contract_modules")
  target_include_directories(halo_exchange_contract_test PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/halo_contract_modules")
  get_target_property(ASTR_HALO_CONTRACT_LIBRARIES astr LINK_LIBRARIES)
  target_link_libraries(halo_exchange_contract_test PRIVATE ${ASTR_HALO_CONTRACT_LIBRARIES})
  target_compile_definitions(halo_exchange_contract_test PRIVATE ASTR_BUILD_TESTING)
  if(ASTR_WITH_AIR5_CHEMISTRY)
    target_compile_definitions(halo_exchange_contract_test PRIVATE ASTR_AIR5_CHEMISTRY)
  endif()
  if(ASTR_HAS_MPI_CUDA_QUERY)
    target_compile_definitions(halo_exchange_contract_test PRIVATE ASTR_MPI_CUDA_AWARE_QUERY)
  endif()
endif()

option(ASTR_BUILD_MPI_COMPLETION_TRACE "Build optional MPI completion profiling interposer" OFF)
if(ASTR_BUILD_MPI_COMPLETION_TRACE AND NOT BUILD_TESTING)
  message(FATAL_ERROR "ASTR_BUILD_MPI_COMPLETION_TRACE requires BUILD_TESTING=ON")
endif()
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
install(FILES
  ../scripts/output/export_checkpoint.py
  ../scripts/output/repair_series.py
  ../scripts/output/combine_series.py
  ../scripts/output/input.output.tgv.example
  ../scripts/output/input.output.tgv.derived.example
  ../scripts/output/input.output.air5.derived.example
  ../scripts/output/README.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/astr/output
  COMPONENT OutputTools)
if(ASTR_WITH_CATALYST)
  install(FILES
    ../scripts/insitu/tgv_pipeline.py
    ../scripts/insitu/tgv_streamlines.py
    ../scripts/insitu/egl_identity.py
    ../scripts/insitu/image_publication.py
    ../scripts/insitu/product_dispatch.py
    DESTINATION ${CMAKE_INSTALL_DATADIR}/astr/insitu
    COMPONENT InsituTools)
  if(ASTR_WITH_INSITU_DEVICE_RENDERING)
    install(FILES ../scripts/insitu/device_render_pipeline.py
      DESTINATION ${CMAKE_INSTALL_DATADIR}/astr/insitu
      COMPONENT InsituTools)
  endif()
endif()

if(BUILD_TESTING)
add_executable(rectilinear_metric_halo_probe EXCLUDE_FROM_ALL
  rectilinear_metric_halo.F90 ../tests/gpu_validation/rectilinear_metric_halo_probe.F90)
set_target_properties(rectilinear_metric_halo_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/metric_halo_probe_modules")
target_include_directories(rectilinear_metric_halo_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/metric_halo_probe_modules")

add_executable(cfl_spectrum_probe EXCLUDE_FROM_ALL
  cfl_spectrum.F90 ../tests/gpu_validation/cfl_spectrum_probe.F90)
set_target_properties(cfl_spectrum_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cfl_probe_modules")
target_include_directories(cfl_spectrum_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/cfl_probe_modules")

add_executable(centered_diffusion_response_probe EXCLUDE_FROM_ALL
  constdef.F90 commtype.F90 commvar.F90 commfunc.F90 singleton.F90 derivative.F90
  ../tests/gpu_validation/centered_diffusion_response_probe.F90)
set_target_properties(centered_diffusion_response_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/diffusion_response_modules")
target_include_directories(centered_diffusion_response_probe PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/diffusion_response_modules")
endif()
