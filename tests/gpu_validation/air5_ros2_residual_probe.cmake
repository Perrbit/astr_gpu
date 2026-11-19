# Generated sources belong only to the EXCLUDE_FROM_ALL numerical experiment.
set(air5_residual_dir "${CMAKE_CURRENT_BINARY_DIR}/air5_ros2_residual_probe_sources")
set(air5_residual_tests "${CMAKE_SOURCE_DIR}/tests/gpu_validation")
foreach(kinetics chemistry_kinetics.F90 ../src_gpu/chemistry_kinetics_gpu.cuf)
  get_filename_component(unit "${kinetics}" NAME)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${kinetics}" "${air5_residual_dir}/${unit}.original" COPYONLY)
  file(READ "${air5_residual_dir}/${unit}.original" source_text)
  if(unit MATCHES "[.]cuf$")
    set(old_solve air5_lu_solve_6_gpu)
    set(fixed_step "  attributes(device) subroutine air5_ros2_fixed_step_gpu(")
    set_source_files_properties("${air5_residual_dir}/${unit}" PROPERTIES COMPILE_DEFINITIONS ASTR_ROS2_RESIDUAL_GPU)
  else()
    set(old_solve air5_lu_solve_6)
    set(fixed_step "  pure subroutine air5_ros2_fixed_step(")
  endif()
  string(FIND "${source_text}" "${fixed_step}" step_pos)
  if(step_pos LESS 0)
    message(FATAL_ERROR "Missing fixed ROS-2 step in ${unit}")
  endif()
  string(REPLACE "${fixed_step}" "#include \"air5_ros2_residual_solve.inc\"\n${fixed_step}" source_text "${source_text}")
  foreach(stage k1 k2)
    set(old_call "call ${old_solve}(lu,pivots,rhs,${stage},status)")
    string(FIND "${source_text}" "${old_call}" call_pos)
    if(call_pos LESS 0)
      message(FATAL_ERROR "Missing ROS-2 solve for ${stage} in ${unit}")
    endif()
    string(REPLACE "${old_call}" "call air5_probe_solve_once(matrix,lu,pivots,rhs,${stage},status)" source_text "${source_text}")
  endforeach()
  file(GENERATE OUTPUT "${air5_residual_dir}/${unit}" CONTENT "${source_text}")
endforeach()
add_executable(air5_ros2_residual_probe EXCLUDE_FROM_ALL
  chemistry_air5_data.F90 chemistry_core.F90 chemistry_properties.F90
  ../src_gpu/gpu_check.cuf ../src_gpu/chemistry_core_gpu.cuf ../src_gpu/chemistry_relaxation_gpu.cuf
  "${air5_residual_dir}/chemistry_kinetics.F90" "${air5_residual_dir}/chemistry_kinetics_gpu.cuf"
  "${air5_residual_tests}/air5_ros2_residual_words.F90" "${air5_residual_tests}/chemistry_ros2_compensation_gpu_probe.cuf")
set_target_properties(air5_ros2_residual_probe PROPERTIES
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/air5_ros2_residual_probe_modules")
target_include_directories(air5_ros2_residual_probe PRIVATE "${air5_residual_tests}" "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/air5_ros2_residual_probe_modules")
set_source_files_properties("${air5_residual_tests}/air5_ros2_residual_words.F90"
  PROPERTIES COMPILE_OPTIONS "-Kieee;-Mnofma;-gpu=nofma,noflushz")
target_link_libraries(air5_ros2_residual_probe PRIVATE MPI::MPI_Fortran)
