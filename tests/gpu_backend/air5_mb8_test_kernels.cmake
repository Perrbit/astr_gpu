# Build mirrors of the real kernels, using the same private headers as the
# component probes. No production kernel or runtime admission is changed.
set(mb8_probe_source "${CMAKE_SOURCE_DIR}/tests/gpu_backend")
set(mb8_probe_build "${PROJECT_BINARY_DIR}/gpu_backend")
set(mb8_kernel_build "${CMAKE_CURRENT_BINARY_DIR}/mb8_test")
set(air5_candidate_sources "")
foreach(kernel IN ITEMS convection filter boundary)
  set(source "${mb8_kernel_build}/chemistry_${kernel}")
  configure_file("chemistry_${kernel}.cu" "${source}.source" COPYONLY)
  file(READ "${source}.source" code)
  set(original "${code}")
  if(kernel STREQUAL "convection")
    string(REPLACE [=[#include "chemistry_flux_math.cuh"]=]
      "#include \"${mb8_probe_source}/air5_nr1_budget_math.cuh\"" code "${code}")
    set_source_files_properties("${source}.cu" PROPERTIES
      INCLUDE_DIRECTORIES "${mb8_probe_build}/nr1_component")
  elseif(kernel STREQUAL "filter")
    string(REPLACE [=[#include "chemistry_filter_math.cuh"]=]
      "#include \"${mb8_probe_source}/air5_nr2_capacity_math.cuh\"" code "${code}")
    string(REPLACE "astr::air5::close_species_flux(high+5,lower,upper,mass,candidate+5)"
      "astr::air5::nr2_capacity::project(high+5,lower,upper,mass,candidate+5)" code "${code}")
  else()
    string(REPLACE [=[#include "chemistry_characteristic_math.cuh"]=]
      "#include \"${mb8_probe_build}/nr4_component/chemistry_characteristic_math.cuh\"\n#include \"${mb8_probe_source}/air5_nr4_fixed_acoustic.cuh\""
      code "${code}")
    string(REPLACE "    const int at=i+nx*k;\n"
      "    bool incoming[2];int temperature_region;\n    nr4_fixed::classify(local,preserve,incoming,temperature_region);\n    const int at=i+nx*k;\n"
      code "${code}")
    string(REPLACE "remaining,rate,result,reference_sound)" "remaining,rate,result,reference_sound,incoming)" code "${code}")
  endif()
  if(code STREQUAL original)
    message(FATAL_ERROR "MB8 test mirror no longer matches chemistry_${kernel}.cu")
  endif()
  file(GENERATE OUTPUT "${source}.cu" CONTENT "${code}")
  set_source_files_properties("${source}.cu" PROPERTIES GENERATED TRUE)
  list(APPEND air5_candidate_sources "${source}.cu")
endforeach()
