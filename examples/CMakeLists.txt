add_executable(serial_cartesian serial_cartesian/main.cpp)
target_link_libraries(serial_cartesian PRIVATE storm)
set_target_properties(serial_cartesian PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/serial_cartesian")

set(MADVORO_EXCEPTION_SOURCES
  ${STORM_DEPS_DIR}/MadVoro/exception/MadVoroException.cpp
  ${STORM_DEPS_DIR}/MadVoro/exception/InvalidArgumentException.cpp
  ${STORM_DEPS_DIR}/MadVoro/exception/SizeException.cpp
)
if(STORM_WITH_MPI)
  add_executable(hohlraum_parallel hohlraum_parallel/main.cpp ${MADVORO_EXCEPTION_SOURCES})
  target_link_libraries(hohlraum_parallel PRIVATE storm)
  target_include_directories(hohlraum_parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hohlraum)
  set_target_properties(hohlraum_parallel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hohlraum_parallel")
endif()

foreach(MNUM 1 2 3)
  add_executable(marshak_wave_${MNUM} marshak_wave_${MNUM}/main.cpp)
  target_link_libraries(marshak_wave_${MNUM} PRIVATE storm)
  target_include_directories(marshak_wave_${MNUM} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave)
  target_compile_definitions(marshak_wave_${MNUM} PRIVATE STORM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave_${MNUM}")
  set_target_properties(marshak_wave_${MNUM} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave_${MNUM}")
endforeach()

if(STORM_WITH_MPI)
  add_executable(marshak_wave_4 marshak_wave_4/main.cpp)
  target_link_libraries(marshak_wave_4 PRIVATE storm)
  target_include_directories(marshak_wave_4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave)
  target_compile_definitions(marshak_wave_4 PRIVATE STORM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave_4")
  set_target_properties(marshak_wave_4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave_4")

  add_executable(cartesian_parallel_check cartesian_parallel_check/main.cpp)
  target_link_libraries(cartesian_parallel_check PRIVATE storm)
  set_target_properties(cartesian_parallel_check PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cartesian_parallel_check")
  if(BUILD_TESTING AND MPIEXEC_EXECUTABLE)
    add_test(NAME cartesian_parallel_check
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:cartesian_parallel_check>)
  endif()
endif()

add_executable(densmore2012 densmore2012/main.cpp)
target_link_libraries(densmore2012 PRIVATE storm)
target_include_directories(densmore2012 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/densmore2012)
target_compile_definitions(densmore2012 PRIVATE STORM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/densmore2012")
set_target_properties(densmore2012 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/densmore2012")

if(STORM_WITH_MPI)
  add_executable(moving_slab moving_slab/main.cpp ${MADVORO_EXCEPTION_SOURCES})
  target_link_libraries(moving_slab PRIVATE storm)
  target_include_directories(moving_slab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/moving_slab)
  set_target_properties(moving_slab PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/moving_slab")
endif()
