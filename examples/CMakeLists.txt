add_executable(serial_cartesian serial_cartesian/main.cpp)
target_link_libraries(serial_cartesian PRIVATE storm)

set(MADVORO_EXCEPTION_SOURCES
  ${STORM_DEPS_DIR}/MadVoro/exception/MadVoroException.cpp
  ${STORM_DEPS_DIR}/MadVoro/exception/InvalidArgumentException.cpp
  ${STORM_DEPS_DIR}/MadVoro/exception/SizeException.cpp
)
add_executable(hohlraum hohlraum/main.cpp ${MADVORO_EXCEPTION_SOURCES})
target_link_libraries(hohlraum PRIVATE storm)
target_include_directories(hohlraum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hohlraum)

if(STORM_WITH_MPI)
  add_executable(hohlraum_parallel hohlraum_parallel/main.cpp ${MADVORO_EXCEPTION_SOURCES})
  target_link_libraries(hohlraum_parallel PRIVATE storm)
  target_include_directories(hohlraum_parallel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hohlraum)
endif()

foreach(MNUM 1 2 3 4)
  add_executable(marshak_wave_${MNUM} marshak_wave_${MNUM}/main.cpp)
  target_link_libraries(marshak_wave_${MNUM} PRIVATE storm)
  target_include_directories(marshak_wave_${MNUM} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave)
  target_compile_definitions(marshak_wave_${MNUM} PRIVATE STORM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/marshak_wave_${MNUM}")
endforeach()

add_executable(densmore2012 densmore2012/main.cpp)
target_link_libraries(densmore2012 PRIVATE storm)
target_include_directories(densmore2012 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/densmore2012)
target_compile_definitions(densmore2012 PRIVATE STORM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/densmore2012")
