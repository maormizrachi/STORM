find_package(Kokkos CONFIG REQUIRED)

get_target_property(_storm_gpu_target_type storm TYPE)
if(_storm_gpu_target_type STREQUAL "INTERFACE_LIBRARY")
  set(_storm_gpu_scope INTERFACE)
else()
  set(_storm_gpu_scope PUBLIC)
endif()

target_compile_definitions(storm ${_storm_gpu_scope} STORM_WITH_GPU)
target_link_libraries(storm ${_storm_gpu_scope} Kokkos::kokkos)

add_library(storm_gpu_compile_test OBJECT compile_test.cpp)
target_link_libraries(storm_gpu_compile_test PRIVATE storm)
