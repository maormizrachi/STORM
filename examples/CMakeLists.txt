add_executable(serial_cartesian serial_cartesian/main.cpp)
target_link_libraries(serial_cartesian PRIVATE storm)

add_executable(hohlraum hohlraum/main.cpp)
target_link_libraries(hohlraum PRIVATE storm)
target_include_directories(hohlraum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/hohlraum)
