# Build one executable for every regression example.
#
# REGRESSION_INFO is deliberately the source of truth here as well as for
# regression_tests/run_all.sh.  A new example therefore only needs a
# directory containing REGRESSION_INFO and a main source file; every C/C++
# source file in that directory is compiled into the resulting executable.

set(_storm_example_info_files "")
file(GLOB _storm_example_info_files CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*/REGRESSION_INFO")
list(SORT _storm_example_info_files)

# MadVoro is header-heavy, but examples using its Voronoi implementation also
# need its exception translation units. STORM_MADVORO_DIR resolves either the
# standalone dependency or a parent project's in-tree MadVoro checkout.
set(_storm_madvoro_root "")
if(DEFINED STORM_MADVORO_DIR AND
   EXISTS "${STORM_MADVORO_DIR}/exception")
  set(_storm_madvoro_root "${STORM_MADVORO_DIR}")
endif()

set(_storm_madvoro_exception_sources "")
if(_storm_madvoro_root)
  file(GLOB _storm_madvoro_exception_sources CONFIGURE_DEPENDS
    "${_storm_madvoro_root}/exception/*.cpp")
endif()

foreach(_storm_info_file IN LISTS _storm_example_info_files)
  get_filename_component(_storm_example_dir "${_storm_info_file}" DIRECTORY)
  get_filename_component(_storm_example_name "${_storm_example_dir}" NAME)

  # Runtime regression variants may reuse a target built from another example
  # directory.  This keeps one source of truth for cases that differ only by
  # command-line arguments, such as IMC and DDMC.
  file(STRINGS "${_storm_info_file}" _storm_build_target_lines
    REGEX "^[ \\t]*BUILD_TARGET[ \\t]*=")
  if(_storm_build_target_lines)
    list(GET _storm_build_target_lines 0 _storm_build_target_line)
    string(REGEX REPLACE
      "^[^=]*=[ \\t]*[\"']?([^\"' \\t]+)[\"']?.*$" "\\1"
      _storm_build_target "${_storm_build_target_line}")
    if(TARGET "${_storm_build_target}")
      message(STATUS "STORM regression example: ${_storm_example_name} reuses ${_storm_build_target}")
      continue()
    endif()
  endif()

  # REGRESSION_INFO is shell-sourceable, but CMake only needs the TAGS value
  # to decide whether an MPI-only example is available in this build.
  file(STRINGS "${_storm_info_file}" _storm_tags_lines
    REGEX "^[ \\t]*TAGS[ \\t]*=")
  string(TOLOWER "${_storm_tags_lines}" _storm_tags)
  if(_storm_tags MATCHES "mpi")
    set(_storm_example_requires_mpi ON)
  else()
    set(_storm_example_requires_mpi OFF)
  endif()

  if(_storm_example_requires_mpi AND NOT STORM_WITH_MPI AND NOT MPI)
    message(STATUS
      "Skipping ${_storm_example_name}: REGRESSION_INFO marks it as MPI "
      "but this build has MPI disabled")
    continue()
  endif()

  file(GLOB _storm_example_sources CONFIGURE_DEPENDS
    "${_storm_example_dir}/*.c"
    "${_storm_example_dir}/*.cc"
    "${_storm_example_dir}/*.cpp"
    "${_storm_example_dir}/*.cxx")
  list(SORT _storm_example_sources)

  set(_storm_example_has_main OFF)
  foreach(_storm_example_source IN LISTS _storm_example_sources)
    get_filename_component(_storm_example_source_name
      "${_storm_example_source}" NAME)
    if(_storm_example_source_name MATCHES "^main\\.(c|cc|cpp|cxx)$")
      set(_storm_example_has_main ON)
    endif()
  endforeach()

  if(NOT _storm_example_has_main)
    message(WARNING
      "Skipping ${_storm_example_name}: ${_storm_info_file} exists, but "
      "the directory has no main source file")
    continue()
  endif()

  set(_storm_example_extra_sources "")
  set(_storm_example_include_dirs "${_storm_example_dir}")

  set(_storm_example_uses_madvoro OFF)
  foreach(_storm_example_source IN LISTS _storm_example_sources)
    file(STRINGS "${_storm_example_source}" _storm_madvoro_includes
      REGEX "MadVoro/")
    if(_storm_madvoro_includes)
      set(_storm_example_uses_madvoro ON)
    endif()

  endforeach()

  if(_storm_example_uses_madvoro)
    list(APPEND _storm_example_extra_sources
      ${_storm_madvoro_exception_sources})
  endif()

  if(TARGET "${_storm_example_name}")
    message(FATAL_ERROR
      "Cannot create STORM example target ${_storm_example_name}: "
      "the target name is already in use")
  endif()

  add_executable("${_storm_example_name}"
    ${_storm_example_sources}
    ${_storm_example_extra_sources})
  target_link_libraries("${_storm_example_name}" PRIVATE storm)
  target_include_directories("${_storm_example_name}" PRIVATE
    ${_storm_example_include_dirs})
  target_compile_definitions("${_storm_example_name}" PRIVATE
    "STORM_DATA_DIR=\"${_storm_example_dir}\"")
  set_target_properties("${_storm_example_name}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${_storm_example_dir}")

  if(NOT _storm_example_requires_mpi AND STORM_WITH_MPI)
    target_compile_options("${_storm_example_name}" PRIVATE
      -USTORM_WITH_MPI -UMADVORO_WITH_MPI -URICH_MPI
      -USPATIAL_DS_WITH_MPI -U__WITH_MPI)
  endif()

  message(STATUS "STORM regression example: ${_storm_example_name}")
endforeach()
