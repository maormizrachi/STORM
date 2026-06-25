cmake_minimum_required(VERSION 3.14)
project(RDMont
  VERSION 1.0.0
  LANGUAGES CXX
  DESCRIPTION "Monte Carlo particle transport library"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# ---------- Options ----------
option(RDMONT_WITH_MPI   "Build with MPI support"  OFF)
option(RDMONT_WITH_HDF5  "Build with HDF5 support" OFF)
option(RDMONT_BUILD_EXAMPLES "Build example programs" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------- External dependencies directory ----------
if(DEFINED RDMONT_DEPS_DIR)
  get_filename_component(_deps "${RDMONT_DEPS_DIR}" ABSOLUTE)
  include_directories("${_deps}/mpi_utils")
  include_directories("${_deps}/MadCart")
endif()

# ---------- Required dependencies ----------
find_package(Boost 1.74.0 REQUIRED)

# ---------- Optional dependencies ----------
if(RDMONT_WITH_MPI)
  find_package(MPI 2.0 REQUIRED COMPONENTS CXX)
endif()

if(RDMONT_WITH_HDF5)
  set(HDF5_PREFER_PARALLEL ${RDMONT_WITH_MPI})
  find_package(HDF5 1.8.0 REQUIRED COMPONENTS CXX HL)
endif()

# ---------- Library sources ----------
set(RDMONT_SOURCES "")

if(RDMONT_WITH_MPI)
  list(APPEND RDMONT_SOURCES
    utils/RankSync.cpp
    manager/parallel/ReallocationAgent.cpp
  )
endif()

if(RDMONT_SOURCES)
  add_library(rdmont ${RDMONT_SOURCES})
else()
  add_library(rdmont INTERFACE)
endif()

target_include_directories(rdmont INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(rdmont INTERFACE Boost::boost)

if(RDMONT_WITH_MPI)
  target_compile_definitions(rdmont PUBLIC RDMONT_WITH_MPI RICH_MPI)
  target_link_libraries(rdmont PUBLIC MPI::MPI_CXX)
endif()

if(RDMONT_WITH_HDF5)
  target_compile_definitions(rdmont PUBLIC RDMONT_WITH_HDF5)
  target_include_directories(rdmont PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(rdmont PUBLIC ${HDF5_LIBRARIES})
endif()

# ---------- Examples ----------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND RDMONT_BUILD_EXAMPLES)
  include(CTest)
endif()

if(RDMONT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------- Install & export (skip when embedded) ----------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rdmont
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
    PATTERN ".git" EXCLUDE
    PATTERN "examples" EXCLUDE
    PATTERN "cmake" EXCLUDE
  )

  install(TARGETS rdmont EXPORT RDMontTargets)
  install(EXPORT RDMontTargets
    NAMESPACE RDMont::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RDMont
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RDMontConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/RDMontConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RDMont
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/RDMontConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/RDMontConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/RDMontConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RDMont
  )
endif()
