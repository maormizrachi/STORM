cmake_minimum_required(VERSION 3.14)
project(STORM
  VERSION 1.0.0
  LANGUAGES CXX C
  DESCRIPTION "Scalable Three-dimensional Optimized RDMA Monte Carlo"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -g1" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g1" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}   -g3"   CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g3"   CACHE STRING "" FORCE)
endif()

# ---------- Options ----------
option(STORM_WITH_MPI   "Build with MPI support"  OFF)
option(STORM_WITH_IBV   "Build with IBV (InfiniBand Verbs) for RDMA transport (requires MPI)" OFF)
option(STORM_WITH_VTK   "Build with VTK support for mesh output (requires MadVoro VTK headers)" OFF)
option(STORM_WITH_HDF5  "Build with HDF5 support" OFF)
option(STORM_BUILD_EXAMPLES "Build example programs" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------- External dependencies directory ----------
# STORM root first so its headers take priority over identically-named deps headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT DEFINED STORM_DEPS_DIR)
  set(STORM_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
endif()
if(IS_DIRECTORY "${STORM_DEPS_DIR}")
  get_filename_component(_deps "${STORM_DEPS_DIR}" ABSOLUTE)
  include_directories("${_deps}")
  if(IS_DIRECTORY "${_deps}/MadVoro")
    include_directories("${_deps}/MadVoro")
  endif()
  if(IS_DIRECTORY "${_deps}/MadVoro/range/finders")
    include_directories("${_deps}/MadVoro/range/finders")
  endif()
endif()

# ---------- Required dependencies ----------
find_package(Boost 1.74.0 REQUIRED)
find_package(OpenMP QUIET)

# ---------- Optional dependencies ----------
if(STORM_WITH_MPI)
  find_package(MPI 2.0 REQUIRED COMPONENTS CXX)
endif()

if(STORM_WITH_IBV)
  if(NOT STORM_WITH_MPI)
    message(FATAL_ERROR "STORM_WITH_IBV requires STORM_WITH_MPI=ON")
  endif()
  find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
endif()

if(STORM_WITH_VTK)
  find_package(VTK 9.3.0 REQUIRED
    COMPONENTS CommonCore CommonColor FiltersGeneral FiltersSources IOXML InteractionStyle
  )
  if(STORM_WITH_MPI)
    find_package(VTK REQUIRED COMPONENTS IOParallelXML ParallelMPI)
  endif()
endif()

if(STORM_WITH_HDF5)
  set(HDF5_PREFER_PARALLEL ${STORM_WITH_MPI})
  find_package(HDF5 1.8.0 REQUIRED COMPONENTS CXX HL)
endif()

# ---------- Library sources ----------
set(STORM_SOURCES "")

if(STORM_WITH_MPI)
  list(APPEND STORM_SOURCES
    utils/RankSync.cpp
    manager/parallel/ReallocationAgent.cpp
  )
  if(EXISTS "${_deps}/mpi_utils/AmountManager.cpp")
    list(APPEND STORM_SOURCES "${_deps}/mpi_utils/AmountManager.cpp")
  endif()
  if(EXISTS "${_deps}/rma/DistributedMutex.cpp")
    list(APPEND STORM_SOURCES "${_deps}/rma/DistributedMutex.cpp")
  endif()
  if(STORM_WITH_IBV AND EXISTS "${_deps}/rma/IBVContext.cpp")
    list(APPEND STORM_SOURCES "${_deps}/rma/IBVContext.cpp")
  endif()
endif()

if(STORM_SOURCES)
  add_library(storm ${STORM_SOURCES})
else()
  add_library(storm INTERFACE)
endif()

target_include_directories(storm INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(storm INTERFACE Boost::boost)
if(OpenMP_CXX_FOUND)
  target_link_libraries(storm INTERFACE OpenMP::OpenMP_CXX)
endif()

if(STORM_WITH_MPI)
  target_compile_definitions(storm PUBLIC STORM_WITH_MPI MADVORO_WITH_MPI RICH_MPI SPATIAL_DS_WITH_MPI __WITH_MPI)
  target_link_libraries(storm PUBLIC MPI::MPI_CXX)
endif()

if(STORM_WITH_IBV)
  target_compile_definitions(storm PUBLIC __WITH_IBV)
  target_link_libraries(storm PUBLIC ${IBVERBS_LIBRARY})
endif()

if(STORM_WITH_VTK)
  target_compile_definitions(storm PUBLIC MADVORO_WITH_VTK)
  target_link_libraries(storm PUBLIC ${VTK_LIBRARIES})
endif()

if(STORM_WITH_HDF5)
  target_compile_definitions(storm PUBLIC STORM_WITH_HDF5)
  target_include_directories(storm PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(storm PUBLIC ${HDF5_LIBRARIES})
endif()

# ---------- Examples ----------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND STORM_BUILD_EXAMPLES)
  include(CTest)
endif()

if(STORM_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---------- Install & export (skip when embedded) ----------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/storm
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
    PATTERN ".git" EXCLUDE
    PATTERN "examples" EXCLUDE
    PATTERN "cmake" EXCLUDE
  )

  install(TARGETS storm EXPORT STORMTargets)
  install(EXPORT STORMTargets
    NAMESPACE STORM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/STORM
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/STORMConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/STORMConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/STORM
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/STORMConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/STORMConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/STORMConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/STORM
  )
endif()
