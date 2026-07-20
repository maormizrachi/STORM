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

  # --- Compiler flags (aligned with RICH/config/compiler_flags.cmake) ---

  # Release: -O2 (not -O3 -- avoids aggressive FP reassociation), minimal debug info
  set(CMAKE_C_FLAGS_RELEASE   "-O2 -DNDEBUG -g1" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -g1" CACHE STRING "" FORCE)

  # Debug: no optimisation, full debug info
  set(CMAKE_C_FLAGS_DEBUG   "-O0 -g3 -gdwarf-3" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -gdwarf-3 -D_GLIBCXX_ASSERTIONS" CACHE STRING "" FORCE)

  # Common flags for all build types
  string(APPEND CMAKE_CXX_FLAGS " -fno-omit-frame-pointer")
  string(APPEND CMAKE_CXX_FLAGS " -fstrict-aliasing")
  string(APPEND CMAKE_CXX_FLAGS " -fno-common")
  string(APPEND CMAKE_CXX_FLAGS " -rdynamic")
  string(APPEND CMAKE_CXX_FLAGS " -DOMPI_SKIP_MPICXX")

  string(APPEND CMAKE_C_FLAGS " -fno-omit-frame-pointer")
  string(APPEND CMAKE_C_FLAGS " -fstrict-aliasing")
  string(APPEND CMAKE_C_FLAGS " -fno-common")
  string(APPEND CMAKE_C_FLAGS " -rdynamic")
endif()

# ---------- Options ----------
option(STORM_WITH_MPI    "Build with MPI support"  OFF)
option(STORM_NO_IBV      "Disable IBV auto-detection (InfiniBand Verbs)" OFF)
option(STORM_NO_OFI      "Disable OFI auto-detection (OpenFabrics Interfaces / libfabric)" OFF)
option(STORM_WITH_VTK    "Build with VTK support for mesh output (requires MadVoro VTK headers)" OFF)
option(STORM_WITH_HDF5   "Build with HDF5 support" OFF)
option(STORM_BUILD_EXAMPLES "Build example programs" OFF)
option(STORM_BUILD_TESTS "Build focused STORM unit tests" OFF)

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

# --- Helper: derive prefix hints from LD_LIBRARY_PATH ---
# CMake does not search LD_LIBRARY_PATH, so we strip trailing /lib or /lib64
# from each entry and offer the parent directories as HINTS.
set(_ld_prefixes "")
if(DEFINED ENV{LD_LIBRARY_PATH})
  string(REPLACE ":" ";" _ld_dirs "$ENV{LD_LIBRARY_PATH}")
  foreach(_d IN LISTS _ld_dirs)
    get_filename_component(_parent "${_d}" DIRECTORY)
    list(APPEND _ld_prefixes "${_parent}")
  endforeach()
  list(REMOVE_DUPLICATES _ld_prefixes)
endif()

set(STORM_WITH_IBV OFF)
if(STORM_WITH_MPI AND NOT STORM_NO_IBV)
  find_library(IBVERBS_LIBRARY ibverbs HINTS ${_ld_prefixes} PATH_SUFFIXES lib lib64)
  if(IBVERBS_LIBRARY)
    set(STORM_WITH_IBV ON)
    message(STATUS "Found libibverbs: ${IBVERBS_LIBRARY} — IBV support enabled")
  else()
    message(STATUS "libibverbs not found — IBV support disabled (P2P/MPI-RMA fallback available)")
  endif()
endif()
if(STORM_NO_IBV)
  message(STATUS "IBV explicitly disabled via STORM_NO_IBV")
endif()

set(STORM_WITH_OFI OFF)
if(STORM_WITH_MPI AND NOT STORM_NO_OFI)
  set(_ofi_hints "")
  if(DEFINED ENV{FABRIC_ROOT})
    list(APPEND _ofi_hints "$ENV{FABRIC_ROOT}")
  endif()
  if(DEFINED ENV{FABRIC_DIR})
    list(APPEND _ofi_hints "$ENV{FABRIC_DIR}")
  endif()
  # Try pkg-config (Cray modules set PKG_CONFIG_PATH but not FABRIC_ROOT)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(_OFI_PC QUIET fabric)
    if(_OFI_PC_FOUND AND _OFI_PC_PREFIX)
      list(APPEND _ofi_hints "${_OFI_PC_PREFIX}")
    endif()
  endif()
  # Also try prefixes derived from LD_LIBRARY_PATH
  list(APPEND _ofi_hints ${_ld_prefixes})
  find_library(FABRIC_LIBRARY fabric HINTS ${_ofi_hints} PATH_SUFFIXES lib lib64)
  find_path(FABRIC_INCLUDE_DIR rdma/fabric.h HINTS ${_ofi_hints} PATH_SUFFIXES include)
  if(FABRIC_LIBRARY AND FABRIC_INCLUDE_DIR)
    set(STORM_WITH_OFI ON)
    message(STATUS "Found libfabric: ${FABRIC_LIBRARY} — OFI support enabled")
  else()
    message(STATUS "libfabric not found — OFI support disabled")
  endif()
endif()
if(STORM_NO_OFI)
  message(STATUS "OFI explicitly disabled via STORM_NO_OFI")
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
  if(STORM_WITH_OFI AND EXISTS "${_deps}/rma/OFIContext.cpp")
    list(APPEND STORM_SOURCES "${_deps}/rma/OFIContext.cpp")
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

if(STORM_WITH_OFI)
  target_compile_definitions(storm PUBLIC __WITH_OFI)
  target_include_directories(storm PUBLIC ${FABRIC_INCLUDE_DIR})
  target_compile_options(storm PUBLIC
    "$<$<COMPILE_LANGUAGE:C>:-I${FABRIC_INCLUDE_DIR}>"
    "$<$<COMPILE_LANGUAGE:CXX>:-I${FABRIC_INCLUDE_DIR}>"
  )
  target_link_libraries(storm PUBLIC ${FABRIC_LIBRARY})
endif()

message(STATUS "STORM RDMA backends: IBV=${STORM_WITH_IBV}, OFI=${STORM_WITH_OFI}")

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

if(STORM_BUILD_TESTS AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(CTest)
  if(BUILD_TESTING)
    add_executable(storm_ddmc_math_tests tests/DDMCMathTests.cpp)
    target_link_libraries(storm_ddmc_math_tests PRIVATE storm)
    add_test(NAME storm_ddmc_math_tests COMMAND storm_ddmc_math_tests)

    add_executable(storm_compton_observer_tests tests/ComptonObserverTests.cpp)
    target_link_libraries(storm_compton_observer_tests PRIVATE storm)
    add_test(NAME storm_compton_observer_tests COMMAND storm_compton_observer_tests)
  endif()
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
