
cmake_minimum_required(VERSION 3.0)

# Basic settings
# ==============

project(FrictionQPotSpringBlock)

option(BUILD_TESTS "${PROJECT_NAME} Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME} Build examples" OFF)

# Version
# =======

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/FrictionQPotSpringBlock/config.h"
             _FRICTIONQPOTSPRINGBLOCK_version_defines
     REGEX "#define FRICTIONQPOTSPRINGBLOCK_VERSION_(MAJOR|MINOR|PATCH)")

foreach(ver ${_FRICTIONQPOTSPRINGBLOCK_version_defines})
    if(ver MATCHES "#define FRICTIONQPOTSPRINGBLOCK_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(_FRICTIONQPOTSPRINGBLOCK_VERSION_${CMAKE_MATCH_1}
            "${CMAKE_MATCH_2}"
            CACHE INTERNAL "")
    endif()
endforeach()

set(_FRICTIONQPOTSPRINGBLOCK_VERSION ${_FRICTIONQPOTSPRINGBLOCK_VERSION_MAJOR}.${_FRICTIONQPOTSPRINGBLOCK_VERSION_MINOR}.${_FRICTIONQPOTSPRINGBLOCK_VERSION_PATCH})

message(STATUS "Building FrictionQPotSpringBlock v${_FRICTIONQPOTSPRINGBLOCK_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(QPot REQUIRED)
find_package(fmt REQUIRED)

add_library(FrictionQPotSpringBlock INTERFACE)

target_include_directories(FrictionQPotSpringBlock INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(FrictionQPotSpringBlock INTERFACE xtensor GooseFEM QPot fmt::fmt)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

install(TARGETS FrictionQPotSpringBlock EXPORT FrictionQPotSpringBlock-targets)

install(
    EXPORT FrictionQPotSpringBlock-targets
    FILE FrictionQPotSpringBlockTargets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FrictionQPotSpringBlock")

set(_FRICTIONQPOTSPRINGBLOCK ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/FrictionQPotSpringBlockConfigVersion.cmake"
    VERSION ${_FRICTIONQPOTSPRINGBLOCK_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_FRICTIONQPOTSPRINGBLOCK})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/FrictionQPotSpringBlockConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/FrictionQPotSpringBlockConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/FrictionQPotSpringBlock")

# Add builds
# ==========

include("FrictionQPotSpringBlockConfig.cmake")

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
