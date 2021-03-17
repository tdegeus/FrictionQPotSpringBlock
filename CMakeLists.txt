cmake_minimum_required(VERSION 3.1...3.19)

# Basic settings
# ==============

project(FrictionQPotSpringBlock)

option(BUILD_TESTS "${PROJECT_NAME}: Build tests" OFF)
option(BUILD_EXAMPLES "${PROJECT_NAME}: Build examples" OFF)
option(BUILD_DOCS "${PROJECT_NAME}: Build docs (include dependencies)" OFF)
option(CHECK_DOCS "${PROJECT_NAME}: Check docs (only this library, throw error for every warning)" OFF)

# Version
# =======

execute_process(
    COMMAND python -c "from setuptools_scm import get_version; print(get_version())"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE FRICTIONQPOTSPRINGBLOCK_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "Building ${PROJECT_NAME} ${FRICTIONQPOTSPRINGBLOCK_VERSION}")

# Set target
# ==========

find_package(xtensor REQUIRED)
find_package(GooseFEM REQUIRED)
find_package(QPot REQUIRED)

add_library(FrictionQPotSpringBlock INTERFACE)

target_include_directories(FrictionQPotSpringBlock INTERFACE
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

target_link_libraries(FrictionQPotSpringBlock INTERFACE xtensor GooseFEM QPot)

# Installation
# ============

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION include)

configure_file("include/${PROJECT_NAME}/version.h"
               "${CMAKE_CURRENT_BINARY_DIR}/version.h"
                @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/version.h"
        DESTINATION "include/${PROJECT_NAME}/")

install(TARGETS FrictionQPotSpringBlock EXPORT FrictionQPotSpringBlock-targets)

install(
    EXPORT FrictionQPotSpringBlock-targets
    FILE "${PROJECT_NAME}Targets.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

set(_FRICTIONQPOTSPRINGBLOCK ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${FRICTIONQPOTSPRINGBLOCK_VERSION}
    COMPATIBILITY AnyNewerVersion)

set(CMAKE_SIZEOF_VOID_P ${_FRICTIONQPOTSPRINGBLOCK})

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Add builds
# ==========

include("${PROJECT_NAME}Config.cmake")
include(CTest)

# Tests

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Examples

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Check docs: only this library, throw error for every warning

if(CHECK_DOCS)
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_EXCLUDE_SYMBOLS detail)
    set(DOXYGEN_CASE_SENSE_NAMES YES)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED YES)
    set(DOXYGEN_WARN_AS_ERROR YES)

    set(DOXYGEN_ALIASES
        "rst=\\verbatim embed:rst:leading-asterisk"
        "endrst=\\endverbatim"
        "license=@par License:")

    doxygen_add_docs(check_docs "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

# Build docs: include dependencies

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    find_package(GMatTensor REQUIRED)

    get_target_property(QPot_INCLUDE QPot INTERFACE_INCLUDE_DIRECTORIES)

    set(DOXYGEN_EXCLUDE_SYMBOLS detail)
    set(DOXYGEN_CASE_SENSE_NAMES YES)
    set(DOXYGEN_USE_MATHJAX YES)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
    set(DOXYGEN_MACRO_EXPANSION YES)
    set(DOXYGEN_SOURCE_BROWSER YES)
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

    set(DOXYGEN_ALIASES
        "rst=\\verbatim embed:rst:leading-asterisk"
        "endrst=\\endverbatim"
        "license=@par License:")

    set(DOXYGEN_STRIP_FROM_INC_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${QPot_INCLUDE}")

    set(DOXYGEN_STRIP_FROM_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${QPot_INCLUDE}")

    doxygen_add_docs(docs
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${QPot_INCLUDE}/QPot")
endif()
