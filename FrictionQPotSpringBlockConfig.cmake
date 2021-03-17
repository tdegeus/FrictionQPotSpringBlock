# FrictionQPotSpringBlock cmake module
#
# This module sets the target:
#
#     FrictionQPotSpringBlock
#
# In addition, it sets the following variables:
#
#     FrictionQPotSpringBlock_FOUND - true if the library is found
#     FrictionQPotSpringBlock_VERSION - the library's version
#     FrictionQPotSpringBlock_INCLUDE_DIRS - directory containing the library's headers
#
# The following support targets are defined to simplify things:
#
#     FrictionQPotSpringBlock::compiler_warnings - enable compiler warnings
#     FrictionQPotSpringBlock::assert - enable library assertions
#     FrictionQPotSpringBlock::debug - enable all assertions (slow)

include(CMakeFindDependencyMacro)

# Define target "FrictionQPotSpringBlock"

if(NOT TARGET FrictionQPotSpringBlock)
    include("${CMAKE_CURRENT_LIST_DIR}/FrictionQPotSpringBlockTargets.cmake")
endif()

# Define "FrictionQPotSpringBlock_INCLUDE_DIRS"

get_target_property(
    FrictionQPotSpringBlock_INCLUDE_DIRS
    FrictionQPotSpringBlock
    INTERFACE_INCLUDE_DIRECTORIES)

# Find dependencies

find_dependency(xtensor)
find_dependency(QPot)

# Define support target "FrictionQPotSpringBlock::compiler_warnings"

if(NOT TARGET FrictionQPotSpringBlock::compiler_warnings)
    add_library(FrictionQPotSpringBlock::compiler_warnings INTERFACE IMPORTED)
    if(MSVC)
        set_property(
            TARGET FrictionQPotSpringBlock::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            /W4)
    else()
        set_property(
            TARGET FrictionQPotSpringBlock::compiler_warnings
            PROPERTY INTERFACE_COMPILE_OPTIONS
            -Wall -Wextra -pedantic -Wno-unknown-pragmas)
    endif()
endif()

# Define support target "FrictionQPotSpringBlock::assert"

if(NOT TARGET FrictionQPotSpringBlock::assert)
    add_library(FrictionQPotSpringBlock::assert INTERFACE IMPORTED)
    set_property(
        TARGET FrictionQPotSpringBlock::assert
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        FRICTIONQPOTFEM_ENABLE_ASSERT)
endif()

# Define support target "FrictionQPotSpringBlock::debug"

if(NOT TARGET FrictionQPotSpringBlock::debug)
    add_library(FrictionQPotSpringBlock::debug INTERFACE IMPORTED)
    set_property(
        TARGET FrictionQPotSpringBlock::debug
        PROPERTY INTERFACE_COMPILE_DEFINITIONS
        XTENSOR_ENABLE_ASSERT
        QPOT_ENABLE_ASSERT
        FRICTIONQPOTFEM_ENABLE_ASSERT)
endif()
