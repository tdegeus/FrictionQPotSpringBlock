
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotSpringBlock-test)
    find_package(FrictionQPotSpringBlock REQUIRED CONFIG)
endif()

set(WARNINGS ON)
set(ASSERT ON)
set(DEBUG ON)
set(XSIMD OFF)

set(CMAKE_BUILD_TYPE Release)

find_package(Catch2 REQUIRED)
add_executable(main main.cpp Line1d.cpp)
target_link_libraries(main PRIVATE Catch2::Catch2 FrictionQPotSpringBlock)

if(WARNINGS)
    target_link_libraries(main PRIVATE FrictionQPotSpringBlock::compiler_warnings)
endif()

if(ASSERT)
    target_link_libraries(main PRIVATE FrictionQPotSpringBlock::assert)
endif()

if(DEBUG)
    target_link_libraries(main PRIVATE FrictionQPotSpringBlock::debug)
endif()

if(XSIMD)
    find_package(xtensor REQUIRED)
    find_package(xsimd REQUIRED)
    target_link_libraries(main PRIVATE xtensor::optimize xtensor::use_xsimd)
endif()
