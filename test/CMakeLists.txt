
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotSpringBlock-test)
    find_package(FrictionQPotSpringBlock REQUIRED CONFIG)
endif()

set(WARNINGS ON)
set(ASSERT ON)
set(DEBUG ON)

set(CMAKE_BUILD_TYPE Release)

set(testname "unit-tests")

find_package(Catch2 REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd QUIET)

add_executable(${testname}
    main.cpp
    Line1d.cpp)

target_link_libraries(${testname} PRIVATE
    Catch2::Catch2
    FrictionQPotSpringBlock
    FrictionQPotSpringBlock::compiler_warnings)

if(ASSERT)
    target_link_libraries(${testname} PRIVATE FrictionQPotSpringBlock::assert QPot::assert)
endif()

if(DEBUG)
    target_link_libraries(${testname} PRIVATE FrictionQPotSpringBlock::debug)
endif()

if(xsimd_FOUND)
    target_link_libraries(${testname} PRIVATE xtensor::optimize xtensor::use_xsimd)
endif()

add_test(NAME ${testname} COMMAND ${testname})
