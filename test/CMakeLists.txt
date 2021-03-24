
cmake_minimum_required(VERSION 3.0)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotSpringBlock-test)
    find_package(FrictionQPotSpringBlock REQUIRED CONFIG)
endif()

set(WARNINGS ON)
set(ASSERT ON)
set(DEBUG ON)
option(XSIMD "Use xsimd optimisations" ON)

set(CMAKE_BUILD_TYPE Release)

set(test_name "unit-tests")

find_package(Catch2 REQUIRED)
find_package(xtensor REQUIRED)

add_executable(${test_name}
    main.cpp
    Line1d.cpp)

target_link_libraries(${test_name} PRIVATE
    Catch2::Catch2
    FrictionQPotSpringBlock
    FrictionQPotSpringBlock::compiler_warnings)

if(ASSERT)
    target_link_libraries(${test_name} PRIVATE FrictionQPotSpringBlock::assert QPot::assert)
endif()

if(DEBUG)
    target_link_libraries(${test_name} PRIVATE FrictionQPotSpringBlock::debug)
endif()

if(XSIMD)
    find_package(xtensor REQUIRED)
    find_package(xsimd REQUIRED)
    target_link_libraries(${test_name} PRIVATE xtensor::optimize xtensor::use_xsimd)
endif()

add_test(NAME ${test_name} COMMAND ${test_name})
