
cmake_minimum_required(VERSION 3.16)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(FrictionQPotSpringBlock-examples)
    find_package(FrictionQPotSpringBlock REQUIRED CONFIG)
endif()

set(WARNINGS ON)
set(ASSERT OFF)

set(CMAKE_BUILD_TYPE Release)

file(GLOB APP_SOURCES *.cpp)

foreach(testsourcefile ${APP_SOURCES})

    string(REPLACE ".cpp" "" testname ${testsourcefile})
    get_filename_component(testname ${testname} NAME)

    add_executable(${testname} ${testsourcefile})

    set(HIGHFIVE_USE_BOOST 0)
    set(HIGHFIVE_USE_XTENSOR 1)

    find_package(xtensor REQUIRED)
    find_package(xsimd REQUIRED)
    find_package(HighFive REQUIRED)
    find_package(fmt REQUIRED)

    target_link_libraries(${testname} PUBLIC
        FrictionQPotSpringBlock
        xtensor::optimize
        xtensor::use_xsimd
        fmt::fmt
        HighFive)

    if(WARNINGS)
        target_link_libraries(${testname} PRIVATE FrictionQPotSpringBlock::compiler_warnings)
    endif()

    if(ASSERT)
        target_link_libraries(${testname} PRIVATE FrictionQPotSpringBlock::assert)
    endif()

endforeach()

