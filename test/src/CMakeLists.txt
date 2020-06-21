cmake_minimum_required(VERSION 3.5)

add_executable(apg_test test.cpp)

target_link_libraries(apg_test gtest gtest_main)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O0")