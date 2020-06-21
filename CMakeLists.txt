cmake_minimum_required(VERSION 3.16)
project(dbscan)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

add_executable(dbscan main.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(test)