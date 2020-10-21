cmake_minimum_required(VERSION 3.12)

project(tinylr)

add_library(tinylr INTERFACE)
target_include_directories(tinylr INTERFACE ${CMAKE_SOURCE_DIR}/include)

add_executable(test test.cpp)
target_link_libraries(test tinylr)

add_executable(bench benchmark.cpp)
target_link_libraries(bench tinylr)

target_compile_definitions(bench PUBLIC DIAG_INVERT=true LR_ITERATIONS=1000000 PIVOT_STRATEGY=absmax STATIC_DIM=3)