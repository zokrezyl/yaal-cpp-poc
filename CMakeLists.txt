cmake_minimum_required(VERSION 3.16)
project(yaal-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for maximum performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Benchmark executable
add_executable(yaal_benchmark src/benchmark.cpp)
target_compile_options(yaal_benchmark PRIVATE -mavx2 -mbmi)

# Make sure we use release mode by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
