cmake_minimum_required(VERSION 3.16)
project(fpinn_black_scholes VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FPINN_USE_OPENMP "Parallelise the loss/gradient over collocation points with OpenMP" ON)
option(FPINN_NATIVE "Optimise for the build machine's CPU (-march=native)" OFF)
option(FPINN_BUILD_TESTS "Build the unit tests" ON)

add_library(fpinn
  src/special_functions.cpp
  src/caputo.cpp
  src/problem.cpp
  src/black_scholes.cpp
  src/fd_solver.cpp
  src/mlp.cpp
  src/optim.cpp
  src/fpinn.cpp
)
target_include_directories(fpinn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fpinn PRIVATE -Wall -Wextra -Wpedantic)
  if(FPINN_NATIVE)
    target_compile_options(fpinn PUBLIC -march=native)
  endif()
endif()

if(FPINN_USE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fpinn PUBLIC OpenMP::OpenMP_CXX)
  else()
    message(STATUS "OpenMP not found - building single-threaded")
  endif()
endif()
if(NOT TARGET OpenMP::OpenMP_CXX AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fpinn PRIVATE -Wno-unknown-pragmas)
endif()

add_executable(fpinn_bs apps/fpinn_bs.cpp)
target_link_libraries(fpinn_bs PRIVATE fpinn)

add_executable(fd_bs apps/fd_bs.cpp)
target_link_libraries(fd_bs PRIVATE fpinn)

if(FPINN_BUILD_TESTS)
  enable_testing()
  foreach(t test_special test_caputo test_mlp test_fd_solver test_fpinn)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE fpinn)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
