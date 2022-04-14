cmake_minimum_required(VERSION 3.21)
project(intrinsic)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-mavx")

add_executable(intrinsic main.cpp)
