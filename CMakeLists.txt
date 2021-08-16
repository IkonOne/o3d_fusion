cmake_minimum_required(VERSION 3.21)
project(O3DF)

find_package(OpenCV 4.5.2 EXACT REQUIRED)
find_package(depthai CONFIG REQUIRED)

include(CMakeIncludes.txt)

add_subdirectory(samples)