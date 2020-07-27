cmake_minimum_required(VERSION 3.14)
project(ekf_gps)
#这个是做什么的？
set(CMAKE_CXX_STANDARD 14)
include_directories("/usr/local/Cellar/eigen/3.3.7/include/eigen3")

add_executable(ekf_gps main.cpp run_fusion.cpp run_fusion.hpp fusion.cpp fusion.hpp geo_ned.hpp geo_ned.cpp ekf.cpp ekf.hpp utils.cpp utils.hpp)
#add_executable(ekf_gps main_test.cpp geo_ned.cpp geo_ned.hpp)