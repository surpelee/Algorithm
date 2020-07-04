cmake_minimum_required(VERSION 3.16)
project(Algrorithm)

set(CMAKE_CXX_STANDARD 17)

include_directories(./include)
aux_source_directory(src DRC_SRC)

add_executable(Algrorithm main.cpp ${DRC_SRC})