cmake_minimum_required(VERSION 3.20)
project(QuiltToNative LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -fmodules-ts")
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

set(SOURCES 
main.cpp
)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)
configure_file("${CMAKE_SOURCE_DIR}/kernel.cl" "${CMAKE_CURRENT_BINARY_DIR}/kernel.cl" COPYONLY)
