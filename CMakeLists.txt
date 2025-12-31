# Copyright (c) Qijizhifeng. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.15)
project(pypto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (Debug or Release)" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set up libbacktrace as an external project
include(ExternalProject)

set(LIBBACKTRACE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libbacktrace")
set(LIBBACKTRACE_INSTALL_DIR "${CMAKE_BINARY_DIR}/3rdparty/libbacktrace")
set(LIBBACKTRACE_BUILD_DIR "${CMAKE_BINARY_DIR}/3rdparty/libbacktrace/build")

# Create installation directories
file(MAKE_DIRECTORY ${LIBBACKTRACE_INSTALL_DIR}/include)
file(MAKE_DIRECTORY ${LIBBACKTRACE_INSTALL_DIR}/lib)

ExternalProject_Add(project_libbacktrace
    PREFIX ${LIBBACKTRACE_BUILD_DIR}
    SOURCE_DIR ${LIBBACKTRACE_SOURCE_DIR}
    BINARY_DIR ${LIBBACKTRACE_BUILD_DIR}
    CONFIGURE_COMMAND ${LIBBACKTRACE_SOURCE_DIR}/configure
    "--prefix=${LIBBACKTRACE_INSTALL_DIR}"
    --with-pic
    BUILD_COMMAND make -j${CMAKE_BUILD_PARALLEL_LEVEL}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS "${LIBBACKTRACE_INSTALL_DIR}/lib/libbacktrace.a"
    "${LIBBACKTRACE_INSTALL_DIR}/include/backtrace.h"
)

# Create imported target
add_library(libbacktrace STATIC IMPORTED)
set_target_properties(libbacktrace PROPERTIES
    IMPORTED_LOCATION ${LIBBACKTRACE_INSTALL_DIR}/lib/libbacktrace.a
    INTERFACE_INCLUDE_DIRECTORIES ${LIBBACKTRACE_INSTALL_DIR}/include
)
add_dependencies(libbacktrace project_libbacktrace)

# PyPTO core sources
set(PYPTO_SOURCES
    src/core/backtrace.cpp
    src/core/error.cpp
)

# Create the pybind11 module
pybind11_add_module(_pypto_core
    python/bindings.cpp
    ${PYPTO_SOURCES}
)

# Link with libbacktrace
target_include_directories(_pypto_core PRIVATE ${LIBBACKTRACE_INSTALL_DIR}/include)
target_link_libraries(_pypto_core PRIVATE libbacktrace)

if(APPLE)
    find_program(DSYMUTIL dsymutil)
    mark_as_advanced(DSYMUTIL)
    add_custom_command(
        TARGET _pypto_core
        POST_BUILD
        COMMAND ${DSYMUTIL} ARGS $<TARGET_FILE:_pypto_core>
        COMMENT "[COMMAND] dsymutil $<TARGET_FILE:_pypto_core>"
        VERBATIM
    )
    message(STATUS "running dsymutil on macos to generate debugging symbols for backtraces")
endif()

# Set output directory
set_target_properties(_pypto_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
