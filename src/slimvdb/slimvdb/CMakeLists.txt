# MIT License
#
# Copyright (c) 2025 Anja Sheppard, University of Michigan
# Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.18)
project(SLIMVDB VERSION 0.1.6 LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA)

set(SLIMVDB_NCLASSES 512 CACHE STRING "Number of semantic classes")
set(SLIMVDB_LANGUAGE "OPEN" CACHE STRING "Language of semantic classes (OPEN or CLOSED)")

if(SLIMVDB_LANGUAGE STREQUAL "OPEN")
  set(LANGUAGE_ENUM OPEN)
  set(LANGUAGE_STR "slimvdb::Language::OPEN")
else()
  set(LANGUAGE_ENUM CLOSED)
  set(LANGUAGE_STR "slimvdb::Language::CLOSED")
endif()

# --- Configure Config.h ---
set(CONFIG_H_IN ${CMAKE_CURRENT_SOURCE_DIR}/Config.h.in)
set(CONFIG_H_OUT ${CMAKE_CURRENT_SOURCE_DIR}/Config.h)

configure_file(${CONFIG_H_IN} ${CONFIG_H_OUT} @ONLY)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "Configuring SLIMVDB with LANGUAGE=${LANGUAGE_STR}, SLIMVDB_NCLASSES=${SLIMVDB_NCLASSES}")

# Set CUDA compilation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

add_library(slimvdb STATIC)
add_library(SLIMVDB::slimvdb ALIAS slimvdb)

target_include_directories(slimvdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

cuda_add_library(cuda_lib STATIC nanovdb_utils/nanovdb.cu)
add_library(SLIMVDB::cuda_lib ALIAS cuda_lib)
target_include_directories(cuda_lib PRIVATE
  /usr/local/cuda/include
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nanovdb_utils>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 14)


if(NOT TARGET Eigen3::Eigen OR NOT TARGET OpenVDB::openvdb)
  message(STATUS "Building only SLIMVDB C++ module")
  include(GNUInstallDirs)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/OpenVDB")
  set(SLIMVDB_STANDALONE ON)
  find_package(Eigen3 REQUIRED)
  find_package(OpenVDB REQUIRED)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_sources(slimvdb PRIVATE VDBVolume.cpp VDBVolumeWrapper.cpp)
target_compile_features(slimvdb PUBLIC cxx_std_11)
target_compile_options(slimvdb PRIVATE -Wall -Wextra)
target_compile_definitions(slimvdb PRIVATE "NANOVDB_USE_OPENVDB")
set_target_properties(slimvdb PROPERTIES PUBLIC_HEADER "VDBVolume.h")
target_include_directories(slimvdb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} 
                                   PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(slimvdb PUBLIC Eigen3::Eigen OpenVDB::openvdb cuda_lib ${OpenCV_LIBS} ${TORCH_LIBRARIES})
install(FILES VDBVolume.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimvdb)
install(FILES MarchingCubesConst.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimvdb)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/Config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimvdb)
install(FILES utils/Utils.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/slimvdb/utils)

# NOTE: Installation is still work in progress, therefore we only support dev machines with all the 3rdparty
# dependencies locally installed
if((USE_SYSTEM_EIGEN3 AND USE_SYSTEM_OPENVDB) OR SLIMVDB_STANDALONE)
  include(cmake/Install.cmake)
endif()
