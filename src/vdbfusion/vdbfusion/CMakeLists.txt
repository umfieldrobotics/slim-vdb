# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
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
project(VDBFusion VERSION 0.1.6 LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA)

# Include nanovdb common.h
include_directories(/usr/local/cuda/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(/home/anjashep-frog-lab/Research/vdbfusion_mapping/vdbfusion/vdbfusion/nanovdb_utils)

# Set CUDA compilation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extended-lambda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

add_library(vdbfusion STATIC)
add_library(VDBFusion::vdbfusion ALIAS vdbfusion)
cuda_add_library(cuda_lib STATIC nanovdb_utils/nanovdb.cu)
add_library(VDBFusion::cuda_lib ALIAS cuda_lib)

# In case we are building this as a standalone project, we should look here the dependencies
if(NOT TARGET Eigen3::Eigen OR NOT TARGET OpenVDB::openvdb)
  message(STATUS "Building only VDBFusion C++ module")
  include(GNUInstallDirs)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/OpenVDB")
  set(VDBFUSION_STANDALONE ON)
  find_package(Eigen3 REQUIRED)
  find_package(OpenVDB REQUIRED)
endif()

include_directories( ${OpenCV_INCLUDE_DIRS} )

target_sources(vdbfusion PRIVATE MarchingCubes.cpp VDBVolume.cpp)
target_compile_features(vdbfusion PUBLIC cxx_std_11)
target_compile_options(vdbfusion PRIVATE -Wall -Wextra)
target_compile_definitions(vdbfusion PRIVATE "NANOVDB_USE_OPENVDB")
set_target_properties(vdbfusion PROPERTIES PUBLIC_HEADER "VDBVolume.h")
target_include_directories(
  vdbfusion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(vdbfusion PUBLIC Eigen3::Eigen OpenVDB::openvdb cuda_lib ${OpenCV_LIBS})

# NOTE: Installation is still work in progress, therefore we only support dev machines with all the 3rdparty
# dependencies locally installed
if((USE_SYSTEM_EIGEN3 AND USE_SYSTEM_OPENVDB) OR VDBFUSION_STANDALONE)
  include(cmake/Install.cmake)
endif()
