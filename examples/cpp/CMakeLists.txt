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

cmake_minimum_required(VERSION 3.17)
project(SLIMVDB_Examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_NVCC_EXECUTABLE /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_HOME /usr/local/cuda)
set(CUDA_HOME /usr/local/cuda)

add_compile_definitions(NCLASSES=26)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  find_package(SLIMVDB REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "-g") 

set(BOOST_VERSION_REQUIRED "1.80.0")

find_package(TBB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Pull dependencies used only for the C++ examples
include(3rdparty/find_dependencies.cmake)
add_subdirectory(datasets)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 --extended-lambda)  # Add --extended-lambda flag

add_executable(kitti_pipeline kitti_pipeline.cpp)
target_compile_features(kitti_pipeline PUBLIC cxx_std_17)
target_link_libraries(kitti_pipeline TBB::tbb SLIMVDB::slimvdb argparse::argparse fmt::fmt igl::core datasets yaml-cpp ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(kitti_pipeline PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(scenenet_pipeline scenenet_pipeline.cpp)
target_compile_features(scenenet_pipeline PUBLIC cxx_std_17)
target_link_libraries(scenenet_pipeline TBB::tbb SLIMVDB::slimvdb argparse::argparse fmt::fmt igl::core datasets yaml-cpp ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(scenenet_pipeline PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(realworld_pipeline realworld_pipeline.cpp)
target_compile_features(realworld_pipeline PUBLIC cxx_std_17)
target_link_libraries(realworld_pipeline TBB::tbb SLIMVDB::slimvdb argparse::argparse fmt::fmt igl::core datasets yaml-cpp ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
set_target_properties(realworld_pipeline PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
