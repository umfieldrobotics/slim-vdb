// MIT License
//
// Copyright (c) 2025 Anja Sheppard, University of Michigan
// Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// OpenVDB
#include <openvdb/openvdb.h>

// pybind11
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// std stuff
#include <Eigen/Core>
#include <memory>
#include <vector>

#ifdef PYOPENVDB_SUPPORT
#include "pyopenvdb.h"
#endif

#include "stl_vector_eigen.h"
#include "slimvdb/VDBVolumeWrapper.h"
#include "slimvdb/utils/Utils.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);

namespace py = pybind11;
using namespace py::literals;

namespace slimvdb {

PYBIND11_MODULE(slimvdb_pybind, m) {
    py::enum_<slimvdb::Language>(m, "Language")
        .value("CLOSED", slimvdb::Language::CLOSED)
        .value("OPEN", slimvdb::Language::OPEN)
        .export_values();

    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_VectorEigen3d", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    auto vector3ivector = pybind_eigen_vector_of_vector<Eigen::Vector3i>(
        m, "_VectorEigen3i", "std::vector<Eigen::Vector3i>",
        py::py_array_to_vectors_int<Eigen::Vector3i>);

    auto vectorUint32 = pybind_vector_of_uint32<uint32_t>(
        m, "_VectorUint32", "std::vector<uint32_t>");

    py::class_<VDBVolumeWrapper, std::shared_ptr<VDBVolumeWrapper>> vdb_volume(
        m, "_VDBVolumeWrapper",
        "This is the low level C++ bindings, all the methods and "
        "constructor defined within this module (starting with a ``_`` "
        "should not be used. Please refer to the python Processor class to "
        "check how to use the API");
    vdb_volume
        .def(py::init<float, float, bool, float, slimvdb::Language>(), "voxel_size"_a, "sdf_trunc"_a,
             "space_carving"_a, "min_weight"_a, "language"_a)
        // Basic integrate: points + origin
        .def(
            "_integrate",
            [](VDBVolumeWrapper& self, const std::vector<Eigen::Vector3d>& points,
               const Eigen::Vector3d& origin) {                                
                self.Integrate(points, origin);
            },
            "points"_a, "origin"_a)
        // Integrate: points + origin + weight
        .def(
            "_integrate",
            [](VDBVolumeWrapper& self, const std::vector<Eigen::Vector3d>& points,
               const Eigen::Vector3d& origin, float weight) {
                self.Integrate(points, origin, weight);
            },
            "points"_a, "origin"_a, "weight"_a)
        // Integrate: points + uint32 labels + origin
        .def(
            "_integrate",
            [](VDBVolumeWrapper& self, const std::vector<Eigen::Vector3d>& points,
               const std::vector<uint32_t>& labels, const Eigen::Vector3d& origin) {
                self.Integrate(points, labels, origin);
            },
            "points"_a, "labels"_a, "origin"_a)
        // Integrate: points + vector<vector<float>> labels + origin (for OPEN language)
        .def(
            "_integrate",
            [](VDBVolumeWrapper& self, const std::vector<Eigen::Vector3d>& points,
               const std::vector<std::vector<float>>& labels, const Eigen::Vector3d& origin) {
                self.Integrate(points, labels, origin);
            },
            "points"_a, "labels"_a, "origin"_a)
#ifdef PYOPENVDB_SUPPORT
        // Integrate: grid
        .def(
            "_integrate",
            [](VDBVolumeWrapper& self, openvdb::FloatGrid::Ptr grid) {
                self.Integrate(grid);
            },
            "grid"_a)
        // Integrate: grid + weight
        .def(
            "_integrate",
            [](VDBVolumeWrapper& self, openvdb::FloatGrid::Ptr grid, float weight) {
                self.Integrate(grid, weight);
            },
            "grid"_a, "weight"_a)
#endif
        // Render
        .def(
            "_render",
            [](VDBVolumeWrapper& self, const std::vector<double>& origin_vec,
               const std::vector<double>& rot_quat_vec, int index, int render_img_width,
               int render_img_height, float min_range, float max_range, float p_threshold) {
                self.Render(origin_vec, rot_quat_vec, index, render_img_width,
                           render_img_height, min_range, max_range, p_threshold);
            },
            "origin_vec"_a, "rot_quat_vec"_a, "index"_a, "render_img_width"_a,
            "render_img_height"_a, "min_range"_a, "max_range"_a, "p_threshold"_a)
        // UpdateTSDF: with weighting function
        .def(
            "_update_tsdf",
            [](VDBVolumeWrapper& self, const float& sdf, const std::vector<int>& ijk,
               const std::function<float(float)>& weighting_function) {
                self.UpdateTSDF(sdf, ijk, weighting_function);
            },
            "sdf"_a, "ijk"_a, "weighting_function"_a)
        // UpdateTSDF: without weighting function
        .def(
            "_update_tsdf",
            [](VDBVolumeWrapper& self, const float& sdf, const std::vector<int>& ijk) {
                self.UpdateTSDF(sdf, ijk);
            },
            "sdf"_a, "ijk"_a)
        // Extract VDB grids to file
        .def(
            "_extract_vdb_grids",
            [](const VDBVolumeWrapper& self, const std::string& filename) {
                openvdb::io::File(filename).write({self.tsdf(), self.weights()});
            },
            "filename"_a)
#ifndef PYOPENVDB_SUPPORT
        .def_property_readonly_static("PYOPENVDB_SUPPORT_ENABLED", [](py::object) { return false; })
#else
        .def_property_readonly_static("PYOPENVDB_SUPPORT_ENABLED", [](py::object) { return true; })
        .def("_prune", &VDBVolumeWrapper::Prune, "min_weight"_a)
        .def_property_readonly("_tsdf", &VDBVolumeWrapper::tsdf)
        .def_property_readonly("_weights", &VDBVolumeWrapper::weights)
        .def_property_readonly("_semantics", &VDBVolumeWrapper::semantics)
#endif
        .def_readwrite("_voxel_size", &VDBVolumeWrapper::voxel_size_)
        .def_readwrite("_sdf_trunc", &VDBVolumeWrapper::sdf_trunc_)
        .def_readwrite("_space_carving", &VDBVolumeWrapper::space_carving_)
        .def_readonly("_language", &VDBVolumeWrapper::language_);
}

}  // namespace slimvdb