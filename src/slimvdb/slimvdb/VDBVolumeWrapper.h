// MIT License
//
// Copyright (c) 2025 Anja Sheppard, University of Michigan
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

// This class is a wrapper for the templatd VDBVolume class. 
// IT IS ONLY USED IN THE PYTHON BINDINGS TO MAKE USER CODING MORE STRAIGHTFORWARD.

#pragma once

#include <variant>
#include <memory>
#include <vector>
#include <functional>

#include "VDBVolume.h"

namespace slimvdb {

/// @brief Wrapper for VDBVolume class 
// Used only in the python bindings to avoid templated pything bindings
class VDBVolumeWrapper {
public:
    VDBVolumeWrapper(float voxel_size, float sdf_trunc, bool space_carving, float min_weight, Language language);

    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const Eigen::Vector3d& origin);

    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const Eigen::Vector3d& origin,
                   float weight);

    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<uint32_t>& labels,
                   const Eigen::Vector3d& origin);

    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<std::vector<float>>& labels,
                   const Eigen::Vector3d& origin);

#ifdef PYOPENVDB_SUPPORT
    void Integrate(openvdb::FloatGrid::Ptr grid);
    void Integrate(openvdb::FloatGrid::Ptr grid, float weight);
#endif

    void Render(const std::vector<double>& origin_vec,
                const std::vector<double>& rot_quat_vec,
                int index,
                int width,
                int height,
                float min_range,
                float max_range,
                float p_threshold);

    void UpdateTSDF(const float& sdf, const std::vector<int>& ijk);

    void UpdateTSDF(const float& sdf, const std::vector<int>& ijk,
                    const std::function<float(float)>& weighting_function);

    openvdb::FloatGrid::Ptr Prune(float min_weight) const;

    std::shared_ptr<openvdb::FloatGrid> tsdf() const;
    std::shared_ptr<openvdb::FloatGrid> weights() const;

#ifdef PYOPENVDB_SUPPORT
    std::shared_ptr<decltype(std::declval<VDBVolume<OPEN>>().semantics_)> semantics() const;
#endif

    float voxel_size_;
    float sdf_trunc_;
    bool space_carving_;
    Language language_;

private:
    std::variant<std::shared_ptr<VDBVolume<CLOSED>>, std::shared_ptr<VDBVolume<OPEN>>> volume_;
};

} // namespace slimvdb