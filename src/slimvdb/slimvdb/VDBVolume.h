/*
 * MIT License
 *
 * # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <openvdb/openvdb.h>

#include <Eigen/Core>
#include <functional>
#include <tuple>

namespace slimvdb {

class VDBVolume {
public:
    VDBVolume(float voxel_size, float sdf_trunc, bool space_carving, float min_weight);
    ~VDBVolume() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud and its labels into the current
    /// tsdf_ volume.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<uint32_t>& labels,
                   const Eigen::Vector3d& origin,
                   const std::function<float(float)>& weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const Eigen::Vector3d& origin,
                   const std::function<float(float)>& weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void inline Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Matrix4d& extrinsics,
                          const std::function<float(float)>& weighting_function) {
        const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
        Integrate(points, origin, weighting_function);
    }

    /// @brief Integrates a new (globally aligned) PointCloud into the current
    /// tsdf_ volume.
    void inline Integrate(const std::vector<Eigen::Vector3d>& points,
                          const std::vector<uint32_t>& labels,
                          const Eigen::Matrix4d& extrinsics,
                          const std::function<float(float)>& weighting_function) {
        const Eigen::Vector3d& origin = extrinsics.block<3, 1>(0, 3);
        Integrate(points, labels, origin, weighting_function);
    }

    /// @brief Integrate incoming TSDF grid inside the current volume using the TSDF equations
    void Integrate(openvdb::FloatGrid::Ptr grid,
                   const std::function<float(float)>& weighting_function);

    /// @brief Render volume from viewpoint of origin and display image
    void Render(const std::vector<double> origin_vec,
                const std::vector<double> rot_quat_vec,
                const int index,
                const int render_img_width,
                const int render_img_height,
                const float min_range,
                const float max_range);

    /// @brief Fuse a new given sdf value at the given voxel location, thread-safe
    void UpdateTSDF(const float& sdf,
                    const openvdb::Coord& voxel,
                    const std::function<float(float)>& weighting_function);

    /// @brief Prune TSDF grids, ideal utility to cleanup a D(x) volume before exporting it
    openvdb::FloatGrid::Ptr Prune(float min_weight) const;

    /// @brief Extracts a TriangleMesh as the iso-surface in the actual volume
    [[nodiscard]] std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>, std::vector<int>>
    ExtractTriangleMesh(bool fill_holes = true, float min_weight = 0.5) const;

    /// @brief Extracts a PointCloud from the actual volume
    [[nodiscard]] std::pair<std::vector<Eigen::Vector3d>, std::vector<int>>
    ExtractPointCloud(bool fill_holes = true, float min_weight = 0.5) const;

public:
    /// VDBVolume public properties
    float voxel_size_;
    float sdf_trunc_;
    bool space_carving_;
    float min_weight_;
    static constexpr uint16_t num_semantic_classes_ = NCLASSES;  // MUST be defined at compile-time

    /// OpenVDB Grids modeling the signed distance field and the weight grid
    openvdb::FloatGrid::Ptr tsdf_;
    openvdb::FloatGrid::Ptr weights_;
    openvdb::VecXIGrid<num_semantic_classes_>::Ptr semantics_;
};

}  // namespace slimvdb
