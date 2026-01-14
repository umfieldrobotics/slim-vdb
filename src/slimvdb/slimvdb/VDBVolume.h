/*
 * MIT License
 *
 * Copyright (c) 2025 Anja Sheppard, University of Michigan
 * Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
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
#include <fstream>
#include <tuple>

#include "MarchingCubesConst.h"
#include "utils/Utils.h"
#include "Config.h"

namespace slimvdb {

/// @brief Compile-time alias for the OpenVDB vector grid type based on language
/// @tparam L The language mode (CLOSED or OPEN)
/// @tparam S Number of semantic classes / feature space dimensions
/// Resolves to `openvdb::VecXIGrid<S>` for CLOSED, or `openvdb::VecXFGrid<S>` for OPEN
template <Language L, uint16_t S>
using VecXGrid = typename std::conditional_t<L == CLOSED, openvdb::VecXIGrid<S>, openvdb::VecXFGrid<S>>;

/// @brief Pointer type alias for the grid, corresponding to Grid<L,S>::Ptr
/// @tparam L The language mode
/// @tparam S Number of semantic classes / feature space dimensions
template <Language L, uint16_t S>
using VecXGridPtr = typename std::conditional_t<L == CLOSED, openvdb::VecXIGrid<S>, openvdb::VecXFGrid<S>>::Ptr;

/// @brief Compile-time alias for semantic label storage type based on language
/// @tparam L The language mode
/// @tparam S Number of semantic classes / feature space dimensions
/// Resolves to `openvdb::VecXI32<S>` for CLOSED, or `openvdb::VecXF32<S>` for OPEN
template <Language L, uint16_t S>
using LabelT = std::conditional_t<L == CLOSED, openvdb::VecXI32<S>, openvdb::VecXF32<S>>;


/// @brief Helper struct for extra variables
/// Specializations of this struct contain members only present for OPEN volumes, nothing extra for CLOSED
template <Language L>
struct VDBVolumeExtra {};

/// @brief Specialization of VDBVolumeExtra for OPEN volumes
/// Contains members specific to open-set semantics: covariances and embeddings
template <>
struct VDBVolumeExtra<OPEN> {
    /// @brief Dummy class for extra variables for open-set semantics
    VecXGridPtr<OPEN, NCLASSES> covariances_;
    float* h_embeddings_;
};


/// @brief Volumetric TSDF representation with semantic information
/// @tparam L Compile-time semantic language selection (CLOSED or OPEN)
/// Inherits optional members from VDBVolumeExtra<L> (covariances/embeddings only for OPEN)
template<Language L>
class VDBVolume : public VDBVolumeExtra<L>{
public:
    VDBVolume(float voxel_size, float sdf_trunc, bool space_carving, float min_weight);
    ~VDBVolume() = default;

public:
    /// @brief Integrates a new (globally aligned) PointCloud and its labels into the current
    /// tsdf_ volume with closed set semantics.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<uint32_t>& labels,
                   const Eigen::Vector3d& origin,
                   const std::function<float(float)>& weighting_function);

    /// @brief Integrates a new (globally aligned) PointCloud and its labels into the current
    /// tsdf_ volume with open set semantics.
    void Integrate(const std::vector<Eigen::Vector3d>& points,
                   const std::vector<std::vector<float>>& labels,
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
                const float max_range,
                const float p_threshold);

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
    ExtractPointCloud(bool fill_holes = true, float min_weight = 0.5, const float p_threshold=0.0) const;

public:
    /// VDBVolume public properties
    static constexpr Language language_ = L;  // MUST be defined at compile-time

    float voxel_size_;
    float sdf_trunc_;
    bool space_carving_;
    float min_weight_;
    static constexpr uint16_t num_semantic_classes_ = NCLASSES;  // MUST be defined at compile-time
    uint16_t num_open_semantic_classes_ = 0;

    /// OpenVDB Grids modeling the signed distance field and the weight grid
    openvdb::FloatGrid::Ptr tsdf_;
    openvdb::FloatGrid::Ptr weights_;
    VecXGridPtr<L, NCLASSES> semantics_;
    // GridPtr covariances_; (THIS IS DEFINED IN THE BASE CLASS VDBVolumeExtra)
};

	
inline std::tuple<float*, uint16_t> read_embeddings (const int S) {
    // Read in encodings per class from file
    std::vector<std::vector<float>> embeddings;
    std::ifstream file("/media/anjashep-frog-lab/DATA1/SLIMVDB_DATASETS/scenenet/train/2/scenenet_clip_embeddings.txt"); // THIS MUST BE HARDCODED TO BE READ AT COMPILE-TIME, FIX FOR YOUR PATH
    if (!file) {
        printf("Error opening clip embeddings file: %s\n", std::strerror(errno));
        std::exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string word;
        
        // Read the class name (or word) and ignore it.
        iss >> word;
        
        std::vector<float> features;
        features.reserve(S); //TODO FIX
        
        // Read each float value.
        float value;
        while (iss >> value) {
            features.push_back(value);
        }
        
        embeddings.push_back(std::move(features));
    }
    file.close();
    // Step 2: Flatten the data into a contiguous host array.
    // Allocate host memory for a contiguous array of floats.
    uint16_t num_open_semantic_classes = embeddings.size();
    size_t totalFloats = num_open_semantic_classes * S;
    float* h_embeddings = new float[totalFloats];
    
    for (size_t i = 0; i < embeddings.size(); ++i) {
        // Copy S floats to the contiguous array.
        std::copy(embeddings[i].begin(), embeddings[i].begin() + S,
                  h_embeddings + i * S);
    }

    return std::make_tuple(h_embeddings, num_open_semantic_classes);
}

}  // namespace slimvdb
