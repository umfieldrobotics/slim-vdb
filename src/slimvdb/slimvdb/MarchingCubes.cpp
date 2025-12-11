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

#include <Eigen/Core>
#include <memory>
#include <unordered_map>
#include <vector>

#include "MarchingCubesConst.h"
#include "VDBVolume.h"

namespace openvdb {
static const openvdb::Coord shift[8] = {
    openvdb::Coord(0, 0, 0), openvdb::Coord(1, 0, 0), openvdb::Coord(1, 1, 0),
    openvdb::Coord(0, 1, 0), openvdb::Coord(0, 0, 1), openvdb::Coord(1, 0, 1),
    openvdb::Coord(1, 1, 1), openvdb::Coord(0, 1, 1),
};
}

// Taken from <open3d/utility/Eigen.h>
namespace {
template <typename T>
struct hash_eigen {
    std::size_t operator()(T const& matrix) const {
        size_t seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
}  // namespace

namespace slimvdb {
std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>, std::vector<int>>
VDBVolume::ExtractTriangleMesh(bool fill_holes, float min_weight) const {
    // implementation of marching cubes, based on Open3D
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> triangles;
    std::vector<int> tri_labels;

    double half_voxel_length = voxel_size_ * 0.2;
    // Map of "edge_index = (x, y, z, 0) + edge_shift" to "global vertex index"
    std::unordered_map<Eigen::Vector4i, int, hash_eigen<Eigen::Vector4i>, std::equal_to<>,
                       Eigen::aligned_allocator<std::pair<const Eigen::Vector4i, int>>>
        edgeindex_to_vertexindex;
    int edge_to_index[12];

    auto tsdf_acc = tsdf_->getAccessor();
    auto weights_acc = weights_->getAccessor();
    auto labels_acc = semantics_->getAccessor();
    for (auto iter = tsdf_->beginValueOn(); iter; ++iter) {
        int cube_index = 0;
        float f[8];
        const openvdb::Coord& voxel = iter.getCoord();
        const int32_t x = voxel.x();
        const int32_t y = voxel.y();
        const int32_t z = voxel.z();
        for (int i = 0; i < 8; i++) {
            openvdb::Coord idx = voxel + openvdb::shift[i];
            if (!fill_holes) {
                if (weights_acc.getValue(idx) == 0.0f) {
                    cube_index = 0;
                    break;
                }
            }
            if (weights_acc.getValue(idx) < min_weight) {
                cube_index = 0;
                break;
            }
            f[i] = tsdf_acc.getValue(idx);
            if (f[i] < 0.0f) {
                cube_index |= (1 << i);
            }
        }
        if (cube_index == 0 || cube_index == 255) {
            continue;
        }
        for (int i = 0; i < 12; i++) {
            if ((edge_table[cube_index] & (1 << i)) != 0) {
                Eigen::Vector4i edge_index = Eigen::Vector4i(x, y, z, 0) + edge_shift[i];
                if (edgeindex_to_vertexindex.find(edge_index) == edgeindex_to_vertexindex.end()) {
                    edge_to_index[i] = (int)vertices.size();
                    edgeindex_to_vertexindex[edge_index] = (int)vertices.size();
                    Eigen::Vector3d pt(half_voxel_length + voxel_size_ * edge_index(0),
                                       half_voxel_length + voxel_size_ * edge_index(1),
                                       half_voxel_length + voxel_size_ * edge_index(2));
                    double f0 = std::abs((double)f[edge_to_vert[i][0]]);
                    double f1 = std::abs((double)f[edge_to_vert[i][1]]);
                    pt(edge_index(3)) += f0 * voxel_size_ / (f0 + f1);
                    vertices.push_back(pt /* + origin_*/);
                } else {
                    edge_to_index[i] = edgeindex_to_vertexindex.find(edge_index)->second;
                }
            }
        }
        for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
            auto label = labels_acc.getValue(openvdb::Coord(voxel.x(), voxel.y(), voxel.z())); // this should be the semantic label for that voxel
            triangles.emplace_back(edge_to_index[tri_table[cube_index][i]],
                                   edge_to_index[tri_table[cube_index][i + 2]],
                                   edge_to_index[tri_table[cube_index][i + 1]]);

            // go from one-hot label back to normal
            int max_i = 0;
            for (int i = 0; i < num_semantic_classes_; i++) {
                if (label[i] > label[max_i]) max_i = i;
            }
            tri_labels.emplace_back(max_i);
            // tri_labels.emplace_back(label);
        }
    }


    return std::make_tuple(vertices, triangles, tri_labels);
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<int>>
VDBVolume::ExtractPointCloud(bool fill_holes, float min_weight) const {

    std::vector<Eigen::Vector3d> points;
    std::vector<int> labels;

    auto tsdf_acc    = tsdf_->getAccessor();
    auto weights_acc = weights_->getAccessor();
    auto labels_acc  = semantics_->getAccessor();

    for (auto iter = tsdf_->beginValueOn(); iter; ++iter) {
        const openvdb::Coord& coord = iter.getCoord();
        float w = weights_acc.getValue(coord);
        if (!fill_holes) {
            if (w == 0.0f) {
                continue;
            }
        }
        if (w < min_weight) {
            continue;
        }

        double px = voxel_size_ * (coord.x() + 0.5);
        double py = voxel_size_ * (coord.y() + 0.5);
        double pz = voxel_size_ * (coord.z() + 0.5);

        points.emplace_back(px, py, pz);

        auto label_vec = labels_acc.getValue(coord);

        // Argmax:
        int max_i = 0;
        float max_val = label_vec[0];
        for (int i = 1; i < num_semantic_classes_; i++) {
            if (label_vec[i] > max_val) {
                max_val = label_vec[i];
                max_i = i;
            }
        }

        labels.push_back(max_i);
    }

    return {points, labels};
}

}  // namespace slimvdb
