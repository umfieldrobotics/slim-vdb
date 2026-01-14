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

#include "VDBVolume.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>

#include "nanovdb_utils/common.h"
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/cuda/CudaDeviceBuffer.h>
#include <nanovdb/NanoVDB.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <execution>

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

namespace {

float ComputeSDF(const Eigen::Vector3d& origin,
                 const Eigen::Vector3d& point,
                 const Eigen::Vector3d& voxel_center) {
    const Eigen::Vector3d v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3d v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);
    const double sign = proj / std::abs(proj);
    return static_cast<float>(sign * dist);
}

Eigen::Vector3d GetVoxelCenter(const openvdb::Coord& voxel, const openvdb::math::Transform& xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return Eigen::Vector3d(v_wf.x(), v_wf.y(), v_wf.z());
}

}  // namespace

namespace slimvdb {

template <Language L>
VDBVolume<L>::VDBVolume(float voxel_size, float sdf_trunc, bool space_carving, float min_weight)
    : voxel_size_(voxel_size), sdf_trunc_(sdf_trunc), space_carving_(space_carving), min_weight_(min_weight) {
    tsdf_ = openvdb::FloatGrid::create(sdf_trunc_);
    tsdf_->setName("D(x): signed distance grid");
    tsdf_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    tsdf_->setGridClass(openvdb::GRID_LEVEL_SET);

    weights_ = openvdb::FloatGrid::create(0.0f);
    weights_->setName("W(x): weights grid");
    weights_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    weights_->setGridClass(openvdb::GRID_UNKNOWN);

    semantics_ = VecXGrid<L, num_semantic_classes_>::create(LabelT<L, num_semantic_classes_>());
    semantics_->setName("A(x): semantics grid");
    semantics_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    semantics_->setGridClass(openvdb::GRID_UNKNOWN);

    // Only for OPEN (evaluated at compile-time)
    if constexpr (L == OPEN) {
        this->covariances_ = VecXGrid<L, num_semantic_classes_>::create();
        this->covariances_->setName("S(x): covariances grid");
        this->covariances_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
        this->covariances_->setGridClass(openvdb::GRID_UNKNOWN);

        auto [embeddings, num_open_semantic_classes] = read_embeddings(num_semantic_classes_);
        this->h_embeddings_ = embeddings;
        this->num_open_semantic_classes_ = num_open_semantic_classes;
    }
}

template <Language L>
void VDBVolume<L>::UpdateTSDF(const float& sdf,
                           const openvdb::Coord& voxel,
                           const std::function<float(float)>& weighting_function) {
    using AccessorRW = openvdb::tree::ValueAccessorRW<openvdb::FloatTree>;
    if (sdf > -sdf_trunc_) {
        AccessorRW tsdf_acc = AccessorRW(tsdf_->tree());
        AccessorRW weights_acc = AccessorRW(weights_->tree());
        const float tsdf = std::min(sdf_trunc_, sdf);
        const float weight = weighting_function(sdf);
        const float last_weight = weights_acc.getValue(voxel);
        const float last_tsdf = tsdf_acc.getValue(voxel);
        const float new_weight = weight + last_weight;
        const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
        tsdf_acc.setValue(voxel, new_tsdf);
        weights_acc.setValue(voxel, new_weight);
    }
}

template <Language L>
void VDBVolume<L>::Integrate(openvdb::FloatGrid::Ptr grid,
                          const std::function<float(float)>& weighting_function) {
    for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
        const auto& sdf = iter.getValue();
        const auto& voxel = iter.getCoord();
        this->UpdateTSDF(sdf, voxel, weighting_function);
    }
}

/* slimvdb::CLOSED: Use Dirichlet-Categorical Bayesian update */
template <>
void VDBVolume<slimvdb::CLOSED>::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const std::vector<uint32_t>& labels,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    if (labels.empty()) {
        std::cerr << "Semantic labels provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Get the "unsafe" version of the grid acessors
    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();
    auto alpha_acc = semantics_->getUnsafeAccessor();

    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        int idx = &point - &points[0];
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                // Cumulative TSDF and weight update
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);

                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_weight * last_tsdf + weight * tsdf) / (new_weight);

                // Bayesian semantic class update
                uint16_t label = (uint16_t)(labels[idx] & 0xFFFF); // lower 16 bits
                auto alpha = alpha_acc.getValue(voxel); // last_alpha
                alpha[label] += 1; // new_alpha

                // Update VDB volumes
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);
                alpha_acc.setValue(voxel, alpha);
            }
        } while (dda.step());
    });
}

/* slimvdb::OPEN: Use Normal Inverse Gamma Bayesian update */
template <>
void VDBVolume<slimvdb::OPEN>::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const std::vector<std::vector<float>>& labels,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }
    if (labels.empty()) {
        std::cerr << "Semantic labels provided is empty\n";
        return;
    }
    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());
    // Get the "unsafe" version of the grid acessors
    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();
    auto mean_acc = semantics_->getUnsafeAccessor();
    auto covs_acc = this->covariances_->getUnsafeAccessor();
    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        int idx = &point - &points[0];
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();
        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;
        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                // Cumulative TSDF and weight update
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);

                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_weight * last_tsdf + weight * tsdf) / (new_weight);

                // Bayesian semantic class update
                // update mean (mu): ( last_weight / weight ) * curr_mu + (1 / weight) * labels[idx]
                auto curr_mu = mean_acc.getValue(voxel);
                double zeta = last_weight / new_weight;
                double kappa  = 1.0         / new_weight;
                openvdb::VecXF32<curr_mu.size> new_mu;
                for (size_t j = 0; j < curr_mu.size; j++) {
                    new_mu[j] = static_cast<float>(zeta * static_cast<double>(curr_mu[j]) + kappa * static_cast<double>(labels[idx][j])); // cast to double and back to avoid overflow
                }

                // update beta (covariance) with |Z| = 1: curr_beta + (last_weight / weight) * ((labels[idx] - curr_mu) @ (labels[idx] - curr_mu).T)
                auto curr_beta = covs_acc.getValue(voxel);
                openvdb::VecXF32<curr_beta.size> new_beta;
                for (size_t j = 0; j < curr_beta.size; j++) { // update diagonal
                    new_beta[j] = curr_beta[j] + zeta * ((labels[idx][j] - curr_mu[j]) * (labels[idx][j] - curr_mu[j]));
                }

                // Update VDB volumes
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);
                mean_acc.setValue(voxel, new_mu);
                covs_acc.setValue(voxel, new_beta);
            }
        } while (dda.step());
    });
}

template <Language L>
void VDBVolume<L>::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Get the "unsafe" version of the grid acessors
    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();

    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);
                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);
            }
        } while (dda.step());
    });
}

template <Language L>
void VDBVolume<L>::Render(const std::vector<double> origin_vec, const std::vector<double> rot_quat_vec, const int index, const int render_img_width, const int render_img_height, 
                                                                                                        const float min_range, const float max_range, const float p_threshold) {
    // Render image and display
    std::clog << "\nFrame #" << index << std::endl;

    auto timer_imgbuff0 = std::chrono::high_resolution_clock::now();
    BufferT imageBuffer;
    imageBuffer.init(3 * render_img_width * render_img_height * sizeof(float)); // needs to be a 3 channel image
    auto timer_imgbuff1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed0 = timer_imgbuff1 - timer_imgbuff0;
    std::clog << "Image buffer creation took: " << elapsed0.count() << " ms" << std::endl;

    auto timer_nanovdbconv0 = std::chrono::high_resolution_clock::now();
    openvdb::CoordBBox bbox;

    auto handle = nanovdb::tools::createNanoGrid<openvdb::FloatGrid, float, BufferT>(*tsdf_);
    auto label_handle = nanovdb::tools::createNanoGrid<VecXGrid<L, num_semantic_classes_>, LabelT<L, num_semantic_classes_>, BufferT>(*semantics_);
    nanovdb::GridHandle<BufferT> weight_handle;
    nanovdb::GridHandle<BufferT> beta_handle;
    if constexpr (L == slimvdb::OPEN) {
        weight_handle = nanovdb::tools::createNanoGrid<openvdb::FloatGrid, float, BufferT>(*weights_);
        beta_handle = nanovdb::tools::createNanoGrid<VecXGrid<L, num_semantic_classes_>, LabelT<L, num_semantic_classes_>, BufferT>(*this->covariances_);
    }

    auto timer_nanovdbconv1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = timer_nanovdbconv1 - timer_nanovdbconv0;
    std::clog << "Conversion to NanoVDB took: " << elapsed1.count() << " ms" << std::endl;

    auto timer_render0 = std::chrono::high_resolution_clock::now();

    if constexpr (L == slimvdb::CLOSED) {
        runNanoVDB<L, num_semantic_classes_>(handle, label_handle, weight_handle, beta_handle, render_img_width, render_img_height, imageBuffer, index, origin_vec, rot_quat_vec, min_range, max_range, p_threshold);
    }
    else { // L == slimvdb::OPEN
#if defined(NANOVDB_USE_CUDA)
        runNanoVDB<L, num_semantic_classes_>(handle, label_handle, weight_handle, beta_handle, render_img_width, render_img_height, imageBuffer, index, origin_vec, rot_quat_vec, min_range, max_range, p_threshold, this->h_embeddings_, this->num_open_semantic_classes_);
#else
        runNanoVDB<L, num_semantic_classes_>(handle, label_handle, weight_handle, beta_handle, render_img_width, render_img_height, imageBuffer, index, origin_vec, rot_quat_vec, min_range, max_range, p_threshold, this->h_embeddings_, this->num_open_semantic_classes_);
#endif
    }

    auto timer_render1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = timer_render1 - timer_render0;
    std::clog << "NanoVDB rendering took: " << elapsed2.count() << " ms" << std::endl;
}

template <Language L>
openvdb::FloatGrid::Ptr VDBVolume<L>::Prune(float min_weight) const {
    const auto weights = weights_->tree();
    const auto tsdf = tsdf_->tree();
    const auto background = sdf_trunc_;
    openvdb::FloatGrid::Ptr clean_tsdf = openvdb::FloatGrid::create(sdf_trunc_);
    clean_tsdf->setName("D(x): Pruned signed distance grid");
    clean_tsdf->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    clean_tsdf->setGridClass(openvdb::GRID_LEVEL_SET);
    clean_tsdf->tree().combine2Extended(tsdf, weights, [=](openvdb::CombineArgs<float>& args) {
        if (args.aIsActive() && args.b() > min_weight) {
            args.setResult(args.a());
            args.setResultIsActive(true);
        } else {
            args.setResult(background);
            args.setResultIsActive(false);
        }
    });
    return clean_tsdf;
}

template <Language L>
std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3i>, std::vector<int>>
VDBVolume<L>::ExtractTriangleMesh(bool fill_holes, float min_weight) const {
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
    auto alpha_acc = semantics_->getAccessor();
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
            auto label = alpha_acc.getValue(openvdb::Coord(voxel.x(), voxel.y(), voxel.z())); // this should be the semantic label for that voxel
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

template <Language L>
std::pair<std::vector<Eigen::Vector3d>, std::vector<int>>
VDBVolume<L>::ExtractPointCloud(bool fill_holes, float min_weight, const float p_threshold) const {

    std::vector<Eigen::Vector3d> points;
    std::vector<int> labels;

    auto tsdf_acc    = tsdf_->getAccessor();
    auto weights_acc = weights_->getAccessor();
    auto alpha_acc  = semantics_->getAccessor();

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

        auto label_vec = alpha_acc.getValue(coord);

        int max_i = 0;

        if constexpr (L == CLOSED) {
            // Argmax:
            float max_val = label_vec[0];
            for (int i = 1; i < num_semantic_classes_; i++) {
                if (label_vec[i] > max_val) {
                    max_val = label_vec[i];
                    max_i = i;
                }
            }
        }
        else { // L == slimvdb::OPEN
            auto beta_acc  = this->covariances_->getAccessor();
            auto beta = beta_acc.getValue(coord);

            auto m = label_vec;

            float lambda = w;
            float nu = w / 2;

            // Compute log posterior predictive per class
            float logp[num_open_semantic_classes_];
            float max_logp = -1e30f;

            for (int c = 0; c < num_open_semantic_classes_; ++c) {
                // compute log predictive Student-t likelihood
                float lp = 0.f;
                const float dof = 2.0f * nu;

                lp += -0.5f * NCLASSES * logf(dof * M_PI);
                for (int j = 0; j < NCLASSES; ++j) {
                    float s2 = beta[j] * (lambda + 1.0f) / (lambda * nu);
                    float x  = (this->h_embeddings_[c * NCLASSES + j] - m[j]) / sqrtf(s2);
                    lp += -0.5f * (dof + 1.f) * log1pf((x * x) / dof);
                    // (dropped constants since they cancel in softmax)
                }
                logp[c] = lp;
                if (lp > max_logp)
                    max_logp = lp;
            }

            // Softmax normalization for posterior probabilities
            float denom = 0.f;
            for (int c = 0; c < num_open_semantic_classes_; ++c)
                denom += expf(logp[c] - max_logp);

            // Most probable class
            int best_class = 0;
            float best_prob = 0.0;
            float p = 0.0;
            for (int c = 0; c < num_open_semantic_classes_; ++c) {
                p = expf(logp[c] - max_logp) / denom;
                if (p > best_prob) {
                    best_prob = p;
                    best_class = c;
                }
            }

            // Threshold probabilities
            if (best_prob < p_threshold)
                best_class = 0; // "unknown"
        }

        labels.push_back(max_i);
    }

    return {points, labels};
}

}  // namespace slimvdb


template class slimvdb::VDBVolume<slimvdb::Language::CLOSED>;
template class slimvdb::VDBVolume<slimvdb::Language::OPEN>;
