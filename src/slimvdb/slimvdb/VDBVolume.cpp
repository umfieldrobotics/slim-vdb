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

#if defined(NANOVDB_USE_CUDA)
using BufferT = nanovdb::cuda::DeviceBuffer;
#else
using BufferT = nanovdb::HostBuffer;
#endif

template <uint16_t S>
using LabelT = openvdb::VecXI32<S>;

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

VDBVolume::VDBVolume(float voxel_size, float sdf_trunc, bool space_carving, float min_weight)
    : voxel_size_(voxel_size), sdf_trunc_(sdf_trunc), space_carving_(space_carving), min_weight_(min_weight) {
    tsdf_ = openvdb::FloatGrid::create(sdf_trunc_);
    tsdf_->setName("D(x): signed distance grid");
    tsdf_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    tsdf_->setGridClass(openvdb::GRID_LEVEL_SET);

    weights_ = openvdb::FloatGrid::create(0.0f);
    weights_->setName("W(x): weights grid");
    weights_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    weights_->setGridClass(openvdb::GRID_UNKNOWN);

    semantics_ = openvdb::VecXIGrid<num_semantic_classes_>::create(LabelT<num_semantic_classes_>());
    semantics_->setName("A(x): semantics grid");
    semantics_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    semantics_->setGridClass(openvdb::GRID_UNKNOWN);
}

void VDBVolume::UpdateTSDF(const float& sdf,
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

void VDBVolume::Integrate(openvdb::FloatGrid::Ptr grid,
                          const std::function<float(float)>& weighting_function) {
    for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
        const auto& sdf = iter.getValue();
        const auto& voxel = iter.getCoord();
        this->UpdateTSDF(sdf, voxel, weighting_function);
    }
}

void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
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
    auto labels_acc = semantics_->getUnsafeAccessor();

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
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);
                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
                uint16_t semantic_label = (uint16_t)(labels[idx] & 0xFFFF); // lower 16 bits
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);
                auto label_one_hot = labels_acc.getValue(voxel);
                label_one_hot[semantic_label] += 1;
                labels_acc.setValue(voxel, label_one_hot);
            }
        } while (dda.step());
    });
}

void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
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

void VDBVolume::Render(const std::vector<double> origin_vec, const std::vector<double> rot_quat_vec, const int index, const int render_img_width, const int render_img_height, const float min_range, const float max_range) {
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
    auto label_handle = nanovdb::tools::createNanoGrid<openvdb::VecXIGrid<num_semantic_classes_>, LabelT<num_semantic_classes_>, BufferT>(*semantics_);

    auto timer_nanovdbconv1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = timer_nanovdbconv1 - timer_nanovdbconv0;
    std::clog << "Conversion to NanoVDB took: " << elapsed1.count() << " ms" << std::endl;

    runNanoVDB(handle, label_handle, render_img_width, render_img_height, imageBuffer, index, origin_vec, rot_quat_vec, min_range, max_range);
}

openvdb::FloatGrid::Ptr VDBVolume::Prune(float min_weight) const {
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
}  // namespace slimvdb
