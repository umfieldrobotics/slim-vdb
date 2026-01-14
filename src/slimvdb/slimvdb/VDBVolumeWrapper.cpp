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

// This class is a wrapper for the templated VDBVolume class. 
// IT IS ONLY USED IN THE PYTHON BINDINGS TO MAKE USER CODING MORE STRAIGHTFORWARD.

#include "VDBVolumeWrapper.h"

namespace slimvdb {

VDBVolumeWrapper::VDBVolumeWrapper(float voxel_size, float sdf_trunc, bool space_carving, float min_weight,
                                   Language language)
    : voxel_size_(voxel_size), sdf_trunc_(sdf_trunc), space_carving_(space_carving), language_(language) 
{
    if(language == CLOSED) {
        volume_ = std::make_shared<VDBVolume<CLOSED>>(voxel_size, sdf_trunc, space_carving, min_weight);
    } else {
        volume_ = std::make_shared<VDBVolume<OPEN>>(voxel_size, sdf_trunc, space_carving, min_weight);
    }
}

void VDBVolumeWrapper::Integrate(const std::vector<Eigen::Vector3d>& points,
                                const Eigen::Vector3d& origin) 
{
    std::visit([&](auto& vol){
        vol->Integrate(points, origin, [](float){ return 1.0f; });
    }, volume_);
}

void VDBVolumeWrapper::Integrate(const std::vector<Eigen::Vector3d>& points,
                                const Eigen::Vector3d& origin,
                                float weight)
{
    std::visit([&](auto& vol){
        vol->Integrate(points, origin, [=](float){ return weight; });
    }, volume_);
}

void VDBVolumeWrapper::Integrate(const std::vector<Eigen::Vector3d>& points,
                                const std::vector<uint32_t>& labels,
                                const Eigen::Vector3d& origin)
{
    std::visit([&](auto& vol){
        if constexpr (std::is_same_v<std::decay_t<decltype(*vol)>, VDBVolume<CLOSED>>) {
            vol->Integrate(points, labels, origin, [](float){ return 1.0f; });
        } else {
            throw std::runtime_error("OPEN volume requires vector<vector<float>> labels");
        }
    }, volume_);
}

void VDBVolumeWrapper::Integrate(const std::vector<Eigen::Vector3d>& points,
                                const std::vector<std::vector<float>>& labels,
                                const Eigen::Vector3d& origin)
{
    std::visit([&](auto& vol){
        if constexpr (std::is_same_v<std::decay_t<decltype(*vol)>, VDBVolume<OPEN>>) {
            vol->Integrate(points, labels, origin, [](float){ return 1.0f; });
        } else {
            throw std::runtime_error("CLOSED volume does not support vector<vector<float>> labels");
        }
    }, volume_);
}

#ifdef PYOPENVDB_SUPPORT
void VDBVolumeWrapper::Integrate(openvdb::FloatGrid::Ptr grid)
{
    std::visit([&](auto& vol){
        vol->Integrate(grid, [](float){ return 1.0f; });
    }, volume_);
}

void VDBVolumeWrapper::Integrate(openvdb::FloatGrid::Ptr grid, float weight)
{
    std::visit([&](auto& vol){
        vol->Integrate(grid, [=](float){ return weight; });
    }, volume_);
}
#endif

void VDBVolumeWrapper::UpdateTSDF(const float& sdf, const std::vector<int>& ijk)
{
    std::visit([&](auto& vol){
        vol->UpdateTSDF(sdf, openvdb::Coord(ijk[0], ijk[1], ijk[2]), [](float){ return 1.0f; });
    }, volume_);
}

void VDBVolumeWrapper::UpdateTSDF(const float& sdf, const std::vector<int>& ijk, const std::function<float(float)>& weighting_function)
{
    std::visit([&](auto& vol){
        vol->UpdateTSDF(sdf, openvdb::Coord(ijk[0], ijk[1], ijk[2]), weighting_function);
    }, volume_);
}

void VDBVolumeWrapper::Render(const std::vector<double>& origin_vec,
                             const std::vector<double>& rot_quat_vec,
                             int index,
                             int width,
                             int height,
                             float min_range,
                             float max_range,
                             float p_threshold) 
{
    std::visit([&](auto& vol){
        vol->Render(origin_vec, rot_quat_vec, index, width, height, min_range, max_range, p_threshold);
    }, volume_);
}

openvdb::FloatGrid::Ptr VDBVolumeWrapper::Prune(float min_weight) const
{
    return std::visit([&](auto& vol) -> openvdb::FloatGrid::Ptr {
        return vol->Prune(min_weight);
    }, volume_);
}

std::shared_ptr<openvdb::FloatGrid> VDBVolumeWrapper::tsdf() const
{
    return std::visit([](auto& vol){ return vol->tsdf_; }, volume_);
}

std::shared_ptr<openvdb::FloatGrid> VDBVolumeWrapper::weights() const
{
    return std::visit([](auto& vol){ return vol->weights_; }, volume_);
}

#ifdef PYOPENVDB_SUPPORT
std::shared_ptr<decltype(std::declval<VDBVolume<OPEN>>().semantics_)> VDBVolumeWrapper::semantics() const
{
    return std::visit([](auto& vol){ return vol->semantics_; }, volume_);
}
#endif

} // namespace slimvdb