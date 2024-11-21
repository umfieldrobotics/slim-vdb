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

#include <Eigen/Core>
#include <string>
#include <vector>
#include <filesystem>

namespace datasets {

class SceneNetDataset {
public:
    using Point = Eigen::Vector3d;
    using PointCloud = std::vector<Eigen::Vector3d>;

    explicit SceneNetDataset(const std::string& scenenet_root_dir,
                          const std::string& sequence,
                          int n_scans = -1,
                          bool rgbd = false);

    explicit SceneNetDataset(const std::string& scenenet_root_dir,
                          const std::string& sequence,
                          int n_scans = -1,
                          bool apply_pose = true,
                          bool preprocess = true,
                          float min_range = 0.0F,
                          float max_range = std::numeric_limits<float>::max(),
                          bool rgbd = false);

    /// Returns a point cloud and the origin of the sensor in world coordinate frames
    [[nodiscard]] std::tuple<PointCloud, std::vector<uint32_t>, Point> operator[](int idx) const;
    [[nodiscard]] std::size_t size() const { if(rgbd_) return depth_files_.size(); else return scan_files_.size(); }

public:
    bool apply_pose_ = true;
    bool preprocess_ = true;
    float min_range_ = 0.0F;
    float max_range_ = std::numeric_limits<float>::max();
    bool rgbd_ = false;
    std::filesystem::path scenenet_sequence_dir_;

private:
    std::vector<std::string> scan_files_;
    std::vector<std::string> depth_files_;
    std::vector<std::string> gt_label_files_;
    std::vector<std::string> label_files_;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<Eigen::Matrix<int, 1, 28>> semantics_;
};

}  // namespace datasets
