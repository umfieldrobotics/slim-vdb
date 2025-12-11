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


#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include <filesystem>

#include <torch/script.h>
#include <opencv2/opencv.hpp>

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
                          bool rgbd = false,
                          bool realtime_segmentation = false);

    /// Returns a point cloud and the origin of the sensor in world coordinate frames
    [[nodiscard]] std::tuple<PointCloud, std::vector<uint32_t>, Eigen::Matrix4d> operator[](int idx) const;
    [[nodiscard]] std::size_t size() const { if(rgbd_) return depth_files_.size(); else return scan_files_.size(); }
    std::pair<cv::Mat, std::vector<int>> RunInference(const std::string&, const std::string&) const;

public:
    bool apply_pose_ = true;
    bool preprocess_ = true;
    float min_range_ = 0.0F;
    float max_range_ = std::numeric_limits<float>::max();
    bool rgbd_ = false;
    bool realtime_segmentation_ = false;
    std::filesystem::path scenenet_sequence_dir_;
    static constexpr uint16_t num_semantic_classes = NCLASSES; // this MUST be known at compile time

private:
    std::vector<std::string> scan_files_;
    std::vector<std::string> depth_files_;
    std::vector<std::string> gt_label_files_;
    std::vector<std::string> label_files_;
    std::vector<std::string> img_files_;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<Eigen::Matrix<int, 1, num_semantic_classes>> semantics_;
    mutable torch::jit::script::Module model_;
    torch::Device device_ = torch::kCUDA;
    static constexpr int8_t nyu40_to_scenenet_[40] = {
        12,5,6,1,4,9,10,12,13,
        6,8,6,13,10,6,13,6,7,7,
        5,7,3,2,6,11,7,7,7,7,
        7,7,6,7,7,7,7,7,7,6,7
    };
    float fx_ = 0.0;
    float fy_ = 0.0;
    float cx_ = 0.0;
    float cy_ = 0.0;
};

}  // namespace datasets