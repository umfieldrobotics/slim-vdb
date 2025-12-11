// MIT License
//
// Copyright (c) 2024 Anja Sheppard, University of Michigan
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

#include "SceneNetOdometry.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "open3d/Open3D.h"

#include <torch/script.h>

namespace fs = std::filesystem;

namespace {

std::vector<std::string> GetDepthFiles(const fs::path& depth_path, int n_scans) {
    std::vector<std::string> depth_files;
    for (const auto& entry : fs::directory_iterator(depth_path)) {
        if (entry.path().extension() == ".png") {
            depth_files.emplace_back(entry.path().string());
        }
    }
    if (depth_files.empty()) {
        std::cerr << depth_path << "path doesn't have any .png" << std::endl;
        exit(1);
    }
    std::sort(depth_files.begin(), depth_files.end());
    if (n_scans > 0) {
        depth_files.erase(depth_files.begin() + n_scans, depth_files.end());
    }
    return depth_files;
}

std::vector<std::string> GetImgFiles(const fs::path& img_path, int n_scans) {
    std::vector<std::string> label_files;
    for (const auto& entry : fs::directory_iterator(img_path)) {
        if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
            label_files.emplace_back(entry.path().string());
        }
    }
    if (label_files.empty()) {
        std::cerr << "ERROR: " << img_path << " path doesn't have any .png or .jpg" << std::endl;
        exit(1);
    }
    std::sort(label_files.begin(), label_files.end());
    if (n_scans > 0) {
        label_files.erase(label_files.begin() + n_scans, label_files.end());
    }
    return label_files;
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>> ReadSceneNetDepthAndLabels(const std::string& depth_path, const std::string& label_path, float fx, float fy, float cx, float cy, float min_range, float max_range) {
    // Read depth and label
    cv::Mat depth_mat = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    cv::Mat label_mat = cv::imread(label_path, cv::IMREAD_UNCHANGED);

    assert(depth_mat.size() == label_mat.size());

    // Convert cv Mat to vector
    std::vector<float> depth_data;
    std::vector<int> label_data(label_mat.total());

    depth_mat.convertTo(depth_mat, CV_32F, 1.0f/1000.0f); // SceneNet depth values are in millimeters
    depth_data.assign(
        reinterpret_cast<float*>(depth_mat.data),
        reinterpret_cast<float*>(depth_mat.data + depth_mat.total() * depth_mat.elemSize())
    );

    for (size_t i = 0; i < label_mat.total(); ++i) {
        label_data[i] = static_cast<int>(label_mat.data[i]);
    }

    // Project depth image to pointcloud
    std::vector<Eigen::Vector3d> pc_points; // store the 3D points for creating the pointcloud
    std::vector<uint32_t> labels; // store the labels that aren't tossed

    int num_rows = depth_mat.size().height;
    int num_cols = depth_mat.size().width;
    
    for (size_t u = 0; u < num_rows; u++) {
        for (size_t v = 0; v < num_cols; v++) {
            size_t i = num_cols * u + v;

            float d = depth_data[i];

            if (d <= 0 || std::isnan(d)) continue;
            if (d > max_range || d < min_range) continue; // Skip if out of range

            float x_norm = (v - cx) / fx;
            float y_norm = (u - cy) / fy;

            // Depth value as Euclidean distance, need to convert
            float norm_squared = x_norm * x_norm + y_norm * y_norm + 1.0f;
            float z = d / std::sqrt(norm_squared);

            float x = x_norm * z;
            float y = y_norm * z;

            Eigen::Vector3d p {x, y, z};

            pc_points.push_back(p);
            labels.push_back(label_data[i]);
        }
    }


    open3d::geometry::PointCloud pc = open3d::geometry::PointCloud(pc_points);
    std::vector<Eigen::Vector3d> points = pc.points_;

    return std::make_tuple(points, labels);
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>> ReadSceneNetDepthAndPredictLabels(cv::Mat depth_mat, std::vector<int> label_data, float fx, float fy, float cx, float cy, float min_range, float max_range) {
    
    // Convert depth mat
    std::vector<float> depth_data;
    depth_data.assign(
        (float*)depth_mat.datastart,
        (float*)depth_mat.dataend
    );
    
    std::vector<Eigen::Vector3d> pc_points; // store the 3D points for creating the pointcloud
    std::vector<uint32_t> labels; // store the labels that aren't tossed

    int num_rows = depth_mat.size().height;
    int num_cols = depth_mat.size().width;
    
    for (size_t u = 0; u < num_rows; u++) {
        for (size_t v = 0; v < num_cols; v++) {
            size_t i = num_cols * u + v;

            float d = depth_data[i];

            if (d <= 0 || std::isnan(d)) continue;
            if (d > max_range || d < min_range) continue; // Skip if out of range

            float x_norm = (v - cx) / fx;
            float y_norm = (u - cy) / fy;

            // Depth value as Euclidean distance, need to convert
            float norm_squared = x_norm * x_norm + y_norm * y_norm + 1.0f;
            float z = d / std::sqrt(norm_squared);

            float x = x_norm * z;
            float y = y_norm * z;

            Eigen::Vector3d p {x, y, z};

            pc_points.push_back(p);
            labels.push_back(label_data[i]);
        }
    }

    open3d::geometry::PointCloud pc = open3d::geometry::PointCloud(pc_points);
    std::vector<Eigen::Vector3d> points = pc.points_;

    return std::make_tuple(points, labels);
}

void PreProcessCloud(std::vector<Eigen::Vector3d>& points, std::vector<uint32_t>& labels, float min_range, float max_range) {
    bool invert = true;
    std::vector<bool> mask = std::vector<bool>(points.size(), invert);
    size_t pos = 0;
    for (auto & point : points) {
        if (point.norm() > max_range || point.norm() < min_range) {
            mask.at(pos) = false;
        }
        ++pos;
    }
    size_t counter = 0;
    for (size_t i = 0; i < points.size(); i++) {
        if (mask[i]) {
            points.at(counter) = points.at(i);
            labels.at(counter) = labels.at(i);
            ++counter;
        }
    }
    points.resize(counter);
    labels.resize(counter);
}

void TransformPoints(std::vector<Eigen::Vector3d>& points, const Eigen::Matrix4d& transformation) {
    for (auto& point : points) {
        Eigen::Vector4d new_point =
            transformation * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

std::vector<Eigen::Matrix4d> GetGTPoses(const fs::path& poses_file, const fs::path& calib_file, const bool rgbd_) {
    std::vector<Eigen::Matrix4d> poses;

    // auxiliary variables to read the txt files
    std::string ss;
    float P_00, P_01, P_02, P_03, P_10, P_11, P_12, P_13, P_20, P_21, P_22, P_23;

    std::ifstream poses_in(poses_file, std::ios_base::in);
    // clang-format off
    while (poses_in >>
            P_00 >> P_01 >> P_02 >> P_03 >>
            P_10 >> P_11 >> P_12 >> P_13 >>
            P_20 >> P_21 >> P_22 >> P_23) {
        Eigen::Matrix4d P;
        P << P_00, P_01, P_02, P_03,
             P_10, P_11, P_12, P_13,
             P_20, P_21, P_22, P_23,
             0.00, 0.00, 0.00, 1.00;
        poses.emplace_back(P);
    }
    // clang-format on
    return poses; // in camera coordinate frame
}

}  // namespace 
namespace datasets {

SceneNetDataset::SceneNetDataset(const std::string& scenenet_root_dir,
                           const std::string& sequence,
                           int n_scans,
                           bool rgbd) {
    rgbd_ = rgbd;
    // TODO: to be completed
    auto scenenet_root_dir_ = fs::absolute(fs::path(scenenet_root_dir));
    auto scenenet_sequence_dir = fs::absolute(fs::path(scenenet_root_dir) / "train/" / sequence);

    // Read data, cache it inside the class.
    poses_ = GetGTPoses(scenenet_sequence_dir / "poses.txt",
                        scenenet_sequence_dir / "intrinsics.txt", rgbd_);
    scan_files_ = GetDepthFiles(fs::absolute(scenenet_sequence_dir / "depth/"), n_scans);
}

SceneNetDataset::SceneNetDataset(const std::string& scenenet_root_dir,
                           const std::string& sequence,
                           int n_scans,
                           bool apply_pose,
                           bool preprocess,
                           float min_range,
                           float max_range,
                           bool rgbd,
                           bool realtime_segmentation)
    : apply_pose_(apply_pose),
      preprocess_(preprocess),
      min_range_(min_range),
      max_range_(max_range),
      rgbd_(rgbd),
      realtime_segmentation_(realtime_segmentation) {
    auto scenenet_root_dir_ = fs::absolute(fs::path(scenenet_root_dir));
    scenenet_sequence_dir_ = fs::absolute(fs::path(scenenet_root_dir_) / "train/" / sequence);

    // Read data, cache it inside the class.
    poses_ = GetGTPoses(scenenet_sequence_dir_ / "poses.txt",
                        scenenet_sequence_dir_ / "intrinsics.txt", rgbd_);
    const fs::path& calib_file = fs::absolute(scenenet_sequence_dir_ / "intrinsics.txt");
    if(rgbd_) {
        depth_files_ = GetDepthFiles(fs::absolute(scenenet_sequence_dir_ / "depth/"), n_scans);
        if (realtime_segmentation_) {
            img_files_ = GetImgFiles(fs::absolute(scenenet_sequence_dir_ / "photo/"), n_scans);
        }
        else {
            label_files_ = GetImgFiles(fs::absolute(scenenet_sequence_dir_ / "prediction/"), n_scans);
        }
    }
    else {
        std::cerr << "ERROR: There are no pointclouds for SceneNet. Please make sure that the rgbd option in the .yaml file is set to True." << std::endl;
    }

    // Load model
    try {
        model_ = torch::jit::load(fs::absolute(fs::path(scenenet_root_dir) / "weights" / "libtorch_nyuv2.pt"));
        std::cout << "Model loaded successfully\n";
        model_.to(device_);
        model_.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "ERROR: The model did not load successfully.\n";
        exit(0);
    }

    // Read image intrinsic parameters
    std::ifstream calib_in(calib_file, std::ios_base::in);

    if (!calib_in.is_open()) {
        std::cerr << "ERROR: Could not open the file " << calib_file << "\n";
        exit(0);
    }

    std::string line;
    while (std::getline(calib_in, line)) {
        if (line.empty()) continue;

        size_t colonPos = line.find(':');

        std::string key = line.substr(0, colonPos);
        std::string value = line.substr(colonPos + 1);

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "fx") {
            fx_ = std::stod(value);
        } else if (key == "fy") {
            fy_ = std::stod(value);
        } else if (key == "cx") {
            cx_ = std::stod(value);
        } else if (key == "cy") {
            cy_ = std::stod(value);
        }
    }
}

std::pair<cv::Mat, std::vector<int>> SceneNetDataset::RunInference(const std::string& rgb_path, const std::string& depth_path) const {
    // Load and preprocess RGB + depth
    cv::Mat img_mat = cv::imread(rgb_path, cv::IMREAD_COLOR);
    cv::cvtColor(img_mat, img_mat, cv::COLOR_BGR2RGB);
    img_mat.convertTo(img_mat, CV_32F, 1.0f / 255.0f);
    int H = img_mat.rows, W = img_mat.cols;

    cv::Mat depth_mat = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    depth_mat.convertTo(depth_mat, CV_32F, 1.0f/1000.0f); // SceneNet depth values are in millimeters

    // The model requires the dimensions to be a mutiple of 32, so pad image here
    int pad_h = (32 - H % 32) % 32;
    int pad_w = (32 - W % 32) % 32;
    cv::copyMakeBorder(img_mat, img_mat, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, 0);
    cv::copyMakeBorder(depth_mat, depth_mat, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, 0);

    // Convert cv::Mat to Torch tensors
    auto img_tensor = torch::from_blob(img_mat.data, {H+pad_h, W+pad_w, 3}, torch::kFloat32).permute({2, 0, 1}).unsqueeze(0).to(device_); // 1, C, H, W
    auto depth_tensor = torch::from_blob(depth_mat.data, {H+pad_h, W+pad_w}, torch::kFloat32).unsqueeze(0).unsqueeze(0).to(device_); // 1, 1, H, W

    // Forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);
    inputs.push_back(depth_tensor);

    torch::IValue iv = model_.forward(inputs); // shape: 1 x C x h' x w'

    at::Tensor out;
    auto elems = iv.toTuple()->elements();
    out = elems[0].toTensor();

    // Return to original resolution by slicing
    out = out.slice(2, 0, H).slice(3, 0, W);

    // Argmax over channel dimension
    out = out.argmax(1).squeeze(0).to(torch::kCPU).contiguous(); // H x W, int64

    out = out.to(torch::kInt).contiguous();
    int32_t* label_data_ptr = out.data_ptr<int32_t>();
    size_t numel = out.numel();

    // Convert from NYU classes to SceneNet
    for(size_t i = 0; i < numel; ++i) {
        int32_t v = label_data_ptr[i];             // original label 0-39
        label_data_ptr[i] = nyu40_to_scenenet_[v]; // new label 0-13
    }

    std::vector<int> label_data = std::vector<int>(label_data_ptr, label_data_ptr + numel);
    
    return {depth_mat, label_data};
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>, Eigen::Matrix4d> SceneNetDataset::operator[](int idx) const {
    if (rgbd_) {
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<Eigen::Vector3d> points;
        std::vector<uint32_t> semantics;

        if (realtime_segmentation_) {
            auto [depth_data, label_data] = RunInference(img_files_[idx], depth_files_[idx]);
            std::tie(points, semantics) = ReadSceneNetDepthAndPredictLabels(depth_data, label_data, fx_, fy_, cx_, cy_, min_range_, max_range_);
        }
        else {
            std::tie(points, semantics) = ReadSceneNetDepthAndLabels(depth_files_[idx], label_files_[idx], fx_, fy_, cx_, cy_, min_range_, max_range_);
        }

        if (apply_pose_) TransformPoints(points, poses_[idx]);
        auto t2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = t2 - t1;
        if (idx % 50 == 0 || idx == 299) std::cout << idx << " Preprocess time: " << elapsed.count()/1e3 << " ";
        return std::make_tuple(points, semantics, poses_[idx]);
    }
    else {
        std::cerr << "The only datatype option for SceneNet is rgbd!\n";
        exit(0);
    }
}
}  // namespace datasets
