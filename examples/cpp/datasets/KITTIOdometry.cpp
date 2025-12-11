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

#include "KITTIOdometry.h"

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

namespace fs = std::filesystem;

namespace {

std::vector<std::string> GetVelodyneFiles(const fs::path& velodyne_path, int n_scans) {
    std::vector<std::string> velodyne_files;
    for (const auto& entry : fs::directory_iterator(velodyne_path)) {
        if (entry.path().extension() == ".bin") {
            velodyne_files.emplace_back(entry.path().string());
        }
    }
    if (velodyne_files.empty()) {
        std::cerr << velodyne_path << "path doesn't have any .bin" << std::endl;
        exit(1);
    }
    std::sort(velodyne_files.begin(), velodyne_files.end());
    if (n_scans > 0) {
        velodyne_files.erase(velodyne_files.begin() + n_scans, velodyne_files.end());
    }
    return velodyne_files;
}

std::vector<std::string> GetDepthFiles(const fs::path& depth_path, int n_scans) {
    std::vector<std::string> depth_files;
    for (const auto& entry : fs::directory_iterator(depth_path)) {
        if (entry.path().extension() == ".tif") {
            depth_files.emplace_back(entry.path().string());
        }
    }
    if (depth_files.empty()) {
        std::cerr << depth_path << "path doesn't have any .tif" << std::endl;
        exit(1);
    }
    std::sort(depth_files.begin(), depth_files.end());
    if (n_scans > 0) {
        depth_files.erase(depth_files.begin() + n_scans, depth_files.end());
    }
    return depth_files;
}

std::vector<std::string> GetGTLabelFiles(const fs::path& label_path, int n_scans) {
    std::vector<std::string> label_files;
    for (const auto& entry : fs::directory_iterator(label_path)) {
        if (entry.path().extension() == ".txt") {
            label_files.emplace_back(entry.path().string());
        }
    }
    if (label_files.empty()) {
        std::cerr << label_path << "path doesn't have any .txt" << std::endl;
        exit(1);
    }
    std::sort(label_files.begin(), label_files.end());
    if (n_scans > 0) {
        label_files.erase(label_files.begin() + n_scans, label_files.end());
    }
    return label_files;
}

std::vector<std::string> GetLabelFiles(const fs::path& label_path, int n_scans) {
    std::vector<std::string> label_files;
    for (const auto& entry : fs::directory_iterator(label_path)) {
        if (entry.path().extension() == ".png") {
            label_files.emplace_back(entry.path().string());
        }
    }
    if (label_files.empty()) {
        std::cerr << label_path << "path doesn't have any .png" << std::endl;
        exit(1);
    }
    std::sort(label_files.begin(), label_files.end());
    if (n_scans > 0) {
        label_files.erase(label_files.begin() + n_scans, label_files.end());
    }
    return label_files;
}

std::vector<Eigen::Vector3d> ReadKITTIVelodyne(const std::string& path) {
    std::ifstream scan_input(path.c_str(), std::ios::binary);
    assert(scan_input.is_open() && "ReadPointCloud| not able to open file");

    scan_input.seekg(0, std::ios::end);
    uint32_t num_points = scan_input.tellg() / (4 * sizeof(float));
    scan_input.seekg(0, std::ios::beg);

    std::vector<float> values(4 * num_points);
    scan_input.read((char*)&values[0], 4 * num_points * sizeof(float));
    scan_input.close();

    std::vector<Eigen::Vector3d> points;
    points.resize(num_points);
    for (uint32_t i = 0; i < num_points; i++) {
        points[i].x() = values[i * 4];
        points[i].y() = values[i * 4 + 1];
        points[i].z() = values[i * 4 + 2];
    }

    return points; // returned in metric Velodyne coordinate frame
}

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>> ReadKITTIDepthAndLabels(const std::string& depth_path, const std::string& label_path, float fx, float fy, float cx, float cy, bool preprocess, float min_range, float max_range) {
    // Read depth and label
    cv::Mat depth_mat = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
    cv::Mat label_mat = cv::imread(label_path, cv::IMREAD_GRAYSCALE);

    assert(depth_mat.size() == label_mat.size());

    // Convert cv Mat to vector
    std::vector<float> depth_data;
    std::vector<int> label_data(label_mat.total());

    depth_mat.convertTo(depth_mat, CV_32F);
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
            size_t i = num_cols*u+v;

            float z = depth_data[i];
            if (z < 0 || std::isnan(z)) z = 0;
            if (preprocess)
                if (z > max_range || z < min_range) continue; // Skip if out of range

            float x = (v - cx) * z / fx;
            float y = (u - cy) * z / fy;

            Eigen::Vector3d p {x, y, z};

            pc_points.push_back(p);
            labels.push_back(label_data[i]);
        }
    }

    open3d::geometry::PointCloud pc = open3d::geometry::PointCloud(pc_points);
    std::vector<Eigen::Vector3d> points = pc.points_;

    return std::make_tuple(points, labels);
}

std::vector<uint32_t> ReadKITTIGroundTruthLabels(const std::string& path) {
    // Load semantics info from text file
    std::ifstream infile(path.c_str()); // Open the input file
    assert(infile.is_open() && "ReadKITTISemantics| not able to open file");

    std::vector<uint32_t> labels; // Vector to store uint32_t numbers
    std::string line;

    while (std::getline(infile,line)) {
        std::stringstream ss(line);
        std::vector<uint32_t> v;

        std::string word;
        while(ss >> word){ // read in format "inst_label semantic_label" from text file
            v.push_back(std::stoul(word));
        }

        uint32_t label = (static_cast<uint32_t>(v[0]) << 16) | v[1]; // encode instance label in upper 16 bits and semantic label in lower 16 bits

        labels.push_back(label); // push back instance and semantic label in one uint32_t
    }

    infile.close(); // Close the file

    return labels;
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
    Eigen::Matrix4d T_cam_velo = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d T_velo_cam = Eigen::Matrix4d::Zero();

    // auxiliary variables to read the txt files
    std::string ss;
    float P_00, P_01, P_02, P_03, P_10, P_11, P_12, P_13, P_20, P_21, P_22, P_23;

    std::ifstream calib_in(calib_file, std::ios_base::in);
    // clang-format off
    while (calib_in >> ss >>
           P_00 >> P_01 >> P_02 >> P_03 >>
           P_10 >> P_11 >> P_12 >> P_13 >>
           P_20 >> P_21 >> P_22 >> P_23) {
        if (ss == "Tr:") {
            T_cam_velo << P_00, P_01, P_02, P_03,
                          P_10, P_11, P_12, P_13,
                          P_20, P_21, P_22, P_23,
                          0.00, 0.00, 0.00, 1.00;
            T_velo_cam = T_cam_velo.inverse();
        }
    }
    // clang-format on

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
        if (rgbd_) {
            poses.emplace_back(T_velo_cam * P); // IF FROM DEPTH
        }
        else {
            poses.emplace_back(T_velo_cam * P * T_cam_velo); // IF FROM VELODYNE
        }
    }
    // clang-format on
    return poses; // in velodyne coordinate frame
}

}  // namespace 
namespace datasets {

KITTIDataset::KITTIDataset(const std::string& kitti_root_dir,
                           const std::string& sequence,
                           int n_scans,
                           bool rgbd) {
    rgbd_ = rgbd;
    // TODO: to be completed
    auto kitti_root_dir_ = fs::absolute(fs::path(kitti_root_dir));
    auto kitti_sequence_dir = fs::absolute(fs::path(kitti_root_dir) / "sequences" / sequence);

    // Read data, cache it inside the class.
    poses_ = GetGTPoses(kitti_root_dir_ / "poses" / std::string(sequence + ".txt"),
                        kitti_sequence_dir / "calib.txt", rgbd_);
    scan_files_ = GetVelodyneFiles(fs::absolute(kitti_sequence_dir / "velodyne/"), n_scans);
}

KITTIDataset::KITTIDataset(const std::string& kitti_root_dir,
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
    auto kitti_root_dir_ = fs::absolute(fs::path(kitti_root_dir));
    kitti_sequence_dir_ = fs::absolute(fs::path(kitti_root_dir) / "sequences" / sequence);

    // Read data, cache it inside the class.
    poses_ = GetGTPoses(kitti_sequence_dir_ / "poses.txt",
                        kitti_sequence_dir_ / "calib.txt", rgbd_);
    const fs::path& calib_file = fs::absolute(kitti_sequence_dir_ / "intrinsics.txt");
    if(rgbd_) {
        depth_files_ = GetDepthFiles(fs::absolute(kitti_sequence_dir_ / "depth_tif/"), n_scans);
        label_files_ = GetLabelFiles(fs::absolute(kitti_sequence_dir_ / "image_2_labels/"), n_scans);
    }
    else {
        scan_files_ = GetVelodyneFiles(fs::absolute(kitti_sequence_dir_ / "velodyne/"), n_scans);
        gt_label_files_ = GetGTLabelFiles(fs::absolute(kitti_sequence_dir_ / "velodyne_predictions_txt/"), n_scans);
    }

    // Read image intrinsic parameters
    std::ifstream calib_in(calib_file, std::ios_base::in);

    if (!calib_in.is_open()) {
        std::cerr << "Error: Could not open the file " << calib_file << "\n";
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

std::tuple<std::vector<Eigen::Vector3d>, std::vector<uint32_t>, Eigen::Matrix4d> KITTIDataset::operator[](int idx) const {
    if (rgbd_) {
        auto [points, semantics] = ReadKITTIDepthAndLabels(depth_files_[idx], label_files_[idx], fx_, fy_, cx_, cy_, preprocess_, min_range_, max_range_);

        if (apply_pose_) TransformPoints(points, poses_[idx]);
        return std::make_tuple(points, semantics, poses_[idx]);
    }
    else {
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<Eigen::Vector3d> points;
        std::vector<uint32_t> semantics;

        if (realtime_segmentation_) {
            std::cerr << "The only segmentation option for KITTI is non-realtime segmentation (at the moment)!\n";
            exit(0);
        }
        else {
            points = ReadKITTIVelodyne(scan_files_[idx]);
            semantics = ReadKITTIGroundTruthLabels(gt_label_files_[idx]);
        }

        if (preprocess_) PreProcessCloud(points, semantics, min_range_, max_range_);
        if (apply_pose_) TransformPoints(points, poses_[idx]);

        auto t2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = t2 - t1;
        if (idx % 25 == 0 || idx == 99) std::cout << idx << " Preprocess time: " << elapsed.count()/1e3 << " ";
        return std::make_tuple(points, semantics, poses_[idx]);
    }
}
}  // namespace datasets
