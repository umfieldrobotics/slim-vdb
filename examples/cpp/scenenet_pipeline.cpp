// MIT License
//
// # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
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

#include <fmt/core.h>
#include <fmt/format.h>
#include <igl/write_triangle_mesh.h>
#include <openvdb/openvdb.h>
#include <vdbfusion/VDBVolume.h>

#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "datasets/SceneNetOdometry.h"
#include "utils/Config.h"
#include "utils/Iterable.h"
#include "utils/Timers.h"

#include "open3d/Open3D.h"

#include <sstream>
#include <vector>
#include <chrono>


// Namespace aliases
using namespace fmt::literals;
using namespace utils;
namespace fs = std::filesystem;

namespace {

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("SceneNetPipeline");
    argparser.add_argument("scenenet_root_dir").help("The full path to the SceneNet dataset");
    argparser.add_argument("mesh_output_dir").help("Directory to store the resultant mesh");
    argparser.add_argument("--sequence").help("SceneNet Sequence");
    argparser.add_argument("--config")
        .help("Dataset specific config file")
        .default_value<std::string>("config/scenenet.yaml")
        .action([](const std::string& value) { return value; });
    argparser.add_argument("--n_scans")
        .help("How many scans to map")
        .default_value(int(-1))
        .action([](const std::string& value) { return std::stoi(value); });

    try {
        argparser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Invalid Arguments " << std::endl;
        std::cerr << err.what() << std::endl;
        std::cerr << argparser;
        exit(0);
    }

    auto scenenet_root_dir = argparser.get<std::string>("scenenet_root_dir"); 
    if (!fs::exists(scenenet_root_dir)) {
        std::cerr << scenenet_root_dir << "path doesn't exists" << std::endl;
        exit(1);
    }
    
    std::cout<< " the direction: " << scenenet_root_dir << std::endl;
    return argparser;
}
}  // namespace


int main(int argc, char* argv[]) {

    // Disable clog (debug) prints
    std::ofstream nullstream;
    std::clog.rdbuf(nullstream.rdbuf());

    auto argparser = ArgParse(argc, argv);

    // VDBVolume configuration
    auto vdbfusion_cfg = 
        vdbfusion::VDBFusionConfig::LoadFromYAML(argparser.get<std::string>("--config")); 
    // Dataset specific configuration
    auto scenenet_cfg = datasets::SceneNetConfig::LoadFromYAML(argparser.get<std::string>("--config"));

    openvdb::initialize();

    // SceneNet stuff
    auto n_scans = argparser.get<int>("--n_scans");
    auto scenenet_root_dir = fs::path(argparser.get<std::string>("scenenet_root_dir"));
    auto sequence = argparser.get<std::string>("--sequence"); 

    // Initialize dataset 
    const auto dataset =
        datasets::SceneNetDataset(scenenet_root_dir, sequence, n_scans, scenenet_cfg.apply_pose_,
                               scenenet_cfg.preprocess_, scenenet_cfg.min_range_, scenenet_cfg.max_range_, scenenet_cfg.rgbd_);

    fmt::print("Integrating {} scans\n", dataset.size());
    vdbfusion::VDBVolume tsdf_volume(vdbfusion_cfg.voxel_size_, vdbfusion_cfg.sdf_trunc_,
                                     vdbfusion_cfg.space_carving_, vdbfusion_cfg.min_weight_);
    timers::FPSTimer<10> timer;
    //modification--------------------------------cuda-------------------------------------------------------
    int index = 0; 
    //modification---------------------------------------------------------------------------------------
    for (const auto& [scan, semantics, pose] : iterable(dataset)) {
        // INTEGRATE //
        auto t1_s = std::chrono::high_resolution_clock::now();
        timer.tic();
        
        Eigen::Vector3d origin = pose.block<3, 1>(0, 3);
        tsdf_volume.Integrate(scan, semantics, origin, [](float /*unused*/) { return 1.0; });
        
        auto t1_e = std::chrono::high_resolution_clock::now();


        // REMDER //
        auto t2_s = std::chrono::high_resolution_clock::now();
        
        std::vector<double> origin_vec = {origin(0), origin(1), origin(2)};
        Eigen::Matrix3d rotation_matrix = pose.block<3,3>(0,0); // extract out translation
        // combine ego rotation with coordinate frame change
        Eigen::Quaterniond e_quat(rotation_matrix);
        Eigen::Quaterniond coord_frame_quat(0, 1, 0, 0);
        e_quat = e_quat * coord_frame_quat;
        std::vector<double> rot_quat_vec = {e_quat.x(), e_quat.y(), e_quat.z(), e_quat.w()};
        tsdf_volume.Render(origin_vec, rot_quat_vec, index);
        index++;
        
        timer.toc();
        auto t2_e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = t1_e - t1_s;
        std::chrono::duration<double, std::milli> elapsed2 = t2_e - t2_s;
        if (index % 1 == 0) std::cout << index << " Integrate time: " << elapsed1.count()/1e3 << " Render time: " << elapsed2.count()/1e3 << std::endl;
    }

    // Store the grid results to disks
    std::string map_name = fmt::format("{out_dir}/scenenet_{seq}_{n_scans}_scans",
                                       "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                       "seq"_a = sequence, "n_scans"_a = n_scans);
    {
        timers::ScopeTimer timer("Writing VDB grid to disk");
        auto tsdf_grid = tsdf_volume.tsdf_;
        std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name);
        openvdb::io::File(filename).write({tsdf_grid});
    }

    std::string map_name_semantics = fmt::format("{out_dir}/scenenet_{seq}_{n_scans}_semantics",
                                       "out_dir"_a = argparser.get<std::string>("mesh_output_dir"),
                                       "seq"_a = sequence, "n_scans"_a = n_scans);
    {
        timers::ScopeTimer timer2("Writing semantic VDB grid to disk");
        auto semantics_grid = tsdf_volume.semantics_;
        std::string filename = fmt::format("{map_name}.vdb", "map_name"_a = map_name_semantics);
        openvdb::io::File(filename).write({semantics_grid});
    }

    // Run marching cubes and save a .ply file
    {
        // for semantics, we will save one mesh for each label--28 in total. the label will be associated with the face. TODO discuss this further
        // triangles[0-2] is for map, triangles[3] is label?
        timers::ScopeTimer timer("Writing Mesh to disk");
        auto [vertices, triangles, labels] = // labels belong to triangles
            tsdf_volume.ExtractTriangleMesh(vdbfusion_cfg.fill_holes_, vdbfusion_cfg.min_weight_);

        // TODO: Fix this!
        Eigen::MatrixXd V(vertices.size(), 3);
        for (size_t i = 0; i < vertices.size(); i++) {
            V.row(i) = Eigen::VectorXd::Map(&vertices[i][0], vertices[i].size());
        }

        // TODO: Also this
        // We want a list of matrices, one for each label
        std::map<int, Eigen::MatrixXi> tri_map; // label: (matrix, #elements)
        std::map<int, int32_t> tri_map_sizes; // label: (matrix, #elements)

        for (size_t i = 0; i < triangles.size(); i++) {
            if (tri_map.find(labels[i]) == tri_map.end()) {
                Eigen::MatrixXi F(triangles.size(), 3);
                tri_map[labels[i]] = F;
                tri_map_sizes[labels[i]] = 0;
            }
            auto curr_num_triangles = tri_map_sizes[labels[i]];
            tri_map[labels[i]].row(curr_num_triangles) = Eigen::VectorXi::Map(&triangles[i][0], triangles[i].size());
            tri_map_sizes[labels[i]]++;
        }

        // Now iterate through the map and save each mesh
        for (auto const& [class_label, matrix] : tri_map) {
            std::string filename = fmt::format("{map_name}_{class_label}.ply", fmt::arg("map_name", map_name), fmt::arg("class_label", class_label));
            // truncate matrix so it's just the number of triangles
            auto num_triangles = tri_map_sizes[class_label];
            Eigen::MatrixXi new_matrix(num_triangles, 3);
            for (size_t i = 0; i < num_triangles; i++) {
                new_matrix.row(i) = matrix.row(i);
            }
            igl::write_triangle_mesh(filename, V, new_matrix, igl::FileEncoding::Ascii);
        }
        
    }

    return 0;
}
