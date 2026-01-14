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

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "yaml-cpp/yaml.h"

namespace slimvdb {
struct SLIMVDBConfig {
    float voxel_size_;
    float sdf_trunc_;
    bool space_carving_;
    int prune_interval_;
    float min_weight_;
    bool fill_holes_;
    float p_threshold_;

    static inline SLIMVDBConfig LoadFromYAML(const std::string& path) {
        std::ifstream config_file(path, std::ios_base::in);
        auto config = YAML::Load(config_file);

        return SLIMVDBConfig{config["voxel_size"].as<float>(),
                               config["sdf_trunc"].as<float>(),
                               config["space_carving"].as<bool>(),
                               config["prune_interval"].as<int>(),
                               config["min_weight"].as<float>(),
                               config["fill_holes"].as<bool>(),
                               config["p_threshold"].as<float>()};
    }
};
}  // namespace slimvdb

namespace datasets {
struct KITTIConfig {
    bool apply_pose_;
    bool preprocess_;
    float min_range_;
    float max_range_;
    bool rgbd_;
    bool realtime_segmentation_;
    int render_img_width_;
    int render_img_height_;
    std::map<int, std::vector<int>> color_map_;

    static inline KITTIConfig LoadFromYAML(const std::string& path) {
        std::ifstream config_file(path, std::ios_base::in);
        auto config = YAML::Load(config_file);

        std::map<int, std::vector<int>> tmp_color_map;
        for (auto color : config["color_map"] )
        {
            tmp_color_map.insert(std::make_pair(color.first.as<int>(), color.second.as<std::vector<int>>()));
        }

        return KITTIConfig{config["apply_pose"].as<bool>(),
                           config["preprocess"].as<bool>(),
                           config["min_range"].as<float>(),
                           config["max_range"].as<float>(),
                           config["rgbd"].as<bool>(),
                           config["realtime_segmentation"].as<bool>(),
                           config["render_img_width"].as<int>(),
                           config["render_img_height"].as<int>(),
                           tmp_color_map};
    }
};

struct SceneNetConfig {
    bool apply_pose_;
    bool preprocess_;
    float min_range_;
    float max_range_;
    bool rgbd_;
    bool realtime_segmentation_;
    int render_img_width_;
    int render_img_height_;
    std::map<int, std::vector<int>> color_map_;

    static inline SceneNetConfig LoadFromYAML(const std::string& path) {
        std::ifstream config_file(path, std::ios_base::in);
        auto config = YAML::Load(config_file);

        std::map<int, std::vector<int>> tmp_color_map;
        for (auto color : config["color_map"] )
        {
            tmp_color_map.insert(std::make_pair(color.first.as<int>(), color.second.as<std::vector<int>>()));
        }

        return SceneNetConfig{config["apply_pose"].as<bool>(),
                           config["preprocess"].as<bool>(),
                           config["min_range"].as<float>(),
                           config["max_range"].as<float>(),
                           config["rgbd"].as<bool>(),
                           config["realtime_segmentation"].as<bool>(),
                           config["render_img_width"].as<int>(),
                           config["render_img_height"].as<int>(),
                           tmp_color_map};
    }
};

struct RealWorldConfig {
    bool apply_pose_;
    bool preprocess_;
    float min_range_;
    float max_range_;
    bool rgbd_;
    bool realtime_segmentation_;
    int render_img_width_;
    int render_img_height_;
    std::map<int, std::vector<int>> color_map_;

    static inline RealWorldConfig LoadFromYAML(const std::string& path) {
        std::ifstream config_file(path, std::ios_base::in);
        auto config = YAML::Load(config_file);

        std::map<int, std::vector<int>> tmp_color_map;
        for (auto color : config["color_map"] )
        {
            tmp_color_map.insert(std::make_pair(color.first.as<int>(), color.second.as<std::vector<int>>()));
        }

        return RealWorldConfig{config["apply_pose"].as<bool>(),
                           config["preprocess"].as<bool>(),
                           config["min_range"].as<float>(),
                           config["max_range"].as<float>(),
                           config["rgbd"].as<bool>(),
                           config["realtime_segmentation"].as<bool>(),
                           config["render_img_width"].as<int>(),
                           config["render_img_height"].as<int>(),
                           tmp_color_map};
    }
};
}  // namespace datasets