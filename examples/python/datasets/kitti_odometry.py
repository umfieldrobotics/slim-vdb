# MIT License
#
# Copyright (c) 2025 Anja Sheppard, University of Michigan
# Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trimesh import transform_points

import sys

sys.path.append("..")

from utils.cache import get_cache, memoize
from utils.config import load_config


class KITTIOdometryDataset:
    def __init__(self, kitti_root_dir: str, sequence: int, config_file: str):
        """Simple KITTI DataLoader to provide a ready-to-run example.

        Heavily inspired in PyLidar SLAM
        """
        # Config stuff
        self.sequence = str(int(sequence)).zfill(2)
        self.config = load_config(config_file)
        self.kitti_sequence_dir = os.path.join(kitti_root_dir, "sequences", self.sequence)
        if self.config['rgbd'] == True and self.config['pointcloud'] == False:
            self.image_dir = os.path.join(self.kitti_sequence_dir, "image_2/")
            self.depth_dir = os.path.join(self.kitti_sequence_dir, "depth_tif/")
            self.label_dir = os.path.join(self.kitti_sequence_dir, "image_2_labels/")
            
            self.image_files = sorted(glob.glob(self.image_dir + "*.png"))
            self.depth_files = sorted(glob.glob(self.depth_dir + "*.tif"))
            if os.path.exists(self.label_dir):
                self.label_files = sorted(glob.glob(self.label_dir + "*.png"))
            else:
                print("No ground truth label files found. Continuing with methods to predict and track semantics.")
                self.label_files = None
        elif self.config['rgbd'] == False and self.config['pointcloud'] == True:
            self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")  
            self.label_dir = os.path.join(self.kitti_sequence_dir, "labels/")
            
            self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
            if os.path.exists(self.label_dir):
                self.label_files = sorted(glob.glob(self.label_dir + "*.label"))
            else:
                print("No ground truth label files found. Continuing with methods to predict and track semantics.")
                self.label_files = None

        else:
            print("ERROR: you must indicate True or False for input types rgbd or pointcloud in the config file!")
            exit(1)

        # Read stuff
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))
        self.poses = self.load_poses(os.path.join(kitti_root_dir, f"poses/{self.sequence}.txt"))

        # Cache
        self.use_cache = True
        self.cache = get_cache(directory="cache/kitti/")
        
        # ConvBKI labels
        self.label_to_id = {(0, 128, 128): 0, (0, 0, 128): 1, (128, 0, 64): 2, (128, 64, 64): 3, (128, 128, 128): 4, (128, 128, 192): 5, 
                            (128, 192, 192): 6, (192, 0, 0): 7, (192, 128, 0): 8, (128, 64, 128): 9, (0, 64, 64): 10}

    def __getitem__(self, idx):
        if self.config['rgbd'] == True and self.config['pointcloud'] == False:
            points, labels = self.rgbds_and_labels(idx)
            return points, labels, self.poses[idx]
        elif self.config['rgbd'] == False and self.config['pointcloud'] == True:
            return self.scans(idx), self.labels(idx), self.poses[idx]
        else:
            print("ERROR: you must indicate True or False for input types rgbd or pointcloud in the config file!")
            exit(1)

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(idx, self.scan_files[idx], self.config)

    def rgbds_and_labels(self, idx):
        return self.read_rgbds(idx, self.image_files[idx], self.depth_files[idx], self.label_files[idx], self.config)

    def labels(self, idx):
        if self.label_files is not None:
            return self.read_labels(self.label_files[idx])
        else:
            return None

    # @memoize()
    def read_point_cloud(self, idx: int, scan_file: str, config: dict):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))
        points = self._correct_scan(points) if config.correct_scan else points[:, :3]
        points = transform_points(points, self.poses[idx]) if config.apply_pose else None
        return points
    
    # @memoize()
    def read_rgbds(self, idx: int, image_file: str, depth_file:str, label_file:str, config: dict):
        depth_data = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        labels_in = self.read_rgbd_labels(label_file)
        valid_mask = (depth_data > 0) & (depth_data <= config['max_range'])
        valid_depth_data = depth_data[valid_mask]
        i_indices, j_indices = np.nonzero(valid_mask)
        z = valid_depth_data
        x = (j_indices - config['cx']) * z / config['fx']
        y = (i_indices - config['cy']) * z / config['fy']
        points = np.column_stack((x, y, z))
        valid_labels_in = labels_in[valid_mask]
        reshaped_labels = valid_labels_in.reshape(-1, valid_labels_in.shape[-1])
        unique_labels, inverse_indices = np.unique(reshaped_labels, axis=0, return_inverse=True)
        label_id_map = np.array([self.label_to_id[tuple(label)] for label in unique_labels])
        labels = label_id_map[inverse_indices].reshape(valid_labels_in.shape[:-1]).astype(np.int32)

        points = self._correct_scan(points) if config.correct_scan else points[:, :3]
        points = transform_points(points, self.poses[idx]) if config.apply_pose else None
        return points, labels

    def read_rgbd_labels(self, label_file):
        return cv2.imread(label_file, cv2.IMREAD_UNCHANGED)

    def read_labels(self, label_file):
        return np.fromfile(label_file, dtype=np.int32).reshape((-1)) # n x 1 vector where each int contains the semantic label in the lower 16 bits and the instance label in the upper 16 bits

    @staticmethod
    def _correct_scan(scan: np.ndarray):
        """Corrects the calibration of KITTI's HDL-64 scan.

        Taken from PyLidar SLAM
        """
        xyz = scan[:, :3]
        n = scan.shape[0]
        z = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (n, 1))
        axes = np.cross(xyz, z)
        # Normalize the axes
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        theta = 0.205 * np.pi / 180.0

        # Build the rotation matrix for each point
        c = np.cos(theta)
        s = np.sin(theta)

        u_outer = axes.reshape(n, 3, 1) * axes.reshape(n, 1, 3)
        u_cross = np.zeros((n, 3, 3), dtype=np.float32)
        u_cross[:, 0, 1] = -axes[:, 2]
        u_cross[:, 1, 0] = axes[:, 2]
        u_cross[:, 0, 2] = axes[:, 1]
        u_cross[:, 2, 0] = -axes[:, 1]
        u_cross[:, 1, 2] = -axes[:, 0]
        u_cross[:, 2, 1] = axes[:, 0]

        eye = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        rotations = c * eye + s * u_cross + (1 - c) * u_outer
        corrected_scan = np.einsum("nij,nj->ni", rotations, xyz)
        return corrected_scan

    def load_poses(self, poses_file): # TODO FIX FOR CAMERA POSES
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            if self.config['rgbd'] == True and self.config['pointcloud'] == False:
                return left
            elif self.config['rgbd'] == False and self.config['pointcloud'] == True:
                return right
            else:
                print("ERROR: you must indicate True or False for input types rgbd or pointcloud in the config file!")
                exit(1)

        poses = pd.read_csv(poses_file, sep=" ", header=None).values
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict