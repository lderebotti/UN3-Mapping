import os
import sys
sys.path.append(os.path.abspath("./"))
import numpy as np
from numpy.linalg import inv
import torch
import open3d as o3d
from utils.config import Config
from torch.utils.data import Dataset
from dataset.data_utils import *
# import cv2

class DataLoader(Dataset):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dataset_name = config.dataset_name
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.device = config.device
        # TODO: can be more flexible like data_path + seq + calib/pose/pc
        self.data_path = config.data_path
        self.pose_path = config.pose_path
        self.calib_path = config.calib_path


        if self.dataset_name == 'kth':
            translation_file = open(self.pose_path)
            self.poses = []
            for line in translation_file:
                values = [float(v) for v in line.strip().split()]
                self.poses.append(values)
        else:
            calib = read_kitti_format_calib(self.calib_path)
            poses_uncalib = None
            if self.pose_path.endswith("txt"):
                poses_uncalib = read_kitti_format_poses(self.pose_path)
                poses_uncalib = np.array(poses_uncalib) # list to np
            else:
                sys.exit(f"Wrong pose file format{self.pose_path}.")
            self.poses = apply_kitti_format_calib(poses_uncalib, inv(calib["Tr"]))
            # if self.config.dataset_name == 'ncd':
            #     self.poses = apply_roate_z(self.poses, -15) # align hash grid

        # load all filenames for lidar data loading.
        self.pc_filenames = load_files_sorted(self.data_path)
        self.total_pc_count = len(self.pc_filenames) # total frames count in the folder

        # TODO: optional data
        self.semantic_on = config.semantic_on
        if self.semantic_on:
            self.label_path = config.label_path
        #     self.label_filenames = load_files_sorted(self.label_path)
            # self.label_filenames = self.label_filenames[start : end : every_n]
    
        # range image
        self.proj_H = config.proj_H
        self.proj_W = config.proj_W
        self.fov_up = config.fov_up
        self.fov_down = config.fov_down

    # load point cloud in self.
    def load_raw_points(self, frame_id):
        # load point cloud (support *pcd, *ply and kitti *bin format)
        frame_filename = os.path.join(self.data_path, self.pc_filenames[frame_id])
        if not self.semantic_on:
            #[N, 3], [N, 4] or [N, 6], may contain color or intensity # here read as numpy array
            point_cloud = read_point_cloud(frame_filename)
            sem_labels_reduced = None
        else:
            label_filename = os.path.join(
                self.label_path,
                self.pc_filenames[frame_id].replace("bin", "label"),
            )
            point_cloud, _, sem_labels_reduced = read_semantic_point_label(
                frame_filename, label_filename
            )
            
        # self.raw_points = point_cloud
        # self.labels = sem_labels_reduced
        return point_cloud, sem_labels_reduced
    
    def crop_frame(self, camera_location = None):
        min_z_th = self.config.min_z
        min_range = self.config.min_range
        max_range = self.config.max_range
        
        if self.dataset_name == 'kth':
            dist = torch.norm(self.points_torch-camera_location, p=2, dim=1)
            filtered_idx = (
                (dist > min_range)
                & (dist < max_range)
            )
        else:
            dist = torch.norm(self.points_torch, dim=1)
            filtered_idx = (
                (dist > min_range)
                & (dist < max_range)
                & (self.points_torch[:, 2] > min_z_th)
            )

        self.points_torch = self.points_torch[filtered_idx]
        self.normals_torch = self.normals_torch[filtered_idx]
    
    def process_frame(self, frame_idx):        
        raw_points, _ = self.load_raw_points(frame_idx)
        
        if self.config.dataset_name == 'kth':
            camera_location = self.poses[frame_idx]
            self.points_torch = torch.tensor(raw_points, device=self.device, dtype=self.dtype)
            # normal estimation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(raw_points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            pcd.orient_normals_towards_camera_location(camera_location) # orient to current location
            self.normals_torch = torch.tensor(np.asarray(pcd.normals), device=self.device, dtype=self.dtype)
            
            self.crop_frame(torch.tensor(camera_location).to(self.device))
            self.points_global_torch = self.points_torch # no need trans
            self.normals_global_torch = self.normals_torch # no need to rotate
        else:
            depth_map, vertex_map, proj_mask = range_projection(
                raw_points, self.fov_up, self.fov_down, self.proj_H, self.proj_W)
            normal_map, normal_mask = gen_normal_map(
                depth_map=depth_map, vertex_map=vertex_map)
            self.depth_map, self.normal_map = depth_map, normal_map
            
            #mask = torch.tensor(normal_mask.reshape(-1), device=self.device, dtype=torch.bool)
            #mask = torch.tensor(proj_mask.reshape(-1), device=self.device, dtype=torch.bool)
            self.points_torch = torch.tensor(vertex_map[proj_mask].reshape(-1,3), device=self.device, dtype=self.dtype)
            self.normals_torch = torch.tensor(normal_map[proj_mask].reshape(-1,3), device=self.device, dtype=self.dtype)
            self.crop_frame()
            self.pose_torch = torch.tensor(self.poses[frame_idx], device=self.device, dtype=self.dtype)
            self.points_global_torch = transform_torch(self.points_torch, self.pose_torch)
            self.normals_global_torch = self.normals_torch @ (self.pose_torch[:3, :3].transpose(-1, -2))
        
    def get_init_pose(self) -> np.array:
        if self.poses is not None:
            return self.poses[0]
        else:
            return np.eye(4)
    
    # def get_gt_pose(self, frame_idx):
    #     # load gt pose in torch
    #     cur_pose_ref = self.poses[frame_idx]
    #     self.pose_torch = torch.tensor(cur_pose_ref, device=self.device, dtype=self.dtype)
        
    #     return self.pose_torch

    def get_gt_translation(self, frame_idx):
        # load gt pose in torch
        if self.dataset_name == 'kth' :
            current_translation = torch.tensor(self.poses[frame_idx], dtype=torch.float32).to(self.device)
        else :
            pose = torch.tensor(self.poses[frame_idx], dtype=torch.float32, device=self.device)
            current_translation = pose[0:-1,-1]
        if self.dataset_name == 'ncd':
            current_translation = rotate_points_z_torch(current_translation, 15)
        return current_translation

    def get_points_global(self) -> torch.tensor:
        # must use process(idx) previously
        if self.dataset_name == 'ncd':
            self.points_global_torch = rotate_points_z_torch(self.points_global_torch, 15)
        return self.points_global_torch
        
    def get_normal_global(self) -> torch.tensor: 
        if self.dataset_name == 'ncd':
            self.normals_global_torch = rotate_points_z_torch(self.normals_global_torch, 15)

        # return normals_global
        return self.normals_global_torch

    def __len__(self):
        return self.total_pc_count

    def __getitem__(self, idx):
        self.process_frame(idx)
        pose = self.get_gt_pose(idx)
        return idx, pose, self.points_torch, self.normals_torch

    # for debug
    def get_range_image(self, idx):
        self.process_frame(idx)
        
        return idx, self.depth_map, self.normal_map

if __name__ == '__main__':

    config = Config()
    config.load(sys.argv[1])
    data_stream = DataLoader(config)
    pc_num = len(data_stream)
    print(f"The data stream contains {pc_num} frames.")
    # from tqdm import tqdm
    # for idx in tqdm(range(pc_num)):
    #     _, depth_map, normal_map = data_stream.get_range_image(idx)
    #     cv2.imshow('depth', depth_map/np.max(depth_map))
    #     normal_map_show = (normal_map+1)/2.0
    #     cv2.imshow('normal', normal_map_show)
    #     cv2.waitKey(1)
    