import math
import sys
import os
from typing import List
import numpy as np
import torch
from numpy.linalg import inv
from natsort import natsorted
from glob import glob
import open3d as o3d

def load_files_sorted(folder):
    """ Load all files in a folder and sort.
    """
    return natsorted(os.listdir(folder))

def read_point_cloud(filename: str) -> np.ndarray:
    # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
    if ".bin" in filename:
        points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
    elif ".ply" in filename or ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points)
    else:
        sys.exit(
            "The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)"
        )
    return points  # as np

def load_nclt_bin(file_path: str):
    def _convert(x_s, y_s, z_s):
        # Copied from http://robots.engin.umich.edu/nclt/python/read_vel_sync.py
        scaling = 0.005
        offset = -100.0

        x = x_s * scaling + offset
        y = y_s * scaling + offset
        z = z_s * scaling + offset
        return x, y, z

    binary = np.fromfile(file_path, dtype=np.int16)
    x = np.ascontiguousarray(binary[::4])
    y = np.ascontiguousarray(binary[1::4])
    z = np.ascontiguousarray(binary[2::4])
    x = x.astype(np.float32).reshape(-1, 1)
    y = y.astype(np.float32).reshape(-1, 1)
    z = z.astype(np.float32).reshape(-1, 1)
    x, y, z = _convert(x, y, z)
    # Flip to have z pointing up
    points = np.concatenate([x, -y, -z], axis=1)
    return points.astype(np.float64)

# now we only support semantic kitti format dataset
def read_semantic_point_label(
    filename: str, label_filename: str, color_on: bool = False
):
    # read point cloud (kitti *.bin format)
    if ".bin" in filename:
        # we also read the intensity channel here
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)[:, :3]
    elif ".ply" in filename or ".pcd" in filename:
        pc_load = o3d.io.read_point_cloud(filename)
        points = np.asarray(pc_load.points)
    else:
        sys.exit("The format of the imported point cloud is wrong (support only *bin)")

    # read point cloud labels (*.label format)
    if ".label" in label_filename:
        labels = np.fromfile(label_filename, dtype=np.uint32).reshape(-1)
    else:
        sys.exit(
            "The format of the imported point labels is wrong (support only *label)"
        )

    labels = labels & 0xFFFF  # only take the semantic part

    # get the reduced label [0-20]
    labels_reduced = np.vectorize(sem_map_function)(labels).astype(
        np.int32
    )  # fast version

    # original label [0-255]
    labels = np.array(labels, dtype=np.int32)

    return points, labels, labels_reduced  # as np

def read_kitti_format_calib(filename: str):
    """
    read calibration file (with the kitti format)
    returns -> dict calibration matrices as 4*4 numpy arrays
    """
    calib = {}
    calib_file = open(filename)

    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        pose = np.zeros((4, 4))

        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()
    return calib


def read_kitti_format_poses(filename: str) -> List[np.ndarray]:
    """
    read pose file (with the kitti format)
    returns -> list, transformation before calibration transformation
    if the format is incorrect, return None
    """
    poses = []
    with open(filename, 'r') as file:            
        for line in file:
            values = line.strip().split()
            if len(values) != 12:
                print('Not a kitti format pose file')
                return None

            values = [float(value) for value in values]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append(pose)
    
    return poses

# def read_tum_format_poses(filename: str):
#     """
#     read pose file (with the tum format), support txt file
#     # timestamp tx ty tz qx qy qz qw
#     returns -> list, transformation before calibration transformation
#     """
#     from pyquaternion import Quaternion

#     poses = []
#     timestamps = []
#     with open(filename, 'r') as file:
#         first_line = file.readline().strip()
        
#         # check if the first line contains any numeric characters
#         # if contain, then skip the first line # timestamp tx ty tz qx qy qz qw
#         if any(char.isdigit() for char in first_line):
#             file.seek(0)
        
#         for line in file: # read each line in the file 
#             values = line.strip().split()
#             if len(values) != 8 and len(values) != 9: 
#                 print('Not a tum format pose file')
#                 return None, None
#             # some tum format pose file also contain the idx before timestamp
#             idx_col =  len(values) - 8 # 0 or 1
#             values = [float(value) for value in values]
#             timestamps.append(values[idx_col])
#             trans = np.array(values[1+idx_col:4+idx_col])
#             quat = Quaternion(np.array([values[7+idx_col], values[4+idx_col], values[5+idx_col], values[6+idx_col]])) # w, i, j, k
#             rot = quat.rotation_matrix
#             # Build numpy transform matrix
#             odom_tf = np.eye(4)
#             odom_tf[0:3, 3] = trans
#             odom_tf[0:3, 0:3] = rot
#             poses.append(odom_tf)
    
#     return poses, timestamps

def apply_kitti_format_calib(poses_np: np.ndarray, calib_T_cl: np.ndarray):
    """Converts from Velodyne to Camera Frame (# T_camera<-lidar)"""
    poses_calib_np = poses_np.copy()
    for i in range(poses_np.shape[0]):
        poses_calib_np[i, :, :] = calib_T_cl @ poses_np[i, :, :] @ inv(calib_T_cl)
    return poses_calib_np

def apply_roate_z(poses_np: np.ndarray, degree = -15):
    # Compute Z-axis rotation matrix
    theta = np.radians(degree)  # Convert degrees to radians
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    poses_rorated = poses_np.copy()
    for i in range(poses_np.shape[0]):
        # Apply calibration
        poses_rorated[i, :, :] =  R_z @ poses_np[i, :, :]
    return poses_rorated

def apply_roate_z(poses_np: np.ndarray, degree = -15):
    # Compute Z-axis rotation matrix
    theta = np.radians(degree)  # Convert degrees to radians
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    poses_rorated = poses_np.copy()
    for i in range(poses_np.shape[0]):
        # Apply calibration
        poses_rorated[i, :, :] =  R_z @ poses_np[i, :, :]
    return poses_rorated


def rotate_points_z_torch(points: torch.Tensor, degree):
    theta = torch.deg2rad(torch.tensor(degree, dtype=torch.float32))
    R_z = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta),  torch.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32).cuda()
    
    rotated_points = points @ R_z

    return rotated_points

# def transform_torch(points: torch.tensor, transformation: torch.tensor):
#     # points [N, 3]
#     # transformation [4, 4]
#     # Add a homogeneous coordinate to each point in the point cloud
#     points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

#     # Apply the transformation by matrix multiplication
#     transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

#     # Remove the homogeneous coordinate from each transformed point
#     transformed_points = transformed_points_homo[:, :3]

#     return transformed_points

def transform_torch(points: torch.tensor, transformation: torch.tensor):
    R = transformation[:3, :3].transpose(-1, -2)
    t = transformation[:3, 3]
    points_global = points @ R + t
    
    return points_global

def range_projection(raw_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=80):
    """ Project a pointcloud into a spherical projection, range image.
        Args:
            raw_vertex: raw point clouds
        Returns: 
            proj_range: projected range image with depth, each pixel contains the corresponding depth
            proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
            proj_mask: each pixel contains the flag indicating whether this pixel correspond a data.
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
    
    # get depth of all points
    depth = np.linalg.norm(raw_vertex, 2, axis=1)
    raw_vertex = raw_vertex[(depth > 0) & (depth < max_range)]
    depth = depth[(depth > 0) & (depth < max_range)]
    
    # get scan components
    scan_x = raw_vertex[:, 0]
    scan_y = raw_vertex[:, 1]
    scan_z = raw_vertex[:, 2]
    
    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    
    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    
    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]
    
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    
    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]
    
    # indices = np.arange(depth.shape[0])
    # indices = indices[order]
    
    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 3), -1,
                            dtype=np.float32)  # [H,W] index (-1 is no data)
    # proj_idx = np.full((proj_H, proj_W), -1,
    #                     dtype=np.int32)  # [H,W] index (-1 is no data)
    
    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z]).T
    #proj_idx[proj_y, proj_x] = indices
    proj_mask = proj_range > 0
    
    return proj_range, proj_vertex, proj_mask

def gen_normal_map(depth_map, vertex_map, delta_ratio = 0.15):
    """Generate a normal image given the range projection of a point cloud.
    Args:
        depth_map: range projection of a point cloud, each pixel contains the corresponding depth
        vertex_map: range projection of a point cloud, each pixel contains the corresponding point (x, y, z)
    Returns:
        normal_map: each pixel contains the corresponding normal
    """
    proj_H = depth_map.shape[0]
    proj_W = depth_map.shape[1]
    normal_map = np.zeros_like(vertex_map, dtype=np.float32)
    normal_mask = np.zeros_like(depth_map, dtype=np.int32)

    # Get the valid depth indices [0] hight [1] width
    valid_indices = np.where(depth_map > 0)
    normal_mask[valid_indices] = 1 # initial value

    # Get the 3D coordinates of the valid points
    p = vertex_map[valid_indices][:, :3]
    depth_p = depth_map[valid_indices]
    # horizontal neighboring points
    #n_u = (valid_indices[1] + 1) % proj_W # ring
    n_u = valid_indices[1] + 1
    n_u[n_u==proj_W] = 0
    p_nu = vertex_map[valid_indices[0], n_u]
    depth_pnu = depth_map[valid_indices[0], n_u]

    # vertial neighboring points
    n_v = valid_indices[0] + 1
    n_v[n_v==proj_H] = proj_H-2 # neighbor at reverse direction
    p_nv = vertex_map[n_v, valid_indices[1]]
    depth_pnv = depth_map[n_v, valid_indices[1]]

    # Calculate the vectors
    d_u = p_nu - p
    d_v = p_nv - p

    # Calculate the normals
    w = np.cross(d_u, d_v)
    normal = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-6)

    # Set the normals in the output image
    normal_map[valid_indices] = normal
    normal_map[-1,:] *= -1.0

    # normal valid mask
    delta_d_u = np.abs(depth_p - depth_pnu)
    delta_d_v = np.abs(depth_p - depth_pnv)
    flag = (delta_d_u < delta_ratio * depth_p) & (delta_d_v <  delta_ratio * depth_p)
    #print(f"find {len(flag)-np.count_nonzero(flag)} invalid normal")
    normal_mask[valid_indices] = flag.astype(int)

    return normal_map, normal_mask

# torch version
def intrinsic_correct(points: torch.tensor, correct_deg=0.0):
    # # This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
    # # the original idea and part of the implementation is taking from CT-ICP(Although IMLS-SLAM
    # # Originally introduced the calibration factor)
    # We set the correct_deg = 0.195 deg for KITTI odom dataset, inline with MULLS #issue 11
    if correct_deg == 0.0:
        return points

    dist = torch.norm(points[:, :3], dim=1)
    kitti_var_vertical_ang = correct_deg / 180.0 * math.pi
    v_ang = torch.asin(points[:, 2] / dist)
    v_ang_c = v_ang + kitti_var_vertical_ang
    hor_scale = torch.cos(v_ang_c) / torch.cos(v_ang)
    points[:, 0] *= hor_scale
    points[:, 1] *= hor_scale
    points[:, 2] = dist * torch.sin(v_ang_c)

    return points

# now only work for semantic kitti format dataset # torch version
def filter_sem_kitti(
    points: torch.tensor,
    sem_labels_reduced: torch.tensor,
    sem_labels: torch.tensor,
    filter_outlier=True,
    filter_moving=False,
):

    # sem_labels_reduced is the reduced labels for mapping (20 classes for semantic kitti)
    # sem_labels is the original semantic label (0-255 for semantic kitti)

    if filter_outlier:  # filter the outliers according to semantic labels
        inlier_mask = sem_labels > 1  # not outlier
    else:
        inlier_mask = sem_labels >= 0  # all

    if filter_moving:
        static_mask = sem_labels < 100  # only for semantic KITTI dataset
        inlier_mask = inlier_mask & static_mask

    points = points[inlier_mask]
    sem_labels_reduced = sem_labels_reduced[inlier_mask]

    return points, sem_labels_reduced

sem_kitti_learning_map = {
    0: 0,  # "unlabeled"
    1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,  # "car"
    11: 2,  # "bicycle"
    13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,  # "motorcycle"
    16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,  # "truck"
    20: 5,  # "other-vehicle"
    30: 6,  # "person"
    31: 7,  # "bicyclist"
    32: 8,  # "motorcyclist"
    40: 9,  # "road"
    44: 10,  # "parking"
    48: 11,  # "sidewalk"
    49: 12,  # "other-ground"
    50: 13,  # "building"
    51: 14,  # "fence"
    52: 20,  # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,  # "lane-marking" to "road" ---------------------------------mapped
    70: 15,  # "vegetation"
    71: 16,  # "trunk"
    72: 17,  # "terrain"
    80: 18,  # "pole"
    81: 19,  # "traffic-sign"
    99: 20,  # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,  # "moving-car"
    253: 7,  # "moving-bicyclist"
    254: 6,  # "moving-person"
    255: 8,  # "moving-motorcyclist"
    256: 5,  # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
    257: 5,  # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
    258: 4,  # "moving-truck"
    259: 5,  # "moving-other-vehicle"
}

def sem_map_function(value):
    return sem_kitti_learning_map.get(value, value)

def set_dataset_path(config, dataset_name: str, seq: str = ''):
    
    config.name = config.name + '_' + dataset_name + '_' + seq
    
    # if config.use_kiss_dataloader:
    #     config.data_loader_name = dataset_name
    #     config.data_loader_seq = seq
    #     print('KISS-ICP data loaders used')
    #     from kiss_icp.datasets import available_dataloaders
    #     print('Available dataloaders:', available_dataloaders())
    # else:
    if dataset_name == "kitti":
        base_path = config.pc_path.rsplit('/', 3)[0]
        config.pc_path = os.path.join(base_path, 'sequences', seq, "velodyne")  # input point cloud folder
        pose_file_name = seq + '.txt'
        config.pose_path = os.path.join(base_path, 'poses', pose_file_name)   # input pose file
        config.calib_path = os.path.join(base_path, 'sequences', seq, "calib.txt")  # input calib file (to sensor frame)
        config.label_path = os.path.join(base_path, 'sequences', seq, "labels") # input point-wise label path, for semantic mapping (optional)
        config.kitti_correction_on = True
        config.correction_deg = 0.195
    elif dataset_name == "mulran":
        config.name = config.name + "_mulran_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "Ouster")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "kitti_carla":
        config.name = config.name + "_kitti_carla_" + seq
        base_path = config.pc_path.rsplit('/', 3)[0]
        config.pc_path = os.path.join(base_path, seq, "generated", "frames")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "generated", "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "generated", "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "ncd":
        config.name = config.name + "_ncd_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "bin")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "ncd128":
        config.name = config.name + "_ncd128_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "ipbcar":
        config.name = config.name + "_ipbcar_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "ouster")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
        config.calib_path = os.path.join(base_path, seq, "calib.txt")  # input calib file (to sensor frame)
    elif dataset_name == "hilti":
        config.name = config.name + "_hilti_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "ply")  # input point cloud folder
        # config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "m2dgr":
        config.name = config.name + "_m2dgr_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "points")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file
    elif dataset_name == "replica":
        config.name = config.name + "_replica_" + seq
        base_path = config.pc_path.rsplit('/', 2)[0]
        config.pc_path = os.path.join(base_path, seq, "rgbd_down_ply")  # input point cloud folder
        #config.pc_path = os.path.join(base_path, seq, "rgbd_ply")  # input point cloud folder
        config.pose_path = os.path.join(base_path, seq, "poses.txt")   # input pose file     