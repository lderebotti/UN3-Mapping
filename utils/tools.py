import os
import torch
import torch.nn as nn

import random
import numpy as np
import open3d as o3d
import time

import matplotlib.cm as cm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers. 
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`  

    Reference: Louis Wiesmann
    """
    _quantization = 1000

    offset = torch.floor(points.min(dim=0)[0]/voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1)**0.5
    dist = dist / dist.max() * (_quantization - 1) # for speed up

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    
    offset = 10**len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset
    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                      device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    idx = idx % offset

    return idx

def updownsampling_voxel(points, indices, counts):
    #to obtain average position of points within a voxel.
    summed_elements = torch.zeros(counts.shape[0], points.shape[-1]).cuda()
    summed_elements = torch.scatter_add(summed_elements, dim=0,
                                        index=indices.unsqueeze(1).repeat(1, points.shape[-1]), src=points)
    updownsample_points = summed_elements / counts.unsqueeze(-1).repeat(1, points.shape[-1])
    return updownsample_points

def transform_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [4, 4]
    # Add a homogeneous coordinate to each point in the point cloud
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1).to(points)], dim=1)

    # Apply the transformation by matrix multiplication
    transformed_points_homo = torch.matmul(points_homo, transformation.to(points).T)

    # Remove the homogeneous coordinate from each transformed point
    transformed_points = transformed_points_homo[:, :3]

    return transformed_points


def transform_batch_torch(points: torch.tensor, transformation: torch.tensor):
    # points [N, 3]
    # transformation [N, 4, 4]
    # N,3,3 @ N,3,1 -> N,3,1 + N,3,1 -> N,3,1 -> N,3

    # Extract rotation and translation components
    rotation = transformation[:, :3, :3].to(points)
    translation = transformation[:, :3, 3:].to(points)

    # Reshape points to match dimensions for batch matrix multiplication
    points = points.unsqueeze(-1)

    # Perform batch matrix multiplication using torch.bmm(), instead of memory hungry matmul
    transformed_points = torch.bmm(rotation, points) + translation

    # Squeeze to remove the last dimension
    transformed_points = transformed_points.squeeze(-1)

    return transformed_points

# normals to colors
def normal2color(normals: np.array):
    
    normal_colors = np.clip((normals+1.0)/2.0, 0.0, 1.0)
    return normal_colors

def error2color(errors: np.array, lower_bound: float, upper_bound: float, cmap=None):

    errors = np.clip((errors-lower_bound)/(upper_bound-lower_bound), 0.0, 1.0)
    # the error larger the color redder
    colors = np.zeros((errors.shape[0], 3))
    colors[:, 0] = 1.0
    colors[:, 1] = 1 - errors
    colors[:, 2] = 1 - errors

    return colors

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()