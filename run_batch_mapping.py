import math
import sys
import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

import open3d as o3d
import numpy as np
from tqdm import tqdm, trange
# import matplotlib.pyplot as plt

from utils.config import Config
from dataset.kitti import DataLoader
from utils.dataSampler import dataSampler
from utils import tools
from utils.mesher import Mesher

from model.neural_voxel_hash import NeuralVoxelHash
from model.gradient_voxel_hash import GradientVoxelHash
from model.decoder import Decoder
from model.loss import *

torch.set_default_dtype(torch.float32)

def batch_mapping(config: Config):

    dataset = DataLoader(config)
    sampler = dataSampler(config)
    feature_field = NeuralVoxelHash(config.feature_dim, \
                                    config.leaf_voxel_size, \
                                    config.voxel_level_num, \
                                    config.scale_up_factor, \
                                    config.hash_buffer_size, \
                                    config.device)
    gradient_field = GradientVoxelHash(config.leaf_voxel_size, \
                                    config.hash_buffer_size, \
                                    config.device)

    sdf_scale = config.logistic_gaussian_ratio * config.sigma_sigmoid_m
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    uncer_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)

    end_points = torch.tensor([], device='cuda', dtype=torch.float32)
    start_points = torch.tensor([], device='cuda', dtype=torch.float32)
    normals_all = torch.tensor([], device='cuda', dtype=torch.float32)
  
    map_points = torch.tensor([], device='cuda', dtype=torch.float32) # mergerd point clodu map
    map_points_o3d = o3d.geometry.PointCloud()

    if config.end_frame == -1:
        config.end_frame = dataset.total_pc_count # [start, end)
    else:
        config.end_frame = min(config.end_frame, dataset.total_pc_count)
    with trange(config.begin_frame, config.end_frame, config.step_frame) as tbar:
        tbar.set_description('Collecting Data')
        for frame_idx in tbar:
            dataset.process_frame(frame_idx)
            frame_points = dataset.get_points_global()
            frame_normals = dataset.get_normal_global()
            frame_translation = dataset.get_gt_translation(frame_idx)

            if config.down_sample:
                down_sampled_id = tools.voxel_down_sample_torch(frame_points, config.vox_down_m)
                frame_points_down = frame_points[down_sampled_id]
                frame_normals_down =  frame_normals[down_sampled_id]
            else:
                frame_points_down = frame_points
            
            # save input point cloud
            map_points = torch.cat((map_points, frame_points_down), dim=0)

            origin_tensor = frame_translation.repeat(frame_points_down.shape[0],1)
            #time_tensor = torch.ones(origin_tensor.shape[0], 1, dtype=ray_times.dtype, device=ray_times.device)*time_step

            sample_points, sample_sdfs, sample_normals, _, _, weight, _ = sampler.ray_sample(origin_tensor,
                                                                    frame_points_down,
                                                                    frame_normals_down)
            surface_sample = sample_points[weight>0]
            feature_field.update(surface_sample)
            gradient_field.update(surface_sample)
            gradient_field.integrate_gradients(sample_points, sample_sdfs, sample_normals)

            end_points = torch.cat((end_points, frame_points_down), dim=0)
            start_points = torch.cat((start_points, origin_tensor), dim=0)
            normals_all = torch.cat((normals_all, frame_normals_down), dim=0)
        
    map_grads_all = gradient_field.get_gradients_batch(map_points)
    map_grads = map_grads_all[:,:3].numpy()
    map_uncer = map_grads_all[:, 3].numpy()
    map_grads_color = tools.normal2color(map_grads)
    map_uncer_color = tools.error2color(map_uncer, lower_bound=0, upper_bound=1.0)
    map_points_o3d.points = o3d.utility.Vector3dVector(map_points.detach().cpu().numpy())
    #map_points_o3d.colors = o3d.utility.Vector3dVector(map_uncer_color)
    map_points_o3d.colors = o3d.utility.Vector3dVector(map_grads_color)
    map_points_o3d = map_points_o3d.voxel_down_sample(voxel_size = config.vox_down_m)
    if config.save_pc_map:
        o3d.io.write_point_cloud(config.output_folder + "/merged_map_pc.ply", map_points_o3d)
        print("merged map pointcloud saved.")

    field_param = list(feature_field.parameters())
    geo_mlp_param = list(geo_mlp.parameters())
    uncer_mlp_param = list(uncer_mlp.parameters())

    field_param_opt_dict = {'params': field_param, 'lr': config.learning_rate}
    geo_mlp_param_opt_dict = {'params': geo_mlp_param, 'lr': config.learning_rate}
    uncer_mlp_param_opt_dict = {'params': uncer_mlp_param, 'lr': config.learning_rate}
    opt = optim.Adam([field_param_opt_dict, geo_mlp_param_opt_dict, uncer_mlp_param_opt_dict], betas=(0.9,0.99), eps = 1e-15)

    # start training
    with tqdm(total=config.epochs) as pbar:
        pbar.set_description('traning')
        for epoch in range(config.epochs):
            random_idx = torch.randperm(end_points.size(0), device='cuda')
        
            this_end_points = end_points[random_idx]

            iterations = math.ceil(this_end_points.shape[0]/config.batch_size)

            with tqdm(total=iterations) as iter_pbar:
                for i in range(iterations):
                    start_idx = config.batch_size*i
                    end_idx = min(config.batch_size*(i+1), random_idx.shape[0])

                    iter_indices = random_idx[start_idx:end_idx]

                    iter_end_points = end_points[iter_indices]
                    iter_start_points = start_points[iter_indices]
                    iter_normals = normals_all[iter_indices]

                    sample_points, sdf_label, _, _, _, weight, rays = sampler.ray_sample(
                                                                                    iter_start_points,
                                                                                    iter_end_points,
                                                                                    iter_normals)
                    
                    # sdf correction with Gradients
                    gradients, _ = gradient_field.get_gradients(sample_points)
                    # grad_weights = torch.clamp(1 - gradients[:, 3], 0.0, 1.0) # lower uncertainty, higher weight
                    if config.proj_correction_on:
                        sdf_label = sampler.correct_sdf(sdf_label, gradients, rays)
                    features, _ = feature_field.get_features(sample_points)
                    sdf_pred = geo_mlp.sdf(features)

                    # calculate the loss
                    cur_loss = 0.0
                    # weight's sign indicate the sample is around the surface or in the free space
                    weight = torch.abs(weight).detach()
                  
                    if config.main_loss_type == "bce":  
                        sdf_loss = sdf_bce_loss(sdf_pred, sdf_label, sdf_scale, weight, config.loss_weight_on)
                    elif config.main_loss_type == "heter":
                        sigma, log_sigma = uncer_mlp.uncertainty(features)
                        sdf_loss = heter_sdf_loss(sdf_pred, sdf_label, sigma, log_sigma)
                    elif config.main_loss_type == "hybrid":
                        sigma, log_sigma = uncer_mlp.uncertainty(features)
                        heter_loss = heter_sdf_loss(sdf_pred, sdf_label, sigma, log_sigma)
                        bce_loss = sdf_bce_loss(sdf_pred, sdf_label, sdf_scale, weight, config.loss_weight_on)
                        sdf_loss = 0.01*heter_loss + bce_loss
                    else:
                        sys.exit("Please choose a valid loss type")
                    
                    # ekinoal loss can be added here but not necessary
                    cur_loss += sdf_loss

                    cur_loss.backward()
                    opt.step()
                    opt.zero_grad()

                    iter_pbar.update(1)
            pbar.update(1)
    
    uncer_max = 1.0
    if config.mesh_recon:
        print('marching cubes ...')
        mesh_filename = config.output_folder + '/mesh.ply'
        mesher = Mesher(config=config,field=feature_field, geo_decoder=geo_mlp, uncer_decoder=uncer_mlp)
        mesher.recon_bbox_mesh(
            map_points=map_points_o3d, 
            voxel_size=config.mesh_resolution,
            mesh_path=mesh_filename)
        if config.main_loss_type == "heter" or config.main_loss_type == "hybrid":
            print('query uncertainty ...')
            _, map_uncer, _, _, mask = mesher.query_points(map_points, query_uncer=True)
            map_uncer = np.clip(map_uncer,0,1)
            map_uncer_color = tools.error2color(map_uncer, lower_bound=0, upper_bound=1.0)
            map_points_o3d.points = o3d.utility.Vector3dVector(map_points.detach().cpu().numpy())
            map_points_o3d.colors = o3d.utility.Vector3dVector(map_uncer_color)
            o3d.io.write_point_cloud(config.output_folder + "/merged_map_pc_uncer.ply", map_points_o3d)
            print("merged map pointcloud saved.")
    
    if config.static_pointcloud:
        total_static_ps = torch.tensor([], device='cuda')
        total_uncer_ps = torch.tensor([], device='cuda')

        if config.end_frame == -1:
            config.end_frame = dataset.total_pc_count  # [start, end)
        else:
            config.end_frame = min(config.end_frame, dataset.total_pc_count)

        while True:
            try:
                # input weight and threshold
                w1 = float(input("Enter weight w1 (0-1, e.g., 0.5): "))
                threshold = float(input("Enter segmentation threshold (e.g., 0.16): "))
                # uncer_thre = float(input("Enter uncer threshold (e.g., 0.1): "))
                uncer_thre = 0.5
                if not (0 <= w1 <= 1):
                    print("w1 must be between 0 and 1. Try again.")
                    continue
                print("uncer_max = ", uncer_max)

                # process each frame
                with trange(config.begin_frame, config.end_frame, config.step_frame) as tbar:
                    tbar.set_description('Dynamic Object Segmentation')
                    for frame_idx in tbar:
                        dataset.process_frame(frame_idx)
                        frame_points = dataset.get_points_global()

                        # get sdf and uncertainty
                        scan_features, _ = feature_field.get_features(frame_points)
                        static_sdf = torch.abs(geo_mlp(scan_features))
                        uncer, _ = uncer_mlp.uncertainty(scan_features)
                        uncer = torch.clamp(uncer,0,0.5)/0.5

                        # calculate mask
                        #mask = (w1 * static_sdf + (1 - w1) * uncer) > threshold
                        mask = w1 * static_sdf/threshold + (1-w1)*uncer > 1.0
                        mask2 = uncer > uncer_thre

                        # extract dynamic and uncertain points
                        dynamic_ps_torch = frame_points[mask].detach()
                        dynamic_ps = o3d.geometry.PointCloud()
                        dynamic_ps.points = o3d.utility.Vector3dVector(dynamic_ps_torch.cpu().numpy())

                        uncer_ps_torch = frame_points[mask2].detach()
                        uncer_ps = o3d.geometry.PointCloud()
                        uncer_ps.points = o3d.utility.Vector3dVector(dynamic_ps_torch.cpu().numpy())

                        static_ps_torch = frame_points[~mask].detach()
                        static_ps = o3d.geometry.PointCloud()
                        static_ps.points = o3d.utility.Vector3dVector(static_ps_torch.cpu().numpy())

                        total_static_ps = torch.cat((total_static_ps, static_ps_torch), 0)
                        total_uncer_ps = torch.cat((total_uncer_ps, uncer_ps_torch), 0)

                # save static points
                device = o3d.core.Device("CPU:0")
                dtype = o3d.core.float32
                static_map = o3d.t.geometry.PointCloud(device)
                static_map.point['positions'] = o3d.core.Tensor(total_static_ps.cpu().numpy(), dtype, device)
                output_path = config.output_folder + f'/static_output_w1_{w1}_thresh_{threshold}.pcd'
                o3d.t.io.write_point_cloud(output_path, static_map, print_progress=False)
                print(f"Static points saved to {output_path}.")
                # clean temp
                total_static_ps = torch.tensor([], device='cuda')
                # total_dynamic_ps = torch.tensor([], device='cuda')
                total_uncer_ps = torch.tensor([], device='cuda')
                # next input
                continue

            except KeyboardInterrupt:
                print("Exiting interactive mode.")
                break
            except Exception as e:
                print(f"Error: {e}. Try again.")

if __name__ == "__main__":
    
    config = Config()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit("No config file.")
    
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    tools.seed_everything(config.random_seed)
    batch_mapping(config)







