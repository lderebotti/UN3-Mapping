import torch
import torch.nn as nn
import utils.tools as tools

class NeuralVoxelHash(nn.Module):

    def __init__(self, feature_dim, leaf_voxel_size, voxel_level_num, scale_up_factor, hash_buffer_size, device) -> None:
        
        super().__init__()
        # feature setting
        self.feature_dim = feature_dim
        self.feature_std = 0.01

        # map structure setting
        self.leaf_voxel_size = leaf_voxel_size
        self.voxel_level_num = voxel_level_num
        self.scale_up_factor = scale_up_factor

        self.dtype = torch.float32
        self.device = device

        self.buffer_size = hash_buffer_size

        # hash function
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=self.device)
        # for point to corners
        self.steps = torch.tensor([[0., 0., 0.], [0., 0., 1.], 
                                   [0., 1., 0.], [0., 1., 1.], 
                                   [1., 0., 0.], [1., 0., 1.], 
                                   [1., 1., 0.], [1., 1., 1.]], dtype=self.dtype, device=self.device)
        
        dilate_dist = 0.5 * leaf_voxel_size
        offsets = torch.tensor([[0, 0, 0], [-1, 0, 0], [1, 0, 0], 
                                [0, -1, 0], [0, 1, 0], 
                                [0, 0, 1], [0, 0, -1]], dtype=self.dtype, device=self.device)
        self.offsets = dilate_dist * offsets

        self.features_layer = nn.ParameterList([])
        self.feature_indexs_layer = []
        self.feature_time_steps = []
        self.hash_grids_layer = [] # hash voxel coordinates (N, 3)
        for l in range(self.voxel_level_num):
            features = nn.Parameter(torch.tensor([],device=self.device))
            time_steps = torch.tensor([], dtype=torch.int64, device=self.device)
            hash_grids = torch.tensor([], dtype=torch.int64, device=self.device)
            feature_indexs = torch.full([self.buffer_size], -1, dtype=torch.int64, 
                                             device=self.device) # -1 for not valid (occupied)
            self.features_layer.append(features)
            self.feature_indexs_layer.append(feature_indexs)
            self.feature_time_steps.append(time_steps)
            self.hash_grids_layer.append(hash_grids)

        self.to(self.device)

    def update(self, points: torch.Tensor):
        for i in range(self.voxel_level_num):
            curr_voxel_size = self.leaf_voxel_size*(self.scale_up_factor**i) # leaf to top
        
            # # if ave point is close to voxel boundary, expand a neighbor voxel to incomplete coverage.
            # offsets = torch.LongTensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]).to(voxels.device)
            # # TODO: try to avoid repeated calculation.
            # updownsampling_points = tools.updownsampling_voxel(points, inverse_indices, counts)
            # for offset in offsets:
            #     offset_axis = offset.nonzero().item()
            #     if offset[offset_axis] > 0:
            #         margin_mask = updownsampling_points[:, offset_axis] % curr_voxel_size > (
            #                 1 - inflate_margin_ratio) * curr_voxel_size
            #     else:
            #         margin_mask = updownsampling_points[:,
            #                     offset_axis] % curr_voxel_size < inflate_margin_ratio * curr_voxel_size
            #     margin_vox = voxels_raw[margin_mask * (counts > voxel_valid_thre)]
            #     voxels_vaild = torch.cat((voxels_vaild, torch.clip(margin_vox + offset, min=0)), dim=0)

            # voxels_unique = torch.unique(voxels_vaild, dim=0)
            # find eight neighbors (TODO mark surface voxels and feature voxels)
            corners = self.to_corners(points, curr_voxel_size)
            #corners = self.voxels_to_corners(voxels_unique)
            unique_corners = torch.unique(corners, dim=0)

            # hash function
            keys = (unique_corners.to(self.primes) * self.primes).sum(-1) % self.buffer_size
            update_mask = (self.feature_indexs_layer[i][keys] == -1) # not initialized term
            new_feature_count = unique_corners[update_mask].shape[0]
            curr_feats_count = self.features_layer[i].shape[0]
            self.feature_indexs_layer[i][keys[update_mask]] = torch.arange(new_feature_count, dtype=torch.int64, 
                                                                        device=self.device) + curr_feats_count
            # append new feature vector
            new_fts = self.feature_std*torch.randn(new_feature_count, self.feature_dim, device=self.device, dtype=self.dtype)
            self.features_layer[i] = nn.Parameter(torch.cat((self.features_layer[i], new_fts),0))
            
            # append initialized hash grids TODO use surface grids or dilated grids
            new_grids = unique_corners[update_mask]
            self.hash_grids_layer[i] = torch.cat((self.hash_grids_layer[i], new_grids),0)

    def update_from_surface_points(self, points: torch.Tensor):
        
        dilate_points = (points.repeat(1,7) + self.offsets.reshape(1,-1)).reshape(-1,3)
        for i in range(self.voxel_level_num):
            curr_voxel_size = self.leaf_voxel_size*(self.scale_up_factor**i) # leaf to top
            corners = self.to_corners(dilate_points, curr_voxel_size)
            unique_corners = torch.unique(corners, dim=0)

            # hash function
            keys = (unique_corners.to(self.primes) * self.primes).sum(-1) % self.buffer_size
            update_mask = (self.feature_indexs_layer[i][keys] == -1) # not initialized term
            new_feature_count = unique_corners[update_mask].shape[0]
            curr_feats_count = self.features_layer[i].shape[0]
            self.feature_indexs_layer[i][keys[update_mask]] = torch.arange(new_feature_count, dtype=torch.int64, 
                                                                        device=self.device) + curr_feats_count
            # append new feature vector
            new_fts = self.feature_std*torch.randn(new_feature_count, self.feature_dim, device=self.device, dtype=self.dtype)
            self.features_layer[i] = nn.Parameter(torch.cat((self.features_layer[i], new_fts),0))
            
            # append initialized hash grids TODO use surface grids or dilated grids
            new_grids = unique_corners[update_mask]
            self.hash_grids_layer[i] = torch.cat((self.hash_grids_layer[i], new_grids),0)

    def get_features(self, query_points): 
        sum_features = torch.zeros(query_points.shape[0], self.feature_dim, device=self.device, dtype=self.dtype)
        valid_mask = torch.ones(query_points.shape[0], device=self.device, dtype=bool)
        for i in range(self.voxel_level_num):
            curr_voxel_size = self.leaf_voxel_size*(self.scale_up_factor**i)
            query_corners = self.to_corners(query_points, curr_voxel_size).to(self.primes)
            query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size
            hash_index_nx8 = self.feature_indexs_layer[i][query_keys].reshape(-1,8)
            featured_query_mask = (hash_index_nx8.min(dim=1)[0]) >= 0
            features_index = hash_index_nx8[featured_query_mask].reshape(-1,1).squeeze(1)
            coeffs = self.interpolat(query_points[featured_query_mask], curr_voxel_size)
            # TODO consider whether the feature is valid
            sum_features[featured_query_mask] += (self.features_layer[i][features_index]*coeffs).reshape(-1,8,self.feature_dim).sum(1)
            
            if i == 0: # leaf voxel
                valid_mask = featured_query_mask
        return sum_features, valid_mask

    def get_valid_mask(self, query_points):
        n = self.voxel_level_num-1
        curr_voxel_size = self.leaf_voxel_size*(self.scale_up_factor**n)
        query_corners = self.to_corners(query_points, curr_voxel_size).to(self.primes)
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size
        hash_index_nx8 = self.feature_indexs_layer[n][query_keys].reshape(-1,8)
        featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

        return featured_query_mask

    # TODO can be more efficient use voxfusion
    def interpolat(self, x, resolution):
        coords = x / resolution
        d_coords = coords - torch.floor(coords)
        tx = d_coords[:,0]
        _1_tx = 1-tx
        ty = d_coords[:,1]
        _1_ty = 1-ty
        tz = d_coords[:,2]
        _1_tz = 1-tz
        p0 = _1_tx*_1_ty*_1_tz
        p1 = _1_tx*_1_ty*tz
        p2 = _1_tx*ty*_1_tz
        p3 = _1_tx*ty*tz
        p4 = tx*_1_ty*_1_tz
        p5 = tx*_1_ty*tz
        p6 = tx*ty*_1_tz
        p7 = tx*ty*tz
        p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.reshape(-1,1)
        return p

    def to_corners(self, points: torch.Tensor, voxel_size):
        grids = torch.floor(points / voxel_size)
        corners = (grids.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners
    
    def voxels_to_corners(self, voxels: torch.Tensor):
        corners = (voxels.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners
