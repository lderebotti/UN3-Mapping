import torch
import torch.nn as nn
import utils.tools as tools

# class GradientVoxelHash(nn.Module):

class GradientVoxelHash():
    def __init__(self, voxel_size, hash_buffer_size, device) -> None:
        # super().__init__()
        # setting
        self.voxel_size = voxel_size
        self.dtype = torch.float32
        self.device = device
        self.buffer_size = hash_buffer_size
        self.feature_dim = 5 # (nx, ny, nz, uncertainty, weight)

        # hash function
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=self.device)
        # for point to corners
        self.steps = torch.tensor([[0., 0., 0.], [0., 0., 1.], 
                                   [0., 1., 0.], [0., 1., 1.], 
                                   [1., 0., 0.], [1., 0., 1.], 
                                   [1., 1., 0.], [1., 1., 1.]], dtype=self.dtype, device=self.device)
   

        #features = nn.Parameter(torch.tensor([],device=self.device))
        self.gradients = torch.tensor([], dtype=self.dtype, device=self.device) 
        self.grad_indexs = torch.full([self.buffer_size], -1, dtype=torch.int64, 
                                            device=self.device) # -1 for not valid (occupied)
        #self.to(self.device)

    def update(self, points: torch.Tensor):
        # create valid surface voxel
        curr_voxel_size = self.voxel_size    
        # find eight neighbors
        corners = self.to_corners(points, curr_voxel_size)
        #corners = self.voxels_to_corners(voxels_unique)
        unique_corners = torch.unique(corners, dim=0)
        
        keys = (unique_corners.to(self.primes) * self.primes).sum(-1) % self.buffer_size
        update_mask = (self.grad_indexs[keys] == -1) # not initialized term
        new_grad_count = unique_corners[update_mask].shape[0]
        curr_len = self.gradients.shape[0]
        self.grad_indexs[keys[update_mask]] = torch.arange(new_grad_count, dtype=torch.int64, 
                                                                    device=self.device) + curr_len
        new_grads = torch.zeros(new_grad_count, self.feature_dim, device=self.device, dtype=self.dtype)
        self.gradients = torch.cat((self.gradients, new_grads),0)
    
    def integrate_gradients(self, points: torch.Tensor, sdfs: torch.Tensor, normals: torch.Tensor):
        """
        Integrate gradients into the hash table following data fusion pipeline.
        comparing to training based gradient fusion, this is more strightforward and faster.
        points: (N, 3), sdfs: (N,), normals: (N, 3)
        """
        weights = 1.0 - torch.abs(sdfs) / 0.4
        weights = torch.clamp(weights, min=0.1, max=1.0)

        curr_voxel_size = self.voxel_size
        corners = torch.round(points/curr_voxel_size) # gradients are anchored at corners.

        keys = (corners.to(self.primes) * self.primes).sum(-1) % self.buffer_size
        update_mask = (self.grad_indexs[keys] != -1) # only update initialized gradients
        update_normals = normals[update_mask]
        update_weights = weights[update_mask]
        grads_index = self.grad_indexs[keys[update_mask]]
        
        curr_grads = self.gradients[grads_index, :3] # nx ny nz
        curr_uncer = self.gradients[grads_index, 3] # uncertainty/stability
        curr_weights = self.gradients[grads_index, -1]
        
        new_weights = curr_weights + update_weights # TODO: add upper bound
        new_grads = (curr_grads * curr_weights.unsqueeze(1) + update_normals * update_weights.unsqueeze(1)) / new_weights.unsqueeze(1)
        new_grads = new_grads/(new_grads.norm(dim=1,keepdim=True)+1e-7)

        d_grad = 1 - torch.abs(torch.nn.functional.cosine_similarity(new_grads, update_normals, dim=1))
        new_uncer = (curr_uncer * curr_weights + d_grad * update_weights) / new_weights
        self.gradients[grads_index, :3] = new_grads.float()
        self.gradients[grads_index, 3] = new_uncer.float()
        self.gradients[grads_index, -1] = new_weights.float()
        
    def get_gradients(self, query_points):
        queried_grads = torch.zeros(query_points.shape[0], 5, device=self.device, dtype=self.dtype)
        
        curr_voxel_size = self.voxel_size
        query_corners = self.to_corners(query_points, curr_voxel_size).to(self.primes)
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size
        
        hash_index_nx8 = self.grad_indexs[query_keys].reshape(-1,8)
        query_mask = (hash_index_nx8.min(dim=1)[0]) > -1
        grads_index = hash_index_nx8[query_mask].reshape(-1,1).squeeze(1)
        coeffs = self.interpolat(query_points[query_mask], curr_voxel_size)
        queried_grads[query_mask] = (self.gradients[grads_index]*coeffs).reshape(-1,8,self.feature_dim).sum(1).float()

        return queried_grads, query_mask # (nx, ny, nz, uncertainty, weight)
    
    def get_gradients_batch(self, query_points, batch_size: int = 20000):
        point_num = query_points.shape[0]
        grads_all = torch.zeros(point_num, 5)
        # TODO torch auto cast?
        for head in range(0, point_num, batch_size):
            sub_points = query_points[head : min(head + batch_size, point_num), :]
            grads, _ = self.get_gradients(sub_points)
            grads_all[head : min(head + batch_size, point_num), :] = grads.detach().cpu()
        return grads_all # (nx, ny, nz, uncertainty, weight)

    def get_valid_mask(self, query_points):
        curr_voxel_size = self.voxel_size
        query_corners = self.to_corners(query_points, curr_voxel_size).to(self.primes)
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size
        hash_index_nx8 = self.grad_indexs[query_keys].reshape(-1,8)
        query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

        return query_mask

    # TODO can be more efficient like voxfusion
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
        origin_corner = torch.floor(points / voxel_size)
        corners = (origin_corner.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners
    
    def voxels_to_corners(self, voxels: torch.Tensor):
        corners = (voxels.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners
