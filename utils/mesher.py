
import numpy as np
import skimage.measure as skm
import matplotlib.cm as cm
from scipy.spatial import KDTree
import time
import torch
import math
import open3d as o3d
from tqdm import tqdm
from utils.config import Config
from model.neural_voxel_hash import NeuralVoxelHash
from model.decoder import Decoder


class Mesher:
    def __init__(
        self,
        config: Config,
        field: NeuralVoxelHash,
        geo_decoder: Decoder,
        uncer_decoder: Decoder = None,
        sem_decoder : Decoder = None,
        color_decoder: Decoder = None
    ):

        self.config = config
        self.field = field
        self.geo_decoder = geo_decoder
        self.uncer_decoder = uncer_decoder
        self.sem_decoder = sem_decoder
        self.color_decoder = color_decoder
        self.device = config.device
        self.dtype = config.dtype
        #self.global_transform = np.eye(4)
    
    def query_points(
        self,
        coord,
        batch_size = 32768,
        query_uncer=False,
        query_sem=False,
        query_color=False,
        query_mask=True
    ):
        """query the sdf value, semantic label and marching cubes mask for points
        Args:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points in the scaled
                kaolin coordinate system [-1,1]
            batch_size: batch size for the inference
        Returns:
            sdf_pred: Ndim numpy array or torch tensor, signed distance value (scaled) at each query point
            sem_pred: Ndim numpy array or torch tenosr, semantic label prediction at each query point
            mc_mask:  Ndim bool numpy array or torch tensor, marching cubes mask at each query point
        """
        sample_count = coord.shape[0]
        sdf_pred = np.zeros(sample_count)
        if query_sem:
            sem_pred = np.zeros(sample_count)
        else:
            sem_pred = None
        if query_uncer:
            uncer_pred = np.zeros(sample_count)
        else:
            uncer_pred = None
        if query_color:
            color_pred = np.zeros((sample_count, 3))
        else:
            color_pred = None
        if query_mask:
            mc_mask = np.zeros(sample_count)
        else:
            mc_mask = None

        with torch.no_grad():  # eval step
            for head in tqdm(range(0, sample_count, batch_size)):
                tail = min(head + batch_size, sample_count)
                batch_coord = coord[head:tail, :].cuda()
                # query
                features, pred_mask = self.field.get_features(batch_coord)
                curr_bs = batch_coord.shape[0]
                batch_sdf = torch.zeros(curr_bs, device=self.device)
                # query the sdf with the feature, only do for valid surface regions.
                batch_sdf[pred_mask] = self.geo_decoder(features.float()[pred_mask])
                sdf_pred[head:tail] = batch_sdf.detach().cpu().numpy()
                
                if query_uncer:
                    batch_uncer = torch.zeros(curr_bs, device=self.device)
                    # query the sdf with the feature, only do for valid surface regions.
                    sigma, _ = self.uncer_decoder.uncertainty(features.float()[pred_mask])
                    batch_uncer[pred_mask] = sigma
                    uncer_pred[head:tail] = batch_uncer.detach().cpu().numpy()
                if query_mask:
                    mc_mask[head:tail] = pred_mask.detach().cpu().numpy()

                if query_sem:
                    batch_sem_prob = self.sem_decoder.sem_label_prob(features)
                    batch_sem = torch.argmax(batch_sem_prob, dim=1)
                    sem_pred[head:tail] = batch_sem.detach().cpu().numpy()
                if query_color:
                    batch_color = self.color_decoder.regress_color(features)
                    color_pred[head:tail] = (
                        batch_color.detach().cpu().numpy().astype(dtype=np.float64)
                    )

        return sdf_pred, uncer_pred, sem_pred, color_pred, mc_mask

    def get_query_from_bbx(self, min_bound, max_bound, voxel_size, pad_voxel=0, skip_top_voxel=0):
        """
        get grid query points inside a given bounding box (bbx)
        Args:
            min_bound: min bound (xyz) of 3D box
            max_bound: max bound (xyz) of 3D box
            voxel_size: scalar, marching cubes voxel size with unit m
        Returns:
            coord: Nx3 torch tensor, the coordinates of all N (axbxc) query points
            voxel_num_xyz: 3dim numpy array, the number of voxels on each axis for the bbx
            voxel_origin: 3dim numpy array the coordinate of the bottom-left corner of the 3d grids
                for marching cubes, in world coordinate system with unit m
        """
        # bbx and voxel_size are all in the world coordinate system
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz / voxel_size) + pad_voxel * 2).astype(np.int_)
        voxel_origin = min_bound - pad_voxel * voxel_size
        # pad an additional voxel underground to gurantee the reconstruction of ground
        voxel_origin[2] -= voxel_size
        voxel_num_xyz[2] += 1
        voxel_num_xyz[2] -= skip_top_voxel

        #voxel_count_total = voxel_num_xyz[0] * voxel_num_xyz[1] * voxel_num_xyz[2]
        # if voxel_count_total > 5e8:  # this value is determined by your gpu memory
        #     print("too many query points, use smaller chunks")
        #     return None, None, None
            # self.cur_device = "cpu" # firstly save in cpu memory (which would be larger than gpu's)
            # print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        # get the vector of all the grid point's 3D coordinates
        coord = torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype)

        return coord, voxel_num_xyz, voxel_origin

    def get_query_from_hor_slice(self, bbx, slice_z, voxel_size):
        """get grid query points inside a given bounding box (bbx) at slice height (slice_z)"""
        # bbx and voxel_size are all in the world coordinate system
        min_bound = bbx.get_min_bound()
        max_bound = bbx.get_max_bound()
        len_xyz = max_bound - min_bound
        voxel_num_xyz = (np.ceil(len_xyz / voxel_size)).astype(np.int_)
        voxel_num_xyz[2] = 1
        voxel_origin = min_bound
        voxel_origin[2] = slice_z

        # query_count_total = voxel_num_xyz[0] * voxel_num_xyz[1]
        # if query_count_total > 1e8:  # avoid gpu memory issue, dirty fix
        #     self.cur_device = (
        #         "cpu"  # firstly save in cpu memory (which would be larger than gpu's)
        #     )
        #     print("too much query points, use cpu memory")
        x = torch.arange(voxel_num_xyz[0], dtype=torch.int16)
        y = torch.arange(voxel_num_xyz[1], dtype=torch.int16)
        z = torch.arange(voxel_num_xyz[2], dtype=torch.int16)

        # order: [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], [0,1,2] ...
        x, y, z = torch.meshgrid(x, y, z, indexing="ij")
        # get the vector of all the grid point's 3D coordinates
        coord = (
            torch.stack((x.flatten(), y.flatten(), z.flatten())).transpose(0, 1).float()
        )
        # transform to world coordinate system
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=self.dtype)

        return coord, voxel_num_xyz, voxel_origin

    def generate_sdf_map(self, coord, sdf_pred, mc_mask):
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        sdf_map_pc = o3d.t.geometry.PointCloud(device)

        coord_np = coord.detach().cpu().numpy()

        # the sdf (unit: m) would be saved in the intensity channel
        sdf_map_pc.point["positions"] = o3d.core.Tensor(coord_np, dtype, device)
        sdf_map_pc.point["intensities"] = o3d.core.Tensor(
            np.expand_dims(sdf_pred, axis=1), dtype, device
        )  # scaled sdf prediction
        if mc_mask is not None:
            # the marching cubes mask would be saved in the labels channel
            sdf_map_pc.point["labels"] = o3d.core.Tensor(
                np.expand_dims(mc_mask, axis=1), o3d.core.int32, device
            )  # mask

        # global transform (to world coordinate system) before output
        if not np.array_equal(self.global_transform, np.eye(4)):
            sdf_map_pc.transform(self.global_transform)

        return sdf_map_pc

    def generate_sdf_map_for_vis(
        self, coord, sdf_pred, mc_mask, min_sdf=-1.0, max_sdf=1.0, cmap="bwr"
    ):  # 'jet','bwr','viridis'

        # do the masking or not
        if mc_mask is not None:
            coord = coord[mc_mask > 0]
            sdf_pred = sdf_pred[mc_mask > 0]

        coord_np = coord.detach().cpu().numpy().astype(np.float64)

        sdf_pred_show = np.clip((sdf_pred - min_sdf) / (max_sdf - min_sdf), 0.0, 1.0)

        color_map = cm.get_cmap(cmap)  # or 'jet'
        colors = color_map(sdf_pred_show)[:, :3].astype(np.float64)

        sdf_map_pc = o3d.geometry.PointCloud()
        sdf_map_pc.points = o3d.utility.Vector3dVector(coord_np)
        sdf_map_pc.colors = o3d.utility.Vector3dVector(colors)
        if not np.array_equal(self.global_transform, np.eye(4)):
            sdf_map_pc.transform(self.global_transform)

        return sdf_map_pc

    def assign_to_bbx(self, sdf_pred, sem_pred, color_pred, mc_mask, voxel_num_xyz):
        """assign the queried sdf, semantic label and marching cubes mask back to the 3D grids in the specified bounding box
        Args:
            sdf_pred: Ndim np.array/torch.tensor
            sem_pred: Ndim np.array/torch.tensor
            mc_mask:  Ndim bool np.array/torch.tensor
            voxel_num_xyz: 3dim numpy array/torch.tensor, the number of voxels on each axis for the bbx
        Returns:
            sdf_pred:  a*b*c np.array/torch.tensor, 3d grids of sign distance values
            sem_pred:  a*b*c np.array/torch.tensor, 3d grids of semantic labels
            mc_mask:   a*b*c np.array/torch.tensor, 3d grids of marching cube masks, marching cubes only on where
                the mask is true
        """
        if sdf_pred is not None:
            sdf_pred = sdf_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if sem_pred is not None:
            sem_pred = sem_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if color_pred is not None:
            color_pred = color_pred.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        if mc_mask is not None:
            mc_mask = mc_mask.reshape(
                voxel_num_xyz[0], voxel_num_xyz[1], voxel_num_xyz[2]
            )

        return sdf_pred, sem_pred, color_pred, mc_mask

    def mc_mesh(self, mc_sdf, mc_mask, voxel_size, mc_origin):
        """use the marching cubes algorithm to get mesh vertices and faces
        Args:
            mc_sdf:  a*b*c np.array, 3d grids of sign distance values
            mc_mask: a*b*c np.array, 3d grids of marching cube masks, marching cubes only on where
                the mask is true
            voxel_size: scalar, marching cubes voxel size with unit m
            mc_origin: 3*1 np.array, the coordinate of the bottom-left corner of the 3d grids for
                marching cubes, in world coordinate system with unit m
        Returns:
            ([verts], [faces]), mesh vertices and triangle faces
        """
        # the input are all already numpy arraies
        verts, faces = np.zeros((0, 3)), np.zeros((0, 3))
        try:
            verts, faces, _, _ = skm.marching_cubes(
                mc_sdf, level=0.0, allow_degenerate=False, mask=mc_mask
            )
            # Whether to allow degenerate (i.e. zero-area) triangles in the
            # end-result. Default True. If False, degenerate triangles are
            # removed, at the cost of making the algorithm slower.
        except:
            pass

        verts = mc_origin + verts * voxel_size

        return verts, faces


    def filter_isolated_vertices(self, mesh, filter_cluster_min_tri=300):
        # print("Cluster connected triangles")
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        # print("Remove the small clusters")
        triangles_to_remove = (
            cluster_n_triangles[triangle_clusters] < filter_cluster_min_tri
        )
        mesh.remove_triangles_by_mask(triangles_to_remove)

        return mesh
    
    def clean_mesh(self, mesh, map_down_pc):
        
        print("begin to clean mesh.")
        points_kd_tree = KDTree(np.asarray(map_down_pc.points))
        #radius = self.config.leaf_voxel_size * 0.90
        radius = 0.12
        print(f"clean radius = {radius:.2f}m.")
        verts = np.asarray(mesh.vertices)
        verts_find_neighbors_num = points_kd_tree.query_ball_point(verts, radius, workers=12, return_length=True)
        mesh.remove_vertices_by_mask(verts_find_neighbors_num < 1)
        print("clean finished")

        return mesh
    
    def clean_mesh_via_mask_field(self, mesh, mask_field):
        
        print("begin to clean mesh.")
        vertices = torch.from_numpy(np.asarray(mesh.vertices)).to('cuda')
        index = torch.from_numpy(np.asarray(mesh.triangles)).to('cuda').long()
        
        bool_mask = torch.zeros(index.shape[0]*3,1,device=index.device)
        tri_vertices = vertices[index.reshape(-1,1)].squeeze(1)

        # Only the observed area is left.
        valid_mask = mask_field.get_valid_mask(tri_vertices)
        bool_mask[valid_mask] = 1

        non_mask = bool_mask.reshape(-1,3).sum(1) < 2.5
        valid_index = index[~non_mask]

        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices.cpu().numpy()),
            o3d.utility.Vector3iVector(valid_index.cpu().numpy())
        )
        print("clean finished")

        return mesh
    
    def recon_bbox_mesh(
        self,
        map_points,
        voxel_size,
        mesh_path,
        estimate_sem=False,
        filter_isolated_mesh=False,
        clean_mesh = True
    ):
        # TODO: map_points can be used as mesher member variable
        # get bounding box from map point cloud (open3d)
        bbox = map_points.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        # reconstruct and save the (semantic) mesh from the feature octree the decoders within a
        # given bounding box.  bbx and voxel_size all with unit m, in world coordinate system
        coord, voxel_num_xyz, voxel_origin = self.get_query_from_bbx(
            min_bound, max_bound, voxel_size, self.config.pad_voxel, self.config.skip_top_voxel
        )

        sdf_pred, _, _, _, mc_mask = self.query_points(coord)

        mc_sdf, _, _, mc_mask = self.assign_to_bbx(
            sdf_pred, None, None, mc_mask, voxel_num_xyz
        ) # reshape to sdf volume
    
        verts, faces = self.mc_mesh(
            mc_sdf, mc_mask.astype(bool), voxel_size, voxel_origin)
        # directly use open3d to get mesh
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces),
        )

        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        # o3d.io.write_triangle_mesh(mesh_path+"before.ply", mesh)
        # o3d.io.write_point_cloud(mesh_path+"points.ply", map_points)
        if clean_mesh:
            mesh = self.clean_mesh(mesh, map_points)
            #mesh = self.clean_mesh_via_mask_field(mesh, mask_field)

        if filter_isolated_mesh:
            mesh = self.filter_isolated_vertices(mesh, self.config.min_cluster_vertices)


        # global transform (to world coordinate system) before output
        # if not np.array_equal(self.global_transform, np.eye(4)):
        #     mesh.transform(self.global_transform)

        # write the mesh to ply file
        if mesh_path is not None:
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print("save the mesh to %s\n" % (mesh_path))

        # if self.config.dataset_name == 'ncd': #rotate back to original frame
        #     theta = np.radians(15)  # Convert degrees to radians
        #     R_z = np.array([
        #     [np.cos(theta), -np.sin(theta), 0],
        #     [np.sin(theta),  np.cos(theta), 0],
        #     [0, 0, 1]
        #     ])
        #     mesh.rotate(R_z, center=(0,0,0))

        return mesh

    # sparse voxel based mesher
    def extract_mesh(self, filename, res = 5, batch_size=8192*4,  offset=None, scale=None):
        t1 = time.time()

        start, end = 0, 1.0 #input voxel coordinate对应corner坐标 采样从0到1
        voxel_size = self.field.leaf_voxel_size
        grid_coord = self.field.hash_grids_layer[0] # leaf voxels coordinate (no metric)
        voxel_xyz = grid_coord * voxel_size
        voxel_num = voxel_xyz.shape[0]
        mc_grid_size = voxel_size/(res-1)

        x = y = z = torch.linspace(start, end, res) #采样5会生成4个间隔 mc_grid_size = voxel_size/(res-1)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + voxel_xyz.unsqueeze(1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        total_sample_num = sampled_xyz.shape[0]
        if total_sample_num == 0:
            print("meshing: empty sampled query points")
            return
        
        mc_mask = np.zeros(total_sample_num)
        sdf_grid = np.zeros(total_sample_num)
        for head in range(0, total_sample_num, batch_size):
            sub_samples = sampled_xyz[head : min(head + batch_size, total_sample_num), 0:3].cuda()
            features, query_mask = self.field.get_features(sub_samples)
            sdf_value = self.geo_decoder(features.float())
            sdf_grid[head : min(head + batch_size, total_sample_num)] = sdf_value.detach().cpu().numpy()
            mc_mask[head : min(head + batch_size, total_sample_num)] = query_mask.detach().cpu().numpy()
            #print(query_mask[query_mask==True].shape[0], query_mask[query_mask==False].shape[0])
        sdf_grid = sdf_grid.reshape(voxel_num, res, res, res)
        mc_mask = mc_mask.reshape(voxel_num, res, res, res)

        num_verts = 0
        total_verts = []
        total_faces = []
        for i in range(voxel_num): # TODO: batch meshing
            sdf_volume = sdf_grid[i]  # (res,res,res)
            sdf_mask = mc_mask[i]
            if np.min(sdf_volume) > 0 or np.max(sdf_volume) < 0:
                continue
            try:
                verts, faces, _, _ = skm.marching_cubes(sdf_volume, level=0.0, spacing=[mc_grid_size]*3, mask=sdf_mask)
                #verts, faces, _, _ = skm.marching_cubes(sdf_volume, level=0.0, spacing=[mc_grid_size]*3)
            except:
                continue
            #verts -= 0.5 # whether to remove?
            verts += voxel_xyz[i].detach().cpu().numpy()
            faces += num_verts
            num_verts += verts.shape[0]

            total_verts += [verts]
            total_faces += [faces]
        total_verts = np.concatenate(total_verts)
        total_faces = np.concatenate(total_faces)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(total_verts)
        mesh.triangles = o3d.utility.Vector3iVector(total_faces)
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        t2 = time.time()
        print(f"meshing takes: {t2-t1:.2f}s")
        o3d.io.write_triangle_mesh(filename, mesh)
        return mesh
