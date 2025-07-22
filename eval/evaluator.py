import os
import sys
sys.path.append(os.path.abspath("./"))
import csv
import numpy as np
import open3d as o3d
from utils.config import Config
from scipy.spatial import KDTree
from utils.tools import get_time

class MeshEvaluator:

    def __init__(self, config: Config, pred_mesh=None):
        
        self.config = config
        self.thre_dist = config.thre_dist
        self.trunc_dist_acc = config.trunc_dist_acc
        self.trunc_dist_com = config.trunc_dist_com
        self.gt_model_path = config.gt_model_path
        # self.thre_dist:float = 0.1
        # self.trunc_dist_acc:float = 0.2
        # self.trunc_dist_com:float = 2.0
        # self.gt_model_path:str = ""
        
        if pred_mesh is None:
            self.pred_mesh = o3d.io.read_triangle_mesh(config.output_folder + '/mesh.ply')
        else:
            self.pred_mesh = pred_mesh

        if config.dataset_name == 'ncd': # transform the mesh back to align with the ground truth
            theta = np.radians(15)  # Convert degrees to radians
            R_z = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
            ])
            self.pred_mesh.rotate(R_z, center=(0,0,0))
            o3d.io.write_triangle_mesh(config.output_folder + '/mesh_ncd.ply', self.pred_mesh)

        self.gt_model = o3d.io.read_point_cloud(self.gt_model_path)
        self.mesh_sample_point_num = len(self.gt_model.points)

    def evaluate(self, save_error_map=False):
        
        # crop mesh according the bounding box of gt model 
        self.pred_mesh = self.crop_mesh()
        # sampling points on the pred_mesh
        sampled_pcd = self.pred_mesh.sample_points_uniformly(number_of_points=self.mesh_sample_point_num, 
                                                             use_triangle_normal=False)
        sampled_points = np.asarray(sampled_pcd.points)
        gt_points = np.asarray(self.gt_model.points)
        
        # Completeness: query distance from the gt to the pred
        com_dists_all = self.distance_p2p_tr(gt_points, sampled_points, self.trunc_dist_com, False)
        #com_dists_all = self.distance_p2p(gt_points, sampled_points)
        if save_error_map:
            self.save_error_map(gt_points, com_dists_all, upper_bound=self.thre_dist)
        com_dists = com_dists_all[com_dists_all < self.trunc_dist_com]
        completenes = com_dists.mean()
        recall = self.get_threshold_percentage(com_dists, self.thre_dist)
        
        # Accuracy: query distance from the pred to the gt
        acc_dists_all = self.distance_p2p_tr(sampled_points, gt_points,  self.trunc_dist_acc, False)
        #acc_dists_all = self.distance_p2p(sampled_points, gt_points)
        acc_dists = acc_dists_all[acc_dists_all < self.trunc_dist_acc]
        accuracy = acc_dists.mean()
        precision = self.get_threshold_percentage(acc_dists, self.thre_dist)

        # Chamfer distance
        chamfer_l1 = 0.5 * (completenes + accuracy)

        # F-Score
        F = 2 * precision * recall / (precision + recall)
        
        print(f"completenes(cm): {completenes*100:.3f}")
        print(f"accuracy(cm): {accuracy*100:.3f}")
        print(f"chamfer_l1(cm): {chamfer_l1*100:.3f}")
        print(f"recall(%): {recall*100: .2f}")
        print(f"precision(%): {precision*100: .2f}")
        print(f"F-score(%): {F*100: .2f}")

        metrics = {"completeness": completenes*100,
                    "accuracy": accuracy*100, 
                    "chamfer_l1": chamfer_l1*100, 
                    "recall": recall*100,
                    "precision": precision*100,
                    "f_score": F*100
                   }
        self.save_metrics(metrics)
        return metrics
    
    def distance_p2p(self, points_src, points_tgt):
        kdtree = KDTree(points_tgt)
        dist, _ = kdtree.query(points_src, workers=12)
        return dist
    
    def distance_p2p_tr(self, points_src, points_tgt, truncation_dist, ignore_outlier):
        """ for each vertex in verts2 find the nearest vertex in points_src
        Args:
            nx3 np.array's
            scalar truncation_dist: points whose nearest neighbor is farther than the distance would not be taken into account
        Returns:
            ([indices], [distances])
        """

        indices = []
        distances = []
        if len(points_tgt) == 0 or len(points_src) == 0:
            return indices, distances

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_tgt)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        truncation_dist_square = truncation_dist**2

        for vert in points_src:
            _, inds, dist_square = kdtree.search_knn_vector_3d(vert, 1)
            
            if dist_square[0] < truncation_dist_square:
                indices.append(inds[0])
                distances.append(np.sqrt(dist_square[0]))
            else:
                if not ignore_outlier:
                    indices.append(inds[0])
                    distances.append(truncation_dist)

        return np.array(distances)

    def get_threshold_percentage(self,dist, threshold):
        in_threshold = (dist <= threshold).mean()
        return in_threshold

    def crop_mesh(self, padding_z = 0.02):
        bbox = self.gt_model.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        min_bound[2] -= padding_z
        max_bound[2] += padding_z
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound) 
        pred_mesh = self.pred_mesh.crop(bbox)
        
        return pred_mesh
    
    def save_error_map(self, points, errors, upper_bound):
        assert points.shape[0] == errors.shape[0]

        errors = np.clip(errors, 0, upper_bound)/upper_bound
        # the error larger the color redder
        colors = np.zeros((errors.shape[0], 3))
        colors[:, 0] = 1.0
        colors[:, 1] = 1 - errors
        colors[:, 2] = 1 - errors
        # save the error_map
        error_map = o3d.geometry.PointCloud()
        error_map.points = o3d.utility.Vector3dVector(points)
        error_map.colors = o3d.utility.Vector3dVector(colors)
        error_map_path = self.config.output_folder + '/error_map.ply'
        o3d.io.write_point_cloud(error_map_path, error_map)
        print("error map saved at " + error_map_path)

        return error_map

    def save_metrics(self, metrics):
        output_csv_path = self.config.output_folder + "/eval.csv"
        csv_columns = ["completeness", "accuracy", "chamfer_l1",\
                        "recall", "precision", "f_score"]
        try:
            with open(output_csv_path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerow(metrics)
        except IOError:
            print("write metrics error")


if __name__ == "__main__":
    
    config = Config()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit("No config file.")
    if not os.path.exists(config.output_folder):
        print("Path does not exists: "+ config.output_folder)
    
    evaluator = MeshEvaluator(config=config)
    T1 = get_time()
    metrics = evaluator.evaluate(save_error_map=True)
    T2 = get_time()
    print(T2-T1)
    print(metrics)

    