import yaml
import os
import torch

class Config:
    def __init__(self):

        # Default values
        # settings
        self.dataset_name: str = ""
        self.data_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame)
        self.label_path: str = "" # input point-wise label path
        self.output_folder: str = ""  # output root folder
        self.random_seed: int = 42

        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = -1  # end at this frame
        self.step_frame: int = 1 # jump frames if necessary

        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.dtype = torch.float32 # default torch tensor data type

        self.semantic_on: bool = False # semantic shine mapping on [semantic]
        self.sem_class_num: int = 20 # semantic class count: 20 for semantic kitti

        # lidar preprocess
        self.min_range: float = 2.75
        self.max_range: float = 80.0
        self.min_z: float = -5.0  # filter for z coordinates (unit: m)
        self.max_z: float = 60.0

        self.down_sample: bool = True  # apply voxel downsampling to input original point clcoud
        self.vox_down_m: float = 0.05 # the voxel size if using voxel downsampling (unit: m)

        # range image parameter (only for lidar sensor)
        self.fov_up: float = 3
        self.fov_down: float = -25
        self.proj_H: int = 64
        self.proj_W: int = 1080

        # neural voxel hash
        self.leaf_voxel_size: float = 0.3
        self.voxel_level_num : int = 2
        self.scale_up_factor : float = 1.5
        self.hash_buffer_size: int = int(2e7)

        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.feature_std: float = 0.0  # grid feature initialization standard deviation
        self.inflate_margin_ratio = 0.1
        self.dilate_ratio = 0.1 # for more stable voxel initialization, dilate distance = dilate_ratio * voxel_size


        # MLP decoder
        self.mlp_bias_on: bool = True
        self.geo_mlp_level: int = 1
        self.geo_mlp_hidden_dim: int = 64
        self.sem_mlp_level: int = 1
        self.sem_mlp_hidden_dim: int = 64
        self.color_mlp_level: int = 1
        self.color_mlp_hidden_dim: int = 64
        self.freeze_after_frame: int = 40  # if the decoder model is not loaded, it would be trained and freezed after such frame number

        # sampling
        # spilt into 3 parts for sampling: close-to-surface (+ exact beam endpoint) + front-surface-freespace + behind-surface-freespace
        self.surface_sample_range_m: float = 0.25 # better to be set according to the noise level (actually as the std for a gaussian distribution)
        self.surface_sample_n: int = 3
        self.free_sample_begin_ratio: float = 0.3 # minimum ray distance ratio in front of the surface 
        self.free_sample_end_dist_m: float = 1.0 # maximum distance behind the surface (unit: m)
        self.free_front_n: int = 2
        self.free_behind_n: int = 1

        # training (mapping) loss
        # the main loss type, select from the sample sdf loss ('bce', 'l1', 'l2', 'zhong') 
        self.main_loss_type: str = 'bce'
        self.sigma_sigmoid_m: float = 0.1 # better to be set according to the noise level (used only for BCE loss as the sigmoid scale factor)
        self.logistic_gaussian_ratio: float = 0.55 # the factor ratio for approximize a Gaussian distribution using the derivative of logistic function

        self.proj_correction_on: bool = False ## conduct projective distance correction based on the sdf gradient or not, True does not work well 
        self.loss_weight_on: bool = False  # if True, the weight would be given to the loss, if False, the weight would be used to change the sigmoid's shape
        self.behind_dropoff_on: bool = False  # behind surface drop off weight
        self.dist_weight_on: bool = True  # weight decrease linearly with the measured distance, reflecting the measurement noise
        self.dist_weight_scale: float = 0.8 # weight changing range [0.6, 1.4]
        
        self.numerical_grad: bool = True # use numerical SDF gradient as in the paper Neuralangelo for the Ekional regularization during mapping
        self.gradient_decimation: int = 10 # use just a part of the points for the ekional loss when using the numerical grad, save computing time
        self.num_grad_step_ratio: float = 0.2 # step as a ratio of the nerual point resolution, length = num_grad_step_ratio * voxel_size_m

        self.ekional_loss_on: bool = True # Ekional regularization (default on)
        self.ekional_add_to: str = 'all' # select from 'all', 'surface', 'freespace', the samples used for Ekional regularization
        self.weight_e: float = 0.5

        self.consistency_loss_on: bool = False # gradient consistency (smoothness) regularization (default off)
        self.weight_c: float = 0.5 # weight for consistency loss, don't mix with the color weight 
        self.consistency_count: int = 1000
        self.consistency_range: float = 0.05 # the neighborhood points would be randomly select within the radius of consistency_range (unit: m)

        self.weight_s: float = 1.0  # weight for semantic classification loss
        self.weight_i: float = 1.0  # weight for color or intensity regression loss

        # mapping
        self.epochs: int = 20
        self.batch_size: int = 16384
        self.learning_rate: float = 0.01

        # output
        self.segmentation_threshold: float = 0.16
        self.static_pointcloud: bool = False
        self.point_cloud_viewer: bool = False
        self.mesh_recon: bool = False
        #self.mesh_dynamic: bool = False
        self.mesh_resolution: float = 0.2
        self.mesh_res: int = 5
        self.save_pc_map: bool = False
        self.save_sdf_slice_map: bool = False
        self.pad_voxel: int = 2 # pad x voxels on each side
        self.skip_top_voxel: int = 2 # slip the top x voxels (mainly for visualization indoor, remove the roof)
        self.mc_mask_on: bool = True # use mask for marching cubes to avoid the artifacts

        # eval mesh
        self.save_error_map: bool = False
        self.thre_dist:float = 0.1
        self.trunc_dist_acc:float = 0.2
        self.trunc_dist_com:float = 2.0
        self.gt_model_path:str = ""


    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # setting
        self.dataset_name = config_args["setting"]["dataset"]
        self.data_path = config_args["setting"]["data_path"] 
        self.pose_path = config_args["setting"]["pose_path"]
        self.calib_path = config_args["setting"]["calib_path"]
        self.label_path = config_args["setting"]["label_path"]
        self.output_folder = config_args["setting"]["output_folder"]  
        self.begin_frame = config_args["setting"]["begin_frame"]
        self.step_frame = config_args["setting"]["step_frame"]
        self.end_frame = config_args["setting"]["end_frame"]
        self.random_seed = config_args["setting"].get("random_seed", self.random_seed)
        self.device = config_args["setting"]["device"]

        # lidar preprocess
        self.min_range = config_args["process"].get("min_range",self.min_range)
        self.max_range = config_args["process"].get("max_range",self.max_range)
        self.min_z = config_args["process"].get("min_z",self.min_z)
        self.max_z = config_args["process"].get("max_z",self.max_z)
        self.down_sample = config_args["process"].get("down_sample", self.down_sample)
        self.vox_down_m = config_args["process"].get("vox_down_m", self.max_range*1e-3)

        # range image parameter (only for lidar sensor)
        self.fov_up = config_args["range_image"].get("fov_up",self.fov_up)
        self.fov_down = config_args["range_image"].get("fov_down",self.fov_down)
        self.proj_H = config_args["range_image"].get("proj_H",self.proj_H)
        self.proj_W = config_args["range_image"].get("proj_W",self.proj_W)

        # neuralvoxel 
        self.leaf_voxel_size = config_args["neural_voxel"]["leaf_voxel_size"]
        self.voxel_level_num = config_args["neural_voxel"]["voxel_level_num"]
        self.scale_up_factor = config_args["neural_voxel"]["scale_up_factor"]
        self.feature_dim = config_args["neural_voxel"]["feature_dim"]

        # decoder
        # number of the level of the mlp decoder
        self.geo_mlp_level = config_args["decoder"].get("mlp_level", self.geo_mlp_level)
        # dimension of the mlp's hidden layer
        self.geo_mlp_hidden_dim = config_args["decoder"].get("mlp_hidden_dim", self.geo_mlp_hidden_dim) 
        # freeze the decoder after runing for x frames (used for incremental mapping to avoid forgeting)
        self.freeze_after_frame = config_args["decoder"].get("freeze_after_frame", self.freeze_after_frame)
        # TODO, now set to the same as geo mlp, but actually can be different
        self.color_mlp_level = self.geo_mlp_level
        self.color_mlp_hidden_dim = self.geo_mlp_hidden_dim
        self.sem_mlp_level = self.geo_mlp_level
        self.sem_mlp_hidden_dim = self.geo_mlp_hidden_dim

        # sampling
        self.surface_sample_range_m = config_args["sampler"].get("surface_sample_range_m", self.vox_down_m * 3.0) 
        self.free_sample_begin_ratio = config_args["sampler"].get("free_sample_begin_ratio", self.free_sample_begin_ratio)
        self.free_sample_end_dist_m = config_args["sampler"].get("free_sample_end_dist_m", self.surface_sample_range_m * 4.0) # this value should be at least 2 times of surface_sample_range_m
        self.surface_sample_n = config_args["sampler"].get("surface_sample_n", self.surface_sample_n)
        self.free_front_n = config_args["sampler"].get("free_front_sample_n", self.free_front_n)
        self.free_behind_n = config_args["sampler"].get("free_behind_sample_n", self.free_behind_n)   

        # loss
        self.main_loss_type = config_args["loss"].get("main_loss_type", "bce")
        self.sigma_sigmoid_m = config_args["loss"].get("sigma_sigmoid_m", self.leaf_voxel_size)
        self.proj_correction_on = config_args["loss"].get("proj_correction_on", self.proj_correction_on)
        self.loss_weight_on = config_args["loss"].get("loss_weight_on", self.loss_weight_on)
        if self.loss_weight_on:
            self.dist_weight_scale = config_args["loss"].get("dist_weight_scale", self.dist_weight_scale)
            # apply "behind the surface" loss weight drop-off or not
            self.behind_dropoff_on = config_args["loss"].get("behind_dropoff_on", self.behind_dropoff_on)
        self.ekional_loss_on = config_args["loss"].get("ekional_loss_on", self.ekional_loss_on) # use ekional loss (norm(gradient) = 1 loss)
        self.weight_e = float(config_args["loss"].get("weight_e", self.weight_e))
        self.numerical_grad = config_args["loss"].get("numerical_grad_on", self.numerical_grad)
        if not self.numerical_grad:
            self.gradient_decimation = 1
        else:
            self.gradient_decimation = config_args["loss"].get("grad_decimation", self.gradient_decimation)
            self.num_grad_step_ratio = config_args["loss"].get("num_grad_step_ratio", self.num_grad_step_ratio)
        self.consistency_loss_on = config_args["loss"].get("consistency_loss_on", self.consistency_loss_on)

        # mapping
        self.epochs = config_args["mapping"]["epochs"] 
        self.batch_size = config_args["mapping"]["batch_size"] 
        self.learning_rate = config_args["mapping"]["learning_rate"]

        # output
        self.segmentation_threshold = config_args["output"]["segmentation_threshold"]
        self.static_pointcloud = config_args["output"]["static_pointcloud"]
        self.point_cloud_viewer =config_args["output"]["point_cloud_viewer"]
        self.mesh_recon = config_args["output"]["mesh_recon"]
        #self.mesh_dynamic = config_args["output"]["mesh_dynamic"]
        self.mesh_resolution = config_args["output"]["mesh_resolution"]
        # self.mesh_res = config_args["output"]["mesh_res"]
        self.save_pc_map = config_args["output"]["save_pc_map"]
        self.save_error_map = config_args["output"]["save_error_map"]
        self.thre_dist = config_args["output"]["thre_dist"]
        self.trunc_dist_acc = config_args["output"]["trunc_dist_acc"]
        self.trunc_dist_com = config_args["output"]["trunc_dist_com"]
        self.gt_model_path = config_args["output"]["gt_model_path"]
