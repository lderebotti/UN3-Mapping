<p align="center">
  <h1 align="center">UN3-Mapping: Uncertainty-Aware Neural Non-Projective Signed Distance Fields for 3D Mapping
</h1>

[click to see the paper](https://ieeexplore.ieee.org/abstract/document/11078897)
```
@article{song2025un3,
  title={UN3-Mapping: Uncertainty-Aware Neural Non-Projective Signed Distance Fields for 3D Mapping},
  author={Song, Shuangfu and Zhao, Junqiao and Veas, Eduardo and Lin, Jiaye and Cao, Qiuyi and Ye, Chen and Feng, Tiantian},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
<img width="1920" height="849" alt="overview" src="https://github.com/user-attachments/assets/0fbbd14c-3f6b-450f-99ff-57249bfad3d8" />
<img width="1024" height="382" alt="qualitative_ucner_grad" src="https://github.com/user-attachments/assets/5c19fd04-f591-4035-84f0-19450f8e8452" />

## Abstract

<details>
  <summary>[Details (click to expand)]</summary>
Building accurate and reliable maps is a critical requirement for autonomous robots. In this letter, we propose UN3-Mapping, an implicit neural mapping method that enables high-quality 3D reconstruction with integrated uncertainty estimation. Our approach employs a hybrid representation: an implicit neural distance field models scene geometry, while an explicit gradient field, optimized from surface normals, derives non-projective signed distance labels from raw range data. These refined distance labels are then used to train our implicit map. For uncertainty estimation, we design an online learning framework to capture the reconstruction uncertainty in a self-supervised manner. Benefiting from the uncertainty-aware map, our method is capable of removing the dynamic obstacles with high uncertainty within the raw point cloud. Extensive experiments show that our approach outperforms existing methods in mapping accuracy and completeness while also exhibiting promising potential for dynamic object segmentation.
</details>

## Installation

We tested our code on Ubuntu 20.04 with an NVIDIA RTX 3090.

### 1. Set up conda environment

```
conda create --name un3 python=3.8
conda activate un3
```

### 2. Install PyTorch

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

The commands depend on your CUDA version. You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).


### 4. Install other dependencies

```
pip install open3d scikit-image tqdm natsort
```

## How to run it

### Clone the repository
```
git clone git@github.com:tiev-tongji/UN3-Mapping.git
cd UN3-Mapping
```

### Demo

```
sh ./scripts/download_maicity.sh
```
Then run:
```
python run_batch_mapping.py config/maicity.yaml
```
Notice that you need to check the config file and set the correct path like `pc_path`, `pose_path` and `calib_path`.
After mapping, you can check the reconstructed mesh and learned uncertainty in the `experiments` folder.

## Evaluation
As reported in the paper, we evaluate our method in surface reconstruction and dynamic object segmentation.

### 3D Surface reconstruction
Please prepare your reconstructed mesh and corresponding ground truth point cloud. Then set the ground truth model path `gt_model_path` in the config file. Then run:
```
python eval/evaluator.py path/to/config.yaml
```

### Dynamic objects segmentation
Following [4DNDF](git@github.com:PRBonn/4dNDF.git), we use [KTH_DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) to evaluate the result of our Dynamic objects segmentation.

Download the data from the official link [here](https://zenodo.org/records/8160051) and unzip it to our data folder as:

```
./4dNDF/
└── data/
    └── KTH_dynamic/
            ├── 00/
            │   ├── gt_cloud.pcd
            │   ├── pcd/
            |   |    ├── 004390.pcd
            |   |    ├── 004391.pcd
            |   |    └── ...
            ├── 05/ 
            ├── av2/
            ├── semindoor/
            └── translations/
```

The benchmark doesn't explicitly provide the pose files, so we extract poses from the data and store them in the `data/translations/` folder.

For evaluating, you need to clone (to somewhere you like) and compile the [KTH_DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) 's repo. The following commands should work if you have ROS-full installed on your machine.

```
git clone --recurse-submodules https://github.com/KTH-RPL/DynamicMap_Benchmark.git
cd DynamicMap_Benchmark/script
mkdir build && cd build
camke ..
make
```

Or check the guidance from the benchmark [here](https://github.com/KTH-RPL/DynamicMap_Benchmark/tree/master/methods).

Then, copy the Python file from `4dNDF/eval/evaluate_single_kth.py` to `/path/to/DynamicMap_Benchmark/scripts/py/eval/`

Take sequence 00 as an example. Run:

```
python static_mapping.py config/kth/00.yaml
```

After training, the static point cloud can be found here: `data/kth/00/static_points.pcd`

To evaluate it, run ( change the `/your/path/to` and `/path/to` to the correct path):

```
cd /your/path/to/DynamicMap_Benchmark/scripts/build/
./export_eval_pcd  /path/to/4dNDF/data/KTH_dynamic/00 static_points.pcd 0.05
```
It will generate the eval point cloud. Finally, Run:

```
python /your/path/to/DynamicMap_Benchmark/scripts/py/eval/evaluate_single_kth.py /path/to/4dNDF/data/KTH_dynamic/00
```
to check the number. For other sequences, we need to change all the `00` to `05`,` av2`, or `semindoor`. You can organize the commands as a bash script to make it more convenient.

## Contact
Feel free to contact me if you have any questions :)
- Song {[1911204@tongji.edu.cn]()}

## Acknowledgment
Our work is mainly built on [4DNDF](git@github.com:PRBonn/4dNDF.git) and [N3-Mapping](https://github.com/tiev-tongji/N3-Mapping). Many thanks to the authors of this excellent work!
We also appreciate the following great open-source works:
- [Voxfield](https://github.com/VIS4ROB-lab/voxfield) (comparison baseline, inspiration)
- [Voxblox](https://github.com/ethz-asl/voxblox) (comparison baseline)
- [PIN_SLAM](https://github.com/PRBonn/PIN_SLAM) (comparison baseline)
- [DynamicMap_Benchmark](https://github.com/KTH-RPL/DynamicMap_Benchmark) (benchmark)

Some implementation is not exactly the same as the paper, but the final performance is very close. The pipeline of incremental mapping still needs to be optimized, and we will update it as soon as possible.
