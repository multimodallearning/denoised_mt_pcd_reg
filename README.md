# denoised_mt_pcd_reg
Source code for our Miccai2023 paper [A denoised Mean Teacher for domain adaptive point cloud registration](https://arxiv.org/abs/2306.14749) [[pdf](https://arxiv.org/pdf/2306.14749.pdf)].

## Dependencies
Please first install the following dependencies
* Python3 (we use 3.9.7)
* pytorch (we use 1.10.2)
* numpy
* yacs
* pyvista
* open3d
* geomloss

Then, you need to compile the `pointnet2_utils` via `cd pointnet2`, `python setup.py install`, `cd ..`.

## Data Preparation
1. Download the 10 cases of the [COPDgene dataset](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html). Create a directory `/datasets/dirlab_copd/raw_data` and move the dataset to this directory. We recommend to create a symlink. Note that we do not require the iamge data but only the landmark annotations in the .txt files for evaluation.
2. Download the 1010 cases of the PVT dataset, following the instructions in this [Github repository](https://github.com/uncbiag/robot). Create a directory `/datasets/pvt/raw_data` and move the dataset to this directory.
3. Execute `cd preprocessing` followed by `python process_copd_lms.py` and `python pvt_keypoint_keypoints_extract.py`. This transforms COPD landmarks to the PVT space and extracts distinctive keypoints from the high-resolution PVT clouds.
4. Afterwards, the data should be organized as follows:
```
    .
    ├── ...
    ├── datasets
    │   ├── dirlab_copd
    │   │   ├── processed_lms
    │   │   │   ├── copd_000001_EXP.vtk
    │   │   │   ├── copd_000001_INSP.vtk
    │   │   │   └── ...
    │   │   └── raw_data
    │   │       ├── copd1_300_eBH_xyz_r1.txt
    │   │       ├── copd1_300_iBH_xyz_r1.txt
    │   │       └── ...
    │   └── pvt
    │       ├── kpts_sigma=0.7_n1=29_n2=15_sigInt=0.1_minPts=9000
    │       │   ├── copd_000001_idx_EXP.vtk
    │       │   ├── copd_000001_idx_INSP.vtk
    │       │   └── ...
    │       ├── kpts_sigma=6.0_n1=29_n2=10_sigInt=0.1_minPts=16384
    │       │   ├── copd_000001_idx_EXP.vtk
    │       │   ├── copd_000001_idx_INSP.vtk
    │       │   └── ...
    │       └── raw_data
    │           ├── copd_000001_EXP.vtk
    │           ├── copd_000001_INSP.vtk
    │           └── ...
    └── ...
```
## Training
1. In `defaults.py`, modify `_C.BASE_DIRECTORY` in line 5 to the root directory where you intend to save the results.
2. In the config files `/configs/CONFIG_TO_SPECIFY.yaml`, you can optionally modify `EXPERIMENT_NAME` in line 1. Models and log files will finally be written to `os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)`.
3. To perform pre-training/train the source-only model, execute `python train.py --config-file configs/def=DEF_pretrain.yaml --train-state pretrain`.
4. To perform domain adaptation, execute `python train.py --config-file configs/def=DEF_ours.yaml --train-state adapt`. The current config file uses our pre-trained models for initialization. If you wish to use your own pre-trained models, you need to modify the paths in `MODEL.WEIGHTS` and `MODEL.TEACHER_WEIGHTS` in the config accordingly.

## Testing
* You can test the model by executing `python test.py --config-file PATH/TO/CONFIG.yaml --model-path PATH/TO/MODEL.pth`. Our pretrained models are provided in the `trained_models` directory.

## Citation
If you find our code useful for your work, please cite the following paper
```latex
@article{bigalke2023denoised,
  title={A denoised Mean Teacher for domain adaptive point cloud registration},
  author={Bigalke, Alexander and Heinrich, Mattias P},
  journal={arXiv preprint arXiv:2306.14749},
  year={2023}
}
```

## Acknowledgements
* Code for data pre-processing has been adapted from https://github.com/uncbiag/robot
* Code for the PointPWC-Net has been adapted from https://github.com/DylanWusee/PointPWC

We thank the authors for sharing their code!

