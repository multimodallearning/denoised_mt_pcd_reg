import os
import numpy as np
import pyvista as pv
import torch
import torch.utils.data
import open3d as o3d


def read_vtk(path):
    data = pv.read(path)
    data_dict = {}
    data_dict["points"] = data.points.astype(np.float32)
    data_dict["faces"] = data.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    for name in data.array_names:
        try:
            data_dict[name] = data[name]
        except:
            pass
    return data_dict


class PVTDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, phase, split, domain):
        self.is_train = True if phase == 'train' else False
        self.split = split
        self.domain = domain
        self.root = 'datasets/pvt'
        self.cloudidx_file_template_large = 'kpts_sigma=6.0_n1=29_n2=10_sigInt=0.1_minPts=16384/copd_{:06d}_idx_{}.pth'
        self.cloudidx_file_template_small = 'kpts_sigma=0.7_n1=29_n2=15_sigInt=0.1_minPts=9000/copd_{:06d}_idx_{}.pth'
        self.cloud_file_template = 'raw_data/copd_{:06d}_{}.vtk'

        if split == 'train':
            self.case_list = np.arange(11, 1011)
            self.case_list = self.case_list[~np.isin(self.case_list, [139, 428, 611, 1002])]
            self.case_list = np.random.default_rng(12345).permutation(self.case_list)[:800]
        elif split == 'val':
            self.case_list = np.arange(11, 1011)
            self.case_list = self.case_list[~np.isin(self.case_list, [139, 428, 611, 1002])]
            self.case_list = np.random.default_rng(12345).permutation(self.case_list)[800:]
        elif split == 'test':
            self.case_list = np.arange(1, 11)
        else:
            raise NotImplementedError()

        self.norm_factor = cfg.INPUT.SCALE_NORM_FACTOR
        self.num_points_small = cfg.INPUT.NUM_POINTS_SMALL
        self.num_points_large = cfg.INPUT.NUM_POINTS_LARGE

        # augmentation parameters
        self.augm_setting = cfg.AUGMENTATIONS

    def __getitem__(self, idx):
        # load input pcds
        case = self.case_list[idx]

        if case <= 10:
            # load input clouds
            pcd_tgt = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, 'EXP')))
            pcd_tgt = torch.tensor(pcd_tgt['points'])
            pcd_tgt_idx = torch.load(os.path.join(self.root, self.cloudidx_file_template_small.format(case, 'EXP')), map_location='cpu')
            pcd_tgt = pcd_tgt[pcd_tgt_idx]
            pcd_tgt = pcd_tgt.numpy()
            pcd_src = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, 'INSP')))
            pcd_src = torch.tensor(pcd_src['points'])
            pcd_src_idx = torch.load(os.path.join(self.root, self.cloudidx_file_template_small.format(case, 'INSP')), map_location='cpu')
            pcd_src = pcd_src[pcd_src_idx]
            pcd_src = pcd_src.numpy()

            # load landmarks
            lm_tgt = read_vtk('datasets/dirlab_copd/processed_lms/copd_{:06d}_EXP.vtk'.format(case))
            lm_tgt = torch.tensor(lm_tgt['points'])
            lm_src = read_vtk('datasets/dirlab_copd/processed_lms/copd_{:06d}_INSP.vtk'.format(case))
            lm_src = torch.tensor(lm_src['points'])

            # prealignment
            mean_tgt = np.mean(pcd_tgt, axis=0)
            std_tgt = np.std(pcd_tgt, axis=0)
            mean_src = np.mean(pcd_src, axis=0)
            std_src = np.std(pcd_src, axis=0)
            pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
            lm_src = (lm_src - mean_src) * std_tgt / std_src + mean_tgt

            # mean center and scale
            mean = np.mean(pcd_tgt, axis=0)
            pcd_tgt = (pcd_tgt - mean) / self.norm_factor
            pcd_src = (pcd_src - mean) / self.norm_factor
            lm_tgt = (lm_tgt - mean) / self.norm_factor
            lm_src = (lm_src - mean) / self.norm_factor

            # subsample point clouds
            if self.num_points_small > 0:
                pts_size_tgt = pcd_tgt.shape[0]
                pts_size_src = pcd_src.shape[0]
                if pts_size_tgt >= self.num_points_small:
                    if self.is_train:
                        raise ValueError()
                    else:
                        permutation_tgt = np.random.default_rng(12345 + 10 * idx).permutation(pts_size_tgt)
                        permutation_src = np.random.default_rng(12345 + 10 * idx + 1).permutation(pts_size_src)
                    pcd_tgt = pcd_tgt[permutation_tgt[:self.num_points_small], :]
                    pcd_src = pcd_src[permutation_src[:self.num_points_small], :]
                else:
                    raise ValueError()

            return np.float32(pcd_src), np.float32(pcd_tgt), (np.float32(lm_src), np.float32(lm_tgt)), '', idx

        else:
            if self.is_train and self.augm_setting.RANDOM_SEED < 0:
                rand = np.random.default_rng().uniform()
            else:
                rand = np.random.default_rng(123456 + 10*idx).uniform()
            which_augment = 'src' if rand < 0.5 else 'tgt'
            if self.domain == 'target':
                which_augment = 'tgt'
            assert which_augment in ['src', 'tgt']

            state = 'EXP' if which_augment == 'tgt' else 'INSP'
            pcd_tgt = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, state)))
            pcd_tgt = torch.tensor(pcd_tgt['points'])
            pcd_tgt_idx = torch.load(os.path.join(self.root, self.cloudidx_file_template_large.format(case, state)), map_location='cpu')
            pcd_tgt = pcd_tgt[pcd_tgt_idx]
            pcd_tgt = pcd_tgt.numpy()


            if self.domain == 'target':
                state = 'EXP' if state == 'INSP' else 'INSP'
                pcd_src = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, state)))
                pcd_src = torch.tensor(pcd_src['points'])
                pcd_src_idx = torch.load(os.path.join(self.root, self.cloudidx_file_template_large.format(case, state)), map_location='cpu')
                pcd_src = pcd_src[pcd_src_idx]
                pcd_src = pcd_src.numpy()

                pcd_src_small = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, state)))
                pcd_src_small = torch.tensor(pcd_src_small['points'])
                pcd_src_idx_small = torch.load(os.path.join(self.root, self.cloudidx_file_template_small.format(case, state)), map_location='cpu')
                pcd_src_small = pcd_src_small[pcd_src_idx_small]
                pcd_src_small = pcd_src_small.numpy()

                state = 'EXP' if state == 'INSP' else 'INSP'
                pcd_tgt_small = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, state)))
                pcd_tgt_small = torch.tensor(pcd_tgt_small['points'])
                pcd_tgt_idx_small = torch.load(os.path.join(self.root, self.cloudidx_file_template_small.format(case, state)), map_location='cpu')
                pcd_tgt_small = pcd_tgt_small[pcd_tgt_idx_small]
                pcd_tgt_small = pcd_tgt_small.numpy()

                # prealignment
                mean_tgt = np.mean(pcd_tgt_small, axis=0)
                std_tgt = np.std(pcd_tgt_small, axis=0)
                mean_src = np.mean(pcd_src_small, axis=0)
                std_src = np.std(pcd_src_small, axis=0)
                pcd_src_small = (pcd_src_small - mean_src) * std_tgt / std_src + mean_tgt
                pcd_src = (pcd_src - mean_src) * std_tgt / std_src + mean_tgt
                # mean center and scale
                mean = np.mean(pcd_tgt_small, axis=0)
                pcd_tgt_small = (pcd_tgt_small - mean) / self.norm_factor
                pcd_src_small = (pcd_src_small - mean) / self.norm_factor
                pcd_src = (pcd_src - mean) / self.norm_factor
                pcd_tgt = (pcd_tgt - mean) / self.norm_factor
                gt_flow = pcd_src - pcd_src  # dummy gt flow as placeholder, real gt flow is neither available nor needed

                permutation_src_small = np.random.default_rng().permutation(pcd_src_small.shape[0])[:self.num_points_small]
                pcd_src_small = pcd_src_small[permutation_src_small, :]
                permutation_tgt_small = np.random.default_rng().permutation(pcd_tgt_small.shape[0])[:self.num_points_small]
                pcd_tgt_small = pcd_tgt_small[permutation_tgt_small, :]

            else:
                mean = np.mean(pcd_tgt, axis=0)
                pcd_tgt = (pcd_tgt - mean) / self.norm_factor
                pcd_src = self.augment(pcd_tgt.copy(), idx)
                gt_flow = pcd_tgt - pcd_src

            # subsample point clouds
            if self.is_train and self.domain == 'target':
                num_points = self.num_points_large
            else:
                num_points = self.num_points_small
            if num_points > 0:
                pts_size_tgt = pcd_tgt.shape[0]
                pts_size_src = pcd_src.shape[0]

                if pts_size_tgt >= num_points:
                    if self.is_train:
                        if self.domain != 'target':
                            assert pts_size_tgt == pts_size_src
                            permutation = np.random.default_rng().permutation(pts_size_tgt)
                            permutation_tgt = permutation[:num_points]
                            permutation_src = permutation[-num_points:]
                        else:
                            permutation_tgt = np.random.default_rng().permutation(pts_size_tgt)[:num_points]
                            permutation_src = np.random.default_rng().permutation(pts_size_src)[:num_points]
                    else:
                        if self.split != 'test':
                            assert pts_size_tgt == pts_size_src
                            permutation = np.random.default_rng(12345 + 10*idx).permutation(pts_size_tgt)
                            permutation_tgt = permutation[:num_points]
                            permutation_src = permutation[-num_points:]
                        else:
                            permutation_tgt = np.random.default_rng(12345 + 10*idx).permutation(pts_size_tgt)[:num_points]
                            permutation_src = np.random.default_rng(12345 + 10*idx + 1).permutation(pts_size_src)[:num_points]
                    pcd_tgt = pcd_tgt[permutation_tgt, :]
                    pcd_src = pcd_src[permutation_src, :]
                    gt_flow = gt_flow[permutation_src, :]
                else:
                    raise ValueError()

            if self.domain == 'target':
                return np.float32(pcd_src_small), np.float32(pcd_tgt_small), np.float32(pcd_src), np.float32(pcd_tgt), np.float32(gt_flow), which_augment, idx
            else:
                return np.float32(pcd_src), np.float32(pcd_tgt), np.float32(gt_flow), which_augment, idx

    def augment(self, pcd, idx):
        setting = self.augm_setting
        seed = setting.RANDOM_SEED
        if setting.METHOD == 'rigid':
            max_transl = setting.MAX_TRANSLATION
            scale_offset = setting.MAX_SCALE_OFFSET
            rot_max = setting.MAX_ROTATION_ANGLE
            if seed < 0:
                transl = np.random.uniform(-1., 1., (1, 3)) * max_transl
                scale = np.random.uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
                rot_angles = np.deg2rad(np.random.uniform(-rot_max, rot_max, 3))
            else:
                transl = np.random.default_rng(seed + 10*idx + 1).uniform(-1., 1., (1, 3)) * max_transl
                scale = np.random.default_rng(seed + 10*idx + 2).uniform(1 - scale_offset, 1 + scale_offset, (1, 3))
                rot_angles = np.deg2rad(np.random.default_rng(seed + 10*idx + 3).uniform(-rot_max, rot_max, 3))

            theta = rot_angles[0]
            rot_mat_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
            theta = rot_angles[1]
            rot_mat_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            theta = rot_angles[2]
            rot_mat_z = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
            rot_mat = np.dot(np.dot(rot_mat_x, rot_mat_y), rot_mat_z)

            pcd_augm = np.dot(pcd, rot_mat) * scale + transl

        elif setting.METHOD == 'multiscale_local_global':
            seed = setting.RANDOM_SEED
            num_control_points_local = setting.NUM_CONTROL_POINTS_LOCAL
            max_control_shift_local = setting.MAX_CONTROL_SHIFT_LOCAL
            kernel_std_local = setting.KERNEL_STD_LOCAL
            global_grid_spacing = setting.GLOBAL_GRID_SPACING
            max_control_shift_global = setting.MAX_CONTROL_SHIFT_GLOBAL
            kernel_std_global = setting.KERNEL_STD_GLOBAL

            if seed < 0:
                local_control_idx = np.random.permutation(pcd.shape[0])[:num_control_points_local]
                local_control_shifts = np.random.uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local
            else:
                local_control_idx =  np.random.default_rng(seed + 10 * idx + 1).permutation(pcd.shape[0])[:num_control_points_local]
                local_control_shifts = np.random.default_rng(seed + 10 * idx + 2).uniform(-1., 1., (num_control_points_local, 3)) * max_control_shift_local

            local_control_pts = pcd[local_control_idx]
            sq_dist = np.sum(np.square(pcd[:, None] - local_control_pts[None]), axis=2)
            weights = np.exp(-0.5 * sq_dist / kernel_std_local**2)
            local_pcd_shifts = np.sum(weights[:, :, None] * local_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
            pcd_augm = pcd + local_pcd_shifts

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(pcd_augm)
            o3d_cloud, _, _ = o3d_cloud.voxel_down_sample_and_trace(global_grid_spacing,
                                                                    min_bound=np.array([-10., -10., -10.]),
                                                                    max_bound=np.array([10., 10., 10.]))

            global_control_pts = np.float32(np.asarray(o3d_cloud.points))
            if seed < 0:
                global_control_shifts = np.random.uniform(-1, 1., (global_control_pts.shape[0], 3)) * max_control_shift_global
            else:
                global_control_shifts = np.random.default_rng(seed + 10 * idx + 3).uniform(-1, 1., (global_control_pts.shape[0], 3)) * max_control_shift_global
            sq_dist = np.sum(np.square(pcd_augm[:, None] - global_control_pts[None]), axis=2)
            weights = np.exp(-0.5 * sq_dist / kernel_std_global ** 2)
            global_pcd_shifts = np.sum(weights[:, :, None] * global_control_shifts[None], axis=1) / np.sum(weights[:, :, None], axis=1)
            pcd_augm = pcd_augm + global_pcd_shifts

        else:
            raise ValueError()

        return pcd_augm

    def __len__(self):
        return len(self.case_list)

    def get_std_ratio(self, idx):
        case = self.case_list[idx]
        if case <= 10:
            # load input clouds
            pcd_tgt = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, 'EXP')))
            pcd_tgt = torch.tensor(pcd_tgt['points'])
            pcd_tgt_idx = torch.load(os.path.join(self.root, self.cloudidx_file_template_small.format(case, 'EXP')), map_location='cpu')
            pcd_tgt = pcd_tgt[pcd_tgt_idx]
            pcd_src = read_vtk(os.path.join(self.root, self.cloud_file_template.format(case, 'INSP')))
            pcd_src = torch.tensor(pcd_src['points'])
            pcd_src_idx = torch.load(os.path.join(self.root, self.cloudidx_file_template_small.format(case, 'INSP')), map_location='cpu')
            pcd_src = pcd_src[pcd_src_idx]

            # prealignment
            std_tgt = pcd_tgt.std(dim=0)
            std_src = pcd_src.std(dim=0)
            std_ratio = std_src / std_tgt

            return std_ratio
        else:
            raise ValueError()
