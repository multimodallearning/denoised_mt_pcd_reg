import numpy as np
import torch
from torch_scatter import scatter
from model.pointconv_util import index_points_gather


# EMA stuff
def update_ema_variables(student, teacher, alpha):
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


class MT_Augmenter(object):
    def __init__(self, cfg):
        self.rot = cfg.DA.MT_ROT
        self.transl = cfg.DA.MT_TRANSL
        self.scale = cfg.DA.MT_SCALE

    def __call__(self, pcd1, pcd2=None):
        B, D, N = pcd1.shape
        dtype = pcd1.dtype
        device = pcd1.device

        angles = torch.deg2rad((torch.rand(B, dtype=dtype, device=device) - 0.5) * 2 * self.rot)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        z = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        rot_mat_x = torch.stack((ones, z, z, z, cos, -sin, z, sin, cos), dim=1).view(B, 3, 3)
        angles = torch.deg2rad((torch.rand(B, dtype=dtype, device=device) - 0.5) * 2 * self.rot)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        z = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        rot_mat_y = torch.stack((cos, z, sin, z, ones, z, -sin, z, cos), dim=1).view(B, 3, 3)
        angles = torch.deg2rad((torch.rand(B, dtype=dtype, device=device) - 0.5) * 2 * self.rot)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        z = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        rot_mat_z = torch.stack((cos, -sin, z, sin, cos, z, z, z, ones), dim=1).view(B, 3, 3)
        rot_mat = torch.bmm(torch.bmm(rot_mat_z, rot_mat_y), rot_mat_x)

        transl = (torch.rand(B, 1, 3, dtype=dtype, device=device) - 0.5) * 2 * self.transl
        scale = (torch.rand(B, 1, 1, dtype=dtype, device=device) - 0.5) * 2 * self.scale + 1

        pcd1 = torch.bmm(pcd1, rot_mat) * scale + transl
        if pcd2 is None:
            return pcd1, rot_mat, transl, scale
        else:
            pcd2 = torch.bmm(pcd2, rot_mat) * scale + transl
            return pcd1, pcd2, rot_mat, transl, scale


def index_points(pc, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B, N, D = pc.shape
    new_points = torch.gather(pc, dim=1, index=idx.unsqueeze(2).repeat(1, 1, D))

    return new_points


def create_syn_pair_from_pred(cfg, pcd_src, pcd_tgt, pred_flow_teach, pcd_src_large, pcd_tgt_large):
    num_points = cfg.INPUT.NUM_POINTS_SMALL
    pcd_src_warp = []
    gt_flow = []
    for b_idx in range(pcd_src.shape[0]):
        sq_dist = (pcd_src[b_idx:b_idx + 1].unsqueeze(1) - pcd_src_large[b_idx:b_idx + 1].unsqueeze(2)).square().sum(dim=3)
        weights = torch.exp(-0.5 * sq_dist / 0.05 ** 2)
        pred_flow_teach_large = (weights.unsqueeze(3) * pred_flow_teach[b_idx:b_idx + 1].unsqueeze(1)).sum(dim=2) / weights.unsqueeze(3).sum(dim=2)
        warped = pcd_src_large[b_idx:b_idx + 1, :num_points] + pred_flow_teach_large[:, :num_points]
        pcd_src_warp.append(warped)
        gt_flow.append(pred_flow_teach_large[:, -num_points:])
    pcd_src_warp = torch.cat(pcd_src_warp, dim=0)
    pcd_src = pcd_src_large[:, -num_points:].contiguous()
    pred_flow_teach = torch.cat(gt_flow, dim=0)

    return pcd_src, pcd_src_warp, pred_flow_teach
