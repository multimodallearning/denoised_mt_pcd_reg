import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import numpy as np
import torch
from torch.utils.data import DataLoader

from defaults import get_cfg_defaults
from pvt_dataset import PVTDataset
from model.ppwc import PointConvSceneFlowPWC8192


def do_inference(config_file, model_path):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    # computational stuff
    batch_size = 1
    num_workers = 0
    if cfg.DEVICE == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    # model
    model = PointConvSceneFlowPWC8192(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)


    # dataset
    val_set = PVTDataset(cfg, phase='val', split='test', domain='target')
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    # Inference
    all_err = []
    all_err_init = []
    model.eval()
    epe_3d = 0
    epe_initial = 0
    for it, data in enumerate(val_loader, 1):
        pcd_src, pcd_tgt, (lm_src, lm_tgt), _, idx = data
        pcd_src = pcd_src.to(device)
        pcd_tgt = pcd_tgt.to(device)
        lm_src = lm_src.to(device)
        lm_tgt = lm_tgt.to(device)

        with torch.no_grad():
            pred_flows, _, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
            pred_flow = pred_flows[0].permute(0,2,1)

        # interpolate predicted displacements to landmarks with Gaussian kernel
        sq_dist = (lm_src.unsqueeze(2).cuda() - pcd_src.unsqueeze(1)).square().sum(dim=3)
        weights = torch.exp(-0.5 * sq_dist / 0.05 ** 2)
        flow_on_lms = (weights.unsqueeze(3) * pred_flow.unsqueeze(1)).sum(dim=2) / weights.unsqueeze(3).sum(dim=2)

        # While we perform inhale-to-exhale registration, comparison methods performed exhale-to-inhale registration.
        # Since landmarks in the inhale clouds are farther apart, we scale our results bei the ratio of the stds to
        # provide a fair comparison.
        corr_fac = val_loader.dataset.get_std_ratio(idx.item())
        corr_fac = corr_fac.view(1,1,3).to(device)

        epe_prealign = ((lm_src - lm_tgt) * corr_fac).square().sum(dim=2).sqrt()
        epe_initial += epe_prealign.mean(1).sum().item()
        all_err_init.append(epe_prealign.view(-1))
        epe_after = ((lm_src + flow_on_lms - lm_tgt) * corr_fac).square().sum(dim=2).sqrt()
        epe_3d += epe_after.mean(1).sum().item()
        all_err.append(epe_after.view(-1))

        print('Case {:02d}; before: {}; after: {}'.format(it, epe_prealign.mean(1).sum().item() * val_loader.dataset.norm_factor,
                                                              epe_after.mean(1).sum().item() * val_loader.dataset.norm_factor))

    epe_3d = epe_3d / len(val_loader.dataset) * val_loader.dataset.norm_factor
    epe_initial = epe_initial / len(val_loader.dataset) * val_loader.dataset.norm_factor

    print('initial error', epe_initial, 'EPEs', epe_3d)
    all_err = torch.cat(all_err).cpu().numpy()
    all_err_init = torch.cat(all_err_init).cpu().numpy()
    print(np.mean(all_err)* val_loader.dataset.norm_factor, np.percentile(all_err, 25)* val_loader.dataset.norm_factor, np.percentile(all_err, 75)* val_loader.dataset.norm_factor)
    print(np.mean(all_err_init)* val_loader.dataset.norm_factor, np.percentile(all_err_init, 25)* val_loader.dataset.norm_factor, np.percentile(all_err_init, 75)* val_loader.dataset.norm_factor)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--model-path",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    do_inference(args.config_file, args.model_path)
