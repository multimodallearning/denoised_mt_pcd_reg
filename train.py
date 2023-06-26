import argparse
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from geomloss import SamplesLoss

from defaults import get_cfg_defaults
from pvt_dataset import PVTDataset
from model.ppwc import PointConvSceneFlowPWC8192, multiScaleLoss, computeChamfer, computeCurvature
from utils import MT_Augmenter, update_ema_variables, create_syn_pair_from_pred


def pretrain(cfg, debug=False):
    root = cfg.BASE_DIRECTORY
    exp_name = cfg.EXPERIMENT_NAME
    out_folder = os.path.join(root, exp_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model_path = os.path.join(out_folder, 'model.pth')
    model_path_ep = os.path.join(out_folder, 'model_ep={}.pth')
    teacher_path_ep = os.path.join(out_folder, 'teacher_alpha={}_ep={}.pth')


    # hyperparameters
    init_lr = cfg.SOLVER.LEARNING_RATE
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    lr_steps = cfg.SOLVER.LR_MILESTONES
    lr_gamma = cfg.SOLVER.LR_LAMBDA
    batch_size = cfg.SOLVER.BATCH_SIZE
    loss_factor = cfg.SOLVER.LOSS_FACTOR

    # computational stuff
    num_workers = 0 if debug else cfg.NUM_WORKERS
    if cfg.DEVICE == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    # model
    alphas = [cfg.DA.MT_TEACHERALPHA]
    num_teachers = len(alphas)
    teachers = []
    model = PointConvSceneFlowPWC8192(cfg).to(device)
    for _ in alphas:
        teacher = PointConvSceneFlowPWC8192(cfg).to(device)
        for param in teacher.parameters():
            param.requires_grad = False
        teachers.append(teacher)

    # optimizer
    optimizer = optim.Adam(model.parameters(), init_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    if cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)
    else:
        raise ValueError()


    # datasets
    train_set = PVTDataset(cfg, phase='train', split='train', domain='source')
    if debug:
        train_set.case_list = train_set.case_list[:8]
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    val_sets = [PVTDataset(cfg, phase='val', split='val', domain='source')]
    if debug:
        for set in val_sets:
            set.case_list = set.case_list[:8]
    val_sets.append(PVTDataset(cfg, phase='val', split='test', domain='target'))
    val_loaders = [DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False) for val_set in val_sets]

    # logging
    validation_log = np.zeros([num_epochs, 1 + len(val_sets) * (num_teachers + 1)])

    for ep in range(1, num_epochs + 1):
        print('Started epoch {}/{}'.format(ep, num_epochs))
        model.train()
        loss_values = []
        start_time = time.time()

        for it, data in enumerate(train_loader, 1):
            pcd_src, pcd_tgt, gt_flow, augm_info, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            gt_flow = gt_flow.to(device)

            with torch.cuda.amp.autocast(enabled=False):
                pred_flows, fps_pc1_idxs, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                loss = multiScaleLoss(pred_flows, gt_flow, fps_pc1_idxs)
                loss_values.append(loss.item())
                loss = loss * loss_factor
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            for alpha, teacher in zip(alphas, teachers):
                update_ema_variables(model, teacher, alpha)

        train_loss = np.mean(loss_values)
        validation_log[ep - 1, 0] = train_loss
        lr_scheduler.step()


        # Validation
        epes_3d = []
        epes_initial = []
        for model_id, model_ in enumerate([model] + teachers):
            model_.eval()
            for loader_id, loader in enumerate(val_loaders):
                epe_3d = 0
                epe_initial = 0
                for it, data in enumerate(loader, 1):
                    pcd_src, pcd_tgt, gt_flow, augm_info, idx = data
                    pcd_src = pcd_src.to(device)
                    pcd_tgt = pcd_tgt.to(device)

                    with torch.cuda.amp.autocast(enabled=False):
                        with torch.no_grad():
                            pred_flows, _, _, _, _ = model_(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                            pred_flow = pred_flows[0].permute(0,2,1)

                    if loader.dataset.split == 'test':
                        lm_src = gt_flow[0].to(device)
                        lm_tgt = gt_flow[1].to(device)
                        sq_dist = (lm_src.unsqueeze(2).cuda() - pcd_src.unsqueeze(1)).square().sum(dim=3)
                        weights = torch.exp(-0.5 * sq_dist / 0.05 ** 2)
                        flow_on_lms = (weights.unsqueeze(3) * pred_flow.unsqueeze(1)).sum(dim=2) / weights.unsqueeze(3).sum(dim=2)
                        epe_initial += (lm_src - lm_tgt).square().sum(dim=2).sqrt().mean(dim=1).sum().item()
                        err_per_sample = (lm_src + flow_on_lms - lm_tgt).square().sum(dim=2).sqrt().mean(dim=1)
                        epe_3d += err_per_sample.sum().item()
                    else:
                        gt_flow = gt_flow.to(device)
                        err_per_sample = (pred_flow - gt_flow).square().sum(dim=2).sqrt().mean(dim=1)
                        epe_3d += err_per_sample.sum().item()
                        epe_initial += gt_flow.square().sum(dim=2).sqrt().mean(dim=1).sum().item()

                epe_3d = epe_3d / len(loader.dataset) * loader.dataset.norm_factor
                epe_initial = epe_initial / len(loader.dataset) * loader.dataset.norm_factor
                epes_initial.append(epe_initial)
                epes_3d.append(epe_3d)
                validation_log[ep - 1, 1 + model_id*len(val_loaders) + loader_id] = epe_3d

        end_time = time.time()
        print('epoch', ep, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.6f' % train_loss,
              'initial error', epes_initial, 'EPEs', epes_3d)

        np.save(os.path.join(out_folder, "validation_history.npy"), validation_log)
        torch.save(model.state_dict(), model_path)
        if ep % cfg.SOLVER.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), model_path_ep.format(ep))
            for alpha, teacher in zip(alphas, teachers):
                torch.save(teacher.state_dict(), teacher_path_ep.format(alpha, ep))


def train_da(cfg, debug=False):
    root = cfg.BASE_DIRECTORY
    exp_name = cfg.EXPERIMENT_NAME
    out_folder = os.path.join(root, exp_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    model_path = os.path.join(out_folder, 'model.pth')
    teacher_path = os.path.join(out_folder, 'teacher.pth')
    model_path_ep = os.path.join(out_folder, 'model_ep={}.pth')
    teacher_path_ep = os.path.join(out_folder, 'teacher_ep={}.pth')


    # hyperparameters
    init_lr = cfg.SOLVER.LEARNING_RATE
    num_epochs = cfg.SOLVER.NUM_EPOCHS
    lr_steps = cfg.SOLVER.LR_MILESTONES
    lr_gamma = cfg.SOLVER.LR_LAMBDA
    batch_size = cfg.SOLVER.BATCH_SIZE

    # DA hyperparameters
    da_method = cfg.DA.METHOD
    loss_factor = cfg.SOLVER.LOSS_FACTOR

    # MT hyperparameters
    augmenter = MT_Augmenter(cfg)
    teacher_alpha = cfg.DA.MT_TEACHERALPHA
    pl_filter_method = cfg.DA.MT_PLFILTERMETHOD

    # computational stuff
    num_workers = 0 if debug else cfg.NUM_WORKERS
    if cfg.DEVICE == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    # model
    model = PointConvSceneFlowPWC8192(cfg).to(device)
    teacher = PointConvSceneFlowPWC8192(cfg).to(device)
    for param in teacher.parameters():
        param.requires_grad = False
    if cfg.MODEL.WEIGHTS != '':
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))
    if cfg.MODEL.TEACHER_WEIGHTS != '':
            teacher.load_state_dict(torch.load(cfg.MODEL.TEACHER_WEIGHTS))
    elif cfg.MODEL.WEIGHTS != '':
        teacher.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))

    # optimizer
    optimizer = optim.Adam(model.parameters(), init_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    if cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_steps, lr_gamma)
    else:
        raise ValueError()

    # datasets
    train_set_source = PVTDataset(cfg, phase='train', split='train', domain='source')
    train_set_target = PVTDataset(cfg, phase='train', split='train', domain='target')
    if debug:
        train_set_source.case_list = train_set_source.case_list[:8]
        train_set_target.case_list = train_set_target.case_list[:8]
    train_loader_source = DataLoader(train_set_source, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    train_loader_target = DataLoader(train_set_target, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    val_sets = [PVTDataset(cfg, phase='val', split='val', domain='source')]
    if debug:
        for set in val_sets:
            set.case_list = set.case_list[:10]
    val_sets.append(PVTDataset(cfg, phase='val', split='test', domain='target'))
    val_loaders = [DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False) for val_set in val_sets]

    # logging
    validation_log = np.zeros([num_epochs, 1 + 2 * len(val_sets) + 1])

    for ep in range(1, num_epochs + 1):
        print('Started epoch {}/{}'.format(ep, num_epochs))
        model.train()
        teacher.train()
        loss_values = []
        loss_values_da = []
        start_time = time.time()

        train_loader_target_iter = iter(train_loader_target)
        for it, data in enumerate(train_loader_source, 1):
            # forward and supervision on labeled source data
            pcd_src, pcd_tgt, gt_flow, augm_info, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            gt_flow = gt_flow.to(device)

            with torch.cuda.amp.autocast(enabled=False):
                pred_flows, fps_pc1_idxs, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                loss = multiScaleLoss(pred_flows, gt_flow, fps_pc1_idxs)
                loss_values.append(loss.item())
                loss = loss * loss_factor
                optimizer.zero_grad()
                scaler.scale(loss).backward()

            # unsupervised training on unlabeled target data
            try:
                data = next(train_loader_target_iter)
            except StopIteration:
                train_loader_target_iter = iter(train_loader_target)
                data = next(train_loader_target_iter)
            pcd_src, pcd_tgt, pcd_src_large, pcd_tgt_large, _, augm_info, idx = data
            pcd_src = pcd_src.to(device)
            pcd_tgt = pcd_tgt.to(device)
            pcd_src_large = pcd_src_large.to(device)
            pcd_tgt_large = pcd_tgt_large.to(device)

            with torch.cuda.amp.autocast(enabled=False):

                if da_method in ['mt_classic', 'mt_cls_gen_joint']:
                    pcd_src_1, pcd_tgt_1, rot_1, transl_1, scale_1 = augmenter(pcd_src, pcd_tgt)
                    pcd_src_2, pcd_tgt_2, rot_2, transl_2, scale_2 = augmenter(pcd_src, pcd_tgt)
                    pred_flows_stud, fps_pc1_idxs, _, _, _ = model(pcd_src_1, pcd_tgt_1, pcd_src_1, pcd_tgt_1)
                    pred_flows_stud = [torch.bmm(flow.permute(0,2,1) / scale_1, rot_1.transpose(1, 2)) for flow in pred_flows_stud]
                    with torch.no_grad():
                        pred_flows_teach, _, _, _, _ = teacher(pcd_src_2, pcd_tgt_2, pcd_src_2, pcd_tgt_2)
                        pred_flows_teach = [torch.bmm(flow.permute(0,2,1) / scale_2, rot_2.transpose(1, 2)) for flow in pred_flows_teach]
                    if pl_filter_method in ['chamfer_select', 'laplacian_select', 'smoothness_select', 'geomloss_select']:
                        with torch.no_grad():
                            if pl_filter_method == 'chamfer_select':
                                dist1, dist2 = computeChamfer((pcd_src + pred_flows_teach[0]).permute(0,2,1), pcd_tgt.permute(0,2,1))
                                chamfer_teach = dist1.mean(dim=1) + dist2.mean(dim=1)
                                dist1, dist2 = computeChamfer((pcd_src + pred_flows_stud[0]).permute(0, 2, 1), pcd_tgt.permute(0,2,1))
                                chamfer_stud = dist1.mean(dim=1) + dist2.mean(dim=1)
                            elif pl_filter_method == 'laplacian_select':
                                chamfer_stud = computeCurvature(pcd_src.permute(0,2,1), pcd_tgt.permute(0,2,1), pred_flows_stud[0].permute(0,2,1)).mean(dim=1)
                                chamfer_teach = computeCurvature(pcd_src.permute(0,2,1), pcd_tgt.permute(0,2,1), pred_flows_teach[0].permute(0,2,1)).mean(dim=1)
                            elif pl_filter_method == 'geomloss_select':
                                geomloss = SamplesLoss(cfg.DA.GEOMLOSS.METHOD, p=cfg.DA.GEOMLOSS.P, blur=cfg.DA.GEOMLOSS.BLUR)
                                chamfer_stud = geomloss(pcd_src + pred_flows_stud[0], pcd_tgt)
                                chamfer_teach = geomloss(pcd_src + pred_flows_teach[0], pcd_tgt)
                            else:
                                raise NotImplementedError()
                            pl_mask = (chamfer_teach < chamfer_stud).view(-1, 1, 1)
                            chamfer_fac = torch.nan_to_num(pl_mask.numel() / pl_mask.sum(), posinf=0.)
                            pred_flows_teach = [flow * pl_mask for flow in pred_flows_teach] # TODO: if used in combination with PL refinement, this needs to be after refine
                        pred_flows_stud = [flow * pl_mask for flow in pred_flows_stud]

                    high_flow_teach = pred_flows_teach[0]
                    pred_flows_stud = [flow.permute(0,2,1) for flow in pred_flows_stud]
                    da_loss = multiScaleLoss(pred_flows_stud, high_flow_teach, fps_pc1_idxs)
                    if pl_filter_method in ['chamfer_select', 'laplacian_select', 'smoothness_select', 'geomloss_select']:
                        da_loss = da_loss * chamfer_fac

                    loss_values_da.append(da_loss.item())
                    da_loss = da_loss * loss_factor
                    scaler.scale(da_loss).backward()

                if da_method in ['mt_generative', 'mt_cls_gen_joint']:
                    with torch.no_grad():
                        pcd_src_1, pcd_tgt_1, rot_1, transl_1, scale_1 = augmenter(pcd_src, pcd_tgt)
                        pred_flows_teach, _, _, _, _ = teacher(pcd_src_1, pcd_tgt_1, pcd_src_1, pcd_tgt_1)
                        pred_flow_teach = torch.bmm(pred_flows_teach[0].permute(0,2,1) / scale_1, rot_1.transpose(1, 2))
                        pcd_src, pcd_src_warp, pred_flow_teach = create_syn_pair_from_pred(cfg, pcd_src, pcd_tgt, pred_flow_teach, pcd_src_large, pcd_tgt_large)
                        pcd_src_augm, pcd_src_warp_augm, rot_2, transl_2, scale_2 = augmenter(pcd_src, pcd_src_warp)
                    pred_flows_stud, fps_pc1_idxs, _, _, _ = model(pcd_src_augm, pcd_src_warp_augm, pcd_src_augm, pcd_src_warp_augm)
                    pred_flows_stud = [torch.bmm(flow.permute(0,2,1) / scale_2, rot_2.transpose(1, 2)) for flow in pred_flows_stud]

                    pred_flows_stud = [flow.permute(0,2,1) for flow in pred_flows_stud]
                    da_loss = multiScaleLoss(pred_flows_stud, pred_flow_teach, fps_pc1_idxs)

                    loss_values_da.append(da_loss.item())
                    da_loss = da_loss * loss_factor
                    scaler.scale(da_loss).backward()

                scaler.step(optimizer)
                scaler.update()

                update_ema_variables(model, teacher, teacher_alpha)

        train_loss = np.mean(loss_values)
        da_loss = np.mean(loss_values_da)
        validation_log[ep - 1, 0] = train_loss
        validation_log[ep - 1, -1] = da_loss
        lr_scheduler.step()


        # Validation
        model.eval()
        teacher.eval()
        epes_3d = []
        epes_3d_teach = []
        for loader_id, loader in enumerate(val_loaders):
            epe_3d = 0
            epe_3d_teach = 0
            for it, data in enumerate(loader, 1):
                pcd_src, pcd_tgt, gt_flow, augm_info, idx = data
                pcd_src = pcd_src.to(device)
                pcd_tgt = pcd_tgt.to(device)

                with torch.cuda.amp.autocast(enabled=False):
                    with torch.no_grad():
                        pred_flows, _, _, _, _ = model(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                        pred_flow = pred_flows[0].permute(0,2,1)
                        pred_flows, _, _, _, _ = teacher(pcd_src, pcd_tgt, pcd_src, pcd_tgt)
                        pred_flow_teach = pred_flows[0].permute(0,2,1)

                if loader.dataset.split == 'test':
                    lm_src = gt_flow[0].to(device)
                    lm_tgt = gt_flow[1].to(device)
                    sq_dist = (lm_src.unsqueeze(2).cuda() - pcd_src.unsqueeze(1)).square().sum(dim=3)
                    weights = torch.exp(-0.5 * sq_dist / 0.05 ** 2)
                    flow_on_lms = (weights.unsqueeze(3) * pred_flow.unsqueeze(1)).sum(dim=2) / weights.unsqueeze(3).sum(dim=2)
                    err_per_sample = (lm_src + flow_on_lms - lm_tgt).square().sum(dim=2).sqrt().mean(dim=1)
                    epe_3d += err_per_sample.sum().item()
                    flow_on_lms_teach = (weights.unsqueeze(3) * pred_flow_teach.unsqueeze(1)).sum(dim=2) / weights.unsqueeze(3).sum(dim=2)
                    err_per_sample = (lm_src + flow_on_lms_teach - lm_tgt).square().sum(dim=2).sqrt().mean(dim=1)
                    epe_3d_teach += err_per_sample.sum().item()
                else:
                    gt_flow = gt_flow.to(device)
                    err_per_sample = (pred_flow - gt_flow).square().sum(dim=2).sqrt().mean(dim=1)
                    epe_3d += err_per_sample.sum().item()
                    err_per_sample = (pred_flow_teach - gt_flow).square().sum(dim=2).sqrt().mean(dim=1)
                    epe_3d_teach += err_per_sample.sum().item()


            epe_3d = epe_3d / len(loader.dataset) * loader.dataset.norm_factor
            epes_3d.append(epe_3d)
            epe_3d_teach = epe_3d_teach / len(loader.dataset) * loader.dataset.norm_factor
            epes_3d_teach.append(epe_3d_teach)
            validation_log[ep - 1, 1 + 2*loader_id : 3 + 2*loader_id] = [epe_3d_teach, epe_3d]

        end_time = time.time()
        print('epoch', ep, 'duration', '%0.3f' % ((end_time - start_time) / 60.), 'train_loss', '%0.6f' % train_loss,
               'da_loss', '%0.6f' % da_loss, 'teacher EPEs', epes_3d_teach, 'student EPEs', epes_3d)

        np.save(os.path.join(out_folder, "validation_history.npy"), validation_log)
        torch.save(model.state_dict(), model_path)
        torch.save(teacher.state_dict(), teacher_path)

        if ep % cfg.SOLVER.CHECKPOINT_INTERVAL == 0:
            torch.save(model.state_dict(), model_path_ep.format(ep))
            torch.save(teacher.state_dict(), teacher_path_ep.format(ep))


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
        "--debug",
        default=False,
        metavar="FILE",
        help="gpu to train on",
        type=bool,
    )
    parser.add_argument(
        "--train-state",
        default='',
        type=str,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.train_state == 'pretrain':
        print('START PRETRAINING')
        pretrain(cfg, debug=args.debug)
    elif args.train_state == 'adapt':
        print('START DOMAIN ADAPTATION')
        train_da(cfg, debug=args.debug)
    else:
        raise ValueError()