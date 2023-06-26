import torch
import pyvista as pv
import numpy as np
import os


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


def foerstner_nms(pcd, sigma, neigh_1, neigh_2, min_points, sigma_interval):
    pcd = pcd.cuda()
    knn = torch.zeros(len(pcd), neigh_1).long().cuda()
    knn_dist = torch.zeros(len(pcd), neigh_1).float().cuda()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            chk = torch.chunk(torch.arange(len(pcd)).cuda(), 192)
            for i in range(len(chk)):
                dist = (pcd[chk[i]].unsqueeze(1) - pcd.unsqueeze(0)).pow(2).sum(-1).sqrt()
                q = torch.topk(dist, neigh_1, dim=1, largest=False)
                knn[chk[i]] = q[1][:, :]
                knn_dist[chk[i]] = q[0][:, :]

    curr_points = 0
    curr_sigma = sigma
    while curr_points < min_points:
        exp_score = torch.exp(-knn_dist[:, :].pow(2) * curr_sigma ** 2).mean(1)
        knn_score = torch.max(exp_score[knn[:, :neigh_2]], 1)[0]
        valid_idx = (knn_score == exp_score).nonzero(as_tuple=True)[0]
        curr_points = valid_idx.shape[0]
        curr_sigma += sigma_interval
    return valid_idx


def process_dataset(min_points, sigma, sigma_interval, neigh1, neigh2):
    tgt_root = '../datasets/pvt/'
    folder = 'kpts_sigma={}_n1={}_n2={}_sigInt={}_minPts={}'.format(sigma, neigh1, neigh2, sigma_interval, min_points)
    save_path_exp = os.path.join(tgt_root, folder, 'copd_{:06d}_idx_EXP.pth')
    save_path_insp = os.path.join(tgt_root, folder, 'copd_{:06d}_idx_INSP.pth')

    if not os.path.exists(os.path.join(tgt_root, folder)):
        os.makedirs(os.path.join(tgt_root, folder))

    case_list = np.arange(1, 1011)
    case_list = case_list[~np.isin(case_list, [139, 428, 611, 1002])]

    for case in case_list:
        print('Case {:04d}'.format(case))
        pcd_exp = read_vtk('../datasets/pvt/raw_data/copd_{:06d}_EXP.vtk'.format(case))
        pcd_exp = torch.tensor(pcd_exp['points'])
        pcd_exp_idx = foerstner_nms(pcd_exp, sigma, neigh1, neigh2, min_points, sigma_interval)
        torch.save(pcd_exp_idx, save_path_exp.format(case))
        print(pcd_exp_idx.shape)

        pcd_insp = read_vtk('../datasets/pvt/raw_data/copd_{:06d}_INSP.vtk'.format(case))
        pcd_insp = torch.tensor(pcd_insp['points'])
        pcd_insp_idx = foerstner_nms(pcd_insp, sigma, neigh1, neigh2, min_points, sigma_interval)
        torch.save(pcd_insp_idx, save_path_insp.format(case))
        print(pcd_insp_idx.shape)


if __name__ == "__main__":
    process_dataset(min_points=9000, sigma=0.7, sigma_interval=0.1, neigh1=29, neigh2=15)
    process_dataset(min_points=16384, sigma=6.0, sigma_interval=0.1, neigh1=29, neigh2=10)
