### from https://github.com/uncbiag/robot

import os
import json
import numpy as np
import pyvista as pv


"""
High resolution to low resoltuion dirlab mapping
1. flip the last dimension of the high resolution image
2. take the high resoltuion (max between the insp and exp )
3.  padding at the end
Landmark mapping
loc[z] = (high_image.shape[z] - low_index[z]*4 + 1.5)*high_spacing + high_origin ( 2 seems better in practice)
"""

COPD_ID={
    "copd1":  "copd_000001",
    "copd2":  "copd_000002",
    "copd3":  "copd_000003",
    "copd4":  "copd_000004",
    "copd5":  "copd_000005",
    "copd6":  "copd_000006",
    "copd7":  "copd_000007",
    "copd8":  "copd_000008",
    "copd9":  "copd_000009",
    "copd10": "copd_000010"
}

ID_COPD={
    "copd_000001":"copd1",
    "copd_000002":"copd2",
    "copd_000003":"copd3",
    "copd_000004":"copd4",
    "copd_000005":"copd5",
    "copd_000006":"copd6",
    "copd_000007":"copd7",
    "copd_000008":"copd8",
    "copd_000009":"copd9",
    "copd_000010":"copd10"
}

#in sitk coord
COPD_spacing = {"copd1": [0.625, 0.625, 2.5],
                "copd2": [0.645, 0.645, 2.5],
                "copd3": [0.652, 0.652, 2.5],
                "copd4": [0.590, 0.590, 2.5],
                "copd5": [0.647, 0.647, 2.5],
                "copd6": [0.633, 0.633, 2.5],
                "copd7": [0.625, 0.625, 2.5],
                "copd8": [0.586, 0.586, 2.5],
                "copd9": [0.664, 0.664, 2.5],
                "copd10": [0.742, 0.742, 2.5]}

# in sitk coord
COPD_low_shape = {"copd1": [512, 512, 121],
              "copd2": [512, 512, 102],
              "copd3": [512, 512, 126],
              "copd4": [512, 512, 126],
              "copd5": [512, 512, 131],
              "copd6": [512, 512, 119],
              "copd7": [512, 512, 112],
              "copd8": [512, 512, 115],
              "copd9": [512, 512, 116],
              "copd10":[512, 512, 135]}


# in sitk coord
COPD_high_insp_shape = {"copd1": [512, 512, 482],
              "copd2": [512, 512, 406],
              "copd3": [512, 512, 502],
              "copd4": [512, 512, 501],
              "copd5": [512, 512, 522],
              "copd6": [512, 512, 474],
              "copd7": [512, 512, 446],
              "copd8": [512, 512, 458],
              "copd9": [512, 512, 461],
              "copd10":[512, 512, 535]}


# in sitk coord
COPD_high_exp_shape = {"copd1": [512, 512, 473],
              "copd2": [512, 512, 378],
              "copd3": [512, 512, 464],
              "copd4": [512, 512, 461],
              "copd5": [512, 512, 522],
              "copd6": [512, 512, 461],
              "copd7": [512, 512, 407],
              "copd8": [512, 512, 426],
              "copd9": [512, 512, 380],
              "copd10":[512, 512, 539]}

COPD_info = {"copd1": {"insp":{'size': [512, 512, 482],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -310.625]},
                        "exp":{'size': [512, 512, 473],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -305.0]}},
              "copd2":  {"insp":{'size': [512, 512, 406],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-176.9, -165.0, -254.625]},
                        "exp":{'size': [512, 512, 378],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-177.0, -165.0, -237.125]}},
              "copd3":  {"insp":{'size': [512, 512, 502],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -343.125]},
                        "exp":{'size': [512, 512, 464],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -319.375]}},
              "copd4":  {"insp":{'size': [512, 512, 501],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -308.25]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -283.25]}},
              "copd5":  {"insp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]},
                        "exp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]}},
              "copd6":  {"insp":{'size': [512, 512, 474],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -299.625]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -291.5]}},
              "copd7":  {"insp":{'size': [512, 512, 446],'spacing': [0.625, 0.625, 0.625], 'origin': [-150.7, -160.0, -301.375]},
                        "exp":{'size': [512, 512, 407],'spacing': [0.625, 0.625, 0.625], 'origin': [-151.0, -160.0, -284.25]}},
              "copd8":  {"insp":{'size': [512, 512, 458],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -313.625]},
                        "exp":{'size': [512, 512, 426],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -294.625]}},
              "copd9":  {"insp":{'size': [512, 512, 461],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -310.25]},
                        "exp":{'size': [512, 512, 380],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -259.625]}},
              "copd10": {"insp":{'size': [512, 512, 535],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -355.0]},
                        "exp":{'size': [512, 512, 539],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -346.25]}}
              }

CENTER_BIAS = 2


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)



def read_landmark_index(f_path):
    """
    :param f_path: the path to the file containing the position of points.
    Points are deliminated by '\n' and X,Y,Z of each point are deliminated by '\t'.
    :return: numpy list of positions.
    """

    with open(f_path) as fp:
        content = fp.read().split('\n')

        # Read number of points from second
        count = len(content) - 1

        # Read the points
        points = np.ndarray([count, 3], dtype=np.float32)
        for i in range(count):
            if content[i] == "":
                break
            temp = content[i].split('\t')
            points[i, 0] = float(temp[0])
            points[i, 1] = float(temp[1])
            points[i, 2] = float(temp[2])
        return points


def transfer_landmarks_from_dirlab_to_high(dirlab_index, high_shape):
    new_index= dirlab_index.copy()
    new_index[:,-1] =(high_shape[-1]- dirlab_index[:,-1]*4) + CENTER_BIAS
    # new_index[:, -1] = dirlab_index[:, -1] * 4
    return new_index


def transfer_landmarks_from_dirlab_to_high_range(dirlab_index, high_shape):
    index_range_shape = list(dirlab_index.shape)+[2]
    new_index_range= np.zeros(index_range_shape)
    new_index_range[:,0,0] = dirlab_index[:,0] - 0.5
    new_index_range[:,0,1] = dirlab_index[:,0] + 0.5
    new_index_range[:,1,0] = dirlab_index[:,1] - 0.5
    new_index_range[:,1,1] = dirlab_index[:,1] + 0.5
    new_index_range[:,2,0] = (high_shape[-1] - dirlab_index[:,-1]*4) + CENTER_BIAS - 2
    new_index_range[:,2,1] = (high_shape[-1] - dirlab_index[:,-1]*4) + CENTER_BIAS + 2
    return new_index_range



def process_points(point_path, case_id, point_mapped_path, is_insp):
    index = read_landmark_index(point_path)
    copd = ID_COPD[case_id]
    phase = "insp" if is_insp else "exp"
    img_info = COPD_info[copd][phase]
    img_shape_itk, spacing_itk, origin_itk = np.array(img_info["size"]), np.array(img_info["spacing"]), np.array(img_info["origin"])
    downsampled_spacing_itk = np.copy(spacing_itk)
    downsampled_spacing_itk[-1] = downsampled_spacing_itk[-1]*4
    # downsampled_spacing_itk = COPD_spacing[ID_COPD[case_id]]
    print("spatial ratio corrections:")
    print("{} : {},".format(copd,np.array(COPD_spacing[copd])/downsampled_spacing_itk))
    transfered_index = transfer_landmarks_from_dirlab_to_high(index, img_shape_itk)
    transfered_index_range = transfer_landmarks_from_dirlab_to_high_range(index, img_shape_itk)
    physical_points = transfered_index*spacing_itk+origin_itk
    physical_points_range = transfered_index_range*(spacing_itk.reshape(1,3,1)) + origin_itk.reshape(1,3,1)
    # for i in range(len(physical_points)):
    #     physical_points[i][-1] = spacing_itk[-1] * img_shape_itk[-1] - index[i][-1]* COPD_spacing[ID_COPD[case_id]][-1] + origin_itk[-1]
    data = pv.PolyData(physical_points)
    data.point_arrays["idx"] = np.arange(1,301)
    data.save(point_mapped_path)
    np.save(point_mapped_path.replace(".vtk","_range.npy"), physical_points_range)
    name = os.path.split(point_mapped_path)[-1].split(".")[0]
    case_info = {"name":name,"raw_spacing":spacing_itk.tolist(), "raw_origin":origin_itk.tolist(),
                 "dirlab_spacing":downsampled_spacing_itk.tolist(), "raw_shape":img_shape_itk.tolist(), "z_bias":CENTER_BIAS, "dirlab_shape":COPD_low_shape[copd]}
    save_json(point_mapped_path.replace(".vtk","_info.json"), case_info)
    return physical_points


if __name__ == "__main__":
    copd_data_path = '../datasets/dirlab_copd/raw_data'
    processed_output_path = os.path.join('../datasets/dirlab_copd/processed_lms')
    os.makedirs(processed_output_path,exist_ok=True)
    id_list = list(ID_COPD.keys())

    landmark_dirlab_insp_path_list = [os.path.join(copd_data_path,ID_COPD[_id]+"_300_iBH_xyz_r1.txt") for _id in id_list]
    landmark_dirlab_exp_path_list = [os.path.join(copd_data_path,ID_COPD[_id]+"_300_eBH_xyz_r1.txt") for _id in id_list]
    landmark_physical_insp_path_list = [os.path.join(processed_output_path,_id+"_INSP.vtk") for _id in id_list]
    landmark_physical_exp_path_list = [os.path.join(processed_output_path,_id+"_EXP.vtk") for _id in id_list]

    landmark_insp_physical_pos_list = [process_points(landmark_dirlab_insp_path_list[i], id_list[i],landmark_physical_insp_path_list[i], is_insp=True) for i in range(len(id_list))]
    landmark_exp_physical_pos_list = [process_points(landmark_dirlab_exp_path_list[i], id_list[i],landmark_physical_exp_path_list[i], is_insp=False) for i in range(len(id_list))]
    for path in landmark_physical_insp_path_list:
        assert os.path.isfile(path), "the file {} is not exist".format(path)
    for path in landmark_physical_exp_path_list:
        assert os.path.isfile(path), "the file {} is not exist".format(path)
    print("landmarks have been projected into physical space")
