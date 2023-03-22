import os
import itertools
import glob
import pickle
import json
import pandas as pd

import numpy as np
import torch

from .tools import load_body_model, move_to, detach_all, EGOBODY_ROOT


def get_sequence_body_info(seq_name):
    info_file = f"{EGOBODY_ROOT}/data_info_release.csv"
    info_df = pd.read_csv(info_file)
    seq_info = info_df[info_df["recording_name"] == seq_name]
    return seq_info["body_idx_fpv"].values[0]


def get_egobody_split(split):
    split_file = f"{EGOBODY_ROOT}/data_splits.csv"
    split_df = pd.read_csv(split_file)
    if split not in split_df.columns:
        print(f"{split} not in {split_file}")
        return []
    return split_df[split].dropna().tolist()


def get_egobody_seq_paths(seq_name, start=0, end=-1):
    img_dir = get_egobody_img_dir(seq_name)
    # img files are named [timestamp]_frame_[index].jpg
    img_files = sorted(os.listdir(img_dir))
    end = len(img_files) if end < 0 else end
    print(f"FOUND {len(img_files)} FILES FOR SEQ {seq_name}")
    return img_files[start:end]


def get_egobody_seq_names(seq_name, start=0, end=-1):
    img_files = get_egobody_seq_paths(seq_name, start=start, end=end)
    frame_names = ["_".join(x.split(".")[0].split("_")[1:]) for x in img_files]
    return frame_names


def get_egobody_img_dir(seq_name):
    img_dir = f"{EGOBODY_ROOT}/egocentric_color/{seq_name}/**/PV"
    matches = glob.glob(img_dir)
    if len(matches) != 1:
        raise ValueError(f"{img_dir} has {len(matches)} matches!")
    return matches[0]


def get_egobody_keypoints(seq_name, start=0, end=-1):
    img_dir = os.path.dirname(get_egobody_img_dir(seq_name))
    kp_file = f"{img_dir}/keypoints.npz"
    valid_file = f"{img_dir}/valid_frame.npz"

    # missing keypoints aren't included, must fill in
    kp_dict = {}
    valid_dict = {}
    kp_data = np.load(kp_file)
    valid_data = np.load(valid_file)

    zeros = np.zeros_like(kp_data["keypoints"][0])
    for img_path, kps in zip(kp_data["imgname"], kp_data["keypoints"]):
        img_name = os.path.basename(img_path)
        kp_dict[img_name] = kps

    for img_path, valid in zip(valid_data["imgname"], valid_data["valid"]):
        img_name = os.path.basename(img_path)
        valid_dict[img_name] = valid

    img_paths = sorted(glob.glob(f"{img_dir}/PV/*.jpg"))
    end = len(img_paths) + 1 + end if end < 0 else end
    img_names = [os.path.basename(x) for x in img_paths[start:end]]
    kps = np.stack([kp_dict.get(name, zeros) for name in img_names], axis=0)
    valid = np.stack([valid_dict.get(name, False) for name in img_names], axis=0)
    return kps, valid


def load_egobody_smpl_params(seq_name, start=0, end=-1):
    frame_names = get_egobody_seq_names(seq_name, start=start, end=end)
    body_name = get_sequence_body_info(seq_name)
    body_idx, gender = body_name.split(" ")
    smpl_dir = (
        f"{EGOBODY_ROOT}/smpl_interactee_val/{seq_name}/body_idx_{body_idx}/results"
    )
    if not os.path.isdir(smpl_dir):
        raise ValueError(f"EXPECTED BODY DIR {smpl_dir} DOES NOT EXIST")

    print(f"LOADING {len(frame_names)} SMPL PARAMS FROM {smpl_dir}")
    smpl_dict = {"trans": [], "root_orient": [], "pose_body": [], "betas": []}
    for frame in frame_names:
        with open(f"{smpl_dir}/{frame}/000.pkl", "rb") as f:
            # data has global_orient, body_pose, betas, transl
            data = pickle.load(f)
            smpl_dict["trans"].append(torch.from_numpy(data["transl"]))
            smpl_dict["pose_body"].append(torch.from_numpy(data["body_pose"]))
            smpl_dict["root_orient"].append(torch.from_numpy(data["global_orient"]))
            smpl_dict["betas"].append(torch.from_numpy(data["betas"]))
    smpl_dict = {k: torch.cat(v, dim=0)[None] for k, v in smpl_dict.items()}
    smpl_dict["genders"] = [gender]
    return smpl_dict


def load_egobody_intrinsics(seq_name, start=0, end=-1, ret_size_tuple=True):
    path = f"{EGOBODY_ROOT}/slahmr/cameras_gt/{seq_name}/intrinsics.txt"
    assert os.path.isfile(path)
    intrins = np.loadtxt(path)  # (T, 6)
    end = len(intrins) if end < 0 else end
    intrins = intrins[start:end]
    if ret_size_tuple:
        img_size = intrins[0, 4:].astype(int).tolist()  # (2)
        intrins = torch.from_numpy(intrins[:, :4].astype(np.float32))
        return intrins, img_size
    img_size = torch.from_numpy(intrins[:, 4:].astype(int))
    intrins = torch.from_numpy(intrins[:, :4].astype(np.float32))
    return intrins, img_size


def load_egobody_gt_extrinsics(seq_name, start=0, end=-1, ret_4d=True):
    path = f"{EGOBODY_ROOT}/slahmr/cameras_gt/{seq_name}/cam2world.txt"
    assert os.path.isfile(path)
    cam2world = np.loadtxt(path).astype(np.float32)  # (T, 16)
    end = len(cam2world) if end < 0 else end
    cam2world = torch.from_numpy(cam2world[start:end].reshape(-1, 4, 4))
    if ret_4d:
        return cam2world
    return cam2world[:, :3, :3], cam2world[:, :3, 3]


def load_egobody_extrinsics(seq_name, use_intrins=True, start=0, end=-1):
    camera_name = "cameras_intrins" if use_intrins else "cameras_default"
    path = f"{EGOBODY_ROOT}/slahmr/{camera_name}/{seq_name}/cameras.npz"
    assert os.path.isfile(path)
    data = np.load(path)
    w2c = torch.from_numpy(data["w2c"].astype(np.float32))  # (N, 4, 4)
    end = len(w2c) if end < 0 else end
    w2c = w2c[start:end]
    c2w = torch.linalg.inv(w2c)
    return c2w[:, :3, :3], c2w[:, :3, 3]


def load_egobody_meshes(seq_name, device, start=0, end=-1):
    params = load_egobody_smpl_params(seq_name, start=start, end=end)
    _, T = params["trans"].shape[:2]

    with torch.no_grad():
        gender = params["genders"][0]
        body_model = load_body_model(T, "smpl", gender, device)
        smpl_res = body_model(
            trans=params["trans"][0].to(device),
            root_orient=params["root_orient"][0].to(device),
            betas=params["betas"][0].to(device),
            pose_body=params["pose_body"][0].to(device),
        )

    res = {"joints": smpl_res.Jtr, "vertices": smpl_res.v, "faces": smpl_res.f}
    return move_to(detach_all(res), "cpu")


def load_egobody_kinect2holo(seq_name, ret_4d=True):
    # load the transform from kinect12 to holo
    # bodies are recorded in the kinect12 frame
    path = f"{EGOBODY_ROOT}/calibrations/{seq_name}/cal_trans/holo_to_kinect12.json"
    with open(path, "r") as f:
        kinect2holo = np.linalg.inv(np.array(json.load(f)["trans"]))
    kinect2holo = torch.from_numpy(kinect2holo.astype(np.float32))
    if ret_4d:
        return kinect2holo
    return kinect2holo[:3, :3], kinect2holo[:3, 3]
