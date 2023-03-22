import os
import sys
import glob
import json
import joblib
import numpy as np
import smplx
import torch

from slahmr.util.loaders import load_smpl_body_model
from slahmr.util.tensor import move_to, detach_all, to_torch
from slahmr.optim.output import load_result, get_results_paths
from slahmr.geometry.pcl import align_pcl
from slahmr.geometry.rotation import batch_rodrigues

BASE_DIR = os.path.abspath(f"{__file__}/../../../")
JOINT_REG_PATH = f"{BASE_DIR}/_DATA/body_models/J_regressor_h36m.npy"


EGOBODY_ROOT = "/path/to/egobody"
TDPW_ROOT = "/path/to/3DPW"



class JointRegressor(object):
    def __init__(self):
        # (17, 6890)
        R17 = torch.from_numpy(np.load(JOINT_REG_PATH).astype(np.float32))
        # (15,)  adding the root, but will omit
        joint_map_h36m = torch.tensor(
            [0, 6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
        )
        self.regressor = R17[joint_map_h36m]  # (14, 6890)

    def to(self, device):
        self.regressor = self.regressor.to(device)

    def __call__(self, verts):
        """
        NOTE: RETURNS ROOT AS WELL
        :param verts (*, V, 3)
        returns (*, J, 3) 15 joints, the root + the 14 standard ones
        """
        return torch.einsum("nv,...vd->...nd", self.regressor, verts)  # (..., 15, 3)


class Evaluator(object):
    def __init__(self):
        self.metric_names = {
            "ga_jmse": 0,  # global PA joint MSE
            "fa_jmse": 1,  # first-frame PA joint MSE
            "la_jmse": 2,  # local PA joint MSE
            "ga_root": 3,  # global PA root error
            "fa_root": 4,  # first-frame PA root error
            "acc_norm_j": 5,  # mean acceleration norm of joints
        }
        self.num_metrics = len(self.metric_names)

    def compute_metrics(self, gt_all, pred_all, metric_names=None):
        """
        JOINTS WITH THE ROOT (T, 15, 3)
        """
        if metric_names is None:
            metric_names = self.metric_names.keys()
        metric_names = set(metric_names)
        metrics = torch.ones(self.num_metrics) * torch.inf
        if len(gt_all) < 1:
            return metrics

        # align all joints
        try:
            ga_all = global_align_joints(gt_all, pred_all)  # (T, J, 3)
            fa_all = first_align_joints(gt_all, pred_all)
            la_all = local_align_joints(gt_all, pred_all)
        except Exception as e:
            print("Exception", e)
            return metrics

        # separate joints and root
        pred_root, pred_joints = pred_all[:, :1, :], pred_all[:, 1:, :]
        ga_root, ga_joints = ga_all[:, :1, :], ga_all[:, 1:, :]
        fa_root, fa_joints = fa_all[:, :1, :], fa_all[:, 1:, :]
        la_root, la_joints = la_all[:, :1, :], la_all[:, 1:, :]
        gt_root, gt_joints = gt_all[:, :1, :], gt_all[:, 1:, :]

        if "ga_jmse" in metric_names:
            metrics[self.metric_names["ga_jmse"]] = torch.linalg.norm(
                gt_joints - ga_joints, dim=-1
            ).mean()

        if "fa_jmse" in metric_names:
            metrics[self.metric_names["fa_jmse"]] = torch.linalg.norm(
                gt_joints - fa_joints, dim=-1
            ).mean()

        if "la_jmse" in metric_names:
            metrics[self.metric_names["la_jmse"]] = torch.linalg.norm(
                gt_joints - la_joints, dim=-1
            ).mean()

        if "ga_root" in metric_names:
            metrics[self.metric_names["ga_root"]] = torch.linalg.norm(
                gt_root - ga_root, dim=-1
            ).mean()

        if "fa_root" in metric_names:
            metrics[self.metric_names["fa_root"]] = torch.linalg.norm(
                gt_root - fa_root, dim=-1
            ).mean()

        if "acc_norm_j" in metric_names:
            gt_anorm = compute_accel_norm(gt_joints)  # (T-2, J)
            pred_anorm = compute_accel_norm(pred_joints)
            metrics[self.metric_names["acc_norm_j"]] = torch.linalg.norm(
                gt_anorm - pred_anorm, dim=-1
            ).mean()

        return metrics


def compute_accel_norm(joints):
    """
    :param joints (T, J, 3)
    """
    vel = joints[1:] - joints[:-1]  # (T-1, J, 3)
    acc = vel[1:] - vel[:-1]  # (T-2, J, 3)
    return torch.linalg.norm(acc, dim=-1)


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(
        gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3)
    )
    pred_glob = (
        s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    )
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(
        gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3)
    )
    pred_first = (
        s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    )
    return pred_first


def local_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_loc, R_loc, t_loc = align_pcl(gt_joints, pred_joints)
    pred_loc = (
        s_loc[:, None] * torch.einsum("tij,tnj->tni", R_loc, pred_joints)
        + t_loc[:, None]
    )
    return pred_loc


def load_body_model(batch_size, model_type, gender, device):
    assert model_type in ["smpl", "smplh"]
    if model_type == "smpl":
        num_betas = 10
        ext = "pkl"
        use_vtx_selector = False
    else:
        num_betas = 16
        ext = "npz"
        use_vtx_selector = True

    smpl_path = f"{BASE_DIR}/body_models/{model_type}/{gender}/model.{ext}"
    body_model, fit_gender = load_smpl_body_model(
        smpl_path,
        batch_size,
        num_betas,
        model_type=model_type,
        use_vtx_selector=use_vtx_selector,
        device=device,
    )
    return body_model


def run_smpl(body_model, *args, **kwargs):
    with torch.no_grad():
        results = body_model(*args, **kwargs)
    return {
        "joints": results.Jtr.detach().cpu(),
        "vertices": results.v.detach().cpu(),
        "faces": results.f.detach().cpu(),
    }


def run_smpl_batch(body_model, device, **kwargs):
    model_kwargs = {}
    B = body_model.bm.batch_size
    kwarg_shape = (B,)
    for k, v in kwargs.items():
        kwarg_shape = v.shape[:-1]
        model_kwargs[k] = v.reshape(B, v.shape[-1]).to(device)
    res_flat = run_smpl(body_model, **model_kwargs)
    res = {}
    for k, v in res_flat.items():
        sh = v.shape
        if sh[0] == B:
            v = v.reshape(*kwarg_shape, *sh[1:])
        res[k] = v
    return res


def cat_dicts(dict_list, dim=0):
    """
    concatenate lists of dict of tensors
    """
    keys = set(dict_list[0].keys())
    assert all(keys == set(d.keys()) for d in dict_list)
    return {k: torch.stack([d[k] for d in dict_list], dim=dim) for k in keys}


def load_results_all(phase_dir, device):
    """
    Load all the reconstructed tracks during optimization
    """
    res_path_dict = get_results_paths(phase_dir)
    max_iter = max(res_path_dict.keys())
    if int(max_iter) < 20:
        print("max_iter", max_iter)
        return None

    res = load_result(res_path_dict[max_iter])["world"]
    # results is dict with (B, T, *) tensors
    trans = res["trans"]
    B, T, _ = trans.shape
    root_orient = res["root_orient"]
    pose_body = res["pose_body"]
    betas = res["betas"].reshape(B, 1, -1).expand(B, T, -1)
    body_model = load_body_model(B * T, "smplh", "neutral", device)
    return run_smpl_batch(
        body_model,
        device,
        trans=trans,
        root_orient=root_orient,
        betas=betas,
        pose_body=pose_body,
    )
