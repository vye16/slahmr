import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (Categorical, MixtureSameFamily,
                                 MultivariateNormal, Normal)

from slahmr.geometry import camera as cam_util
from slahmr.geometry.rotation import rotation_matrix_to_angle_axis
from slahmr.body_model import OP_NUM_JOINTS, SMPL_JOINTS, SMPL_PARENTS
from slahmr.util.logger import Logger

CONTACT_HEIGHT_THRESH = 0.08


class StageLoss(nn.Module):
    def __init__(self, loss_weights, **kwargs):
        super().__init__()
        self.cur_optim_step = 0
        self.set_loss_weights(loss_weights)
        self.setup_losses(loss_weights, **kwargs)

    def setup_losses(self, *args, **kwargs):
        raise NotImplementedError

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights
        Logger.log("Stage loss weights set to:")
        Logger.log(self.loss_weights)


class RootLoss(StageLoss):
    def setup_losses(
        self,
        loss_weights,
        ignore_op_joints=None,
        joints2d_sigma=100,
        use_chamfer=False,
        robust_loss="none",
        robust_tuning_const=4.6851,
    ):
        self.joints2d_loss = Joints2DLoss(ignore_op_joints, joints2d_sigma)
        self.points3d_loss = Points3DLoss(use_chamfer, robust_loss, robust_tuning_const)

    def forward(self, observed_data, pred_data, valid_mask=None):
        """
        For fitting just global root trans/orientation.
        Only computes joint/point/vert losses, i.e. no priors.
        """
        stats_dict = dict()
        loss = 0.0

        # Joints in 3D space
        if (
            "joints3d" in observed_data
            and "joints3d" in pred_data
            and self.loss_weights["joints3d"] > 0.0
        ):
            cur_loss = joints3d_loss(
                observed_data["joints3d"], pred_data["joints3d"], valid_mask
            )
            loss += self.loss_weights["joints3d"] * cur_loss
            stats_dict["joints3d"] = cur_loss

        # Select vertices in 3D space
        if (
            "verts3d" in observed_data
            and "verts3d" in pred_data
            and self.loss_weights["verts3d"] > 0.0
        ):
            cur_loss = verts3d_loss(
                observed_data["verts3d"], pred_data["verts3d"], valid_mask
            )
            loss += self.loss_weights["verts3d"] * cur_loss
            stats_dict["verts3d"] = cur_loss

        # All vertices to non-corresponding observed points in 3D space
        if (
            "points3d" in observed_data
            and "points3d" in pred_data
            and self.loss_weights["points3d"] > 0.0
        ):
            cur_loss = self.points3d_loss(
                observed_data["points3d"], pred_data["points3d"]
            )
            loss += self.loss_weights["points3d"] * cur_loss
            stats_dict["points3d"] = cur_loss

        # 2D re-projection loss
        if (
            "joints2d" in observed_data
            and "joints3d_op" in pred_data
            and "cameras" in pred_data
            and self.loss_weights["joints2d"] > 0.0
        ):
            joints2d = cam_util.reproject(
                pred_data["joints3d_op"], *pred_data["cameras"]
            )
            cur_loss = self.joints2d_loss(
                observed_data["joints2d"], joints2d, valid_mask
            )
            loss += self.loss_weights["joints2d"] * cur_loss
            stats_dict["joints2d"] = cur_loss

        # smooth 3d joint motion
        if self.loss_weights["joints3d_smooth"] > 0.0:
            cur_loss = joints3d_smooth_loss(pred_data["joints3d"], valid_mask)
            loss += self.loss_weights["joints3d_smooth"] * cur_loss
            stats_dict["joints3d_smooth"] = cur_loss

        # If we're optimizing cameras, camera reprojection loss
        if "bg2d_err" in pred_data and self.loss_weights["bg2d"] > 0.0:
            cur_loss = pred_data["bg2d_err"]
            loss += self.loss_weights["bg2d"] * cur_loss
            stats_dict["bg2d_err"] = cur_loss

        # camera smoothness
        if "cam_R" in pred_data and self.loss_weights["cam_R_smooth"] > 0.0:
            cam_R = pred_data["cam_R"]  # (T, 3, 3)
            cur_loss = rotation_smoothness_loss(cam_R[1:], cam_R[:-1])
            loss += self.loss_weights["cam_R_smooth"] * cur_loss
            stats_dict["cam_R_smooth"] = cur_loss

        if "cam_t" in pred_data and self.loss_weights["cam_t_smooth"] > 0.0:
            cam_t = pred_data["cam_t"]  # (T, 3, 3)
            cur_loss = translation_smoothness_loss(cam_t[1:], cam_t[:-1])
            loss += self.loss_weights["cam_t_smooth"] * cur_loss
            stats_dict["cam_t_smooth"] = cur_loss

        return loss, stats_dict


def rotation_smoothness_loss(R1, R2):
    R12 = torch.einsum("...ij,...jk->...ik", R2, R1.transpose(-1, -2))
    aa12 = rotation_matrix_to_angle_axis(R12)
    return torch.sum(aa12**2)


def translation_smoothness_loss(t1, t2):
    return torch.sum((t2 - t1) ** 2)


def camera_smoothness_loss(R1, t1, R2, t2):
    """
    :param R1, t1 (N, 3, 3), (N, 3)
    :param R2, t2 (N, 3, 3), (N, 3)
    """
    R12, t12 = cam_util.compose_cameras(R2, t2, *cam_util.invert_camera(R1, t1))
    aa12 = rotation_matrix_to_angle_axis(R12)
    return torch.sum(aa12**2) + torch.sum(t12**2)


"""
Losses are cumulative
SMPLLoss setup is same as RootLoss
"""


class SMPLLoss(RootLoss):
    def forward(self, observed_data, pred_data, nsteps, valid_mask=None):
        """
        For fitting full shape and pose of SMPL.
        nsteps used to scale single-step losses
        """
        loss, stats_dict = super().forward(
            observed_data, pred_data, valid_mask=valid_mask
        )

        # prior to keep latent pose likely
        if "latent_pose" in pred_data and self.loss_weights["pose_prior"] > 0.0:
            cur_loss = pose_prior_loss(pred_data["latent_pose"], valid_mask)
            loss += self.loss_weights["pose_prior"] * cur_loss
            stats_dict["pose_prior"] = cur_loss

        # prior to keep PCA shape likely
        if "betas" in pred_data and self.loss_weights["shape_prior"] > 0.0:
            cur_loss = shape_prior_loss(pred_data["betas"])
            loss += self.loss_weights["shape_prior"] * nsteps * cur_loss
            stats_dict["shape_prior"] = cur_loss

        return loss, stats_dict


"""
MotionLoss also includes SMPLLoss
"""


class MotionLoss(SMPLLoss):
    def setup_losses(
        self,
        loss_weights,
        init_motion_prior=None,
        **kwargs,
    ):
        super().setup_losses(loss_weights, **kwargs)
        if loss_weights["init_motion_prior"] > 0.0:
            self.init_motion_prior_loss = GMMPriorLoss(init_motion_prior)

    def forward(
        self,
        observed_data,
        pred_data,
        cam_pred_data,
        nsteps,
        valid_mask=None,
        init_motion_scale=1.0,
    ):
        """
        For fitting full shape and pose of SMPL with motion prior.

        pred_data is data pertinent to the canonical prior coordinate frame
        cam_pred_data is for the camera coordinate frame

        loss rather than standard normal if given.
        """
        cam_pred_data["latent_pose"] = pred_data["latent_pose"]
        loss, stats_dict = super().forward(
            observed_data, cam_pred_data, nsteps, valid_mask=valid_mask
        )

        # prior to keep latent motion likely
        if "latent_motion" in pred_data and self.loss_weights["motion_prior"] > 0.0:
            # NOTE: latent is NOT synchronized in time,
            # the mask is NOT relevant
            # Generate the async mask to properly mask motion prior loss
            # Helps to calibrate the range of 'good' loss values
            B, T, _ = pred_data["latent_motion"].shape
            device = pred_data["latent_motion"].device
            async_mask = (
                torch.arange(T, device=device)[None].expand(B, -1)
                < valid_mask.sum(dim=1)[:, None]
            )
            cur_loss = motion_prior_loss(
                pred_data["latent_motion"],
                cond_prior=pred_data.get("cond_prior", None),
                mask=async_mask,
            )
            loss += self.loss_weights["motion_prior"] * cur_loss
            stats_dict["motion_prior"] = cur_loss

        # prior to keep initial state likely
        have_init_prior_info = (
            "joints3d_init" in pred_data
            and "joints_vel" in pred_data
            and "trans_vel" in pred_data
            and "root_orient_vel" in pred_data
        )
        if have_init_prior_info and self.loss_weights["init_motion_prior"] > 0.0:
            cur_loss = self.init_motion_prior_loss(
                pred_data["joints3d_init"],
                pred_data["joints_vel"],
                pred_data["trans_vel"],
                pred_data["root_orient_vel"],
            )
            loss += (
                self.loss_weights["init_motion_prior"] * init_motion_scale * cur_loss
            )  # must scale since doesn't scale with more steps
            stats_dict["init_motion_prior"] = cur_loss

        # make sure joints consistent between SMPL and direct motion prior output
        if (
            "joints3d_rollout" in pred_data
            and "joints3d" in pred_data
            and self.loss_weights["joint_consistency"] > 0.0
        ):
            cur_loss = joint_consistency_loss(
                pred_data["joints3d"],
                pred_data["joints3d_rollout"],
                valid_mask,
            )
            loss += self.loss_weights["joint_consistency"] * cur_loss
            stats_dict["joint_consistency"] = cur_loss

        # make sure bone lengths between frames of direct motion prior output are consistent
        if "joints3d_rollout" in pred_data and self.loss_weights["bone_length"] > 0.0:
            cur_loss = bone_length_loss(pred_data["joints3d_rollout"], valid_mask)
            loss += self.loss_weights["bone_length"] * cur_loss
            stats_dict["bone_length"] = cur_loss

        # make sure rolled out joints match observations too
        if (
            "joints3d" in observed_data
            and "joints3d_rollout" in pred_data
            and self.loss_weights["joints3d_rollout"] > 0.0
        ):
            cur_loss = joints3d_loss(
                observed_data["joints3d"], pred_data["joints3d_rollout"], valid_mask
            )
            loss += self.loss_weights["joints3d_rollout"] * cur_loss
            stats_dict["joints3d_rollout"] = cur_loss

        # velocity 0 during contacts
        if (
            self.loss_weights["contact_vel"] > 0.0
            and "contacts_conf" in pred_data
            and "joints3d" in pred_data
        ):
            cur_loss = contact_vel_loss(
                pred_data["contacts_conf"], pred_data["joints3d"], valid_mask
            )
            loss += self.loss_weights["contact_vel"] * cur_loss
            stats_dict["contact_vel"] = cur_loss

        # contacting joints are near the floor
        if (
            self.loss_weights["contact_height"] > 0.0
            and "contacts_conf" in pred_data
            and "joints3d" in pred_data
        ):
            cur_loss = contact_height_loss(
                pred_data["contacts_conf"], pred_data["joints3d"], valid_mask
            )
            loss += self.loss_weights["contact_height"] * cur_loss
            stats_dict["contact_height"] = cur_loss

        # floor is close to the initialization
        if (
            self.loss_weights["floor_reg"] > 0.0
            and "floor_plane" in pred_data
            and "floor_plane" in observed_data
        ):
            cur_loss = floor_reg_loss(
                pred_data["floor_plane"], observed_data["floor_plane"]
            )
            loss += self.loss_weights["floor_reg"] * nsteps * cur_loss
            stats_dict["floor_reg"] = cur_loss

        return loss, stats_dict


def joints3d_loss(joints3d_obs, joints3d_pred, mask=None):
    """
    :param joints3d_obs (B, T, J, 3)
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    B, T, *dims = joints3d_obs.shape
    vis_mask = get_visible_mask(joints3d_obs)
    if mask is not None:
        vis_mask = vis_mask & mask.reshape(B, T, *(1,) * len(dims)).bool()
    loss = (joints3d_obs[vis_mask] - joints3d_pred[vis_mask]) ** 2
    loss = 0.5 * torch.sum(loss)
    return loss


def verts3d_loss(verts3d_obs, verts3d_pred, mask=None):
    """
    :param verts3d_obs (B, T, V, 3)
    :param verts3d_pred (B, T, V, 3)
    :param mask (optional) (B, T)
    """
    B, T, *dims = verts3d_obs.shape
    vis_mask = get_visible_mask(verts3d_obs)
    if mask is not None:
        assert mask.shape == (B, T)
        vis_mask = vis_mask & mask.reshape(B, T, *(1,) * len(dims)).bool()
    loss = (verts3d_obs[vis_mask] - verts3d_pred[vis_mask]) ** 2
    loss = 0.5 * torch.sum(loss)
    return loss


def get_visible_mask(obs_data):
    """
    Given observed data gets the mask of visible data (that actually contributes to the loss).
    """
    return torch.logical_not(torch.isinf(obs_data))


class Joints2DLoss(nn.Module):
    def __init__(self, ignore_op_joints=None, joints2d_sigma=100):
        super().__init__()
        self.ignore_op_joints = ignore_op_joints
        self.joints2d_sigma = joints2d_sigma

    def forward(self, joints2d_obs, joints2d_pred, mask=None):
        """
        :param joints2d_obs (B, T, 25, 3)
        :param joints2d_pred (B, T, 22, 2)
        :param mask (optional) (B, T)
        """
        if mask is not None:
            mask = mask.bool()
            joints2d_obs = joints2d_obs[mask]  # (N, 25, 3)
            joints2d_pred = joints2d_pred[mask]  # (N, 22, 2)

        joints2d_obs_conf = joints2d_obs[..., 2:3]
        if self.ignore_op_joints is not None:
            # set confidence to 0 so not weighted
            joints2d_obs_conf[..., self.ignore_op_joints, :] = 0.0

        # weight errors by detection confidence
        robust_sqr_dist = gmof(
            joints2d_pred - joints2d_obs[..., :2], self.joints2d_sigma
        )
        reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist
        loss = torch.sum(reproj_err)
        return loss


class Points3DLoss(nn.Module):
    def __init__(
        self,
        use_chamfer=False,
        robust_loss="bisquare",
        robust_tuning_const=4.6851,
    ):
        super().__init__()

        if not use_chamfer:
            self.active = False
            return

        self.active = True

        robust_choices = ["none", "bisquare", "gm"]
        if robust_loss not in robust_choices:
            Logger.log(
                "Not a valid robust loss: %s. Please use %s"
                % (robust_loss, str(robust_choices))
            )
            exit()

        from utils.chamfer_distance import ChamferDistance

        self.chamfer_dist = ChamferDistance()

        self.robust_loss = robust_loss
        self.robust_tuning_const = robust_tuning_const

    def forward(self, points3d_obs, points3d_pred):
        if not self.active:
            return torch.tensor(0.0, dtype=torch.float32, device=points3d_obs.device)

        # one-way chamfer
        B, T, N_obs, _ = points3d_obs.size()
        N_pred = points3d_pred.size(2)
        points3d_obs = points3d_obs.reshape((B * T, -1, 3))
        points3d_pred = points3d_pred.reshape((B * T, -1, 3))

        obs2pred_sqr_dist, pred2obs_sqr_dist = self.chamfer_dist(
            points3d_obs, points3d_pred
        )
        obs2pred_sqr_dist = obs2pred_sqr_dist.reshape((B, T * N_obs))
        pred2obs_sqr_dist = pred2obs_sqr_dist.reshape((B, T * N_pred))

        weighted_obs2pred_sqr_dist, w = apply_robust_weighting(
            obs2pred_sqr_dist.sqrt(),
            robust_loss_type=self.robust_loss,
            robust_tuning_const=self.robust_tuning_const,
        )

        loss = torch.sum(weighted_obs2pred_sqr_dist)
        loss = 0.5 * loss
        return loss


def pose_prior_loss(latent_pose_pred, mask=None):
    """
    :param latent_pose_pred (B, T, D)
    :param mask (optional) (B, T)
    """
    # prior is isotropic gaussian so take L2 distance from 0
    loss = latent_pose_pred**2
    if mask is not None:
        loss = loss[mask.bool()]
    loss = torch.sum(loss)
    return loss


def shape_prior_loss(betas_pred):
    # prior is isotropic gaussian so take L2 distance from 0
    loss = betas_pred**2
    loss = torch.sum(loss)
    return loss


def joints3d_smooth_loss(joints3d_pred, mask=None):
    """
    :param joints3d_pred (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    # minimize delta steps
    B, T, *dims = joints3d_pred.shape
    loss = (joints3d_pred[:, 1:, :, :] - joints3d_pred[:, :-1, :, :]) ** 2
    if mask is not None:
        mask = mask.bool()
        mask = mask[:, 1:] & mask[:, :-1]
        loss = loss[mask]
    loss = 0.5 * torch.sum(loss)
    return loss


def motion_prior_loss(latent_motion_pred, cond_prior=None, mask=None):
    """
    :param latent_motion_pred (B, T, D)
    :param cond_prior (optional) (B, T, D, 2) stacked mean and var of post distribution
    :param mask (optional) (B, T)
    """
    if mask is not None:
        latent_motion_pred = latent_motion_pred[mask.bool()]

    if cond_prior is None:
        # assume standard normal
        loss = latent_motion_pred**2
    else:
        pm, pv = cond_prior[..., 0], cond_prior[..., 1]
        if mask is not None:
            mask = mask.bool()
            pm, pv = pm[mask], pv[mask]
        loss = -log_normal(latent_motion_pred, pm, pv)

    return torch.sum(loss)


class GMMPriorLoss(nn.Module):
    def __init__(self, init_motion_prior=None):
        super().__init__()
        if init_motion_prior is None:
            self.active = False
            return

        # build pytorch GMM
        self.active = True
        self.init_motion_prior = dict()
        gmm = build_gmm(*init_motion_prior["gmm"])
        self.init_motion_prior["gmm"] = gmm

    def forward(self, joints, joints_vel, trans_vel, root_orient_vel):
        if not self.active:
            return torch.tensor(0.0, dtype=torch.float32, device=joints.device)

        # create input
        B = joints.size(0)

        joints = joints.reshape((B, -1))
        joints_vel = joints_vel.reshape((B, -1))
        trans_vel = trans_vel.reshape((B, -1))
        root_orient_vel = root_orient_vel.reshape((B, -1))
        init_state = torch.cat([joints, joints_vel, trans_vel, root_orient_vel], dim=-1)

        loss = -self.init_motion_prior["gmm"].log_prob(init_state)
        loss = torch.sum(loss)

        return loss


def build_gmm(gmm_weights, gmm_means, gmm_covs):
    mix = Categorical(gmm_weights)
    comp = MultivariateNormal(gmm_means, covariance_matrix=gmm_covs)
    gmm_distrib = MixtureSameFamily(mix, comp)
    return gmm_distrib


def joint_consistency_loss(smpl_joints3d, rollout_joints3d, mask=None):
    """
    :param smpl_joints3d (B, T, J, 3)
    :param rollout_joints3d (B, T, J, 3)
    :param mask (optional) (B, T)
    """
    loss = (smpl_joints3d - rollout_joints3d) ** 2
    if mask is not None:
        loss = loss[mask.bool()]
    loss = 0.5 * torch.sum(loss)
    return loss


def bone_length_loss(rollout_joints3d, mask=None):
    bones = rollout_joints3d[:, :, 1:]
    parents = rollout_joints3d[:, :, SMPL_PARENTS[1:]]
    bone_lengths = torch.norm(bones - parents, dim=-1)
    loss = bone_lengths[:, 1:] - bone_lengths[:, :-1]
    if mask is not None:
        mask = mask.bool()
        mask = mask[:, 1:] & mask[:, :-1]
        loss = loss[mask]
    loss = 0.5 * torch.sum(loss**2)
    return loss


def contact_vel_loss(contacts_conf, joints3d, mask=None):
    """
    Velocity should be zero at predicted contacts
    """
    delta_pos = (joints3d[:, 1:] - joints3d[:, :-1]) ** 2
    loss = delta_pos.sum(dim=-1) * contacts_conf[:, 1:]
    if mask is not None:
        mask = mask.bool()
        mask = mask[:, 1:] & mask[:, :-1]
        loss = loss[mask]
    loss = 0.5 * torch.sum(loss)

    return loss


def contact_height_loss(contacts_conf, joints3d, mask=None):
    """
    Contacting joints should be near floor
    """
    # won't be exactly on the floor, just near it (since joints are inside the body)
    floor_diff = F.relu(torch.torch.abs(joints3d[:, :, :, 2]) - CONTACT_HEIGHT_THRESH)
    loss = floor_diff * contacts_conf
    if mask is not None:
        loss = loss[mask.bool()]

    loss = torch.sum(loss)
    return loss


def floor_reg_loss(pred_floor_plane, obs_floor_plane):
    """
    Pred floor plane shouldn't deviate from the initial observation
    :param pred_floor_plane (B, 3)
    :param obs_floor_plane (B, 3)
    """
    floor_loss = (pred_floor_plane - obs_floor_plane) ** 2
    floor_loss = 0.5 * torch.sum(floor_loss)

    return floor_loss


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    ​
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    ​
    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    log_prob = (
        -torch.log(torch.sqrt(v))
        - math.log(math.sqrt(2 * math.pi))
        - ((x - m) ** 2 / (2 * v))
    )
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob


def apply_robust_weighting(
    res, robust_loss_type="bisquare", robust_tuning_const=4.6851
):
    """
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    """
    robust_choices = ["none", "bisquare"]
    if robust_loss_type not in robust_choices:
        print(
            "Not a valid robust loss: %s. Please use %s"
            % (robust_loss_type, str(robust_choices))
        )

    w = None
    detach_res = (
        res.clone().detach()
    )  # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == "none":
        w = torch.ones_like(detach_res)
    elif robust_loss_type == "bisquare":
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w


def robust_std(res):
    """
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    """
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B, 1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std


def bisquare_robust_weights(res, tune_const=4.6851):
    """
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    """
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    # NOTE: this should use absolute value, it's ok right now since only used for 3d point cloud residuals
    #   which are guaranteed positive, but generally this won't work)
    outlier_mask = norm_res >= 1.0

    # print(torch.sum(outlier_mask))
    # print('Outlier frac: %f' % (float(torch.sum(outlier_mask)) / res.size(1)))

    w = (1.0 - norm_res**2) ** 2
    w[outlier_mask] = 0.0

    return w


def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)
