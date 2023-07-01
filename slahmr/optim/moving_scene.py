import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slahmr.geometry.rotation import batch_rodrigues, rotation_matrix_to_angle_axis
from slahmr.body_model import KEYPT_VERTS, SMPL_JOINTS, smpl_to_openpose
from slahmr.util.logger import Logger
from slahmr.util.tensor import (detach_all, get_scatter_mask, scatter_intervals,
                         select_intervals)

from .base_scene import BaseSceneModel
from .helpers import compute_world2prior, estimate_floor_planes

J_BODY = len(SMPL_JOINTS) - 1  # no root
CONTACT_ORDERING = [
    "hips",
    "leftLeg",
    "rightLeg",
    "leftFoot",
    "rightFoot",
    "leftToeBase",
    "rightToeBase",
    "leftHand",
    "rightHand",
]
CONTACT_INDS = [SMPL_JOINTS[jname] for jname in CONTACT_ORDERING]
CONTACT_THRESH = 0.5


class MovingSceneModel(BaseSceneModel):
    """
    Scene model of moving people in a shared global reference frame

    Parameters:
        batch_size: number of sequences to optimize
        seq_len: length of the sequences
        pose_prior: VPoser model
        motion_prior: humor model
        init_motion_prior: dict of GMM params to use for prior on initial motion state
        shared_floor: (default True) if true, sequences are in the same reference frame
        fit_gender: (optional) gender of SMPL model
    """

    def __init__(
        self,
        batch_size,
        seq_len,
        body_model,
        pose_prior,
        motion_prior,
        init_motion_prior=None,
        fit_gender="neutral",
        use_init=False,
        opt_cams=False,
        opt_scale=True,
        cam_graph=False,
        est_floor=True,
        floor_type="shared",
        async_tracks=True,
        **kwargs,
    ):
        super().__init__(
            batch_size,
            seq_len,
            body_model,
            pose_prior,
            fit_gender=fit_gender,
            use_init=use_init,
            opt_cams=opt_cams,
            opt_scale=opt_scale,
            cam_graph=cam_graph,
        )
        assert motion_prior is not None
        assert motion_prior.model_data_config in [
            "smpl+joints",
            "smpl+joints+contacts",
        ], "Only smpl+joints motion prior configuration is supported!"

        self.motion_prior = motion_prior
        self.init_motion_prior = init_motion_prior

        # need latent dynamics sequence as well
        self.latent_motion_dim = self.motion_prior.latent_size
        self.cond_prior = self.motion_prior.use_conditional_prior

        # the frame chosen to use for the initial state (first frame by default)
        self.async_tracks = async_tracks
        self.register_buffer("track_start", torch.zeros(self.batch_size))
        self.register_buffer("track_end", torch.ones(self.batch_size) * self.seq_len)

        self.shared_floor = floor_type == "shared"
        self.group_floor = floor_type == "group"
        self.floor_type = floor_type
        print("FLOOR TYPE", floor_type)
        self.est_floor = est_floor

    @property
    def is_motion_active(self):
        return hasattr(self.params, "latent_motion")

    def initialize(self, obs_data, cam_data, param_dict, data_fps):
        """
        we need to also optimize for floor and world scale
        """
        Logger.log("Initializing moving scene model with observed data")

        self.params.set_cameras(
            cam_data,
            opt_scale=self.opt_scale,
            opt_cams=self.opt_cams,
            opt_focal=self.opt_cams,
            **param_dict,
        )
        self.init_floor(obs_data, param_dict)
        self.init_first_state(obs_data, param_dict, data_fps)

    def init_floor(self, obs_data, param_dict):
        if self.est_floor or self.group_floor:
            with torch.no_grad():
                smpl_preds = self.pred_smpl(
                    param_dict["trans"],
                    param_dict["root_orient"],
                    self.latent2pose(param_dict["latent_pose"]),
                    param_dict["betas"],
                )
            floor_plane, floor_idcs = estimate_floor_planes(
                smpl_preds["joints3d"].detach(),
                obs_data["vis_mask"] > 0,
                group=self.group_floor,
                flatten=self.shared_floor,
            )
            if self.group_floor and not self.est_floor:
                # don't use the estimated floor as initial plane
                floor_plane = obs_data["floor_plane"][: len(floor_plane)]
        else:  # fixed shared or separate floors
            num_floors = 1 if self.shared_floor else self.batch_size
            floor_plane = obs_data["floor_plane"][:num_floors]
            if self.shared_floor:
                floor_idcs = torch.zeros(
                    self.batch_size, dtype=torch.long, device=floor_plane.device
                )
            else:
                floor_idcs = torch.arange(
                    self.batch_size, dtype=torch.long, device=floor_plane.device
                )

        Logger.log(f"ESTIMATED FLOORS: {str(floor_plane.detach().cpu())}")
        Logger.log(f"FLOOR IDCS: {str(floor_idcs.detach().cpu())}")
        self.params.set_param("floor_plane", floor_plane.float().detach())
        self.params.set_param(
            "floor_idcs", floor_idcs.long().detach(), requires_grad=False
        )

    def init_first_state(self, obs_data, param_dict, data_fps):
        """
        initialize the latent motion and first state of each track
        using per-frame trajectories of trans, rot, latent_pose, betas
        and observed data
        """
        B, T = self.batch_size, self.seq_len
        param_dict = detach_all(param_dict)

        # select the valid segments of each track; pad the shorter tracks with final element
        if self.async_tracks and "track_interval" in obs_data:
            print("asynchronous tracks")
            interval = obs_data["track_interval"]  # (B, 2)
            start, end = interval[:, 0], interval[:, 1]
            self.track_start, self.track_end = start, end
            param_dict = select_dict_segments(
                param_dict, start, end, names=["trans", "root_orient", "latent_pose"]
            )

        trans = param_dict["trans"]  # (B, T, 3)
        root_orient = param_dict["root_orient"]  # (B, T, 3)
        latent_pose = param_dict["latent_pose"]  # (B, T, D)
        betas = param_dict["betas"]  # (B, b)

        Logger.log(f"SEL TRACKS {trans.shape}, {root_orient.shape}")

        # save each track's first appearance
        self.params.set_param("trans", trans[:, :1])
        Logger.log(f"INITIAL TRANS {trans[:, :1].detach().cpu()}")
        self.params.set_param("root_orient", root_orient[:, :1])
        self.params.set_param("latent_pose", latent_pose[:, :1])
        self.params.set_param("betas", betas)

        # pass the current pose estimates through the motion prior
        self.data_fps = data_fps
        body_pose = self.latent2pose(latent_pose)
        init_latent = self.infer_latent_motion(
            trans, root_orient, body_pose, betas, data_fps
        ).detach()  # (B, T-1, D)
        self.params.set_param("latent_motion", init_latent)

        # estimate velocities in prior frame and save initial state
        trans_vel, joints_vel, root_orient_vel = self.estimate_prior_velocities(
            trans, root_orient, body_pose, betas, data_fps
        )
        self.params.set_param("trans_vel", trans_vel[:, :1].detach())
        self.params.set_param("joints_vel", joints_vel[:, :1].detach())
        self.params.set_param("root_orient_vel", root_orient_vel[:, :1].detach())

    def get_optim_result(self, num_steps=-1):
        res = super().get_optim_result()
        if not self.is_motion_active:
            return res

        num_steps = self.seq_len if num_steps < 0 else num_steps
        prior_res, world_res = self.rollout_latent_motion(
            self.params.latent_motion[:, : num_steps - 1]
        )
        world_res = self.synchronize_preds(world_res, num_steps)
        res["world"].update(world_res)

        prior_res = self.synchronize_preds(prior_res, num_steps)
        res["prior"] = prior_res
        return res

    def estimate_prior_velocities(self, trans, root_orient, body_pose, betas, data_fps):
        """
        compute velocities of trajectory in prior frame (not world frame)
        Transforms the first of each track into the prior frame
        :param trans (B, T, 3)
        :param root_orient (B, T, 3)
        :param body_pose (B, T, Dp)
        :param betas (B, Db)
        """
        with torch.no_grad():
            self.update_world2prior(trans, root_orient, body_pose, betas)
            trans, root_orient = self.apply_world2prior(
                trans, root_orient, body_pose, betas
            )
            smpl_results = self.pred_smpl(trans, root_orient, body_pose, betas)
        return estimate_velocities(
            trans, root_orient, smpl_results["joints3d"], data_fps
        )

    def update_world2prior(self, trans, root_orient, body_pose, betas):
        B = trans.shape[0]
        smpl_data = self.pred_smpl(
            trans[:, :1], root_orient[:, :1], body_pose[:, :1], betas
        )
        floor_plane = self.params.floor_plane[self.params.floor_idcs]
        R, t, height = compute_world2prior(
            floor_plane,
            trans[:, 0],
            root_orient[:, 0],
            smpl_data["joints3d"][:, 0, 0],
        )
        self.world2prior_R = R  # (B, 3, 3)
        self.world2prior_t = t  # (B, 3)
        self.world2prior_root_height = height  # (B)

    def apply_world2prior(self, trans, root_orient, body_pose, betas, inverse=False):
        """
        Applies the world2prior tranformation to trans, root_orient
        If T=1, this function assumes they are at key_frame_idx,
        which we need to compute the offset from the origin
        :param trans (B, T, 3)
        :param root_orient (B, T, 3)
        :param body_pose (B, T, J, 3)
        :param betas (B, b)
        :param inverse (bool) optional, default False
        """
        B, T, _ = trans.size()
        # (B, 3, 3), (B, 3), (B)
        R, t, root_height = (
            self.world2prior_R,
            self.world2prior_t,
            self.world2prior_root_height,
        )
        R_time = R.unsqueeze(1).expand((B, T, 3, 3))
        t_time = t.unsqueeze(1).expand((B, T, 3))
        root_orient_mat = batch_rodrigues(root_orient.reshape(-1, 3)).reshape(
            B, T, 3, 3
        )

        if inverse:
            R_time = R_time.transpose(-1, -2)

        root_orient_mat = torch.matmul(R_time, root_orient_mat)
        root_orient = rotation_matrix_to_angle_axis(
            root_orient_mat.reshape(B * T, 3, 3)
        ).reshape(B, T, 3)

        if inverse:
            # transform so first frame is at origin
            trans = trans - trans[:, :1, :]

            # rotates to world frame
            trans = torch.matmul(R_time, trans[..., None])[..., 0]
            # translate to world frame
            trans = trans - t_time

            return trans, root_orient

        # first transform so the trans of key frame is at origin
        trans = trans + t_time
        # then rotate to canonical frame
        trans = torch.matmul(R_time, trans[..., None])[..., 0]
        # compute the root height after transforming
        cur_smpl_data = self.pred_smpl(trans, root_orient, body_pose, betas)
        cur_root_height = cur_smpl_data["joints3d"][:, 0, 0, 2:3]
        # then apply floor offset so the root joint is at root_height
        height_diff = root_height - cur_root_height
        trans_offset = torch.cat(
            [torch.zeros((B, 2)).to(height_diff), height_diff], axis=1
        )
        trans = trans + trans_offset.reshape((B, 1, 3))

        return trans, root_orient

    def convert_prior_rot_inputs(self, root_orient, body_pose):
        # convert rots
        # body pose and root orient are both in aa
        B, T = root_orient.shape[:2]
        if (
            self.motion_prior.in_rot_rep == "mat"
            or self.motion_prior.in_rot_rep == "6d"
        ):
            root_orient_in = batch_rodrigues(root_orient.reshape(-1, 3)).reshape(
                (B, T, 9)
            )
            body_pose_in = batch_rodrigues(body_pose.reshape(-1, 3)).reshape(
                (B, T, J_BODY * 9)
            )
        if self.motion_prior.in_rot_rep == "6d":
            root_orient_in = root_orient[:, :, :6]
            body_pose_in = body_pose.reshape((B, T, J_BODY, 9))[:, :, :, :6].reshape(
                (B, T, J_BODY * 6)
            )
        return root_orient_in, body_pose_in

    def convert_prior_rot_outputs(self, out_dict):
        keys = ["root_orient", "pose_body"]
        for k in keys:
            out = out_dict[k]
            B, T = out.shape[:2]
            out_dict[k] = rotation_matrix_to_angle_axis(
                out.reshape((-1, 3, 3))
            ).reshape((B, T, -1))
        return out_dict

    def infer_latent_motion(
        self, trans, root_orient, body_pose, betas, data_fps, full_forward_pass=False
    ):
        """
        By default, gets a sequence of z's from the current SMPL optim params.

        If full_forward_pass is true, also samples from the posterior and feeds
        through the motion prior decoder to get all terms needed to calculate the ELBO.
        """
        B, T, _ = trans.size()
        h = 1.0 / data_fps

        # need to first transform into canonical coordinate frame
        self.update_world2prior(trans, root_orient, body_pose, betas)
        trans, root_orient = self.apply_world2prior(
            trans, root_orient, body_pose, betas
        )

        smpl_results = self.pred_smpl(trans, root_orient, body_pose, betas)
        joints = smpl_results["joints3d"]  # (B, T, len(SMPL_JOINTS), 3)
        trans_vel, joints_vel, root_orient_vel = estimate_velocities(
            trans, root_orient, joints, data_fps
        )

        root_orient_in, body_pose_in = self.convert_prior_rot_inputs(
            root_orient, body_pose
        )
        joints_in = joints.reshape((B, T, -1))
        joints_vel_in = joints_vel.reshape((B, T, -1))

        seq_dict = {
            "trans": trans,
            "trans_vel": trans_vel,
            "root_orient": root_orient_in,
            "root_orient_vel": root_orient_vel,
            "pose_body": body_pose_in,
            "joints": joints_in,
            "joints_vel": joints_vel_in,
        }

        infer_results = self.motion_prior.infer_global_seq(
            seq_dict, full_forward_pass=full_forward_pass
        )
        if full_forward_pass:
            # return both the given motion and the one from the forward pass
            # make sure rotations are matrix
            # NOTE: assumes seq_dict is same thing we want to compute loss on
            # Need to change if multiple future steps.
            if self.motion_prior.in_rot_rep != "mat":
                seq_dict["trans"] = batch_rodrigues(root_orient.reshape(-1, 3)).reshape(
                    (B, T, 9)
                )
                seq_dict["pose_body"] = batch_rodrigues(
                    body_pose.reshape(-1, 3)
                ).reshape((B, T, J_BODY * 9))
            # do not need initial step anymore since output will be T-1
            for k, v in seq_dict.items():
                seq_dict[k] = v[:, 1:]
            for k in infer_results.keys():
                if k != "posterior_distrib" and k != "prior_distrib":
                    infer_results[k] = infer_results[k][
                        :, :, 0
                    ]  # only want first output step
            infer_results = (seq_dict, infer_results)
        else:
            prior_z, posterior_z = infer_results
            infer_results = posterior_z[0]  # mean of the approximate posterior

        return infer_results

    def rollout_smpl_steps(self, num_steps=-1):
        # rollout motion given initial state
        num_steps = self.seq_len if num_steps < 0 else num_steps
        latent_motion = self.params.latent_motion[:, : num_steps - 1]

        # roll out tracks from their first appearances
        # NOTE rolled out tracks are unsynced in time
        res, world_res = self.rollout_latent_motion(
            latent_motion, return_prior=self.cond_prior
        )

        # get the smpl predictions for the unsynced tracks
        preds = self.pred_smpl(
            res["trans"], res["root_orient"], res["pose_body"], res["betas"]
        )
        world_preds = self.pred_smpl(
            world_res["trans"],
            world_res["root_orient"],
            world_res["pose_body"],
            world_res["betas"],
        )

        # pass along relevant results
        preds["joints3d_rollout"] = res["joints"]
        preds["latent_pose"] = self.pose2latent(res["pose_body"])
        preds["joints3d_init"] = preds["joints3d"][:, :1]

        if "contacts" in res:
            preds["contacts"] = res["contacts"]
        if "contacts_conf" in res:
            preds["contacts_conf"] = res["contacts_conf"]

        # synchronize the tracklets to the same timesteps
        preds = self.synchronize_preds(preds, num_steps)
        world_preds = self.synchronize_preds(world_preds, num_steps)

        # pass along unsynchronized motion latents
        preds["latent_motion"] = latent_motion
        if "cond_prior" in res:
            preds["cond_prior"] = res["cond_prior"]

        return preds, world_preds

    def rollout_latent_motion(
        self,
        latent_motion,
        return_prior=False,
        return_vel=False,
        num_steps=-1,
        canonicalize_input=False,
    ):
        """
        From the stored initial state, rolls out a sequence from the latent_motion vector
        NOTE: the stored initial states are not at the same time step, but we return all
        predicted sequences starting from time 0.

        If latent_motion is None, samples num_steps into the future sequence from the prior.
        using the mean of the prior rather than random samples.

        If canonicalize_input is True, transform the initial state into the local canonical
        frame before roll out
        """
        # get the first frame state
        # NOTE: first states do not occur at same time step
        trans, root_orient, betas = (
            self.params.trans,
            self.params.root_orient,
            self.params.betas,
        )
        trans_vel, root_orient_vel, joints_vel = (
            self.params.trans_vel,
            self.params.root_orient_vel,
            self.params.joints_vel,
        )
        body_pose = self.latent2pose(self.params.latent_pose)

        B = trans.size(0)
        is_sampling = latent_motion is None
        Tm1 = num_steps if latent_motion is None else latent_motion.size(1)
        if is_sampling and Tm1 <= 0:
            Logger.log("num_steps must be positive to sample!")
            exit()

        # need to first transform initial state into canonical coordinate frame
        self.update_world2prior(trans, root_orient, body_pose, betas)
        trans, root_orient = self.apply_world2prior(
            trans, root_orient, body_pose, betas
        )

        smpl_results = self.pred_smpl(trans, root_orient, body_pose, betas)
        joints = smpl_results["joints3d"]  # (B, T, len(SMPL_JOINTS), 3)

        # update to correct rotations for input
        root_orient_in, body_pose_in = self.convert_prior_rot_inputs(
            root_orient, body_pose
        )
        joints_in = joints.reshape((B, 1, -1))
        joints_vel_in = joints_vel.reshape((B, 1, -1))

        rollout_in_dict = {
            "trans": trans,
            "root_orient": root_orient_in,
            "pose_body": body_pose_in,
            "joints": joints_in,
            "trans_vel": trans_vel,
            "root_orient_vel": root_orient_vel,
            "joints_vel": joints_vel_in,
        }

        roll_output = self.motion_prior.roll_out(
            None,
            rollout_in_dict,
            Tm1,
            z_seq=latent_motion,
            return_prior=return_prior,
            return_z=is_sampling,
            canonicalize_input=canonicalize_input,
            gender=[self.fit_gender] * B,
            betas=betas.reshape((B, 1, -1)),
        )

        pred_dict = prior_out = None
        if return_prior:
            pred_dict, prior_out = roll_output
        else:
            pred_dict = roll_output

        pred_dict = self.convert_prior_rot_outputs(pred_dict)

        # concat with initial state
        trans_out = torch.cat([trans, pred_dict["trans"]], dim=1)
        root_orient_out = torch.cat([root_orient, pred_dict["root_orient"]], dim=1)
        body_pose_out = torch.cat([body_pose, pred_dict["pose_body"]], dim=1)
        joints_out = torch.cat(
            [joints, pred_dict["joints"].reshape((B, Tm1, -1, 3))], dim=1
        )
        out_dict = {
            "trans": trans_out,
            "root_orient": root_orient_out,
            "pose_body": body_pose_out,
            "joints": joints_out,
            "betas": betas,
        }
        if return_vel:
            out_dict["trans_vel"] = torch.cat(
                [trans_vel, pred_dict["trans_vel"]], dim=1
            )
            out_dict["root_orient_vel"] = torch.cat(
                [root_orient_vel, pred_dict["root_orient_vel"]], dim=1
            )
            out_dict["joints_vel"] = torch.cat(
                [joints_vel, pred_dict["joints_vel"].reshape((B, Tm1, -1, 3))],
                dim=1,
            )

        if return_prior:  # return the mean and var of distribution stacked
            pm, pv = prior_out
            out_dict["cond_prior"] = torch.stack([pm, pv], dim=-1)

        if self.motion_prior.model_data_config == "smpl+joints+contacts":
            pred_contacts = pred_dict["contacts"]
            # get binary classification
            contact_conf = torch.sigmoid(pred_contacts)
            pred_contacts = (contact_conf > CONTACT_THRESH).to(torch.float)
            # expand to full body
            full_contact_conf = torch.zeros((B, Tm1, len(SMPL_JOINTS))).to(contact_conf)
            full_contact_conf[:, :, CONTACT_INDS] = (
                full_contact_conf[:, :, CONTACT_INDS] + contact_conf
            )
            full_contacts = torch.zeros((B, Tm1, len(SMPL_JOINTS))).to(pred_contacts)
            full_contacts[:, :, CONTACT_INDS] = (
                full_contacts[:, :, CONTACT_INDS] + pred_contacts
            )
            # repeat first entry for t0
            full_contact_conf = torch.cat(
                [full_contact_conf[:, 0:1], full_contact_conf], dim=1
            )
            full_contacts = torch.cat([full_contacts[:, 0:1], full_contacts], dim=1)
            out_dict["contacts_conf"] = full_contact_conf
            out_dict["contacts"] = full_contacts

        if is_sampling:
            out_dict["z"] = pred_dict["z"]

        cam_dict = {
            "trans": out_dict["trans"],
            "root_orient": out_dict["root_orient"],
            "pose_body": out_dict["pose_body"],
            "betas": out_dict["betas"],
        }

        # also must return trans and root orient in camera frame
        cam_dict["trans"], cam_dict["root_orient"] = self.apply_world2prior(
            out_dict["trans"],
            out_dict["root_orient"],
            out_dict["pose_body"],
            out_dict["betas"],
            inverse=True,
        )
        return out_dict, cam_dict

    def synchronize_preds(self, pred_dict, seg_len):
        """
        synchronize predictions in time, scatter predictions into [0, seq_len)
        """
        if not self.async_tracks:
            # predictions are already synchronized
            return pred_dict, None

        # return time-synchronized predictions
        start = self.track_start
        end = torch.clip(start + seg_len, max=self.track_end)
        T = end.max()
        pred_dict = scatter_dict_segments(pred_dict, start, end, T)
        pred_mask = get_scatter_mask(start, end, T)
        pred_dict["track_mask"] = pred_mask
        return pred_dict


def scatter_dict_segments(data_dict, start, end, T=None, names=None):
    """
    the tracks as they are stored are synchronized by time step
    Uses the start and end to synchronize output predictions
    """
    sync_dict = data_dict.copy()
    min_len = (end - start).min()
    if names is None:
        names = data_dict.keys()
    for name in names:
        val = data_dict[name]
        if not isinstance(val, torch.Tensor) or val.ndim < 3 or val.shape[1] < min_len:
            continue
        sync_dict[name] = scatter_intervals(val, start, end, T)
    return sync_dict


def select_dict_segments(data_dict, start, end, names=None):
    out_data_dict = data_dict.copy()
    min_len = (end - start).min()
    if names is None:
        names = data_dict.keys()
    for name in names:
        val = data_dict[name]
        if not isinstance(val, torch.Tensor) or val.ndim < 3 or val.shape[1] < min_len:
            continue
        # only select for time-series observations
        out_data_dict[name] = select_intervals(val, start, end)
    return out_data_dict


def estimate_velocities(trans, root_orient, joints3d, data_fps):
    """
    Estimates velocity inputs to the motion prior.
    - trans (B, T, 3) root translation
    - root_orient (B, T, 3) aa root orientation
    - joints3d (B, T, len(SMPL_JOINTS), 3) joints3d of SMPL prediction
    """
    B, T, _ = trans.size()
    h = 1.0 / data_fps
    trans_vel = estimate_linear_velocity(trans, h)
    joints_vel = estimate_linear_velocity(joints3d, h)
    root_orient_mat = batch_rodrigues(root_orient.reshape((-1, 3))).reshape(
        (B, T, 3, 3)
    )
    root_orient_vel = estimate_angular_velocity(root_orient_mat, h)
    return trans_vel, joints_vel, root_orient_vel


def estimate_linear_velocity(data_seq, h):
    """
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    """
    # first steps is forward diff (t+1 - t) / h
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / h
    # middle steps are second order (t+1 - t-1) / 2h
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * h)
    # last step is backward diff (t - t-1) / h
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / h

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq


def estimate_angular_velocity(rot_seq, h):
    """
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    """
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, h)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w
