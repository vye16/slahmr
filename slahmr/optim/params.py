import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slahmr.geometry.camera import compose_cameras, invert_camera
from slahmr.geometry.rotation import batch_rodrigues
from slahmr.body_model import SMPL_JOINTS
from slahmr.util.logger import Logger
from slahmr.util.tensor import detach_all

J_BODY = len(SMPL_JOINTS) - 1  # no root


class Params(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.param_names = set()

    def set_param(self, name, val, requires_grad=False):
        print("SETTING PARAM", name, val.shape)
        with torch.no_grad():
            setattr(
                self, name, nn.Parameter(val.contiguous(), requires_grad=requires_grad)
            )
        self.param_names.add(name)

    def get_param(self, name):
        if name not in self.param_names:
            raise ValueError(f"{name} not stored as opt param")
        return getattr(self, name)

    def load_dict(self, param_dict):
        for name, val in param_dict.items():
            self.set_param(name, val, requires_grad=False)

    def get_dict(self):
        return {name: self.get_param_item(name) for name in self.param_names}

    def get_vars(self, names=None):
        if names is None:
            names = self.param_names
        return {name: self.get_param(name) for name in names}

    def get_param_item(self, name):
        with torch.no_grad():
            param = self.get_param(name)
            return param.detach()

    def _set_param_grad(self, name, val: bool):
        if name not in self.param_names:
            raise ValueError(f"{name} not stored as param")
        param = getattr(self, name)
        assert isinstance(param, torch.Tensor)
        param.requires_grad = val

    def set_require_grads(self, names):
        """
        set parameters in names to True, set all others to False
        """
        for name in self.param_names:
            self._set_param_grad(name, False)

        for name in names:
            self._set_param_grad(name, True)

        Logger.log("Set parameter grads:")
        Logger.log(
            {name: getattr(self, name).requires_grad for name in self.param_names}
        )


class CameraParams(Params):
    """
    Parameter container with cameras
    """

    def set_cameras(
        self, cam_data, opt_scale=True, opt_cams=False, opt_focal=False, **kwargs
    ):
        self.opt_scale = opt_scale
        self.opt_cams = opt_cams
        self.opt_focal = opt_focal

        # (T, 3, 3), (T, 3)
        cam_R, cam_t = cam_data["cam_R"], cam_data["cam_t"]
        intrins = cam_data["intrins"]  # (T, 4)

        T = cam_R.shape[0]
        device = cam_R.device

        # assume focal length and center are same for all timesteps
        self.cam_center = intrins[:, 2:]  # (T, 2)
        cam_f = intrins[:, :2]  # (T, 2)
        if self.opt_focal:
            self.set_param("cam_f", cam_f)
        else:
            self.cam_f = cam_f

        self._cam_R = cam_R  # (T, 3, 3)
        self._cam_t = cam_t  # (T, 3)

        world_scale = torch.ones(1, 1, device=device)
        if self.opt_scale:
            if "world_scale" in kwargs:
                world_scale = kwargs["world_scale"]
            self.set_param("world_scale", world_scale)
        else:
            self.world_scale = world_scale

        if self.opt_cams:
            init_delta_R = torch.zeros(T, 3, device=device)
            init_delta_t = torch.zeros(T, 3, device=device)
            if "delta_cam_R" in kwargs:
                print("setting delta_cam_R from kwargs")
                init_delta_R = kwargs["delta_cam_R"]
            if "delta_cam_t" in kwargs:
                print("setting delta_cam_t from kwargs")
                init_delta_t = kwargs["delta_cam_t"]

            self.set_param("delta_cam_R", init_delta_R)
            self.set_param("delta_cam_t", init_delta_t)

    @property
    def intrins(self):  # (4,)
        return torch.cat([self.cam_f[0], self.cam_center[0]], dim=-1).detach().cpu()

    def get_extrinsics(self):
        """
        returns (T, 3, 3), (T, 3)
        """
        cam_R, cam_t = self._cam_R, self._cam_t
        if self.opt_cams:
            dR = batch_rodrigues(self.delta_cam_R)
            cam_R = torch.matmul(cam_R, dR)
            cam_t = cam_t + self.delta_cam_t
        return cam_R, cam_t * self.world_scale

    def get_intrinsics(self):
        """
        returns (T, 4)
        """
        return torch.cat([self.cam_f, self.cam_center], axis=-1)

    def get_cameras(self, idcs=None):
        """
        returns cam_R (B, T, 3, 3) cam_t (B, T, 3), cam_f (T, 2), cam_center (T, 2)
        """
        if idcs is None:
            idcs = np.arange(self._cam_R.shape[0])

        cam_R, cam_t = self.get_extrinsics()
        cam_R, cam_t = cam_R[None, idcs], cam_t[None, idcs]
        cam_f, cam_center = self.cam_f[idcs], self.cam_center[idcs]

        B = self.batch_size
        cam_R = cam_R.repeat(B, 1, 1, 1)
        cam_t = cam_t.repeat(B, 1, 1)
        return cam_R, cam_t, cam_f, cam_center
