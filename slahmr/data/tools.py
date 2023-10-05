import functools
import json
import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from slahmr.body_model import OP_NUM_JOINTS


def read_keypoints(keypoint_fn):
    """
    Only reads body keypoint data of first person.
    """
    empty_kps = np.zeros((OP_NUM_JOINTS, 3), dtype=np.float32)
    if not os.path.isfile(keypoint_fn):
        return empty_kps

    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data["people"]) == 0:
        print("WARNING: Found no keypoints in %s! Returning zeros!" % (keypoint_fn))
        return empty_kps

    person_data = data["people"][0]
    body_keypoints = np.array(person_data["pose_keypoints_2d"], dtype=np.float32)
    body_keypoints = body_keypoints.reshape([-1, 3])
    return body_keypoints


def read_mask_path(path):
    mask_path = None
    if not os.path.isfile(path):
        return mask_path

    with open(path, "r") as f:
        data = json.load(path)

    person_data = data["people"][0]
    if "mask_path" in person_data:
        mask_path = person_data["mask_path"]

    return mask_path


def read_smpl_preds(pred_path, num_betas=10):
    """
    reads the betas, body_pose, global orientation and translation of a smpl prediction
    exported from phalp outputs
    returns betas (10,), body_pose (23, 3), global_orientation (3,), translation (3,)
    """
    pose = np.zeros((23, 3))
    rot = np.zeros(3)
    trans = np.zeros(3)
    betas = np.zeros(num_betas)
    if not os.path.isfile(pred_path):
        return pose, rot, trans, betas

    with open(pred_path, "r") as f:
        data = json.load(f)

    if "body_pose" in data:
        pose = np.array(data["body_pose"], dtype=np.float32)

    if "global_orient" in data:
        rot = np.array(data["global_orient"], dtype=np.float32)

    if "cam_trans" in data:
        trans = np.array(data["cam_trans"], dtype=np.float32)

    if "betas" in data:
        betas = np.array(data["betas"], dtype=np.float32)

    return pose, rot, trans, betas


def load_smpl_preds(pred_paths, interp=True, num_betas=10):
    vis_mask = np.array([os.path.isfile(x) for x in pred_paths])
    vis_idcs = np.where(vis_mask)[0]

    # load single image smpl predictions
    stack_fnc = functools.partial(np.stack, axis=0)
    # (N, 23, 3), (N, 3), (N, 3), (N, 10)
    pose, orient, trans, betas = map(
        stack_fnc, zip(*[read_smpl_preds(p, num_betas=num_betas) for p in pred_paths])
    )
    if not interp:
        return pose, orient, trans, betas

    # interpolate the occluded tracks
    orient_slerp = Slerp(vis_idcs, Rotation.from_rotvec(orient[vis_idcs]))
    trans_interp = interp1d(vis_idcs, trans[vis_idcs], axis=0)
    betas_interp = interp1d(vis_idcs, betas[vis_idcs], axis=0)

    tmin, tmax = min(vis_idcs), max(vis_idcs) + 1
    times = np.arange(tmin, tmax)
    orient[times] = orient_slerp(times).as_rotvec()
    trans[times] = trans_interp(times)
    betas[times] = betas_interp(times)

    # interpolate for each joint angle
    for i in range(pose.shape[1]):
        pose_slerp = Slerp(vis_idcs, Rotation.from_rotvec(pose[vis_idcs, i]))
        pose[times, i] = pose_slerp(times).as_rotvec()

    return pose, orient, trans, betas
