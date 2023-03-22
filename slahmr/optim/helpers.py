import numpy as np
import torch

from slahmr.body_model import OP_EDGE_LIST
from slahmr.geometry.rotation import batch_rodrigues
from slahmr.geometry.plane import (
    parse_floor_plane,
    compute_plane_intersection,
    fit_plane,
)


def compute_world2prior(floor_plane, trans, root_orient, origin):
    """
    Computes rotation and translation from the camera frame to the canonical coordinate system
    used by the motion and initial state priors.
    - floor_plane : B x 3
    - trans : B x 3
    - root_orient : B x 3
    - origin: B x 3 desired origin (first joint of human model)
    - returns R (B, 3, 3), t (B, 3), root_height (B)
    """
    B = trans.size(0)
    if floor_plane.size(1) == 3:
        floor_plane_4d = parse_floor_plane(floor_plane)
    else:
        floor_plane_4d = floor_plane
    floor_normal = floor_plane_4d[:, :3]
    floor_trans, _ = compute_plane_intersection(trans, -floor_normal, floor_plane_4d)

    # compute prior frame axes within the camera frame
    # up is the floor_plane normal
    up_axis = floor_normal
    # right is body -x direction projected to floor plane
    root_orient_mat = batch_rodrigues(root_orient)
    body_right = -root_orient_mat[:, :, 0]
    floor_body_right, s = compute_plane_intersection(trans, body_right, floor_plane_4d)
    right_axis = floor_body_right - floor_trans
    # body right may not actually intersect - in this case must negate axis because we have the -x
    right_axis = torch.where(s.reshape((B, 1)) < 0, -right_axis, right_axis)
    right_axis = right_axis / torch.norm(right_axis, dim=1, keepdim=True)
    # forward is their cross product
    fwd_axis = torch.linalg.cross(up_axis, right_axis)
    fwd_axis = fwd_axis / torch.norm(fwd_axis, dim=1, keepdim=True)

    # prior frame is right, fwd, up
    prior_R = torch.stack([right_axis, fwd_axis, up_axis], dim=2)
    world2prior_R = prior_R.transpose(-1, -2)

    # translation takes translation to origin plus offset to the floor
    world2prior_t = -trans

    # compute the distance from root to ground plane
    _, s_root = compute_plane_intersection(origin, -floor_normal, floor_plane_4d)
    root_height = s_root.reshape(B, 1)

    if False:
        # apply transform to origin
        origin_prior = torch.matmul(world2prior_R, (origin - trans)[..., None])[..., 0]
        root_height = s_root.reshape(B, 1) - origin_prior[:, 2:3]

    return world2prior_R, world2prior_t, root_height


def find_cliques(edges):
    cliques = []
    N = edges.shape[0]
    empty = np.ones(N, dtype=bool)
    while empty.sum() > 0:
        idcs = np.where(empty)[0]
        i = idcs[0]
        cur_clique = [i]
        for j in idcs[1:]:
            if all(edges[c, j] for c in cur_clique):
                cur_clique.append(j)
        empty[cur_clique] = False
        cliques.append(cur_clique)
    return cliques


def estimate_floor_planes(
    smpl_joints, valid_mask, thresh=0.5, group=False, flatten=False
):
    """
    :param smpl_joints (B, T, J, 3)
    :param valid_mask (B, T)
    :param thresh (optional, default 0.5) threshold for separating
    :param group (optional, default False) whether to group the planes
    :param flatten (optional, default False) whether to estimate one plane for everyone
    returns floor_plane (C, 3), floor_idcs (B,)
    """
    B, T = valid_mask.shape
    device = valid_mask.device
    feet_joints = smpl_joints[..., 10:12, :]  # (B, T, 2, 3)
    points = [feet_joints[b, valid_mask[b]].reshape(-1, 3) for b in range(B)]

    if flatten:
        planes = fit_plane(torch.cat(points, dim=0))[None]
        labels = torch.zeros(B, device=device, dtype=torch.long)
        return planes[..., :3] * planes[..., 3:], labels

    # estimate floors independently
    planes = torch.stack([fit_plane(pts) for pts in points], dim=0)  # (B, 4)
    normals, offsets = planes[..., :3], planes[..., 3:]  # (B, 3), (B, 1)
    sgn = torch.sign(offsets)
    normals, offsets = sgn * normals, sgn * offsets
    planes = normals * offsets  # (B, 3)
    labels = torch.arange(B, device=device)

    if not group:
        return planes, labels  # (B, 3), (B,)

    # asymetric, project each plane vector onto the other normals
    diffs = offsets - (planes[None] * normals[:, None]).sum(dim=-1)
    print(diffs)
    edges = diffs < thresh
    edges = edges.T & edges
    clusters = find_cliques(edges)  # list of indices of clusters
    planes = []
    labels = torch.zeros(B, dtype=torch.long, device=device)
    for c, idcs in enumerate(clusters):
        plane = fit_plane(torch.cat([points[i] for i in idcs], dim=0))
        planes.append(plane)
        labels[idcs] = c
    planes = torch.stack(planes, dim=0)  # (C, 4)
    return planes[..., :3] * planes[..., 3:], labels


def estimate_initial_trans(body_pose, joints3d_op, joints2d_op, focal):
    """
    use focal length and bone lengths to approximate distance from camera
    (based on PROX https://github.com/mohamedhassanmus/prox/blob/master/prox/fitting.py)
    :param body_pose
    :param joints3d_op OP keypoints in 3d
    :param joints2d_op OP keypoints in 2d
    """
    B, T, N, _ = joints2d_op.shape
    joints2d_obs = joints2d_op[:, :, :, :2]  # (B, T, N, 3)
    joints2d_conf = joints2d_op[:, :, :, 2]

    # find least-occluded 2d frame
    num_2d_vis = torch.sum(joints2d_conf > 0.0, dim=2)
    best_2d_idx = torch.max(num_2d_vis, dim=1)[1]

    # calculate bone lengths and confidence in each bone length
    bone3d = []
    bone2d = []
    conf2d = []
    for pair in OP_EDGE_LIST:
        diff3d = torch.norm(
            joints3d_op[:, 0, pair[0], :] - joints3d_op[:, 0, pair[1], :],
            dim=1,
        )  # does not change over time
        diff2d = torch.norm(
            joints2d_obs[:, :, pair[0], :] - joints2d_obs[:, :, pair[1], :],
            dim=2,
        )
        minconf2d = torch.min(
            joints2d_conf[:, :, pair[0]], joints2d_conf[:, :, pair[1]]
        )
        bone3d.append(diff3d)
        bone2d.append(diff2d)
        conf2d.append(minconf2d)

    bone3d = torch.stack(bone3d, dim=1)
    bone2d = torch.stack(bone2d, dim=2)
    bone2d = bone2d[np.arange(B), best_2d_idx, :]
    conf2d = torch.stack(conf2d, dim=2)
    conf2d = conf2d[np.arange(B), best_2d_idx, :]

    # mean over all bones
    mean_bone3d = torch.mean(bone3d, dim=1)
    mean_bone2d = torch.mean(bone2d * (conf2d > 0.0), dim=1)

    # approx z based on ratio
    init_z = focal * (mean_bone3d / mean_bone2d)
    init_trans = torch.zeros(B, T, 3)
    init_trans[:, :, 2] = init_z.unsqueeze(1).expand(B, T).detach()

    return init_trans
