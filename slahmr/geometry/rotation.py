import torch
import numpy as np
from torch.nn import functional as F


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """
    Taken from https://github.com/mkocabas/VIBE/blob/master/lib/utils/geometry.py
    Calculates the rotation matrices for a batch of rotation vectors
    - param rot_vecs: torch.tensor (N, 3) array of N axis-angle vectors
    - returns R: torch.tensor (N, 3, 3) rotation matrices
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def quaternion_mul(q0, q1):
    """
    EXPECTS WXYZ
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    r0, r1 = q0[..., :1], q1[..., :1]
    v0, v1 = q0[..., 1:], q1[..., 1:]
    r = r0 * r1 - (v0 * v1).sum(dim=-1, keepdim=True)
    v = r0 * v1 + r1 * v0 + torch.linalg.cross(v0, v1)
    return torch.cat([r, v], dim=-1)


def quaternion_inverse(q, eps=1e-8):
    """
    EXPECTS WXYZ
    :param q (*, 4)
    """
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    mag = torch.square(q).sum(dim=-1, keepdim=True) + eps
    return conj / mag


def quaternion_slerp(t, q0, q1, eps=1e-8):
    """
    :param t (*, 1)  must be between 0 and 1
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    dims = q0.shape[:-1]
    t = t.view(*dims, 1)

    q0 = F.normalize(q0, p=2, dim=-1)
    q1 = F.normalize(q1, p=2, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # make sure we give the shortest rotation path (< 180d)
    neg = dot < 0
    q1 = torch.where(neg, -q1, q1)
    dot = torch.where(neg, -dot, dot)
    angle = torch.acos(dot)

    # if angle is too small, just do linear interpolation
    collin = torch.abs(dot) > 1 - eps
    fac = 1 / torch.sin(angle)
    w0 = torch.where(collin, 1 - t, torch.sin((1 - t) * angle) * fac)
    w1 = torch.where(collin, t, torch.sin(t * angle) * fac)
    slerp = q0 * w0 + q1 * w1
    return slerp


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert rotation matrix to Rodrigues vector
    """
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    :param quaternion (*, 4) expects WXYZ
    :returns angle_axis (*, 3)
    """
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    Taken from https://github.com/kornia/kornia, based on
    https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
    https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247
    :param quaternion (N, 4) expects WXYZ order
    returns rotation matrix (N, 3, 3)
    """
    # normalize the input quaternion
    quaternion_norm = F.normalize(quaternion, p=2, dim=-1, eps=1e-12)
    *dims, _ = quaternion_norm.shape

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(*dims, 3, 3)
    return matrix


def angle_axis_to_quaternion(angle_axis):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert angle axis to quaternion in WXYZ order
    :param angle_axis (*, 3)
    :returns quaternion (*, 4) WXYZ order
    """
    theta_sq = torch.sum(angle_axis**2, dim=-1, keepdim=True)  # (*, 1)
    # need to handle the zero rotation case
    valid = theta_sq > 0
    theta = torch.sqrt(theta_sq)
    half_theta = 0.5 * theta
    ones = torch.ones_like(half_theta)
    # fill zero with the limit of sin ax / x -> a
    k = torch.where(valid, torch.sin(half_theta) / theta, 0.5 * ones)
    w = torch.where(valid, torch.cos(half_theta), ones)
    quat = torch.cat([w, k * angle_axis], dim=-1)
    return quat


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    :param rotation_matrix (N, 3, 3)
    """
    *dims, m, n = rotation_matrix.shape
    rmat_t = torch.transpose(rotation_matrix.reshape(-1, m, n), -1, -2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q.reshape(*dims, 4)
