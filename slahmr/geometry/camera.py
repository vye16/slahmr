import torch
import numpy as np


def perspective_projection(
    points, focal_length, camera_center, rotation=None, translation=None
):
    """
    Adapted from https://github.com/mkocabas/VIBE/blob/master/lib/models/spin.py
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        focal_length (bs, 2): Focal length
        camera_center (bs, 2): Camera center
        rotation (bs, 3, 3): OPTIONAL Camera rotation
        translation (bs, 3): OPTIONAL Camera translation
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    if rotation is not None and translation is not None:
        # Transform points
        points = torch.einsum("bij,bkj->bki", rotation, points)
        points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[..., 2:3]

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[:, :, :-1]


def reproject(points3d, cam_R, cam_t, cam_f, cam_center):
    """
    reproject points3d into the scene cameras
    :param points3d (B, T, N, 3)
    :param cam_R (B, T, 3, 3)
    :param cam_t (B, T, 3)
    :param cam_f (T, 2)
    :param cam_center (T, 2)
    """
    B, T, N, _ = points3d.shape
    points3d = torch.einsum("btij,btnj->btni", cam_R, points3d)
    points3d = points3d + cam_t[..., None, :]  # (B, T, N, 3)
    points2d = points3d[..., :2] / points3d[..., 2:3]
    points2d = cam_f[None, :, None] * points2d + cam_center[None, :, None]
    return points2d


def focal2fov(focal, R):
    """
    :param focal, focal length
    :param R, either W / 2 or H / 2
    """
    return 2 * np.arctan(R / focal)


def fov2focal(fov, R):
    """
    :param fov, field of view in radians
    :param R, either W / 2 or H / 2
    """
    return R / np.tan(fov / 2)


def compute_lookat_box(bb_min, bb_max, intrins):
    """
    The center and distance to a scene with bb_min, bb_max
    to place a camera with given intrinsics
    :param bb_min (3,)
    :param bb_max (3,)
    :param intrinsics, (fx, fy, cx, cy) of camera
    :param view_angle (optional) viewing angle in radians (elevation)
    """
    fx, fy, cx, cy = intrins
    bb_min, bb_max = torch.tensor(bb_min), torch.tensor(bb_max)
    center = 0.5 * (bb_min + bb_max)
    size = torch.linalg.norm(bb_max - bb_min)
    cam_dist = np.sqrt(fx**2 + fy**2) / np.sqrt(cx**2 + cy**2)
    cam_dist = 0.75 * size * cam_dist
    return center, cam_dist


def lookat_origin(cam_dist, view_angle=-np.pi / 6):
    """
    :param cam_dist (float)
    :param view_angle (float)
    """
    cam_dist = np.abs(cam_dist)
    view_angle = np.abs(view_angle)
    pos = cam_dist * torch.tensor([0, np.sin(view_angle), np.cos(view_angle)])
    rot = rotx(view_angle)
    return rot, pos


def lookat_matrix(source_pos, target_pos, up):
    """
    IMPORTANT: USES RIGHT UP BACK XYZ CONVENTION
    :param source_pos (*, 3)
    :param target_pos (*, 3)
    :param up (3,)
    """
    *dims, _ = source_pos.shape
    up = up.reshape(*(1,) * len(dims), 3)
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)
    back = normalize(target_pos - source_pos)
    right = normalize(torch.linalg.cross(up, back))
    up = normalize(torch.linalg.cross(back, right))
    R = torch.stack([right, up, back], dim=-1)
    return make_4x4_pose(R, source_pos)


def normalize(x):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


def invert_camera(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    returns Ri (*, 3, 3), ti (*, 3)
    """
    R, t = torch.tensor(R), torch.tensor(t)
    Ri = R.transpose(-1, -2)
    ti = -torch.einsum("...ij,...j->...i", Ri, t)
    return Ri, ti


def compose_cameras(R1, t1, R2, t2):
    """
    composes [R1, t1] and [R2, t2]
    :param R1 (*, 3, 3)
    :param t1 (*, 3)
    :param R2 (*, 3, 3)
    :param t2 (*, 3)
    """
    R = torch.einsum("...ij,...jk->...ik", R1, R2)
    t = t1 + torch.einsum("...ij,...j->...i", R1, t2)
    return R, t


def matmul_nd(A, x):
    """
    multiply batch matrix A to batch nd tensors
    :param A (B, m, n)
    :param x (B, *dims, m)
    """
    B, m, n = A.shape
    assert len(A) == len(x)
    assert x.shape[-1] == m
    B, *dims, _ = x.shape
    return torch.matmul(A.reshape(B, *(1,) * len(dims), m, n), x[..., None])[..., 0]


def view_matrix(z, up, pos):
    """
    :param z (*, 3) up (*, 3) pos (*, 3)
    returns (*, 4, 4)
    """
    *dims, _ = z.shape
    x = normalize(torch.linalg.cross(up, z))
    y = normalize(torch.linalg.cross(z, x))
    bottom = (
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )

    return torch.cat([torch.stack([x, y, z, pos], dim=-1), bottom], dim=-2)


def average_pose(poses):
    """
    :param poses (N, 4, 4)
    returns average pose (4, 4)
    """
    center = poses[:, :3, 3].mean(0)
    up = normalize(poses[:, :3, 1].sum(0))
    z = normalize(poses[:, :3, 2].sum(0))
    return view_matrix(z, up, center)


def project_so3(M, eps=1e-4):
    """
    :param M (N, *, 3, 3)
    """
    N, *dims, _, _ = M.shape
    M = M * (1 + torch.rand(N, *dims, 1, 3, device=M.device))
    U, D, Vt = torch.linalg.svd(M)  # (N, *, 3, 3), (N, *, 3), (N, *, 3, 3)
    detuvt = torch.linalg.det(torch.matmul(U, Vt))  # (N, *)
    S = torch.cat(
        [torch.ones(N, *dims, 2, device=M.device), detuvt[..., None]], dim=-1
    )  # (N, *, 3)
    return torch.matmul(U, torch.matmul(torch.diag_embed(S), Vt))


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def normalize(x):
    return x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def relative_pose_c2w(Rwc1, Rwc2, twc1, twc2):
    """
    compute relative pose from cam 1 to cam 2 given c2w pose matrices
    :param Rwc1, Rwc2 (N, 3, 3) cam1, cam2 to world rotations
    :param twc1, twc2 (N, 3) cam1, cam2 to world translations
    returns R21 (N, 3, 3) t21 (N, 3)
    """
    twc1 = twc1.view(-1, 3, 1)
    twc2 = twc2.view(-1, 3, 1)
    Rc2w = Rwc2.transpose(-1, -2)  # world to c2
    tc2w = -torch.matmul(Rc2w, twc2)
    Rc2c1 = torch.matmul(Rc2w, Rwc1)
    tc2c1 = tc2w + torch.matmul(Rc2w, twc1)
    return Rc2c1, tc2c1[..., 0]


def relative_pose_w2c(Rc1w, Rc2w, tc1w, tc2w):
    """
    compute relative pose from cam 1 to cam 2 given w2c camera matrices
    :param Rc1w, Rc2w (N, 3, 3) world to cam1, cam2 rotations
    :param tc1w, tc2w (N, 3) world to cam1, cam2 translations
    """
    tc1w = tc1w.view(-1, 3, 1)
    tc2w = tc2w.view(-1, 3, 1)
    # we keep the world to cam transforms
    Rwc1 = Rc1w.transpose(-1, -2)  # c1 to world
    twc1 = -torch.matmul(Rwc1, tc1w)
    Rc2c1 = torch.matmul(Rc2w, Rwc1)  # c1 to c2
    tc2c1 = tc2w + torch.matmul(Rc2w, twc1)
    return Rc2c1, tc2c1[..., 0]


def project(xyz_c, center, focal, eps=1e-5):
    """
    :param xyz_c (*, 3) 3d point in camera coordinates
    :param focal (1)
    :param center (*, 2)
    return (*, 2)
    """
    return focal * xyz_c[..., :2] / (xyz_c[..., 2:3] + eps) + center  # (N, *, 2)


def convert_yup(xyz):
    """
    converts points in x right y down z forward to x right y up z back
    :param xyz (*, 3)
    """
    x, y, z = torch.split(xyz[..., :3], 1, dim=-1)
    return torch.cat([x, -y, -z], dim=-1)


def inv_project(uv, z, center, focal, yup=True):
    """
    :param uv (*, 2)
    :param z (*, 1)
    :param center (*, 2)
    :param focal (1)
    :returns (*, 3)
    """
    uv = uv - center
    if yup:
        return z * torch.cat(
            [uv[..., :1] / focal, -uv[..., 1:2] / focal, -torch.ones_like(uv[..., :1])],
            dim=-1,
        )  # (N, *, 3)

    return z * torch.cat(
        [uv / focal, torch.ones_like(uv[..., :1])],
        dim=-1,
    )  # (N, *, 3)
