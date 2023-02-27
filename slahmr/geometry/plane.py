import torch


def fit_plane(points):
    """
    :param points (*, N, 3)
    returns (*, 3) plane parameters (returns in normal * offset format)
    """
    *dims, N, D = points.shape
    mean = points.mean(dim=-2, keepdim=True)
    # (*, N, D), (*, D), (*, D, D)
    U, S, Vh = torch.linalg.svd(points - mean)
    normal = Vh[..., -1, :]  # (*, D)
    offset = torch.einsum("...ij,...j->...i", points, normal)  # (*, N)
    offset = offset.mean(dim=-1, keepdim=True)
    return torch.cat([normal, offset], dim=-1)


def get_plane_transform(up, ground_plane=None, xyz_orig=None):
    """
    get R, t rigid transform from plane and desired origin
    :param up (3,) up vector of coordinate frame
    :param ground_plane (4) (a, b, c, d) where a,b,c is the normal
    :param xyz_orig (3) desired origin
    """
    R = torch.eye(3)
    t = torch.zeros(3)
    if ground_plane is None:
        return R, t

    # compute transform between world up vector and passed in floor
    ground_plane = torch.as_tensor(ground_plane)
    ground_plane = torch.sign(ground_plane[3]) * ground_plane

    normal = ground_plane[:3]
    normal = normal / torch.linalg.norm(normal)
    v = torch.linalg.cross(up, normal)
    ang_sin = torch.linalg.norm(v)
    ang_cos = up.dot(normal)
    skew_v = torch.as_tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = torch.eye(3) + skew_v + (skew_v @ skew_v) * ((1.0 - ang_cos) / (ang_sin**2))

    # project origin onto plane
    if xyz_orig is None:
        xyz_orig = torch.zeros(3)
    t, _ = compute_plane_intersection(xyz_orig, -normal, ground_plane)

    return R, t


def parse_floor_plane(floor_plane):
    """
    Takes floor plane in the optimization form (Bx3 with a,b,c * d) and parses into
    (a,b,c,d) from with (a,b,c) normal facing "up in the camera frame and d the offset.
    """
    floor_offset = torch.norm(floor_plane, dim=-1, keepdim=True)
    floor_normal = floor_plane / floor_offset

    # in camera system -y is up, so floor plane normal y component should never be positive
    #       (assuming the camera is not sideways or upside down)
    neg_mask = floor_normal[..., 1:2] > 0.0
    floor_normal = torch.where(
        neg_mask.expand_as(floor_normal), -floor_normal, floor_normal
    )
    floor_offset = torch.where(neg_mask, -floor_offset, floor_offset)
    floor_plane_4d = torch.cat([floor_normal, floor_offset], dim=-1)

    return floor_plane_4d


def compute_plane_intersection(point, direction, plane):
    """
    Given a ray defined by a point in space and a direction,
    compute the intersection point with the given plane.
    Detect intersection in either direction or -direction.
    Note, ray may not actually intersect with the plane.

    Returns the intersection point and s where
    point + s * direction = intersection_point. if s < 0 it means
    -direction intersects.

    - point : B x 3
    - direction : B x 3
    - plane : B x 4 (a, b, c, d) where (a, b, c) is the normal and (d) the offset.
    """
    dims = point.shape[:-1]
    plane_normal = plane[..., :3]
    plane_off = plane[..., 3]
    s = (plane_off - bdot(plane_normal, point)) / (bdot(plane_normal, direction) + 1e-4)
    itsct_pt = point + s.reshape((-1, 1)) * direction
    return itsct_pt, s


def bdot(A1, A2, keepdim=False):
    """
    Batched dot product.
    - A1 : B x D
    - A2 : B x D.
    Returns B.
    """
    return (A1 * A2).sum(dim=-1, keepdim=keepdim)
