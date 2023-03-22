from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points    
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / (points[:,:,-1].unsqueeze(-1))
    
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]
        
def compute_uvsampler(vt, ft, tex_size=6):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    uv = obj2nmr_uvmap(ft, vt, tex_size=tex_size)
    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv

def obj2nmr_uvmap(ft, vt, tex_size=6):
    """
    Converts obj uv_map to NMR uv_map (F x T x T x 2),
    where tex_size (T) is the sample rate on each face.
    """
    # This is F x 3 x 2
    uv_map_for_verts = vt[ft]

    # obj's y coordinate is [1-0], but image is [0-1]
    uv_map_for_verts[:, :, 1] = 1 - uv_map_for_verts[:, :, 1]

    # range [0, 1] -> [-1, 1]
    uv_map_for_verts = (2 * uv_map_for_verts) - 1

    alpha = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    beta = np.arange(tex_size, dtype=np.float64) / (tex_size - 1)
    import itertools
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])

    # Compute alpha, beta (this is the same order as NMR)
    v2 = uv_map_for_verts[:, 2]
    v0v2 = uv_map_for_verts[:, 0] - uv_map_for_verts[:, 2]
    v1v2 = uv_map_for_verts[:, 1] - uv_map_for_verts[:, 2]
    # Interpolate the vertex uv values: F x 2 x T*2
    uv_map = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 2, 1)

    # F x T*2 x 2  -> F x T x T x 2
    uv_map = np.transpose(uv_map, (0, 2, 1)).reshape(-1, tex_size, tex_size, 2)

    return uv_map
