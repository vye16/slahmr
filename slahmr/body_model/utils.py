import torch
from .specs import SMPL_JOINTS


def run_smpl(body_model, trans, root_orient, body_pose, betas=None):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : (optional) B x D
    """
    B, T, _ = trans.shape
    bm_batch_size = body_model.bm.batch_size
    assert bm_batch_size % B == 0
    seq_len = bm_batch_size // B
    bm_num_betas = body_model.bm.num_betas
    J_BODY = len(SMPL_JOINTS) - 1  # all joints except root
    if T == 1:
        # must expand to use with body model
        trans = trans.expand(B, seq_len, 3)
        root_orient = root_orient.expand(B, seq_len, 3)
        body_pose = body_pose.expand(B, seq_len, J_BODY * 3)
    elif T != seq_len:
        trans, root_orient, body_pose = zero_pad_tensors(
            [trans, root_orient, body_pose], seq_len - T
        )
    if betas is None:
        betas = torch.zeros(B, bm_num_betas, device=trans.device)
    betas = betas.reshape((B, 1, bm_num_betas)).expand((B, seq_len, bm_num_betas))
    smpl_body = body_model(
        pose_body=body_pose.reshape((B * seq_len, -1)),
        pose_hand=None,
        betas=betas.reshape((B * seq_len, -1)),
        root_orient=root_orient.reshape((B * seq_len, -1)),
        trans=trans.reshape((B * seq_len, -1)),
    )
    return {
        "joints": smpl_body.Jtr.reshape(B, seq_len, -1, 3)[:, :T],
        "vertices": smpl_body.v.reshape(B, seq_len, -1, 3)[:, :T],
        "faces": smpl_body.f,
    }


def zero_pad_tensors(pad_list, pad_size):
    """
    Assumes tensors in pad_list are B x T x D and pad temporal dimension
    """
    B = pad_list[0].size(0)
    new_pad_list = []
    for pad_idx, pad_tensor in enumerate(pad_list):
        padding = torch.zeros((B, pad_size, pad_tensor.size(2))).to(pad_tensor)
        new_pad_list.append(torch.cat([pad_tensor, padding], dim=1))
    return new_pad_list
