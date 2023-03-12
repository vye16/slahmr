import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import BatchNorm2d

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,2,3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

class SMPLHead(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, cfg):
        super(SMPLHead, self).__init__()
        self.cfg = cfg
        npose = 6 * (cfg.SMPL.NUM_BODY_JOINTS + 1)
        self.npose = npose
        in_channels = cfg.MODEL.SMPL_HEAD.IN_CHANNELS
        self.pool = cfg.MODEL.SMPL_HEAD.POOL
        self.fc1 = nn.Linear(in_channels + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(cfg.MODEL.SMPL_HEAD.SMPL_MEAN_PARAMS)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam)
        
    def forward(self, x, n_iter=3):
        batch_size = x.shape[0]

        init_body_pose = self.init_body_pose.expand(batch_size, -1)
        init_betas = self.init_betas.expand(batch_size, -1)
        init_cam = self.init_cam.expand(batch_size, -1)

        # pooling type
        if self.pool == 'max':
            print("pooling")
            xf = x.max(3)[0].max(2)[0]
        elif self.pool == 'pooled':
            xf = x
        else:
            xf = x.mean(dim=(2,3))
        xf = xf.view(xf.size(0), -1)
        
        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(n_iter):
            xc = torch.cat([xf, pred_body_pose, pred_betas, pred_cam],1)
            xc = F.relu(self.fc1(xc))
            xc = self.drop1(xc)
            xc = F.relu(self.fc2(xc))
            xc = self.drop2(xc)
            pred_body_pose = self.decpose(xc) + pred_body_pose
            pred_betas = self.decshape(xc) + pred_betas
            pred_cam = self.deccam(xc) + pred_cam
            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        pred_smpl_params_list = {}
        pred_smpl_params_list['body_pose'] = torch.cat([rot6d_to_rotmat(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_body_pose_list], dim=0)
        pred_smpl_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smpl_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_body_pose = rot6d_to_rotmat(pred_body_pose).view(batch_size, self.cfg.SMPL.NUM_BODY_JOINTS+1, 3, 3)

        pred_smpl_params = {'global_orient': pred_body_pose[:, [0]],
                             'body_pose': pred_body_pose[:, 1:],
                             'betas': pred_betas}
        
        return pred_smpl_params, pred_cam, pred_smpl_params_list
