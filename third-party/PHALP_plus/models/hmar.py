import torch
import torch.nn as nn
import numpy as np

from .backbones import resnet
from .heads.mesh import *
from .heads.smpl_head import SMPLHead
from .heads.apperence_head import TextureHead
from .heads.encoding_head import EncodingHead
from .joint_mapper import JointMapper, smpl_to_openpose
from .smplx import create
from .utils import perspective_projection

from yacs.config import CfgNode as CN
import neural_renderer as nr
# from .renderer import Renderer
from models.pose_transformer import Pose_transformer

from .utils import *

class HMAR(nn.Module):
    
    def __init__(self, config, opt):
        super(HMAR, self).__init__()
        with open(config, 'r') as f:
            cfg = CN.load_cfg(f); cfg.freeze()
            
        self.cfg = cfg
        self.opt = opt

        nz_feat  = 512; tex_size = 6
        img_H    = 256; img_W    = 256
        
        texture_file         = np.load(self.cfg.SMPL.TEXTURE)
        self.faces_cpu       = texture_file['smpl_faces'].astype('uint32')
        
        vt                   = texture_file['vt']
        ft                   = texture_file['ft']
        uv_sampler           = compute_uvsampler(vt, ft, tex_size=6)
        uv_sampler           = torch.tensor(uv_sampler, dtype=torch.float)
        uv_sampler           = uv_sampler.unsqueeze(0)

        self.F               = uv_sampler.size(1)   
        self.T               = uv_sampler.size(2) 
        self.uv_sampler      = uv_sampler.view(-1, self.F, self.T*self.T, 2)
        self.backbone        = resnet(cfg.MODEL.BACKBONE, num_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS, pretrained=True, opt=self.opt)
        self.texture_head    = TextureHead(self.uv_sampler, self.cfg, img_H=img_H, img_W=img_W)
        self.encoding_head   = EncodingHead(opt=opt, img_H=img_H, img_W=img_W) 
    
    
        smpl_params  = {k.lower(): v for k,v in dict(cfg.SMPL).items()}
        joint_mapper = JointMapper(smpl_to_openpose(model_type=cfg.SMPL.MODEL_TYPE))
        self.smpl    = create(**smpl_params,
                                  batch_size=1,
                                  joint_mapper = joint_mapper,
                                  create_betas=False,
                                  create_body_pose=False,
                                  create_global_orient=False,
                                  create_left_hand_pose=False,
                                  create_right_hand_pose=False,
                                  create_expression=False,
                                  create_leye_pose=False,
                                  create_reye_pose=False,
                                  create_jaw_pose=False,
                                  create_transl=False)
        
        self.nmr_renderer = nr.Renderer(dist_coeffs=None, orig_size=256,
                                          image_size=256,
                                          light_intensity_ambient=1.0,
                                          light_intensity_directional=0.0,
                                          anti_aliasing=False, far=200)
        
        # self.py_render            = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=256, faces=self.faces_cpu)
        self.nmr_size             = 256
        self.pyrender_size        = 256
        self.smpl_head            = SMPLHead(cfg)
        self.smpl_head.pool       = 'pooled'
        self.device               = "cuda"
        
        if("P" in self.opt.predict):
            opt.pose_transformer_size = 2048
            self.pose_transformer     = Pose_transformer(opt)
            checkpoint_file = torch.load("_DATA/hmmr_v2_weights.pt")
                
            state_dict_filt = {k[11:]: v for k, v in checkpoint_file['model'].items() if ("relational" in k)}  
            self.pose_transformer.relational.load_state_dict(state_dict_filt, strict=True)

            state_dict_filt = {k[18:]: v for k, v in checkpoint_file['model'].items() if ("smplx_head_future" in k)}  
            self.pose_transformer.smpl_head_prediction.load_state_dict(state_dict_filt, strict=False)


    def forward(self, x):
        feats, skips    = self.backbone(x)
        flow            = self.texture_head(skips)
        uv_image        = self.flow_to_texture(flow, x)
        
        pose_embeddings = feats.max(3)[0].max(2)[0]
        pose_embeddings = pose_embeddings.view(x.size(0),-1)
        with torch.no_grad():
            pred_smpl_params, _, _ = self.smpl_head(pose_embeddings)

        out = {
            "uv_image"  : uv_image,
            "flow"      : flow,
            "pose_emb"  : pose_embeddings,
            "pose_smpl" : pred_smpl_params,
        }
        return out    
    
    def flow_to_texture(self, flow_map, img_x):
        batch_size = flow_map.size(0)
        flow_map   = flow_map.permute(0,2,3,1)
        uv_images  = torch.nn.functional.grid_sample(img_x, flow_map)
        return uv_images
    
    def autoencoder_hmar(self, x, en=True):
        if(en==True):
            if(self.opt.encode_type=="3c"):
                return self.encoding_head(x[:, :3, :, :], en=en)
            else:
                return self.encoding_head(x, en=en)
        else:
            return self.encoding_head(x, en=en)

    
    def render_3d(self, pose_embeddings, color, center=np.array([128, 128]), img_size = 256, scale = None, location=None, texture=None, image=None, use_image=False, render=True, full_shape=True, engine="NMR"):
        
        if(scale is not None): pass
        else: scale = np.ones((pose_embeddings.size(0), 1))*256

        with torch.no_grad():
            pred_smpl_params, pred_cam, _ = self.smpl_head(pose_embeddings[:, 2048:].float())

        batch_size             = pose_embeddings.shape[0]
        dtype                  = pred_cam.dtype
        focal_length           = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device, dtype=dtype)
 
        smpl_output            = self.smpl(**{k: v.float() for k,v in pred_smpl_params.items()}, pose2rot=False)
        pred_vertices          = smpl_output.vertices
        pred_joints            = smpl_output.joints
        
        if(location is not None):
            pred_cam_t         = torch.tensor(location*1.0, dtype=dtype, device=self.device) #location
        else:
            pred_cam_t         = torch.stack([pred_cam[:,1], pred_cam[:,2], 2*focal_length[:, 0]/(pred_cam[:,0]*torch.tensor(scale[:, 0], dtype=dtype, device=self.device) + 1e-9)], dim=1)
            pred_cam_t[:, :2] += torch.tensor(center-img_size/2., dtype=dtype, device=self.device) * pred_cam_t[:, [2]] / focal_length

        # initialize camera params and mesh faces for NMR
        K = torch.eye(3, device='cuda')
        K[0, 0] = K[1, 1]  = self.cfg.EXTRA.FOCAL_LENGTH
        K[2, 2] = 1
        K[1, 2] = K[0, 2]  = img_size/2.0
                                      
                                      
        K = K.unsqueeze(0).repeat(batch_size, 1, 1)  # to BS
        R = torch.eye(3, device='cuda').unsqueeze(0)
        t = torch.zeros(3, device='cuda').unsqueeze(0) 
        face_tensor = torch.tensor(self.faces_cpu.astype(np.int64), dtype=torch.long, device='cuda').unsqueeze_(0)
        face_tensor = face_tensor.repeat(batch_size, 1, 1)
        

        # transform vertices to world coordinates
        pred_cam_t_bs         = pred_cam_t.unsqueeze(1).repeat(1, pred_vertices.size(1), 1)
        verts                 = pred_vertices + pred_cam_t_bs

        mask_model = []
        loc_       = 0
        if(engine=="NMR"):
            
            if(render):
                if(texture is not None):
                    texture_vert = torch.nn.functional.grid_sample(texture, self.uv_sampler.repeat(batch_size,1,1,1).cuda())
                    texture_vert = texture_vert.view(texture_vert.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1).contiguous()
                    texture      = texture_vert.unsqueeze(4).expand(-1, -1, -1, -1, 6, -1) 
                else:
                    texture               = torch.ones(batch_size, 3, 256, 256).cuda() 
                    for i_ in range(texture.shape[0]):
                        if(full_shape):
                            texture[i_, 0, :, :]          = color[i_, 2]
                            texture[i_, 1, :, :]          = color[i_, 1]
                            texture[i_, 2, :, :]          = color[i_, 0]
                        else:
                            texture[i_, :, :, : ]         = -1000.0
                            texture[i_, 0, :152, :153]    = color[i_, 2]
                            texture[i_, 1, :152, :153]    = color[i_, 1]
                            texture[i_, 2, :152, :153]    = color[i_, 0]
                            texture[i_, 0, 120:152, :159] = color[i_, 2]
                            texture[i_, 1, 120:152, :159] = color[i_, 1]
                            texture[i_, 2, 120:152, :159] = color[i_, 0]

                    texture_vert = torch.nn.functional.grid_sample(texture, self.uv_sampler.repeat(batch_size,1,1,1).cuda())
                    texture_vert = texture_vert.view(texture_vert.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1).contiguous()
                    texture      = texture_vert.unsqueeze(4).expand(-1, -1, -1, -1, 6, -1) 

                rgb_from_pred, depth, mask = self.nmr_renderer(verts.cuda(), face_tensor.int().cuda(), textures=texture, K=K, R=R, t=t, dist_coeffs=torch.tensor([[0., 0., 0., 0., 0.]], device=self.device))
                loc_3 = rgb_from_pred<-10
                predicted_depth            = depth
                predicted_mask             = mask
                predicted_closest_depth, closest_channel = torch.min(predicted_depth, dim=0)
                for i in range(mask.shape[0]):
                    loc_  = closest_channel==i
                    loc_2 = mask[i]>0
                    mask_ = torch.logical_and(loc_2, torch.logical_not(loc_3[i, 0, :, :]))
                    mask_ = torch.logical_and(loc_, mask_)
                    mask_model.append(mask_)

                mask_model = torch.stack(mask_model, 0)
            else:
                rgb_from_pred, depth, mask = 0, 0, 0

            zeros_  = torch.zeros(batch_size, 1, 3).cuda()
            pred_joints = torch.cat((pred_joints, zeros_), 1)

            camera_center          = torch.zeros(batch_size, 2)
            pred_keypoints_2d_smpl = perspective_projection(pred_joints, rotation=torch.eye(3,).unsqueeze(0).expand(batch_size, -1, -1).cuda(),
                                                            translation=pred_cam_t.cuda(),
                                                            focal_length=focal_length / img_size,
                                                            camera_center=camera_center.cuda())  

            pred_keypoints_2d_smpl = (pred_keypoints_2d_smpl+0.5)*img_size

            return rgb_from_pred, [mask, mask_model, loc_], pred_keypoints_2d_smpl, pred_joints, pred_cam_t

        if(render and engine=="PYR"):
            rgb_from_pred, validmask = self.py_render.visualize_all(pred_vertices.cpu().numpy(), pred_cam_t_bs.cpu().numpy(), color, image, use_image=use_image)
            return rgb_from_pred, validmask, 0, 0 
        
        
    def reset_pyrender(self, image_size):
        self.py_render            = Renderer(focal_length=self.cfg.EXTRA.FOCAL_LENGTH, img_res=image_size, faces=self.faces_cpu)
        self.pyrender_size        = image_size

    def reset_nmr(self, image_size):
        self.nmr_renderer = nr.Renderer(dist_coeffs=None, 
                                           orig_size=image_size,
                                           image_size=image_size,
                                           light_intensity_ambient=0.8, 
                                           far=1000
                                          )   
        self.nmr_size      = image_size

        
  











