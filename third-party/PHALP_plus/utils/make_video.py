import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import copy
import cv2
import numpy as np
from utils.utils import get_colors
 
RGB_tuples = get_colors()


def numpy_to_torch_image(ndarray):
    torch_image              = torch.from_numpy(ndarray)
    torch_image              = torch_image.unsqueeze(0)
    torch_image              = torch_image.permute(0, 3, 1, 2)
    torch_image              = torch_image[:, [2,1,0], :, :]
    return torch_image


def render_frame_main_online(opt, phalp_tracker, image_name, final_visuals_dic, dataset,number_of_windows=4, downsample=1, storage_folder=None, track_id=0):

    t_           = final_visuals_dic['time']
    cv_image     = final_visuals_dic['frame']
    tracked_ids  = final_visuals_dic["tid"]
    tracked_bbox = final_visuals_dic["bbox"]
    tracked_loca = final_visuals_dic["prediction_loca"]
    tracked_pose = [final_visuals_dic["prediction_pose"], final_visuals_dic["pose"]]
    tracked_appe = [final_visuals_dic["uv"], final_visuals_dic["prediction_uv"]]
    tracked_time = final_visuals_dic["tracked_time"]
    
    
    number_of_windows = 1
    res               = 1440

    img_height, img_width, _      = cv_image.shape
    new_image_size                = max(img_height, img_width)
 
    if(phalp_tracker.HMAR.nmr_size!=opt.res*opt.render_up_scale):
        phalp_tracker.HMAR.reset_nmr(opt.res*opt.render_up_scale)
    
    new_image_size_x              = opt.res*opt.render_up_scale
    ratio                         = 1.0*opt.res/max(img_height, img_width)*opt.render_up_scale
    
    delta_w                       = new_image_size - img_width
    delta_h                       = new_image_size - img_height
    top, bottom, left, right      = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
    resized_image                 = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    resized_image_bbox            = copy.deepcopy(resized_image)
    resized_image_small           = cv2.resize(resized_image, (opt.res*opt.render_up_scale, opt.res*opt.render_up_scale))
    scale_                        = res/img_width
    frame_size                    = (number_of_windows*res, int(img_height*(scale_)))

    rendered_image_1              = numpy_to_torch_image(np.array(resized_image_bbox)/255.)
    rendered_image_1x             = numpy_to_torch_image(np.array(resized_image_bbox)/255.)
    
    if(len(tracked_ids)>0):
        tracked_time              = np.array(tracked_time)
        tracked_pose_single       = np.array(tracked_pose[1])
        tracked_pose              = np.array(tracked_pose[0])
        tracked_pose_single       = torch.from_numpy(tracked_pose_single).cuda()
        tracked_pose              = torch.from_numpy(tracked_pose).cuda()
        tracked_bbox              = np.array(tracked_bbox)
        tracked_center            = tracked_bbox[:, :2] + tracked_bbox[:, 2:]/2.0 + [left, top]
        tracked_scale             = np.max(tracked_bbox[:, 2:], axis=1)
        tracked_loca              = np.array(tracked_loca)
        
        tracked_loca_xy           = tracked_loca[:, :90]
        tracked_loca_xy           = np.reshape(tracked_loca_xy, (-1,45,2))
        tracked_loca_xy           = tracked_loca_xy[:, 44, :]*new_image_size

        tracked_loca              = tracked_loca[:, 90:93]
        tracked_loca[:, 2]       /= opt.render_up_scale
        
        if "HUMAN" in opt.render_type:
            ids_x                 = tracked_time==0
        elif "GHOST" in opt.render_type:
            ids_x                 = tracked_time>-100

        tracked_ids_x             = np.array(tracked_ids)
        tracked_ids_x             = tracked_ids_x[ids_x]

        tracked_appe_single       = np.array(tracked_appe[0])
        tracked_appe              = np.array(tracked_appe[1])
        tracked_appe_single       = torch.from_numpy(tracked_appe_single).float().cuda()
        tracked_appe              = torch.from_numpy(tracked_appe).float().cuda()
        uv_maps                   = tracked_appe_single[:, :3, :, :]
        scale_x                   = tracked_scale
        scale_x                   = np.reshape(scale_x, (len(scale_x), 1))
        
        

        rendered_ = 0
        with torch.no_grad():
            if(len(tracked_ids_x)>0):
                rendered_                     = 1
                if("SMOOTH" in opt.render_type):
                    rendered_uv, mask_, _, _, _   = phalp_tracker.HMAR.render_3d(tracked_pose[ids_x, :], 
                                                                                 np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                 img_size   = new_image_size_x, 
                                                                                 location   = torch.from_numpy(tracked_loca[ids_x, :]).cuda(),
                                                                                 render     = True, 
                                                                                 full_shape = "FULL" in opt.render_type)
                    
                else:
                    rendered_uv, mask_, _, _, _   = phalp_tracker.HMAR.render_3d(tracked_pose_single[ids_x, :], 
                                                                                 np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                 center     = ratio*tracked_center[ids_x, :], 
                                                                                 img_size   = new_image_size_x, 
                                                                                 scale      = ratio*scale_x[ids_x, :], 
                                                                                 render     = True, 
                                                                                 full_shape = "FULL" in opt.render_type)

            if(rendered_ == 1):
                rendered_uv                   = rendered_uv[:, [2,1,0], :, :]
                a_mask, m_mask, f_mask        = mask_
                m_mask                        = m_mask.unsqueeze(1).repeat(1, 3, 1, 1)
                loc_mask                      = m_mask==0
                rendered_uv[loc_mask]         = 0.0
                rendered_uv                   = rendered_uv.sum(0, keepdim=True)
                if(ratio!=1.0):rendered_uv    = F.interpolate(rendered_uv.detach().cpu(), size=max(img_height, img_width))
                rendered_image_1              = rendered_uv[:, :, top:top+img_height, left:left+img_width]
                rendered_image_1x             = rendered_image_1x[:, :, top:top+img_height, left:left+img_width]
                rendered_image_4x             = copy.deepcopy(rendered_image_1x)#*0 + 1
                loc_xx                        = rendered_image_1>0
                rendered_image_4x[loc_xx]     = 0.0
                if("FAST" in opt.render_type):
                    rendered_image_1              =  rendered_image_1 + rendered_image_4x
            else:
                rendered_image_1              = rendered_image_1[:, :, top:top+img_height, left:left+img_width]

            rendered_image_1x             = rendered_image_1x[:, :, top:top+img_height, left:left+img_width]

        if("FAST" not in opt.render_type):
            if(phalp_tracker.HMAR.pyrender_size!=opt.res*opt.render_up_scale):
                phalp_tracker.HMAR.reset_pyrender(opt.res*opt.render_up_scale)
            resized_image_small             = cv2.resize(resized_image, (opt.res*opt.render_up_scale, opt.res*opt.render_up_scale))
            if(len(tracked_ids_x)>0):
                if("SMOOTH" in opt.render_type):
                    rendered_image_3, valid_mask, _, _  = phalp_tracker.HMAR.render_3d(tracked_pose[ids_x, :], 
                                                                                       np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                       img_size   = new_image_size_x,
                                                                                       location   = torch.from_numpy(tracked_loca[ids_x, :]).cuda(),
                                                                                       image      = (0*resized_image_small)/255.0, 
                                                                                       render     = True, 
                                                                                       use_image  = True,
                                                                                       engine     = "PYR" )
                    
                else:
                    rendered_image_3, valid_mask, _, _  = phalp_tracker.HMAR.render_3d(tracked_pose_single[ids_x, :], 
                                                                                       np.array(RGB_tuples[list(tracked_ids_x)])/255.0, 
                                                                                       center     = ratio*tracked_center[ids_x, :], 
                                                                                       img_size   = new_image_size_x, 
                                                                                       scale      = ratio*scale_x[ids_x, :], 
                                                                                       image      = (0*resized_image_small)/255.0, 
                                                                                       render     = True, 
                                                                                       use_image  = True,
                                                                                       engine     = "PYR" )
                
                
                
                
                rendered_image_3           = cv2.resize(rendered_image_3, (max(img_height, img_width), max(img_height, img_width)))
                rendered_image_3           = numpy_to_torch_image(np.array(rendered_image_3))[:, :, top:top+img_height, left:left+img_width]

                valid_mask                 = np.repeat(valid_mask, 3, 2)
                valid_mask                 = np.array(valid_mask, dtype=int)
                valid_mask                 = np.array(valid_mask, dtype=float)
                valid_mask                 = cv2.resize(valid_mask, (max(img_height, img_width), max(img_height, img_width)))
                valid_mask                 = numpy_to_torch_image(np.array(valid_mask))[:, :, top:top+img_height, left:left+img_width]

                loc_b = valid_mask==1
                rendered_image_5x             = copy.deepcopy(rendered_image_1x)#*0 + 1
                rendered_image_5x[loc_b]      = 0
                rendered_image_3[torch.logical_not(loc_b)] = 0
                rendered_image_3              = rendered_image_3 + rendered_image_5x
                
                
            else:
                rendered_image_3              = copy.deepcopy(rendered_image_1x)

    else:
        rendered_image_3              = copy.deepcopy(rendered_image_1x)
        rendered_image_1              = rendered_image_1[:, :, top:top+img_height, left:left+img_width]

    if("FAST" not in opt.render_type):
        grid_img = make_grid(torch.cat([rendered_image_3], 0), nrow=10)
    else:
        grid_img = make_grid(torch.cat([rendered_image_1], 0), nrow=10)

    grid_img = grid_img[[2,1,0], :, :]
    ndarr    = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    cv_ndarr = cv2.resize(ndarr, frame_size)
    cv2.putText(cv_ndarr, str(t_), (20,20), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,255))

    return cv_ndarr, frame_size


