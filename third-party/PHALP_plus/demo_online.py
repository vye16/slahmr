import os
import cv2
import time
import joblib
import argparse
import warnings
import traceback
import numpy as np
from tqdm import tqdm

from PHALP import PHALP_tracker
from models.pose_model import PoseModel
from deep_sort_ import nn_matching
from deep_sort_.detection import Detection
from deep_sort_.tracker import Tracker

from utils.make_video import render_frame_main_online
from utils.utils import FrameExtractor, str2bool
from pytube import YouTube

warnings.filterwarnings('ignore')
  
        
def test_tracker(opt, phalp_tracker: PHALP_tracker):
    
    eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
    history_keys    = ['appe', 'loca', 'pose', 'uv'] if opt.render else []
    prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if opt.render else []
    extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'mask_name', 'conf']
    extra_keys_2    = ['smpl', '3d_joints', 'camera', 'embedding', 'vitpose']
    history_keys    = history_keys + extra_keys_1 + extra_keys_2
    visual_store_   = eval_keys + history_keys + prediction_keys
    tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']
    
                                                
    if(not(opt.overwrite) and os.path.isfile('out/' + opt.storage_folder + '/results/' + str(opt.video_seq) + '.pkl')): return 0
    print(opt.storage_folder + '/results/' + str(opt.video_seq))
    
    try:
        os.makedirs('out/' + opt.storage_folder, exist_ok=True)  
        os.makedirs('out/' + opt.storage_folder + '/results', exist_ok=True)  
        os.makedirs('out/' + opt.storage_folder + '/_TMP', exist_ok=True)  
    except: pass
    
    phalp_tracker.eval()
    phalp_tracker.HMAR.reset_nmr(opt.res)    

    # initialize vitpose model
    device = 'cuda'
    cpm = PoseModel(device)

    metric  = nn_matching.NearestNeighborDistanceMetric(opt, opt.hungarian_th, opt.past_lookback)
    tracker = Tracker(opt, metric, max_age=opt.max_age_track, n_init=opt.n_init, phalp_tracker=phalp_tracker, dims=[4096, 4096, 99])  
        
    try: 
        
        main_path_to_frames = opt.base_path + '/' + opt.video_seq + opt.sample
        list_of_frames      = np.sort([i for i in os.listdir(main_path_to_frames) if '.jpg' in i])
        list_of_frames      = list_of_frames if opt.start_frame==-1 else list_of_frames[opt.start_frame:opt.end_frame]
        list_of_shots       = phalp_tracker.get_list_of_shots(main_path_to_frames, list_of_frames)
        
        tracked_frames          = []
        final_visuals_dic       = {}

        for t_, frame_name in enumerate(tqdm(list_of_frames)):
            if(opt.verbose): 
                print('\n\n\nTime: ', opt.video_seq, frame_name, t_, time.time()-time_ if t_>0 else 0 )
                time_ = time.time()
            
            image_frame               = cv2.imread(main_path_to_frames + '/' + frame_name)
            img_height, img_width, _  = image_frame.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            opt.shot                  = 1 if t_ in list_of_shots else 0

            ############ detection ##############
            pred_bbox, pred_masks, pred_scores, mask_names, gt = phalp_tracker.get_detections(image_frame, frame_name, t_)

            ############ vitpose #############
            vitposes_out = cpm.predict_pose(image_frame[:,:,::-1], [np.concatenate([pred_bbox, pred_scores[:,None]], axis=1)])
            #vis = cpm.visualize_pose_results(image_frame[:,:,::-1], vitposes_out)
            #cv2.imwrite('test.jpg', vis[:,:,::-1])

            ############ HMAR ##############                      
            detections = []                                       
            for bbox, mask, score, mask_name, gt_id, vitpose_out in zip(pred_bbox, pred_masks, pred_scores, mask_names, gt, vitposes_out):
                if bbox[2]-bbox[0]<50 or bbox[3]-bbox[1]<100: continue
                detection_data = phalp_tracker.get_human_apl(image_frame, mask, bbox, score, [main_path_to_frames, frame_name], mask_name, t_, measurments, gt_id)
                vitpose_2d = np.zeros([25, 3])                    
                vitpose_2d[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = vitpose_out['keypoints']
                detection_data['vitpose'] = vitpose_2d
                detections.append(Detection(detection_data))

            ############ tracking ##############
            tracker.predict()
            tracker.update(detections, t_, frame_name, opt.shot)

            ############ record the results ##############
            final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': opt.shot})
            if(opt.render): final_visuals_dic[frame_name]['frame'] = image_frame
            for key_ in visual_store_: final_visuals_dic[frame_name][key_] = []
            
            for tracks_ in tracker.tracks:
                if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                if(not(tracks_.is_confirmed())): continue
                
                track_id        = tracks_.track_id
                track_data_hist = tracks_.track_data['history'][-1]
                track_data_pred = tracks_.track_data['prediction']

                final_visuals_dic[frame_name]['tid'].append(track_id)
                final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                for hkey_ in history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                for pkey_ in prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])

                if(tracks_.time_since_update==0):
                    final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                    final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                    
                    if(tracks_.hits==opt.n_init):
                        for pt in range(opt.n_init-1):
                            track_data_hist_ = tracks_.track_data['history'][-2-pt]
                            track_data_pred_ = tracks_.track_data['prediction']
                            frame_name_      = tracked_frames[-2-pt]
                            final_visuals_dic[frame_name_]['tid'].append(track_id)
                            final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                            final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_time'].append(0)

                            for hkey_ in history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                            for pkey_ in prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])

                            
                            
            ############ save the video ##############
            if(opt.render and t_>=opt.n_init):
                d_ = opt.n_init+1 if(t_+1==len(list_of_frames)) else 1
                for t__ in range(t_, t_+d_):
                    frame_key          = list_of_frames[t__-opt.n_init]
                    rendered_, f_size  = render_frame_main_online(opt, phalp_tracker, frame_key, final_visuals_dic[frame_key], opt.track_dataset, track_id=-100)      
                    if(t__-opt.n_init in list_of_shots): cv2.rectangle(rendered_, (0,0), (f_size[0], f_size[1]), (0,0,255), 4)
                    if(t__-opt.n_init==0):
                        file_name      = 'out/' + opt.storage_folder + '/PHALP_' + str(opt.video_seq) + '_'+ str(opt.detection_type) + '.mp4'
                        video_file     = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, frameSize=f_size)
                    video_file.write(rendered_)
                    del final_visuals_dic[frame_key]['frame']
                    for tkey_ in tmp_keys_:  del final_visuals_dic[frame_key][tkey_] 

        joblib.dump(final_visuals_dic, 'out/' + opt.storage_folder + '/results/' + opt.track_dataset + "_" + str(opt.video_seq) + opt.post_fix  + '.pkl')
        if(opt.use_gt): joblib.dump(tracker.tracked_cost, 'out/' + opt.storage_folder + '/results/' + str(opt.video_seq) + '_' + str(opt.start_frame) + '_distance.pkl')
        if(opt.render): video_file.release()
        
    except Exception as e: 
        print(e)
        print(traceback.format_exc())     


class options():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description='PHALP_pixel Tracker')
        self.parser.add_argument('--batch_id', type=int, default='-1')
        self.parser.add_argument('--track_dataset', type=str, default='posetrack')
        self.parser.add_argument('--predict', type=str, default='APL')
        self.parser.add_argument('--storage_folder', type=str, default='Videos_v20.000')
        self.parser.add_argument('--distance_type', type=str, default='A5')
        self.parser.add_argument('--use_gt', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--overwrite', type=str2bool, nargs='?', const=True, default=False)

        self.parser.add_argument('--alpha', type=float, default=0.1)
        self.parser.add_argument('--low_th_c', type=float, default=0.95)
        self.parser.add_argument('--hungarian_th', type=float, default=100.0)
        self.parser.add_argument('--track_history', type=int, default=7)
        self.parser.add_argument('--max_age_track', type=int, default=20)
        self.parser.add_argument('--n_init',  type=int, default=1)
        self.parser.add_argument('--max_ids', type=int, default=50)
        self.parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--detect_shots', type=str2bool, nargs='?', const=True, default=False)
        
        self.parser.add_argument('--base_path', type=str)
        self.parser.add_argument('--video_seq', type=str, default='_DATA/posetrack/list_videos_val.npy')
        self.parser.add_argument('--youtube_id', type=str, default="xEH_5T9jMVU")
        self.parser.add_argument('--all_videos', type=str2bool, nargs='?', const=True, default=True)
        self.parser.add_argument('--store_mask', type=str2bool, nargs='?', const=True, default=True)

        self.parser.add_argument('--render', type=str2bool, nargs='?', const=True, default=False)
        self.parser.add_argument('--render_type', type=str, default='HUMAN_HEAD_FAST')
        self.parser.add_argument('--render_up_scale', type=int, default=2)
        self.parser.add_argument('--res', type=int, default=256)
        self.parser.add_argument('--downsample',  type=int, default=1)
        
        self.parser.add_argument('--encode_type', type=str, default='3c')
        self.parser.add_argument('--cva_type', type=str, default='least_square')
        self.parser.add_argument('--past_lookback', type=int, default=1)
        self.parser.add_argument('--mask_type', type=str, default='feat')
        self.parser.add_argument('--detection_type', type=str, default='mask2')
        self.parser.add_argument('--start_frame', type=int, default='1000')
        self.parser.add_argument('--end_frame', type=int, default='1100')
        self.parser.add_argument('--store_extra_info', type=str2bool, nargs='?', const=True, default=False)
    
    def parse(self):
        self.opt          = self.parser.parse_args()
        self.opt.sample   = ''
        self.opt.post_fix = ''
        return self.opt
   

if __name__ == '__main__':
    
    opt                   = options().parse()
    phalp_tracker         = PHALP_tracker(opt)
    phalp_tracker.cuda()
    phalp_tracker.eval()

    if(opt.track_dataset=='youtube'):   
        
        video_id = opt.youtube_id
        video    = "youtube_" + video_id

        os.system("rm -rf " + "_DEMO/" + video)
        os.makedirs("_DEMO/" + video, exist_ok=True)    
        os.makedirs("_DEMO/" + video + "/img", exist_ok=True)    
        youtube_video = YouTube('https://www.youtube.com/watch?v=' + video_id)
        print(f'Title: {youtube_video.title}')
        print(f'Duration: {youtube_video.length / 60:.2f} minutes')
        youtube_video.streams.get_by_itag(136).download(output_path = "_DEMO/" + video, filename="youtube.mp4")
        fe = FrameExtractor("_DEMO/" + video + "/youtube.mp4")
        print('Number of frames: ', fe.n_frames)
        fe.extract_frames(every_x_frame=1, img_name='', dest_path= "_DEMO/" + video + "/img/", start_frame=1100, end_frame=1300)

        opt.base_path       = '_DEMO/'
        opt.video_seq       = video
        opt.sample          =  '/img/'
        test_tracker(opt, phalp_tracker)


            
