import numpy as np
from tqdm import tqdm
import motmetrics as mm
from utils.utils_measure import _from_dense
import trackeval as trackeval
import sys
from tqdm import tqdm
import joblib

    
def evaluate_trackers(results_dir, method="phalp", dataset="posetrack", make_video=0):   
    if(dataset=="posetrack"): data_gt = joblib.load('_DATA/posetrack/gt_data.pickle')     ; base_dir = "_DATA/posetrack/posetrack_data/"
    if(dataset=="mupots"):    data_gt = joblib.load('_DATA/mupots/gt_data.pickle')        ; base_dir = "_DATA/mupots/mupots_data/"
    if(dataset=="ava"):       data_gt = joblib.load('_DATA/ava/gt_data.pickle')           ; base_dir = "_DATA/ava/ava_data/"
        
    data_all              = {}
    total_annoated_frames = 0
    total_detected_frames = 0
    
    if(method=='phalp'):
        for video_ in data_gt.keys():
            try: 
                PHALP_predictions = joblib.load(results_dir + video_ + ".pkl")
            except: 
                continue
            list_of_gt_frames = np.sort(list(data_gt[video_].keys()))
            tracked_frames    = list(PHALP_predictions.keys())
            data_all[video_]  = {}
            for i in range(len(list_of_gt_frames)):
                frame_        = list_of_gt_frames[i]
                total_annoated_frames += 1
                if(frame_ in tracked_frames):
                    tracked_data = PHALP_predictions[frame_]
                    if(len(data_gt[video_][frame_][0])>0):
                        if(dataset=="ava"):  assert data_gt[video_][frame_][0][0][0].split("/")[-1] == frame_
                        else:                assert data_gt[video_][frame_][0][0].split("/")[-1]    == frame_
                        if(len(tracked_data[0])==0):   
                            data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], [], [], []]
                        else:
                            data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], frame_, tracked_data[0], tracked_data[1]] 
                            total_detected_frames   += 1
                else:
                    data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], [], [], []]; print("Error!")
                    
    print("Total annoated frames ", total_annoated_frames)
    print("Total detected frames ", total_detected_frames)
    joblib.dump(data_all, results_dir + '/'+str(dataset)+'_'+str(method)+'.pkl')        
        
    # #########################################################################################################
    # #########################################################################################################
    # ###############################             Evaluate             #######################################
    # #########################################################################################################
    # #########################################################################################################
  
    use_hota     = True

    accumulators = []   
    TOTAL_AssA   = []
    TOTAL_DetA   = []
    TOTAL_HOTA   = []
    for video in tqdm(list(data_all.keys())):
        
        
        acc = mm.MOTAccumulator(auto_id=True)
        accumulators.append(acc)

        
        ############# HOTA evaluation code
        if(use_hota):
            T                = len(data_all[video].keys())
            gt_ids_hota      = np.zeros((T, 500))
            pr_ids_hota      = np.zeros((T, 500))
            similarity_hota  = np.zeros((T, 500, 500))
            gt_available     = []   
            hota_metric      = trackeval.metrics.HOTA()
            start_           = 0
        
        list_of_predictions  = []
        for t, frame in enumerate(data_all[video].keys()):
            data        = data_all[video][frame]
            pt_ids      = data[5]
            for p_ in pt_ids:
                list_of_predictions.append(p_)
        list_of_predictions = np.unique(list_of_predictions)        
            
        for t, frame in enumerate(data_all[video].keys()):

            data = data_all[video][frame]

            gt_ids      = data[1]
            gt_ids_new  = data[3]
            gt_bbox     = data[2]
            pt_ids_      = data[5]
            pt_bbox_     = data[6]
            
            pt_ids  = []
            pt_bbox = []
            for p_, b_ in zip(pt_ids_, pt_bbox_):
                loc= np.where(list_of_predictions==p_)[0][0]
                pt_ids.append(loc)
                pt_bbox.append(b_)

            if(len(gt_ids_new)>0):
                
                cost_ = mm.distances.iou_matrix(gt_bbox, pt_bbox, max_iou=0.99)
                
                accumulators[-1].update(
                                                    gt_ids_new,  # Ground truth objects in this frame
                                                    pt_ids,      # Detector hypotheses in this frame
                                                    cost_
                        )

                cost_[np.isnan(cost_)] = 1
                ############# HOTA evaluation code
                if(use_hota):
                    gt_available.append(t)
                    for idx_gt in gt_ids_new:
                        gt_ids_hota[t][idx_gt] = 1

                    for idx_pr in pt_ids:
                        pr_ids_hota[t][idx_pr] = 1

                    for i, idx_gt in enumerate(gt_ids_new):
                        for j, idx_pr in enumerate(pt_ids):
                            similarity_hota[t][idx_gt][idx_pr] = 1-cost_[i][j]

        if(use_hota):
            gt_ids_hota     = gt_ids_hota[gt_available, :]
            pr_ids_hota     = pr_ids_hota[gt_available, :]
            similarity_hota = similarity_hota[gt_available, :]

            data = _from_dense(
                    num_timesteps  =  len(gt_available),
                    num_gt_ids     =  np.sum(np.sum(gt_ids_hota, 0)>0),
                    num_tracker_ids=  np.sum(np.sum(pr_ids_hota, 0)>0),
                    gt_present     =  gt_ids_hota,
                    tracker_present=  pr_ids_hota,
                    similarity     =  similarity_hota,
            )
            
            results = hota_metric.eval_sequence(data)    
            TOTAL_AssA.append(np.mean(results['AssA']))
            TOTAL_DetA.append(np.mean(results['DetA']))
            TOTAL_HOTA.append(np.mean(results['HOTA']))



    mh = mm.metrics.create()

    summary = mh.compute_many(
        accumulators,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True
    )

    ID_switches = summary['num_switches']['OVERALL']
    MOTA        = summary['mota']['OVERALL']
    PRCN        = summary['precision']['OVERALL']
    RCLL        = summary['recall']['OVERALL']

    strsummary  = mm.io.render_summary(
        summary,
        formatters = mh.formatters,
        namemap    = mm.io.motchallenge_metric_names
    )

    results_ID_switches       = summary['num_switches']['OVERALL']
    results_mota              = summary['mota']['OVERALL']
    
    print(strsummary)
    print("ID switches  ", results_ID_switches)
    print("MOTA         ", results_mota)
    
    if(use_hota):
        results_AssA              = np.mean(TOTAL_AssA)
        results_DetA              = np.mean(TOTAL_DetA)
        results_HOTA              = np.mean(TOTAL_HOTA)    
        print("AssA         ", results_AssA)
        print("DetA         ", results_DetA)
        print("HOTA         ", results_HOTA)
    
    
    text_file = open(results_dir + '/str_summary.txt', "w")
    n = text_file.write(strsummary)
    text_file.close()

    
    return strsummary, summary, TOTAL_AssA, TOTAL_HOTA
    
            
        
if __name__ == '__main__':
    
    results_dir = str(sys.argv[1])
    method      = str(sys.argv[2])
    dataset     = str(sys.argv[3])
    strsummary, summary, TOTAL_AssA, TOTAL_HOTA = evaluate_trackers(results_dir, method=method, dataset=dataset, make_video=0)
