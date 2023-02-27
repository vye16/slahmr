"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import
import torch
import numpy as np
from . import linear_assignment
from .track import Track

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, opt, metric, max_age=30, n_init=3, phalp_tracker=None, dims=None):
        self.opt              = opt
        self.metric           = metric
        self.max_age          = max_age
        self.n_init           = n_init
        self.tracks           = []
        self._next_id         = 1
        self.tracked_cost     = {}
        self.phalp_tracker    = phalp_tracker
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.phalp_tracker, increase_age=True)

    def update(self, detections, frame_t, image_name, shot):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        matches, unmatched_tracks, unmatched_detections, statistics = self._match(detections)
        self.tracked_cost[frame_t] = [statistics[0], matches, unmatched_tracks, unmatched_detections, statistics[1], statistics[2], statistics[3], statistics[4]] 
        if(self.opt.verbose): print(np.round(np.array(statistics[0]), 2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], detection_idx, shot)
        self.accumulate_vectors([i[0] for i in matches], features=self.opt.predict)
 
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.accumulate_vectors(unmatched_tracks, features=self.opt.predict)
    
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], detection_idx)
            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() or t.is_tentative()]
        appe_features, loca_features, pose_features, uv_maps, targets = [], [], [], [], []
        for track in self.tracks:
            if not (track.is_confirmed() or track.is_tentative()): continue
                    
                         
            appe_features += [track.track_data['prediction']['appe'][-1]]
            loca_features += [track.track_data['prediction']['loca'][-1]]
            pose_features += [track.track_data['prediction']['pose'][-1]]
            uv_maps       += [track.track_data['prediction']['uv'][-1]]
            targets       += [track.track_id]
            
            
        self.metric.partial_fit(np.asarray(appe_features), np.asarray(loca_features), np.asarray(pose_features), np.asarray(uv_maps), np.asarray(targets), active_targets)
        
        return matches
        

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            appe_emb          = np.array([dets[i].detection_data['appe'] for i in detection_indices])
            loca_emb          = np.array([dets[i].detection_data['loca'] for i in detection_indices])
            pose_emb          = np.array([dets[i].detection_data['pose'] for i in detection_indices])
            uv_maps           = np.array([dets[i].detection_data['uv'] for i in detection_indices])
            targets           = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix       = self.metric.distance([appe_emb, loca_emb, pose_emb, uv_maps], targets, dims=[self.A_dim, self.P_dim, self.L_dim], phalp_tracker=self.phalp_tracker)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        
        # Associate confirmed tracks using appearance features.
        matches, unmatched_tracks, unmatched_detections, cost_matrix = linear_assignment.matching_simple(gated_metric, self.metric.matching_threshold, self.max_age, self.tracks, detections, confirmed_tracks)


        track_gt   = [t.track_data['history'][-1]['ground_truth'] for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_gt  = [d.detection_data['ground_truth'] for i, d in enumerate(detections)]

        track_idt  = [i for i, t in enumerate(self.tracks) if t.is_confirmed() or t.is_tentative()]
        detect_idt = [i for i, d in enumerate(detections)]
        
        if(self.opt.use_gt): 
            matches = []
            for t_, t_gt in enumerate(track_gt):
                for d_, d_gt in enumerate(detect_gt):
                    if(t_gt==d_gt): matches.append([t_, d_])
            t_pool = [t_ for (t_, _) in matches]
            d_pool = [d_ for (_, d_) in matches]
            unmatched_tracks     = [t_ for t_ in track_idt if t_ not in t_pool]
            unmatched_detections = [d_ for d_ in detect_idt if d_ not in d_pool]
            return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]
        
        return matches, unmatched_tracks, unmatched_detections, [cost_matrix, track_gt, detect_gt, track_idt, detect_idt]

    
    
    def _initiate_track(self, detection, detection_id):
        new_track = Track(self.opt, self._next_id, self.n_init, self.max_age, 
                          detection_data=detection.detection_data, 
                          detection_id=detection_id, 
                          dims=[self.A_dim, self.P_dim, self.L_dim])
        new_track.add_predicted()
        self.tracks.append(new_track)
        self._next_id += 1

                
                
                
                
    def accumulate_vectors(self, track_ids, features="APL"):

        a_features = []; p_features = []; l_features = []; t_features = []; l_time     = []; confidence = []; is_tracks  = 0; p_data = []
        for track_idx in track_ids:
            t_features.append([self.tracks[track_idx].track_data['history'][i]['time'] for i in range(self.opt.track_history)])
            l_time.append(self.tracks[track_idx].time_since_update)
                
            if("L" in features):  l_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['loca'] for i in range(self.opt.track_history)]))
            if("P" in features):  p_features.append(np.array([self.tracks[track_idx].track_data['history'][i]['pose'] for i in range(self.opt.track_history)]))
            if("P" in features):  t_id = self.tracks[track_idx].track_id; p_data.append([[data['xy'][0], data['xy'][1], data['scale'], data['scale'], data['time'], t_id] for data in self.tracks[track_idx].track_data['history']])
            if("L" in features):  confidence.append(np.array([self.tracks[track_idx].track_data['history'][i]['conf'] for i in range(self.opt.track_history)]))
            is_tracks                       = 1

            
        l_time         = np.array(l_time)
        t_features     = np.array(t_features)
        if("P" in features): p_features     = np.array(p_features)
        if("P" in features): p_data         = np.array(p_data)
        if("L" in features): l_features     = np.array(l_features)
        if("L" in features): confidence     = np.array(confidence)
        
        
        if(is_tracks):
            with torch.no_grad():
                if("P" in features): p_pred = self.phalp_tracker.forward_for_tracking([p_features, p_data, t_features], "P", l_time)
                if("L" in features): l_pred = self.phalp_tracker.forward_for_tracking([l_features, t_features, confidence], "L", l_time)    
                
            for p_id, track_idx in enumerate(track_ids):
                self.tracks[track_idx].add_predicted(pose=p_pred[p_id] if("P" in features) else None, 
                                                     loca=l_pred[p_id] if("L" in features) else None)
                
        