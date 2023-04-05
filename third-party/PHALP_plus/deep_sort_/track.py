"""
Modified code from https://github.com/nwojke/deep_sort
"""

import numpy as np
import copy
import scipy.signal as signal
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque
    
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted   = 3


class Track:
    """
    Mark this track as missed (no association at the current time step).
    """

    def __init__(self, opt, track_id, n_init, max_age, detection_data=None, detection_id=None, dims=None):
        self.opt               = opt
        self.track_id          = track_id
        self.hits              = 1
        self.age               = 1
        self.time_since_update = 0
        self.time_init         = detection_data["time"]
        self.state             = TrackState.Tentative            
        
        self._n_init           = n_init
        self._max_age          = max_age
        
        if(dims is not None):
            self.A_dim = dims[0]
            self.P_dim = dims[1]
            self.L_dim = dims[2]
        
        self.track_data        = {"history": deque(maxlen=self.opt.track_history) , "prediction":{}}
        for _ in range(self.opt.track_history):
            self.track_data["history"].append(detection_data)
            
        self.track_data['prediction']['appe'] = deque([detection_data['appe']], maxlen=self.opt.n_init+1)
        self.track_data['prediction']['loca'] = deque([detection_data['loca']], maxlen=self.opt.n_init+1)
        self.track_data['prediction']['pose'] = deque([detection_data['pose']], maxlen=self.opt.n_init+1)
        self.track_data['prediction']['uv']   = deque([copy.deepcopy(detection_data['uv'])], maxlen=self.opt.n_init+1)

    def predict(self, phalp_tracker, increase_age=True):
        if(increase_age):
            self.age += 1; self.time_since_update += 1
            
    def add_predicted(self, appe=None, pose=None, loca=None, uv=None):
        appe_predicted = copy.deepcopy(appe.numpy()) if(appe is not None) else copy.deepcopy(self.track_data['history'][-1]['appe'])
        loca_predicted = copy.deepcopy(loca.numpy()) if(loca is not None) else copy.deepcopy(self.track_data['history'][-1]['loca'])
        pose_predicted = copy.deepcopy(pose.numpy()) if(pose is not None) else copy.deepcopy(self.track_data['history'][-1]['pose'])
        
        self.track_data['prediction']['appe'].append(appe_predicted)
        self.track_data['prediction']['loca'].append(loca_predicted)
        self.track_data['prediction']['pose'].append(pose_predicted)

    def update(self, detection, detection_id, shot):             

        self.track_data["history"].append(copy.deepcopy(detection.detection_data))
        if(shot==1): 
            for tx in range(self.opt.track_history):
                self.track_data["history"][-1-tx]['loca'] = copy.deepcopy(detection.detection_data['loca'])

        if("T" in self.opt.predict):
            mixing_alpha_                      = self.opt.alpha*(detection.detection_data['conf']**2)
            ones_old                           = self.track_data['prediction']['uv'][-1][3:, :, :]==1
            ones_new                           = self.track_data['history'][-1]['uv'][3:, :, :]==1
            ones_old                           = np.repeat(ones_old, 3, 0)
            ones_new                           = np.repeat(ones_new, 3, 0)
            ones_intersect                     = np.logical_and(ones_old, ones_new)
            ones_union                         = np.logical_or(ones_old, ones_new)
            good_old_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_old)
            good_new_ones                      = np.logical_and(np.logical_not(ones_intersect), ones_new)
            new_rgb_map                        = np.zeros((3, 256, 256))
            new_mask_map                       = np.zeros((1, 256, 256))-1
            new_mask_map[ones_union[:1, :, :]] = 1.0
            new_rgb_map[ones_intersect]        = (1-mixing_alpha_)*self.track_data['prediction']['uv'][-1][:3, :, :][ones_intersect] + mixing_alpha_*self.track_data['history'][-1]['uv'][:3, :, :][ones_intersect]
            new_rgb_map[good_old_ones]         = self.track_data['prediction']['uv'][-1][:3, :, :][good_old_ones] 
            new_rgb_map[good_new_ones]         = self.track_data['history'][-1]['uv'][:3, :, :][good_new_ones] 
            self.track_data['prediction']['uv'].append(np.concatenate((new_rgb_map , new_mask_map), 0))
        else:
            self.track_data['prediction']['uv'].append(self.track_data['history'][-1]['uv'])
            
        
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def smooth_bbox(self, bbox):
        kernel_size = 5
        sigma       = 3
        bbox        = np.array(bbox)
        smoothed    = np.array([signal.medfilt(param, kernel_size) for param in bbox.T]).T
        out         = np.array([gaussian_filter1d(traj, sigma) for traj in smoothed.T]).T
        return list(out)