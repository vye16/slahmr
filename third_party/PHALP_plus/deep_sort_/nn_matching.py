"""
Modified code from https://github.com/nwojke/deep_sort
"""

import numpy as np
import copy

def _pdist_l2(a, b):
    """Compute pair-wise squared l2 distances between points in `a` and `b`.""" 
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))

    return r2

def _pdist(opt, a, b, dims, phalp_tracker):

    a_appe, a_loca, a_pose, a_uv        = [], [], [], []
    for i_ in range(len(a)):
        a_appe.append(a[i_][0]); 
        a_loca.append(a[i_][1]); 
        a_pose.append(a[i_][2]); 
        a_uv.append(a[i_][3])

    b_appe, b_loca, b_pose, b_uv    = b[0], b[1], b[2], b[3]
    a_uv, b_uv      = np.asarray(a_uv),   copy.deepcopy(np.asarray(b_uv))
    a_appe, b_appe  = np.asarray(a_appe), copy.deepcopy(np.asarray(b_appe))
    a_loca, b_loca  = np.asarray(a_loca), copy.deepcopy(np.asarray(b_loca))
    a_pose, b_pose  = np.asarray(a_pose), copy.deepcopy(np.asarray(b_pose))
 
    track_pose      = a_pose
    detect_pose     = b_pose
    pose_distance   = _pdist_l2(track_pose[:, 2048:], detect_pose[:, 2048:])
    

    track_location  = np.reshape(a_loca[:, :-9], (-1, 45, 2))
    detect_location = np.reshape(b_loca[:, :-9], (-1, 45, 2))
    track_loc       = track_location[:, 44, :]
    detect_loc      = detect_location[:, 44, :]
    loc_distance    = np.sqrt(_pdist_l2(track_loc, detect_loc))

    al              = np.reshape(a_loca[:, -9:], (-1, 3, 3))
    bl              = np.reshape(b_loca[:, -9:], (-1, 3, 3))
    c_x             = al[:, [1], 0]
    c_y             = al[:, [1], 1]
    c_xy            = np.sqrt((c_x**2+c_y**2))
    c_xy            = np.tile(c_xy, (1, len(b_appe)))
    
    al_nearness     = al[:, [0], -1]
    bl_nearness     = bl[:, [0], -1]
    cl_nearness     = al[:, [1], -1]
    nearness        = np.matmul(al_nearness, 1.0/bl_nearness.T)
    loc_            = nearness>1
    nearness[loc_]  = 1.0/nearness[loc_]
    n_log           = -np.log(nearness)
    cn_log          = np.abs(np.log(1.0/cl_nearness)) 
    c_nlog          = np.tile(cn_log, (1, len(b_appe)))

    r_texture = np.zeros((len(a_appe), len(b_appe)))
    if("T" in opt.predict):
        for ix in range(len(a_appe)):
            for iy in range(len(b_appe)):
                accc = copy.deepcopy(a_uv[ix])
                bccc = copy.deepcopy(b_uv[iy])
                xu, yu, cu = phalp_tracker.get_uv_distance(accc, bccc)
                r_texture[ix, iy]  = np.sum((xu-yu)**2)*100
                if(opt.distance_type=="EQ_010x"): 
                    r_texture[ix, iy] *= np.exp((1-cu)/2)
            
    elif("A" in opt.predict):
        track_appe      = a_appe/10**3
        detect_appe     = b_appe/10**3
        r_texture       = _pdist_l2(track_appe, detect_appe)
        
    if(opt.distance_type=="A0"): return r_texture
    if(opt.distance_type=="P0"): return pose_distance
    if(opt.distance_type=="L0"): return loc_distance
    if(opt.distance_type=="LC"): return c_xy
    if(opt.distance_type=="N0"): return n_log
    if(opt.distance_type=="NC"): return c_nlog
    
    if(opt.distance_type=="EQ_010"):  
        betas     = [3.8303, 1.5207, 0.4930, 4.5831]; c=1
        pose_distance[pose_distance>1.2] = 1.2
        if(opt.shot==1): 
            # For best performance under the shot change scenario
            # you can set look_back to 20.
            #betas           = [7.8303, 1.5207, 1e100, 1e100]; c=1
            # treat each shot independently
            betas           = [100000*7.8303, 100000*1.5207, 1e100, 1e100]; c=1 
            track_appe      = a_appe/10**3
            detect_appe     = b_appe/10**3
            r_texture       = _pdist_l2(track_appe, detect_appe)
   
    xy_cxy_distance     = loc_distance/(0.1 + c*np.tanh(c_xy))/betas[2]
    n_cn_log_distance   = n_log/(0.1 + c*np.tanh(c_nlog))/betas[3]
    ruv2                = (1+r_texture*betas[0]) * (1+pose_distance*betas[1]) * np.exp(xy_cxy_distance) * np.exp(n_cn_log_distance) 
        
    return ruv2
    
def _nn_euclidean_distance_min(opt, x, y, dims, phalp_tracker):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray./
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances_a = _pdist(opt, x, y, dims, phalp_tracker)
    return np.maximum(0.0, distances_a.min(axis=0))


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, opt, matching_threshold, budget=None):
        
        self.opt                = opt
        self._metric            = _nn_euclidean_distance_min
        self.matching_threshold = matching_threshold
        self.budget             = budget
        self.samples            = {}
        
        
    def partial_fit(self, appe_features, loca_features, pose_features, uv_maps, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        for appe_feature, loca_feature, pose_feature, uv_map, target in zip(appe_features, loca_features, pose_features, uv_maps, targets):
            self.samples.setdefault(target, []).append([appe_feature, loca_feature, pose_feature, uv_map])
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, detection_features, targets, dims=None, phalp_tracker=None):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        cost_matrix_a = np.zeros((len(targets), len(detection_features[0])))
        for i, target in enumerate(targets):
            cost_matrix_a[i, :] = self._metric(self.opt, self.samples[target], detection_features, dims, phalp_tracker)
        
        return cost_matrix_a
