from __future__ import absolute_import

import numpy as np


def normalize(x, dim=-1):
    norm1 = x / np.linalg.norm(x, axis=dim, keepdims=True)
    return norm1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def _from_dense(num_timesteps, num_gt_ids, num_tracker_ids, gt_present, tracker_present, similarity):
    gt_subset = [np.flatnonzero(gt_present[t, :]) for t in range(num_timesteps)]
    tracker_subset = [np.flatnonzero(tracker_present[t, :]) for t in range(num_timesteps)]
    similarity_subset = [
            similarity[t][gt_subset[t], :][:, tracker_subset[t]]
            for t in range(num_timesteps)
    ]
    data = {
            'num_timesteps': num_timesteps,
            'num_gt_ids': num_gt_ids,
            'num_tracker_ids': num_tracker_ids,
            'num_gt_dets': np.sum(gt_present),
            'num_tracker_dets': np.sum(tracker_present),
            'gt_ids': gt_subset,
            'tracker_ids': tracker_subset,
            'similarity_scores': similarity_subset,
    }
    return data
