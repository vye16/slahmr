import os
import glob
import json
import joblib
import numpy as np
import torch

from slahmr.data.tools import read_keypoints


def associate_phalp_track_dirs(
    phalp_dir, img_dir, track_ids, gt_kps, start=0, end=-1, debug=False
):
    """
    Associate the M track_ids with G GT tracks
    returns (M, T) array of best matching GT person index
    :param phalp_dir (str) directory with phalp track folders
    :param img_dir (str) directory with source images
    :param track ids (list) M tracks to match
    :param gt_kps (G, T, J, 3) gt keypoints for G people, T times, J joints
    :param start (optional int default 0)
    :param end (optional int default -1)
    """
    img_names = sorted([os.path.splitext(x)[0] for x in os.listdir(img_dir)])
    N = len(img_names)
    end = N + 1 + end if end < 0 else end
    sel_imgs = img_names[start:end]
    G, T = gt_kps.shape[:2]  # G num people, T num frames
    assert len(sel_imgs) == T, f"found {len(sel_imgs)} frames, expected {T}"

    track_ids = [f"{int(tid):03d}" for tid in track_ids]
    M = len(track_ids)
    # find the best matching GT track for each PHALP track
    match_idcs = torch.full((M, T), -1)
    for t, frame_name in enumerate(sel_imgs):
        track_kps = []  # get track keypoints
        for tid in track_ids:
            kp_path = f"{phalp_dir}/{tid}/{frame_name}_keypoints.json"
            track_kps.append(read_keypoints(kp_path))
        track_kps = np.stack(track_kps, axis=0)  # (M, 25, 3)
        for g in range(G):
            kp_gt = gt_kps[g, t].T.numpy()  # (18, 3)
            m = associate_keypoints(kp_gt, track_kps, debug=debug)
            if m == -1:
                continue
            match_idcs[m, t] = g
    return match_idcs


def associate_phalp_track_data(
    phalp_file, track_ids, gt_kps, start=0, end=-1, debug=False
):
    """
    Get the best GT person for each phalp track
    :param phalp_file (path) to phalp result pickle file
    :param gt_kps (G, T, 3, 18) gt keypoints
    :param track_ids (list) of phalp track ids
    :param start (optional int)
    :param end (optional int)
    return (M, T) array the matching GT person index for each phalp track
    """
    data = joblib.load(phalp_file)
    img_names = sorted(data.keys())
    N = len(img_names)  # number of frames
    end = N + 1 + end if end < 0 else end
    sel_imgs = img_names[start:end]

    G, T = gt_kps.shape[:2]  # G num people, T num frames
    assert len(sel_imgs) == T, f"found {len(sel_imgs)} frames, expected {T}"

    M = len(track_ids)
    track_idcs = {tid: m for m, tid in enumerate(track_ids)}
    # get the best matching GT track for each PHALP track
    match_idcs = torch.full((M, T), -1)
    for t, frame_name in enumerate(sel_imgs):
        frame_data = data[frame_name]
        for g in range(G):
            kp_gt = gt_kps[g, t].T.numpy()  # (18, 3)
            # get the best track ID for the GT person
            tid = associate_frame_dict(frame_data, kp_gt, track_ids, debug=debug)
            if tid == -1:
                continue
            m = track_idcs[tid]
            match_idcs[m, t] = g
    return match_idcs


def associate_keypoints(gt_kps, track_kps, debug=False):
    """
    :param gt_bbox (25, 3)
    :param track_bboxes (M, 25, 3)
    return the index of the best overlapping track bbox
    """
    gt_kps = gt_kps[gt_kps[:, 2] > 0, :2]
    if len(gt_kps) < 1:
        return -1
    bb_min, bb_max = gt_kps.min(axis=0), gt_kps.max(axis=0)
    gt_bbox = np.concatenate([bb_min, bb_max], axis=-1)  # (4,)

    track_kps = track_kps[..., :2]  # (M, 25, 2)
    track_min, track_max = track_kps.min(axis=1), track_kps.max(axis=1)
    track_bboxes = np.concatenate([track_min, track_max], axis=-1)  # (M, 4)

    ious = np.stack([compute_iou(bb, gt_bbox)[0] for bb in track_bboxes], axis=0)
    return np.argmax(ious)


def associate_frame_dict(frame_data, gt_kps, track_ids, debug=False):
    """
    For the GT keypoints, find the PHALP track in track_ids with best overlap
    :param frame_data (dict) PHALP output data
    :param gt_kps (25, 3)
    :param track_ids (list of N) PHALP track ids to search over
    return the id in track_ids with the biggest overlap with gt_kps
    """
    gt_kps = gt_kps[gt_kps[:, 2] > 0, :2]
    if len(gt_kps) < 1:
        return -1
    bb_min, bb_max = gt_kps.min(axis=0), gt_kps.max(axis=0)
    gt_bbox = np.concatenate([bb_min, bb_max], axis=-1)  # (4,)

    # use strs for track ids
    tid_strs = [str(tid) for tid in track_ids]
    # get the list indices of the PHALP tracks
    track_idcs = {
        str(int(tid)): i
        for i, tid in enumerate(frame_data["tid"])
        if tid in frame_data["tracked_ids"]
    }
    # select the track with the biggest overlap with the gt kps
    ious = []
    for tid in track_ids:
        if tid not in track_idcs:
            ious.append(0)
            continue
        bb = frame_data["bbox"][track_idcs[tid]]  # (min_x, min_y, w, h)
        bbox = np.concatenate([bb[:2], bb[:2] + bb[2:]], axis=-1)
        iou = compute_iou(bbox, gt_bbox)[0]
        ious.append(iou)
    ious = np.stack(ious, axis=0)
    idx = np.argmax(ious)
    if debug:
        print(track_ids[idx], track_ids, ious)
    return track_ids[idx]


def compute_iou(bb1, bb2):
    """
    :param bb1 (..., 4) top left x, y bottom right x y
    :param bb2 (..., 4) top left x, y bottom right x y
    return (...) IOU
    """
    x11, y11, x12, y12 = np.split(bb1, 4, axis=-1)
    x21, y21, x22, y22 = np.split(bb2, 4, axis=-1)
    x1 = np.maximum(x11, x21)
    y1 = np.maximum(y11, y21)
    x2 = np.minimum(x12, x22)
    y2 = np.minimum(y12, y22)
    intersect = np.maximum((x2 - x1) * (y2 - y1), 0)
    union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - intersect
    return intersect / (union + 1e-6)
