import os
import joblib
import json

import cv2
import numpy as np


def export_phalp_predictions(res_path, target_dir):
    """
    exports phalp output results into a directory for each track
    of JSON files with each frame's SMPL prediction
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"exporting phalp predictions from {res_path} to {target_dir}")

    tracklet_data = joblib.load(res_path)
    for frame, track_data in tracklet_data.items():
        name = os.path.splitext(frame.split("/")[-1])[0]
        track_dicts = unpack_frame(track_data)
        for tid, pred_dict in track_dicts.items():
            track_dir = os.path.join(target_dir, f"{tid:03d}")
            os.makedirs(track_dir, exist_ok=True)
            pred_path = os.path.join(track_dir, f"{name}_smpl.json")
            with open(pred_path, "w") as f:
                json.dump(pred_dict, f, indent=1)


def export_vitpose_keypoints(res_path, target_dir):
    """
    exports each track of each frame into its own json of keypoints
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"exporting keypoints from {res_path} to {target_dir}")

    tracklet_data = joblib.load(res_path)
    for frame, track_data in tracklet_data.items():
        name = os.path.splitext(frame.split("/")[-1])[0]
        tids = track_data["tid"]
        valid_tids = track_data["tracked_ids"]
        kps_all = track_data["extra_data"]
        # mask_paths = track_data["mask_name"]
        for i, tid in enumerate(tids):
            if tid not in valid_tids:
                continue
            track_dir = os.path.join(target_dir, f"{tid:03d}")
            os.makedirs(track_dir, exist_ok=True)
            kp_path = os.path.join(track_dir, f"{name}_keypoints.json")
            kp_dict = {
                "people": [
                    {
                        "pose_keypoints_2d": kps_all[i].tolist(),
                        # "mask_path": mask_paths[i],
                    }
                ],
            }
            with open(kp_path, "w") as f:
                json.dump(kp_dict, f, indent=1)


def export_shot_changes(res_path, target_path):
    """
    Exports the phalp output of shot changes into a JSON file
    dict with the shot index for each frame in the sequence
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"exporting shot changes to {target_path}")

    tracklet_data = joblib.load(res_path)
    frames = sorted(tracklet_data.keys())
    shots = np.cumsum([tracklet_data[frame]["shot"] for frame in frames])
    shots = shots.astype(int).tolist()
    shot_dict = {frame.split("/")[-1]: shot for frame, shot in zip(frames, shots)}
    with open(target_path, "w") as f:
        json.dump(shot_dict, f, indent=1)


def unpack_frame(track_data):
    """
    Unpack one frame of track data from the phalp output into JSON files for each track.
    Track data contains body pose, global translation and rotation,
    and keypoints for all tracks in the frame.
    Each track is unpacked into folders with a JSON for each frame.
    """
    out_dict = {}
    for i, tid in enumerate(track_data["tid"]):
        if tid not in track_data["tracked_ids"]:
            continue
        cam = track_data["camera_bbox"][i].squeeze()  # (3,)
        H, W = track_data["size"][i]
        focal = 0.5 * (H + W)

        # every tracklet has its own bounding box
        cx, cy = track_data["center"][i]  # (2,)
        bbox = track_data["bbox"][i]  # (4,)
        scale = max(bbox[-2:])

        tz = 2 * focal / (scale * cam[0] + 1e-6)
        tx = cam[1] + tz / focal * (cx - W / 2)
        ty = cam[2] + tz / focal * (cy - H / 2)
        cam_trans = np.array([tx, ty, tz])

        smpl = track_data["smpl"][i]
        betas = smpl["betas"].squeeze()  # (10,)
        body_pose = smpl["body_pose"].squeeze()  # (23, 3, 3)
        body_pose_aa = np.stack(
            [cv2.Rodrigues(x)[0].squeeze() for x in body_pose], axis=0
        )  # (23, 3)
        global_orient = smpl["global_orient"].squeeze()  # (3, 3)
        global_orient_aa = cv2.Rodrigues(global_orient)[0].squeeze()  # (3,)

        out_dict[tid] = {
            "betas": betas.tolist(),
            "body_pose": body_pose_aa.tolist(),
            "global_orient": global_orient_aa.tolist(),
            "cam_trans": cam_trans.tolist(),
        }

    return out_dict


def export_sequence_results(phalp_res_path, track_dir, shot_path):
    export_phalp_predictions(phalp_res_path, track_dir)
    export_vitpose_keypoints(phalp_res_path, track_dir)
    export_shot_changes(phalp_res_path, shot_path)
