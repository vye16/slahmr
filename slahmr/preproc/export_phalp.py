import os
import joblib
import json

import cv2
import numpy as np


def export_phalp_predictions(data_dir, seq, phalp_out, pred_out):
    """
    exports phalp output results into a directory for each track
    of JSON files with each frame's SMPL prediction
    """
    res_file = os.path.join(data_dir, phalp_out, f"{seq}.pkl")
    pred_dir = os.path.join(data_dir, pred_out, seq)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"exporting phalp predictions from {res_file} to {pred_dir}")

    tracklet_data = joblib.load(res_file)
    for frame, track_data in tracklet_data.items():
        name = os.path.splitext(frame)[0]
        track_dicts = unpack_frame(track_data)
        for tid, pred_dict in track_dicts.items():
            track_dir = os.path.join(pred_dir, f"{tid:03d}")
            os.makedirs(track_dir, exist_ok=True)
            pred_path = os.path.join(track_dir, f"{name}_smpl.json")
            with open(pred_path, "w") as f:
                json.dump(pred_dict, f, indent=1)


def export_vitpose_keypoints(data_dir, seq, phalp_out, kp_out):
    """
    exports each track of each frame into its own json of keypoints
    """
    res_file = os.path.join(data_dir, phalp_out, f"{seq}.pkl")
    kp_dir = os.path.join(data_dir, kp_out, seq)
    os.makedirs(kp_dir, exist_ok=True)
    print(f"exporting keypoints from {res_file} to {kp_dir}")

    tracklet_data = joblib.load(res_file)
    for frame, track_data in tracklet_data.items():
        name = os.path.splitext(frame)[0]
        tids = track_data["tid"]
        valid_tids = track_data["tracked_ids"]
        kps_all = track_data["vitpose"]
        mask_paths = track_data["mask_name"]
        for i, tid in enumerate(tids):
            if tid not in valid_tids:
                continue
            track_dir = os.path.join(kp_dir, f"{tid:03d}")
            os.makedirs(track_dir, exist_ok=True)
            kp_path = os.path.join(track_dir, f"{name}_keypoints.json")
            kp_dict = {
                "people": [
                    {
                        "pose_keypoints_2d": kps_all[i].tolist(),
                        "mask_path": mask_paths[i],
                    }
                ],
            }
            with open(kp_path, "w") as f:
                json.dump(kp_dict, f, indent=1)


def export_shot_changes(data_dir, seq, phalp_out, out_name):
    """
    Exports the phalp output of shot changes into a JSON file
    dict with the shot index for each frame in the sequence
    """
    res_file = os.path.join(data_dir, phalp_out, f"{seq}.pkl")
    out_dir = os.path.join(data_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq}.json")
    print(f"exporting shot changes to {out_path}")

    tracklet_data = joblib.load(res_file)
    frames = sorted(tracklet_data.keys())
    shots = np.cumsum([tracklet_data[frame]["shot"] for frame in frames])
    shots = shots.astype(int).tolist()
    shot_dict = {frame: shot for frame, shot in zip(frames, shots)}
    with open(out_path, "w") as f:
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
        cam = track_data["camera"][i].squeeze()  # (3,)
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


def export_sequence_results(
    data_dir,
    seq,
    res_name="phalp_out/results",
    track_name="track_preds",
    shot_name="shot_idcs",
):
    export_phalp_predictions(data_dir, seq, res_name, track_name)
    export_vitpose_keypoints(data_dir, seq, res_name, track_name)
    export_shot_changes(data_dir, seq, res_name, shot_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--res_name", default="phalp_out/results")
    parser.add_argument("--track_name", default="track_preds")
    parser.add_argument("--shot_name", default="shot_idcs")
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("-j", "--n_workers", type=int, default=8)

    args = parser.parse_args()
    in_dir = os.path.join(args.data_dir, args.name_in)
    seqs_all = sorted([os.path.splitext(x)[0] for x in os.listdir(in_dir)])
    print(f"Exporting {len(seqs_all)} sequences")

    for seq in seqs_all:
        export_sequence_results(
            args.data_dir, seq, args.res_name, args.track_name, args.shot_name
        )
