import os
import glob
import joblib
import itertools

import numpy as np
import pandas as pd

from .associate import associate_frame
from .tools import EGOBODY_ROOT


"""
Script to find the associations of ground truth Egobody tracks with the detected PHALP tracks
Will write a job specification file to ../job_specs with which track IDs to run optimization on
"""

IMG_ROOT = f"{EGOBODY_ROOT}/egocentric_color"
PHALP_DIR = f"{EGOBODY_ROOT}/slahmr/phalp_out/results"


def load_split_sequences(split):
    split_file = "{EGOBODY_ROOT}/data_splits.csv"
    df = pd.read_csv(split_file)
    if split not in df.columns:
        print(f"{split} not in {split_file}")
        return []
    return df[split].dropna().tolist()


def get_egobody_keypoints(img_dir, start, end):
    kp_file = f"{img_dir}/keypoints.npz"
    valid_file = f"{img_dir}/valid_frame.npz"
    img_paths = sorted(glob.glob(f"{img_dir}/PV/*.jpg"))[start:end]
    img_names = [os.path.basename(x) for x in img_paths]

    kp_dict = {}
    valid_dict = {}
    kp_data = np.load(kp_file)
    valid_data = np.load(valid_file)

    zeros = np.zeros_like(kp_data["keypoints"][0])
    for img_path, kps in zip(kp_data["imgname"], kp_data["keypoints"]):
        img_name = os.path.basename(img_path)
        kp_dict[img_name] = kps

    for img_path, valid in zip(valid_data["imgname"], valid_data["valid"]):
        img_name = os.path.basename(img_path)
        valid_dict[img_name] = valid

    kps = np.stack([kp_dict.get(name, zeros) for name in img_names], axis=0)
    valid = np.stack([valid_dict.get(name, False) for name in img_names], axis=0)
    return kps, valid


def select_phalp_tracks(seq_name, img_dir, start, end, debug=False):
    """
    Get the best phalp track for each GT person for each frame
    Returns all phalp tracks that match GT over sequence
    """
    phalp_file = f"{PHALP_DIR}/{seq_name}.pkl"
    track_data = joblib.load(phalp_file)
    img_names = sorted(track_data.keys())
    sel_imgs = img_names[start:end]

    kps_all, valid = get_egobody_keypoints(img_dir, start, end)
    T = len(kps_all)
    assert len(sel_imgs) == T, f"found {len(sel_imgs)} frames, expected {T}"

    track_ids = set()
    for frame in sel_imgs:
        frame_data = track_data[frame]
        for tid in frame_data["tracked_ids"]:
            track_ids.add(str(tid))
    track_ids = list(track_ids)
    M = len(track_ids)
    track_idcs = {tid: m for m, tid in enumerate(track_ids)}

    # get the best matching PHALP track for each GT person
    sel_tracks = set()
    for t, frame_name in enumerate(sel_imgs):
        frame_data = track_data[frame_name]
        # get the best track ID for the GT person
        tid = associate_frame(frame_data, kps_all[t], track_ids, debug=debug)
        if tid == -1:
            continue
        sel_tracks.add(int(tid))
    return list(sel_tracks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--prefix", default="ego")
    args = parser.parse_args()

    seqs = load_split_sequences(args.split)

    job_arg_strs = []
    for seq in seqs:
        img_dir = glob.glob(f"{IMG_ROOT}/{seq}/**/")[0]
        num_imgs = len(glob.glob(f"{img_dir}/PV/*.jpg"))
        splits = list(range(0, num_imgs, args.seq_len))
        splits[-1] = num_imgs  # just add the remainder to the last job
        for start, end in zip(splits[:-1], splits[1:]):
            sel_tracks = select_phalp_tracks(seq, img_dir, start, end)
            if len(sel_tracks) < 1:
                continue
            track_str = "-".join([f"{tid:03d}" for tid in sel_tracks])
            arg_str = (
                f"{seq} data.start_idx={start} data.end_idx={end} "
                f"data.track_ids={track_str}"
            )
            print(arg_str)
            job_arg_strs.append(arg_str)

    with open(
        f"../job_specs/{args.prefix}_{args.split}_len_{args.seq_len}_tracks.txt", "w"
    ) as f:
        f.write("\n".join(job_arg_strs))
