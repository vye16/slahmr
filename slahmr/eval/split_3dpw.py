import os
import glob
import itertools
import joblib

from .eval_3dpw import load_3dpw_params
from .associate import associate_frame
from .tools import TDPW_ROOT


"""
Script to find the associations of ground truth 3DPW tracks with the detected PHALP tracks
Will write a job specification file to ../job_specs with which track IDs to run optimization on
"""

IMG_ROOT = f"{TDPW_ROOT}/imageFiles"
SRC_DIR = f"{TDPW_ROOT}/sequenceFiles"
PHALP_DIR = f"{TDPW_ROOT}/slahmr/phalp_out/results"


def load_split_sequences(split):
    assert split in ["train", "val", "test"]
    split_dir = f"{SRC_DIR}/{split}"
    seq_files = sorted(os.listdir(split_dir))
    return [os.path.splitext(f)[0] for f in seq_files]


def select_phalp_tracks(seq_name, split, start, end, debug=False):
    """
    Select the best phalp track for each GT person for each frame.
    Returns all phalp tracks that match GT over sequence
    """
    phalp_file = f"{PHALP_DIR}/{seq_name}.pkl"
    track_data = joblib.load(phalp_file)
    img_names = sorted(track_data.keys())
    sel_imgs = img_names[start:end]

    gt_params = load_3dpw_params(f"{SRC_DIR}/{split}/{seq_name}.pkl", start, end)
    gt_kps = gt_params["keypts2d"]
    G, T = gt_kps.shape[:2]  # G num people, T num frames
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
        for g in range(G):
            kp_gt = gt_kps[g, t].T.numpy()  # (18, 3)
            # get the best track ID for the GT person
            tid = associate_frame(frame_data, kp_gt, track_ids, debug=debug)
            if tid == -1:
                continue
            sel_tracks.add(int(tid))
    return list(sel_tracks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test", "all"]
    )
    parser.add_argument("--prefix", default="3dpw")
    args = parser.parse_args()

    seqs = load_split_sequences(args.split)

    job_arg_strs = []
    for seq in seqs:
        num_imgs = len(glob.glob(f"{IMG_ROOT}/{seq}/*.jpg"))
        splits = list(range(0, num_imgs, args.seq_len))
        splits[-1] = num_imgs  # just add the remainder to the last job
        for start, end in zip(splits[:-1], splits[1:]):
            sel_tracks = select_phalp_tracks(seq, args.split, start, end)
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
        f"../job_specs/{args.prefix}_{args.split}_len_{args.seq_len}.txt", "w"
    ) as f:
        f.write("\n".join(job_arg_strs))
